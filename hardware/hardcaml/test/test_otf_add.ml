open! Core
open! Hardcaml
open Onnx2fpga_hardcaml

module Make_test (Config : Otf_add.Config) = struct
  open Config
  module Dut = Otf_add.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let expected left right =
    List.map2_exn left right ~f:(fun a b ->
      List.map2_exn
        (Stream_bench.signed_lanes ~lane_bits:in_bits a)
        (Stream_bench.signed_lanes ~lane_bits:in_bits b)
        ~f:(fun a b ->
          let wrap = Model.wrap ~bits:acc_bits in
          let acc = wrap (wrap (a * left_mult) + wrap (b * right_mult) + bias) in
          Model.requant
            ~acc_bits
            ~mult_bits:2
            ~shift
            ~out_min
            ~out_max
            ~out_zp
            ~acc
            ~mult:1)
      |> Stream_bench.of_signed_lanes ~lane_bits:out_bits)
  ;;

  (* Both inputs arrive with their own random gaps; the join takes them together. *)
  let random_traffic ~seed ~beats ~duties:(a_duty, b_duty, m_duty) () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let width = pe * in_bits in
         let left = Stream_bench.beats ~rng ~width beats in
         let right = Stream_bench.beats ~rng ~width beats in
         let a =
           Stream_bench.Source.create
             ~data:inputs.a.data
             ~valid:inputs.a.valid
             ~ready:o.a.ready
             ~duty:a_duty
             left
         in
         let b =
           Stream_bench.Source.create
             ~data:inputs.b.data
             ~valid:inputs.b.valid
             ~ready:o.b.ready
             ~duty:b_duty
             right
         in
         let sink =
           Stream_bench.Sink.create
             ~data:o.m.data
             ~valid:o.m.valid
             ~ready:inputs.m.ready
             ~duty:m_duty
             ~wanted:beats
         in
         let each_cycle () =
           let both = Bits.(!(inputs.a.valid) &: !(inputs.b.valid)) in
           [%test_result: Bits.t] !(o.m.valid) ~expect:both;
           [%test_result: Bits.t] !(o.a.ready) ~expect:Bits.(both &: !(inputs.m.ready));
           [%test_result: Bits.t] !(o.b.ready) ~expect:Bits.(both &: !(inputs.m.ready))
         in
         let cycles =
           Stream_bench.run h ~rng ~each_cycle ~sources:[ a; b ] ~sinks:[ sink ] ()
         in
         [%test_result: Bits.t list]
           (Stream_bench.Sink.received sink)
           ~expect:(expected left right);
         cycles)
  ;;
end

let config
  ?(bias = 0)
  ?(out_zp = 0)
  ?(acc_bits = 27)
  ?(out_min = -128)
  ~pe
  ~left
  ~right
  ~shift
  ()
  : (module Otf_add.Config)
  =
  (module struct
    let pe = pe
    let in_bits = 8
    let out_bits = 8
    let acc_bits = acc_bits
    let left_mult = left
    let right_mult = right
    let shift = shift
    let out_min = out_min
    let out_max = 127
    let bias = bias
    let out_zp = out_zp
  end)
;;

let%expect_test "two streams join under independent backpressure" =
  List.iter
    [ "residual", config ~pe:1 ~left:98284 ~right:51356 ~shift:17 ()
    ; "biased", config ~pe:2 ~left:3000 ~right:(-2500) ~shift:10 ~bias:12345 ~out_zp:9 ()
    ; "unshifted relu", config ~pe:4 ~left:1 ~right:1 ~shift:0 ~acc_bits:10 ~out_min:0 ()
    ; "wrapping", config ~pe:1 ~left:1500 ~right:1500 ~shift:8 ~acc_bits:12 ()
    ]
    ~f:(fun (name, (module Config)) ->
      let module T = Make_test (Config) in
      let cycles =
        List.map
          [ 100, 100, 100; 50, 80, 50; 90, 20, 60; 30, 40, 40 ]
          ~f:(fun duties -> T.random_traffic ~seed:3 ~beats:200 ~duties ())
      in
      print_s [%message name (cycles : int list)]);
  [%expect
    {|
    (residual (cycles (200 1072 1948 4331)))
    (biased (cycles (200 1072 1948 4331)))
    ("unshifted relu" (cycles (200 1051 1861 4555)))
    (wrapping (cycles (200 1072 1948 4331)))
    |}]
;;
