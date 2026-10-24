open! Core
open! Hardcaml
open Onnx2fpga_hardcaml

module type Params = sig
  include Otf_act.Config

  val mult_values : int list
end

module Make_test (Config : Params) = struct
  open Config
  module Dut = Otf_act.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let nf_count = channels / pe

  let expected beats =
    List.mapi beats ~f:(fun index beat ->
      let nf = index % nf_count in
      Stream_bench.signed_lanes ~lane_bits:in_bits beat
      |> List.mapi ~f:(fun p x ->
        Model.requant
          ~acc_bits:(in_bits + 2)
          ~mult_bits
          ~shift
          ~out_min
          ~out_max
          ~out_zp
          ~acc:(Model.wrap ~bits:(in_bits + 2) (x - in_zp))
          ~mult:(List.nth_exn mult_values ((p * nf_count) + nf)))
      |> Stream_bench.of_signed_lanes ~lane_bits:out_bits)
  ;;

  (* The data path is a wire, so the handshake is too: checked on every cycle. *)
  let random_traffic ~seed ~beats ~in_duty ~out_duty () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let sent = Stream_bench.beats ~rng ~width:(pe * in_bits) beats in
         let source =
           Stream_bench.Source.create
             ~data:inputs.s.data
             ~valid:inputs.s.valid
             ~ready:o.s.ready
             ~duty:in_duty
             sent
         in
         let sink =
           Stream_bench.Sink.create
             ~data:o.m.data
             ~valid:o.m.valid
             ~ready:inputs.m.ready
             ~duty:out_duty
             ~wanted:beats
         in
         let each_cycle () =
           [%test_result: Bits.t] !(o.s.ready) ~expect:!(inputs.m.ready);
           [%test_result: Bits.t] !(o.m.valid) ~expect:!(inputs.s.valid)
         in
         let cycles =
           Stream_bench.run h ~rng ~each_cycle ~sources:[ source ] ~sinks:[ sink ] ()
         in
         [%test_result: Bits.t list]
           (Stream_bench.Sink.received sink)
           ~expect:(expected sent);
         cycles)
  ;;
end

let config
  ?(in_zp = 0)
  ?(out_zp = 0)
  ?(shift = 16)
  ?(out_min = -128)
  ?(out_max = 127)
  ?(in_bits = 20)
  ?(mult_bits = 18)
  ?(out_bits = 8)
  ~channels
  ~pe
  seed
  : (module Params)
  =
  let rng = Random.State.make [| seed |] in
  let mult_values = Model.random_ints rng ~bits:mult_bits channels in
  (module struct
    let channels = channels
    let pe = pe
    let in_bits = in_bits
    let out_bits = out_bits
    let mult_bits = mult_bits
    let shift = shift
    let out_min = out_min
    let out_max = out_max
    let in_zp = in_zp
    let out_zp = out_zp
    let mult_values = mult_values
    let mult = Model.words_of_ints ~bits:mult_bits mult_values
  end)
;;

let%expect_test "per channel rescale under random backpressure" =
  List.iter
    [ "folded", config ~channels:8 ~pe:1 ~in_zp:3 ~out_zp:(-5) 1
    ; "pe4", config ~channels:8 ~pe:4 ~shift:0 ~in_zp:(-7) 2
    ; "unfolded relu", config ~channels:6 ~pe:6 ~out_min:0 ~shift:30 3
    ; ( "narrow"
      , config
          ~channels:3
          ~pe:1
          ~in_bits:8
          ~mult_bits:4
          ~shift:2
          ~out_bits:4
          ~out_min:(-8)
          ~out_max:7
          4 )
    ]
    ~f:(fun (name, (module Config)) ->
      let module T = Make_test (Config) in
      let cycles =
        List.map
          [ 100, 100; 50, 50; 20, 90; 90, 20 ]
          ~f:(fun (in_duty, out_duty) ->
            T.random_traffic ~seed:7 ~beats:240 ~in_duty ~out_duty ())
      in
      print_s [%message name (cycles : int list)]);
  [%expect
    {|
    (folded (cycles (240 935 1286 1197)))
    (pe4 (cycles (240 827 1298 1239)))
    ("unfolded relu" (cycles (240 982 1338 1173)))
    (narrow (cycles (240 935 1286 1197)))
    |}]
;;
