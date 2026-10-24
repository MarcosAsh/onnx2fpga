open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

module type Params = sig
  include Otf_dense.Config

  (* Indexed [c][k], the weight from input k to output channel c. *)
  val weight_values : int array array
  val bias_values : int array
  val mult_values : int array
end

let config
  ?(act_bits = 8)
  ?(wgt_bits = 8)
  ?(acc_bits = 20)
  ?(out_bits = 8)
  ?(mult_bits = 18)
  ?(shift = 20)
  ?(out_min = -128)
  ?(out_max = 127)
  ?(out_zp = 0)
  ~mw
  ~mh
  seed
  : (module Params)
  =
  let rng = Random.State.make [| seed |] in
  let weight_values =
    Array.init mh ~f:(fun _ -> Array.of_list (Model.random_ints rng ~bits:wgt_bits mw))
  in
  let bias_values = Array.of_list (Model.random_ints rng ~bits:(acc_bits - 4) mh) in
  let mult_values = Array.of_list (Model.random_ints rng ~bits:mult_bits mh) in
  (module struct
    let mw = mw
    let mh = mh
    let act_bits = act_bits
    let wgt_bits = wgt_bits
    let acc_bits = acc_bits
    let out_bits = out_bits
    let mult_bits = mult_bits
    let shift = shift
    let out_min = out_min
    let out_max = out_max
    let out_zp = out_zp
    let weight_values = weight_values
    let bias_values = bias_values
    let mult_values = mult_values

    let weights =
      Array.to_list weight_values
      |> List.map ~f:(fun row ->
        Stream_bench.of_signed_lanes ~lane_bits:wgt_bits (Array.to_list row))
    ;;

    let bias = Model.words_of_ints ~bits:acc_bits (Array.to_list bias_values)
    let mult = Model.words_of_ints ~bits:mult_bits (Array.to_list mult_values)
  end)
;;

module Make_test (Config : Params) = struct
  open Config
  module Dut = Otf_dense.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let expected x =
    List.init mh ~f:(fun c ->
      let dot =
        List.foldi x ~init:bias_values.(c) ~f:(fun k sum a ->
          sum + (a * weight_values.(c).(k)))
      in
      Model.requant
        ~acc_bits
        ~mult_bits
        ~shift
        ~out_min
        ~out_max
        ~out_zp
        ~acc:(Model.wrap ~bits:acc_bits dot)
        ~mult:mult_values.(c))
    |> Stream_bench.of_signed_lanes ~lane_bits:out_bits
  ;;

  (* A vector every cycle, valid on [duty] percent, each due exactly [latency] later. *)
  let run ?random_initial_state ~seed ~cycles ~duty () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ?random_initial_state
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         inputs.clocking.clear := Bits.vdd;
         Lws.cycle h;
         inputs.clocking.clear := Bits.gnd;
         let in_flight = Queue.create () in
         let checked = ref 0 in
         List.iter (List.range 0 cycles) ~f:(fun cycle ->
           let x = Model.random_ints rng ~bits:act_bits mw in
           let valid = Random.State.int rng 100 < duty in
           inputs.x.value := Stream_bench.of_signed_lanes ~lane_bits:act_bits x;
           inputs.x.valid := Bits.of_bool valid;
           Lws.cycle h;
           Queue.enqueue in_flight (valid, x);
           let got_valid = Bits.to_bool !(o.y.valid) in
           if Queue.length in_flight <= Dut.latency
           then (
             if got_valid then raise_s [%message "valid before any input" (cycle : int)])
           else (
             let sent_valid, sent = Queue.dequeue_exn in_flight in
             if Bool.( <> ) got_valid sent_valid
             then raise_s [%message "valid is not the input delayed" (cycle : int)];
             if sent_valid
             then (
               [%test_result: Bits.t] !(o.y.value) ~expect:(expected sent);
               incr checked)));
         !checked)
  ;;
end

let%expect_test "every vector comes out exactly three cycles later" =
  List.iter
    [ "mlp first layer", config ~mw:32 ~mh:64 1
    ; "relu at a zero point", config ~mw:16 ~mh:8 ~out_zp:(-5) ~out_min:(-5) 2
    ; "one by one", config ~mw:1 ~mh:1 ~acc_bits:17 ~shift:4 3
    ; "no shift", config ~mw:4 ~mh:3 ~acc_bits:18 ~shift:0 ~mult_bits:3 4
    ; "quotient always zero", config ~mw:3 ~mh:2 ~acc_bits:18 ~shift:36 5
    ; ( "narrow"
      , config
          ~mw:6
          ~mh:5
          ~act_bits:4
          ~wgt_bits:3
          ~acc_bits:10
          ~mult_bits:5
          ~shift:3
          ~out_bits:4
          ~out_min:(-8)
          ~out_max:7
          6 )
    ]
    ~f:(fun (name, (module Config)) ->
      let module T = Make_test (Config) in
      let checked =
        List.map [ 100; 60; 5 ] ~f:(fun duty -> T.run ~seed:7 ~cycles:300 ~duty ())
      in
      print_s [%message name ~latency:(T.Dut.latency : int) (checked : int list)]);
  [%expect
    {|
    ("mlp first layer" (latency 3) (checked (297 174 16)))
    ("relu at a zero point" (latency 3) (checked (297 177 13)))
    ("one by one" (latency 3) (checked (297 173 10)))
    ("no shift" (latency 3) (checked (297 188 14)))
    ("quotient always zero" (latency 3) (checked (297 187 12)))
    (narrow (latency 3) (checked (297 185 13)))
    |}]
;;

let%expect_test "random initial state is overwritten by the clear" =
  let (module Config) = config ~mw:8 ~mh:4 ~out_zp:3 8 in
  let module T = Make_test (Config) in
  let checked = T.run ~random_initial_state:`All ~seed:3 ~cycles:200 ~duty:50 () in
  print_s [%message (checked : int)];
  [%expect {| (checked 95) |}]
;;

let%expect_test "waveform of back to back vectors and a gap" =
  let (module Config) = config ~mw:2 ~mh:2 ~acc_bits:17 ~mult_bits:4 ~shift:6 9 in
  let module Dut = Otf_dense.Make (Config) in
  let module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O) in
  let display_rules =
    [ Display_rule.port_name_matches (Glob "x$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "y$*") ~wave_format:(Bit_or Hex)
    ]
  in
  Harness.run
    ~create:Dut.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print ~display_rules ~signals_width:9 ~display_width:90 ~wave_width:2 waves)
    (fun (h @ local) ~inputs ~outputs:_ ->
      inputs.clocking.clear := Bits.vdd;
      Lws.cycle h;
      inputs.clocking.clear := Bits.gnd;
      List.iter [ Some 0x0102; Some 0x7f80; None; Some 0x0303 ] ~f:(fun x ->
        inputs.x.valid := Bits.of_bool (Option.is_some x);
        Bits.(inputs.x.value <--. Option.value x ~default:0);
        Lws.cycle h);
      inputs.x.valid := Bits.gnd;
      Lws.cycle ~n:4 h);
  [%expect
    {|
    ┌Signals┐┌Waves──────────────────────────────────────────────────────────────────────────┐
    │x$valid││      ┌───────────┐     ┌─────┐                                                │
    │       ││──────┘           └─────┘     └─────────────────────────────                   │
    │       ││──────┬─────┬─────┬─────┬───────────────────────────────────                   │
    │x$value││ 0000 │0102 │7F80 │0000 │0303                                                  │
    │       ││──────┴─────┴─────┴─────┴───────────────────────────────────                   │
    │y$valid││                        ┌───────────┐     ┌─────┐                              │
    │       ││────────────────────────┘           └─────┘     └───────────                   │
    │       ││────────────┬───────────┬─────┬─────┬─────┬─────────────────                   │
    │y$value││ 0000       │9270       │956A │807F │9270 │976B                                │
    │       ││────────────┴───────────┴─────┴─────┴─────┴─────────────────                   │
    └───────┘└───────────────────────────────────────────────────────────────────────────────┘
    |}]
;;
