open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

let debug = false

module Make_test (Config : Otf_fifo.Config) = struct
  module Dut = Otf_fifo.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  (* [s.ready] and [m.valid] must match an occupancy count every cycle. *)
  let random_traffic ~seed ~beats ~in_duty ~out_duty () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ~random_initial_state:`All
      ~waves_config:
        (if debug
         then Hardcaml_test_harness.Waves_config.to_test_directory ()
         else Hardcaml_test_harness.Waves_config.no_waves)
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
        let o = Before_and_after_edge.before_edge outputs in
        Stream_bench.clear h inputs.clocking.clear;
        let sent = Stream_bench.beats ~rng ~width:Config.width beats in
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
        let count = ref 0 in
        let each_cycle () =
          let expected_ready = !count <> Config.depth in
          let expected_valid = !count <> 0 in
          [%test_result: bool] (Bits.to_bool !(o.s.ready)) ~expect:expected_ready;
          [%test_result: bool] (Bits.to_bool !(o.m.valid)) ~expect:expected_valid;
          let push = Bits.to_bool !(inputs.s.valid) && expected_ready in
          let pop = expected_valid && Bits.to_bool !(inputs.m.ready) in
          count := !count + Bool.to_int push - Bool.to_int pop
        in
        let cycles =
          Stream_bench.run h ~rng ~each_cycle ~sources:[ source ] ~sinks:[ sink ] ()
        in
        [%test_result: Bits.t list] (Stream_bench.Sink.received sink) ~expect:sent;
        cycles)
  ;;
end

let%expect_test "every depth passes beats through in order under random backpressure" =
  List.iter
    [ 1, 8; 2, 8; 3, 5; 5, 16 ]
    ~f:(fun (depth, width) ->
      let module T =
        Make_test (struct
          let width = width
          let depth = depth
        end)
      in
      let cycles =
        List.map
          [ 100, 100; 50, 50; 20, 90; 90, 20 ]
          ~f:(fun (in_duty, out_duty) ->
            T.random_traffic ~seed:depth ~beats:200 ~in_duty ~out_duty ())
      in
      print_s [%message (depth : int) (cycles : int list)]);
  [%expect
    {|
    ((depth 1) (cycles (400 825 1290 1313)))
    ((depth 2) (cycles (201 565 997 1035)))
    ((depth 3) (cycles (201 504 1011 1122)))
    ((depth 5) (cycles (201 454 1060 1031)))
    |}]
;;

module Two = Make_test (struct
    let width = 8
    let depth = 2
  end)

(* Full holds [s.ready] low while a beat leaves; empty shows a stale word. *)
let%expect_test "waveform of a depth two buffer filling and draining" =
  let display_rules =
    [ Display_rule.port_name_matches (Glob "s$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "m$*") ~wave_format:(Bit_or Hex)
    ]
  in
  Two.Harness.run
    ~create:Two.Dut.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print ~display_rules ~signals_width:9 ~display_width:72 ~wave_width:1 waves)
    (fun (h @ local) ~inputs ~outputs:_ ->
      Stream_bench.clear h inputs.clocking.clear;
      let offer ?(ready = false) data =
        inputs.s.valid := Bits.vdd;
        Bits.(inputs.s.data <--. data);
        inputs.m.ready := Bits.of_bool ready;
        Lws.cycle h
      in
      List.iter [ 0x11; 0x22; 0x33 ] ~f:(fun data -> offer data);
      offer ~ready:true 0x33;
      offer ~ready:true 0x33;
      inputs.s.valid := Bits.gnd;
      inputs.m.ready := Bits.vdd;
      Lws.cycle ~n:3 h;
      offer ~ready:true 0x44;
      inputs.s.valid := Bits.gnd;
      Lws.cycle ~n:3 h);
  [%expect
    {|
    ┌Signals┐┌Waves────────────────────────────────────────────────────────┐
    │       ││────────┬───┬───┬───────────────────────┬─────────────────── │
    │s$data ││ 00     │11 │22 │33                     │44                  │
    │       ││────────┴───┴───┴───────────────────────┴─────────────────── │
    │s$valid││        ┌───────────────────┐           ┌───┐                │
    │       ││────────┘                   └───────────┘   └─────────────── │
    │s$ready││────────────────┐       ┌─────────────────────────────────── │
    │       ││                └───────┘                                    │
    │m$ready││                    ┌─────────────────────────────────────── │
    │       ││────────────────────┘                                        │
    │       ││────────────┬───────────┬───┬───┬───────────┬───┬─────────── │
    │m$data ││ 00         │11         │22 │33 │22         │44 │33          │
    │       ││────────────┴───────────┴───┴───┴───────────┴───┴─────────── │
    │m$valid││            ┌───────────────────┐           ┌───┐            │
    │       ││────────────┘                   └───────────┘   └─────────── │
    └───────┘└─────────────────────────────────────────────────────────────┘
    |}]
;;
