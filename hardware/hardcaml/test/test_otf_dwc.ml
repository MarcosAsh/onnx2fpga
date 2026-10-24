open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

module Make_test (Config : Otf_dwc.Config) = struct
  open Config
  module Dut = Otf_dwc.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let random_traffic ~seed ~elements ~in_duty ~out_duty () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let sent =
           Stream_bench.beats ~rng ~width:(in_elems * elem_bits) (elements / in_elems)
         in
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
             ~wanted:(elements / out_elems)
         in
         let cycles = Stream_bench.run h ~rng ~sources:[ source ] ~sinks:[ sink ] () in
         [%test_result: int list]
           (Model.elements ~bits:elem_bits (Stream_bench.Sink.received sink))
           ~expect:(Model.elements ~bits:elem_bits sent);
         cycles)
  ;;
end

let%expect_test "repacking widens, narrows and passes through under random backpressure" =
  List.iter
    [ 1, 4; 4, 1; 2, 2; 3, 9; 2, 16; 16, 2; 6, 2 ]
    ~f:(fun (in_elems, out_elems) ->
      let module T =
        Make_test (struct
          let in_elems = in_elems
          let out_elems = out_elems
          let elem_bits = 8
          let total_elems = 144
        end)
      in
      let cycles =
        List.map
          [ 100, 100; 50, 50; 20, 90; 90, 20 ]
          ~f:(fun (in_duty, out_duty) ->
            T.random_traffic ~seed:in_elems ~elements:288 ~in_duty ~out_duty ())
      in
      print_s [%message (in_elems : int) (out_elems : int) (cycles : int list)]);
  [%expect
    {|
    ((in_elems 1) (out_elems 4) (cycles (360 724 1559 698)))
    ((in_elems 4) (out_elems 1) (cycles (360 675 587 1437)))
    ((in_elems 2) (out_elems 2) (cycles (144 598 731 848)))
    ((in_elems 3) (out_elems 9) (cycles (128 237 513 264)))
    ((in_elems 2) (out_elems 16) (cycles (162 322 683 257)))
    ((in_elems 16) (out_elems 2) (cycles (162 318 270 769)))
    ((in_elems 6) (out_elems 2) (cycles (192 351 371 742)))
    |}]
;;

module Narrow = Make_test (struct
    let in_elems = 4
    let out_elems = 1
    let elem_bits = 8
    let total_elems = 8
  end)

(* The next beat is not taken until the last slice has gone. *)
let%expect_test "waveform of a beat narrowed four to one" =
  let display_rules =
    [ Display_rule.port_name_matches (Glob "s$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "m$*") ~wave_format:(Bit_or Hex)
    ]
  in
  Narrow.Harness.run
    ~create:Narrow.Dut.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print ~display_rules ~signals_width:9 ~display_width:90 ~wave_width:2 waves)
    (fun (h @ local) ~inputs ~outputs ->
      let o = Before_and_after_edge.before_edge outputs in
      Stream_bench.clear h inputs.clocking.clear;
      inputs.m.ready := Bits.vdd;
      List.iter [ 0x44332211; 0x88776655 ] ~f:(fun data ->
        inputs.s.valid := Bits.vdd;
        Bits.(inputs.s.data <--. data);
        let taken = ref false in
        while not !taken do
          Lws.cycle h;
          taken := Bits.to_bool !(o.s.ready)
        done);
      inputs.s.valid := Bits.gnd;
      Lws.cycle ~n:5 h);
  [%expect
    {|
    ┌Signals┐┌Waves──────────────────────────────────────────────────────────────────────────┐
    │       ││────────────┬─────┬────────────────────────────────────────────────────────────│
    │s$data ││ 00000000   │4433.│88776655                                                    │
    │       ││────────────┴─────┴────────────────────────────────────────────────────────────│
    │s$valid││            ┌───────────────────────────────────┐                              │
    │       ││────────────┘                                   └──────────────────────────────│
    │s$ready││──────────────────┐                       ┌─────┐                       ┌──────│
    │       ││                  └───────────────────────┘     └───────────────────────┘      │
    │m$ready││            ┌──────────────────────────────────────────────────────────────────│
    │       ││────────────┘                                                                  │
    │       ││──────────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──────│
    │m$data ││ 00               │11   │22   │33   │44   │11   │55   │66   │77   │88   │55    │
    │       ││──────────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴──────│
    │m$valid││                  ┌───────────────────────┐     ┌───────────────────────┐      │
    │       ││──────────────────┘                       └─────┘                       └──────│
    └───────┘└───────────────────────────────────────────────────────────────────────────────┘
    |}]
;;
