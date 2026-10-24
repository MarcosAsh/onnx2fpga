open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

module Make_test (Config : Otf_dup.Config) = struct
  open Config
  module Dut = Otf_dup.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  (* Valids and ready follow a per-branch mask, modelled cycle by cycle. *)
  let random_traffic ?random_initial_state ~seed ~beats ~in_duty ~out_duties () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ?random_initial_state
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let sent = Stream_bench.beats ~rng ~width beats in
         let source =
           Stream_bench.Source.create
             ~data:inputs.s.data
             ~valid:inputs.s.valid
             ~ready:o.s.ready
             ~duty:in_duty
             sent
         in
         let sinks =
           List.map3_exn o.m inputs.m out_duties ~f:(fun branch dest duty ->
             Stream_bench.Sink.create
               ~data:branch.data
               ~valid:branch.valid
               ~ready:dest.ready
               ~duty
               ~wanted:beats)
         in
         let served = Array.create ~len:fanout false in
         let each_cycle () =
           let valid = Bits.to_bool !(inputs.s.valid) in
           let readies = List.map inputs.m ~f:(fun dest -> Bits.to_bool !(dest.ready)) in
           List.iteri o.m ~f:(fun f branch ->
             [%test_result: bool]
               (Bits.to_bool !(branch.valid))
               ~expect:(valid && not served.(f)));
           let all_taken =
             List.for_alli readies ~f:(fun f ready -> served.(f) || ready)
           in
           [%test_result: bool] (Bits.to_bool !(o.s.ready)) ~expect:all_taken;
           if valid
           then
             List.iteri readies ~f:(fun f ready ->
               served.(f) <- (not all_taken) && (served.(f) || ready))
         in
         let cycles = Stream_bench.run h ~rng ~each_cycle ~sources:[ source ] ~sinks () in
         List.iter sinks ~f:(fun sink ->
           [%test_result: Bits.t list] (Stream_bench.Sink.received sink) ~expect:sent);
         cycles)
  ;;
end

let%expect_test "every branch gets every beat under independent backpressure" =
  List.iter
    [ 2, 8; 3, 16 ]
    ~f:(fun (fanout, width) ->
      let module T =
        Make_test (struct
          let width = width
          let fanout = fanout
        end)
      in
      let duties = [ 100, [ 100; 100; 100 ]; 80, [ 30; 90; 60 ]; 40, [ 70; 20; 100 ] ] in
      let cycles =
        List.map duties ~f:(fun (in_duty, out_duties) ->
          T.random_traffic
            ~seed:fanout
            ~beats:200
            ~in_duty
            ~out_duties:(List.take out_duties fanout)
            ())
      in
      print_s [%message (fanout : int) (cycles : int list)]);
  [%expect
    {|
    ((fanout 2) (cycles (200 923 2386)))
    ((fanout 3) (cycles (200 930 2621)))
    |}]
;;

let%expect_test "random initial state is overwritten by the clear" =
  let module T =
    Make_test (struct
      let width = 4
      let fanout = 2
    end)
  in
  let cycles =
    T.random_traffic
      ~random_initial_state:`All
      ~seed:1
      ~beats:100
      ~in_duty:70
      ~out_duties:[ 50; 50 ]
      ()
  in
  print_s [%message (cycles : int)];
  [%expect {| (cycles 400) |}]
;;

module Two = Otf_dup.Make (struct
    let width = 8
    let fanout = 2
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Two.I) (Two.O)

(* Branch 1 takes 0xa1 two cycles after branch 0, which releases the input. *)
let%expect_test "waveform of branches accepting on different cycles" =
  let display_rules =
    [ Display_rule.port_name_matches (Glob "s$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "m$*") ~wave_format:(Bit_or Hex)
    ]
  in
  Harness.run
    ~create:Two.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print
        ~display_rules
        ~signals_width:11
        ~display_width:66
        ~wave_width:1
        waves)
    (fun (h @ local) ~inputs ~outputs:_ ->
      Stream_bench.clear h inputs.clocking.clear;
      let step data readies =
        inputs.s.valid := Bits.vdd;
        Bits.(inputs.s.data <--. data);
        List.iter2_exn inputs.m readies ~f:(fun dest ready ->
          dest.ready := Bits.of_bool ready);
        Lws.cycle h
      in
      List.iter
        [ 0xa1, [ true; false ]
        ; 0xa1, [ false; false ]
        ; 0xa1, [ false; true ]
        ; 0xb2, [ true; true ]
        ]
        ~f:(fun (data, readies) -> step data readies);
      inputs.s.valid := Bits.gnd;
      List.iter inputs.m ~f:(fun dest -> dest.ready := Bits.gnd);
      Lws.cycle ~n:2 h);
  [%expect
    {|
    ┌Signals──┐┌Waves────────────────────────────────────────────────┐
    │         ││────────┬───────────┬───────────────                 │
    │s$data   ││ 00     │A1         │B2                              │
    │         ││────────┴───────────┴───────────────                 │
    │s$valid  ││        ┌───────────────┐                            │
    │         ││────────┘               └───────────                 │
    │s$ready  ││                ┌───────┐                            │
    │         ││────────────────┘       └───────────                 │
    │m$ready_0││        ┌───┐       ┌───┐                            │
    │         ││────────┘   └───────┘   └───────────                 │
    │m$ready_1││                ┌───────┐                            │
    │         ││────────────────┘       └───────────                 │
    │         ││────────┬───────────┬───────────────                 │
    │m$data_0 ││ 00     │A1         │B2                              │
    │         ││────────┴───────────┴───────────────                 │
    │         ││────────┬───────────┬───────────────                 │
    │m$data_1 ││ 00     │A1         │B2                              │
    │         ││────────┴───────────┴───────────────                 │
    │m$valid_0││        ┌───┐       ┌───┐                            │
    │         ││────────┘   └───────┘   └───────────                 │
    │m$valid_1││        ┌───────────────┐                            │
    │         ││────────┘               └───────────                 │
    └─────────┘└─────────────────────────────────────────────────────┘
    |}]
;;
