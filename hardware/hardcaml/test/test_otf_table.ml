open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

let config ~lanes ~in_bits ~out_bits ~tables seed : (module Otf_table.Config) =
  let rng = Random.State.make [| seed |] in
  let count = tables in
  (module struct
    let lanes = lanes
    let in_bits = in_bits
    let out_bits = out_bits

    let tables =
      List.init count ~f:(fun _ ->
        Array.of_list (Model.random_ints rng ~bits:out_bits (1 lsl in_bits)))
    ;;

    let lane_table = List.init lanes ~f:(fun _ -> Random.State.int rng count)
  end)
;;

module Make_test (Config : Otf_table.Config) = struct
  open Config
  module Dut = Otf_table.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let expected x =
    List.map2_exn x lane_table ~f:(fun code table ->
      (List.nth_exn tables table).(code + (1 lsl (in_bits - 1))))
  ;;

  (* One stimulus per cycle; every answer must arrive exactly one cycle later. *)
  let run stimuli =
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let previous = ref None in
         let checked = ref 0 in
         List.iter
           (stimuli @ [ false, List.init lanes ~f:(Fn.const 0) ])
           ~f:(fun (valid, x) ->
             inputs.x.value := Stream_bench.of_signed_lanes ~lane_bits:in_bits x;
             inputs.x.valid := Bits.of_bool valid;
             Lws.cycle h;
             let got_valid = Bits.to_bool !(o.y.valid) in
             (match !previous with
              | Some sent ->
                if not got_valid then raise_s [%message "valid is not the input delayed"];
                [%test_result: int list]
                  (Stream_bench.signed_lanes ~lane_bits:out_bits !(o.y.value))
                  ~expect:(expected sent);
                incr checked
              | None -> if got_valid then raise_s [%message "valid without an input"]);
             previous := Option.some_if valid x);
         !checked)
  ;;

  (* Lane [k] walks every code from its own offset, with [valid] high [duty] percent. *)
  let every_code ~duty =
    let rng = Random.State.make [| duty |] in
    List.init (3 lsl in_bits) ~f:(fun step ->
      ( Random.State.int rng 100 < duty
      , List.init lanes ~f:(fun k -> Model.wrap ~bits:in_bits ((step / 3) + (37 * k))) ))
  ;;
end

let%expect_test "every code of every lane reads its own table, one cycle later" =
  List.iter
    [ "one int8 table", config ~lanes:1 ~in_bits:8 ~out_bits:8 ~tables:1 1
    ; "per channel tables", config ~lanes:6 ~in_bits:8 ~out_bits:8 ~tables:3 2
    ; "narrow in, wide out", config ~lanes:3 ~in_bits:4 ~out_bits:12 ~tables:2 3
    ; "one bit", config ~lanes:2 ~in_bits:1 ~out_bits:3 ~tables:2 4
    ; "ten bits", config ~lanes:2 ~in_bits:10 ~out_bits:8 ~tables:2 5
    ]
    ~f:(fun (name, (module Config)) ->
      let module T = Make_test (Config) in
      let checked = List.map [ 100; 40 ] ~f:(fun duty -> T.run (T.every_code ~duty)) in
      print_s [%message name ~latency:(T.Dut.latency : int) (checked : int list)]);
  [%expect
    {|
    ("one int8 table" (latency 1) (checked (768 312)))
    ("per channel tables" (latency 1) (checked (768 312)))
    ("narrow in, wide out" (latency 1) (checked (48 23)))
    ("one bit" (latency 1) (checked (6 2)))
    ("ten bits" (latency 1) (checked (3072 1234)))
    |}]
;;

let%expect_test "waveform of a three bit table on two lanes" =
  let module Config = struct
    let lanes = 2
    let in_bits = 3
    let out_bits = 4
    let tables = [ [| -8; -6; -4; -2; 0; 2; 4; 6 |]; [| 7; 6; 5; 4; 3; 2; 1; 0 |] ]
    let lane_table = [ 0; 1 ]
  end
  in
  let module T = Make_test (Config) in
  let display_rules =
    [ Display_rule.port_name_matches (Glob "x$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "y$*") ~wave_format:(Bit_or Hex)
    ]
  in
  T.Harness.run
    ~create:T.Dut.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print ~display_rules ~signals_width:9 ~display_width:90 ~wave_width:2 waves)
    (fun (h @ local) ~inputs ~outputs:_ ->
      Stream_bench.clear h inputs.clocking.clear;
      List.iter
        [ Some [ -4; -4 ]; Some [ 0; 3 ]; None; Some [ 3; -1 ] ]
        ~f:(fun x ->
          inputs.x.valid := Bits.of_bool (Option.is_some x);
          inputs.x.value
          := Stream_bench.of_signed_lanes ~lane_bits:3 (Option.value x ~default:[ 0; 0 ]);
          Lws.cycle h);
      inputs.x.valid := Bits.gnd;
      Lws.cycle ~n:3 h);
  [%expect
    {|
    ┌Signals┐┌Waves──────────────────────────────────────────────────────────────────────────┐
    │x$valid││            ┌───────────┐     ┌─────┐                                          │
    │       ││────────────┘           └─────┘     └───────────────────────                   │
    │       ││────────────┬─────┬─────┬─────┬─────────────────────────────                   │
    │x$value││ 00         │24   │18   │00   │3B                                              │
    │       ││────────────┴─────┴─────┴─────┴─────────────────────────────                   │
    │y$valid││                  ┌───────────┐     ┌─────┐                                    │
    │       ││──────────────────┘           └─────┘     └─────────────────                   │
    │       ││──────┬───────────┬─────┬─────┬─────┬───────────────────────                   │
    │y$value││ 00   │30         │78   │00   │30   │46                                        │
    │       ││──────┴───────────┴─────┴─────┴─────┴───────────────────────                   │
    └───────┘└───────────────────────────────────────────────────────────────────────────────┘
    |}]
;;
