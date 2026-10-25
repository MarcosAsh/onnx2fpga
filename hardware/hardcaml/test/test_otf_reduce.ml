open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

let config ?(last = false) ?(stages = Otf_reduce.Stages.none) ~lanes ~bits ~op seed
  : (module Otf_reduce.Config)
  =
  let rng = Random.State.make [| seed |] in
  (module struct
    let lanes = lanes
    let bits = bits
    let op = op
    let last = last
    let zero_point = Random.State.int_incl rng (-20) 20
    let mult = (1 lsl 15) + Random.State.int rng (1 lsl 15)
    let mult_bits = 18
    let acc_bits = bits + Int.ceil_log2 (lanes + 1) + 6
    let shift = 16 + Int.ceil_log2 lanes
    let out_zp = Random.State.int_incl rng (-10) 10
    let out_min = -(1 lsl (bits - 1))
    let out_max = (1 lsl (bits - 1)) - 1
    let stages = stages
  end)
;;

module Make_test (Config : Otf_reduce.Config) = struct
  open Config
  module Dut = Otf_reduce.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  (* The compiler's ReduceUnit.execute: the first index of a tie, or the last. *)
  let expected x =
    let codes = Stream_bench.signed_lanes ~lane_bits:bits x in
    let best ~larger =
      List.foldi
        codes
        ~init:(0, List.hd_exn codes)
        ~f:(fun k (at, best) code ->
          let better = if larger then code > best else code < best in
          if better || (last && code = best) then k, code else at, best)
    in
    let index (k, _) = Bits.of_unsigned_int ~width:Dut.out_bits k in
    let value (_, v) = Bits.of_signed_int ~width:Dut.out_bits v in
    match op with
    | Arg_max -> index (best ~larger:true)
    | Arg_min -> index (best ~larger:false)
    | Max -> value (best ~larger:true)
    | Min -> value (best ~larger:false)
    | Sum ->
      let acc = List.sum (module Int) codes ~f:Fn.id - (lanes * zero_point) in
      Bits.of_signed_int
        ~width:bits
        (Model.python_requant ~shift ~out_min ~out_max ~out_zp ~acc ~mult)
  ;;

  let run ~duty =
    let rng = Random.State.make [| duty; lanes |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         Fixed_bench.check
           h
           ~x:inputs.x
           ~y:o.y
           ~latency:Dut.latency
           ~expected
           (List.init 200 ~f:(fun _ ->
              ( Random.State.int rng 100 < duty
              , Stream_bench.random_bits rng ~width:(lanes * bits) ))))
  ;;
end

let%expect_test "every reduction matches the compiler's model, with and without stages" =
  List.iter
    [ ( "argmax"
      , 10
      , 8
      , Otf_reduce.Op.Arg_max
      , false
      , { Otf_reduce.Stages.tree = 3; rescale = 0 } )
    ; "argmax, last of a tie", 10, 3, Arg_max, true, { tree = 2; rescale = 0 }
    ; "argmin", 5, 3, Arg_min, false, { tree = 2; rescale = 0 }
    ; "argmin, last of a tie", 2, 2, Arg_min, true, { tree = 0; rescale = 0 }
    ; "max", 16, 8, Max, false, { tree = 3; rescale = 0 }
    ; "min, one lane", 1, 8, Min, false, Otf_reduce.Stages.none
    ; "sum", 7, 8, Sum, false, { tree = 2; rescale = 3 }
    ]
    ~f:(fun (name, lanes, bits, op, last, most) ->
      let runs =
        List.map [ Otf_reduce.Stages.none; most ] ~f:(fun stages ->
          let (module Config) = config ~last ~stages ~lanes ~bits ~op 5 in
          let module T = Make_test (Config) in
          T.Dut.latency, List.map [ 100; 40 ] ~f:(fun duty -> T.run ~duty))
      in
      let latencies, checked = List.unzip runs in
      print_s
        [%message name (latencies : int list) ~checked:(List.hd_exn checked : int list)]);
  [%expect
    {|
    (argmax (latencies (1 4)) (checked (200 91)))
    ("argmax, last of a tie" (latencies (1 3)) (checked (200 76)))
    (argmin (latencies (1 3)) (checked (200 82)))
    ("argmin, last of a tie" (latencies (1 1)) (checked (200 73)))
    (max (latencies (1 4)) (checked (200 90)))
    ("min, one lane" (latencies (1 1)) (checked (200 83)))
    (sum (latencies (2 7)) (checked (200 82)))
    |}]
;;

let%expect_test "waveform of an argmax over four lanes with a tie" =
  let (module Config) = config ~lanes:4 ~bits:4 ~op:Arg_max 1 in
  let module T = Make_test (Config) in
  let display_rules =
    [ Display_rule.port_name_matches (Glob "x$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "y$*") ~wave_format:(Bit_or Unsigned_int)
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
        [ Some [ 1; 7; -3; 7 ]; Some [ -8; -8; -2; -8 ]; None; Some [ 0; 0; 0; 1 ] ]
        ~f:(fun x ->
          inputs.x.valid := Bits.of_bool (Option.is_some x);
          inputs.x.value
          := Stream_bench.of_signed_lanes
               ~lane_bits:4
               (Option.value x ~default:[ 0; 0; 0; 0 ]);
          Lws.cycle h);
      inputs.x.valid := Bits.gnd;
      Lws.cycle ~n:3 h);
  [%expect
    {|
    ┌Signals┐┌Waves──────────────────────────────────────────────────────────────────────────┐
    │x$valid││            ┌───────────┐     ┌─────┐                                          │
    │       ││────────────┘           └─────┘     └───────────────────────                   │
    │       ││────────────┬─────┬─────┬─────┬─────────────────────────────                   │
    │x$value││ 0000       │7D71 │8E88 │0000 │1000                                            │
    │       ││────────────┴─────┴─────┴─────┴─────────────────────────────                   │
    │y$valid││                  ┌───────────┐     ┌─────┐                                    │
    │       ││──────────────────┘           └─────┘     └─────────────────                   │
    │       ││──────────────────┬─────┬─────┬─────┬───────────────────────                   │
    │y$value││ 0                │1    │2    │0    │3                                         │
    │       ││──────────────────┴─────┴─────┴─────┴───────────────────────                   │
    └───────┘└───────────────────────────────────────────────────────────────────────────────┘
    |}]
;;
