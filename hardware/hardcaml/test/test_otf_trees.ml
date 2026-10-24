open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

let random_int rng ~bits = List.hd_exn (Model.random_ints rng ~bits 1)

(* Splits test equality one time in [equal_one_in]; a branch stops early one time in five. *)
let rec random_tree rng ~depth ~features ~feature_bits ~leaf_bits ~equal_one_in
  : Otf_trees.Tree.t
  =
  let subtree () =
    random_tree rng ~depth:(depth - 1) ~features ~feature_bits ~leaf_bits ~equal_one_in
  in
  if depth = 0 || Random.State.int rng 5 = 0
  then Leaf (random_int rng ~bits:leaf_bits)
  else (
    let split =
      { Otf_trees.Split.feature = Random.State.int rng features
      ; threshold = random_int rng ~bits:feature_bits
      ; equal = Random.State.int rng equal_one_in = 0
      }
    in
    let if_true = subtree () in
    Node { split; if_true; if_false = subtree () })
;;

let config
  ?(feature_bits = 8)
  ?(leaf_bits = 8)
  ?(decision = false)
  ?(equal_one_in = 8)
  ~features
  ~trees
  ~depth
  seed
  : (module Otf_trees.Config)
  =
  let rng = Random.State.make [| seed |] in
  let trees =
    List.init trees ~f:(fun _ ->
      random_tree rng ~depth ~features ~feature_bits ~leaf_bits ~equal_one_in)
  in
  (module struct
    let features = features
    let feature_bits = feature_bits
    let leaf_bits = leaf_bits
    let sum_bits = leaf_bits + Int.ceil_log2 (List.length trees + 1)
    let trees = trees
    let bias = random_int rng ~bits:leaf_bits
    let decision = decision
  end)
;;

module Make_test (Config : Otf_trees.Config) = struct
  open Config
  module Dut = Otf_trees.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let expected x =
    let x = Array.of_list x in
    let total =
      List.fold trees ~init:bias ~f:(fun sum tree -> sum + Otf_trees.Tree.eval tree x)
    in
    if decision
    then Bits.of_bool (total >= 0)
    else Bits.of_signed_int ~width:sum_bits total
  ;;

  (* One stimulus per cycle; every answer must arrive exactly [latency] later. *)
  let run stimuli =
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         inputs.clocking.clear := Bits.vdd;
         Lws.cycle h;
         inputs.clocking.clear := Bits.gnd;
         let in_flight = Queue.create () in
         let checked = ref 0 in
         let ones = ref 0 in
         let cycle (valid, x) =
           inputs.x.value := Stream_bench.of_signed_lanes ~lane_bits:feature_bits x;
           inputs.x.valid := Bits.of_bool valid;
           Lws.cycle h;
           Queue.enqueue in_flight (valid, x);
           let got_valid = Bits.to_bool !(o.y.valid) in
           if Queue.length in_flight <= Dut.latency
           then (if got_valid then raise_s [%message "valid before any input"])
           else (
             let sent_valid, sent = Queue.dequeue_exn in_flight in
             if Bool.( <> ) got_valid sent_valid
             then raise_s [%message "valid is not the input delayed"];
             if sent_valid
             then (
               [%test_result: Bits.t] !(o.y.value) ~expect:(expected sent);
               if Bits.to_bool (Bits.lsb !(o.y.value)) then incr ones;
               incr checked))
         in
         List.iter stimuli ~f:cycle;
         List.iter (List.init Dut.latency ~f:Fn.id) ~f:(fun _ ->
           cycle (false, List.init features ~f:(Fn.const 0)));
         !checked, !ones)
  ;;

  let random ~seed ~cycles ~duty =
    let rng = Random.State.make [| seed |] in
    List.init cycles ~f:(fun _ ->
      Random.State.int rng 100 < duty, Model.random_ints rng ~bits:feature_bits features)
  ;;
end

let%expect_test "every answer comes out exactly latency cycles later" =
  List.iter
    [ "one leaf", config ~features:1 ~trees:1 ~depth:0 1
    ; "three stumps", config ~features:3 ~trees:3 ~depth:1 2
    ; "four trees", config ~features:4 ~trees:4 ~depth:3 3
    ; ( "decision"
      , config
          ~features:6
          ~trees:20
          ~depth:3
          ~decision:true
          ~leaf_bits:6
          ~equal_one_in:4
          4 )
    ; ( "hundred trees of depth four"
      , config ~features:8 ~trees:100 ~depth:4 ~feature_bits:16 ~leaf_bits:16 5 )
    ; "depth six", config ~features:5 ~trees:64 ~depth:6 ~feature_bits:12 6
    ]
    ~f:(fun (name, (module Config)) ->
      let module T = Make_test (Config) in
      let checked =
        List.map [ 100; 60; 5 ] ~f:(fun duty ->
          fst (T.run (T.random ~seed:7 ~cycles:300 ~duty)))
      in
      print_s [%message name ~latency:(T.Dut.latency : int) (checked : int list)]);
  [%expect
    {|
    ("one leaf" (latency 3) (checked (300 175 10)))
    ("three stumps" (latency 3) (checked (300 188 12)))
    ("four trees" (latency 4) (checked (300 190 14)))
    (decision (latency 5) (checked (300 187 13)))
    ("hundred trees of depth four" (latency 6) (checked (300 165 12)))
    ("depth six" (latency 6) (checked (300 178 18)))
    |}]
;;

let%expect_test "every input of a small ensemble, with equality splits" =
  List.iter [ false; true ] ~f:(fun decision ->
    let (module Config) =
      config
        ~features:2
        ~trees:6
        ~depth:3
        ~feature_bits:4
        ~leaf_bits:5
        ~equal_one_in:2
        ~decision
        8
    in
    let module T = Make_test (Config) in
    let all =
      List.init 256 ~f:(fun code ->
        true, [ Model.wrap ~bits:4 code; Model.wrap ~bits:4 (code lsr 4) ])
    in
    let checked, ones = T.run all in
    print_s [%message (decision : bool) (checked : int) (ones : int)]);
  [%expect
    {|
    ((decision false) (checked 256) (ones 97))
    ((decision true) (checked 256) (ones 19))
    |}]
;;

let%expect_test "waveform of two stumps, back to back vectors and a gap" =
  let module Config = struct
    let features = 2
    let feature_bits = 4
    let leaf_bits = 4
    let sum_bits = 6
    let bias = 1

    let trees : Otf_trees.Tree.t list =
      [ Node
          { split = { feature = 0; threshold = 2; equal = false }
          ; if_true = Leaf 3
          ; if_false = Leaf (-2)
          }
      ; Node
          { split = { feature = 1; threshold = -1; equal = true }
          ; if_true = Leaf 7
          ; if_false = Leaf 0
          }
      ]
    ;;

    let decision = false
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
      inputs.clocking.clear := Bits.vdd;
      Lws.cycle h;
      inputs.clocking.clear := Bits.gnd;
      List.iter [ Some 0xf2; Some 0x03; None; Some 0xf5 ] ~f:(fun x ->
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
    │x$value││ 00   │F2   │03   │00   │F5                                                    │
    │       ││──────┴─────┴─────┴─────┴───────────────────────────────────                   │
    │y$valid││                        ┌───────────┐     ┌─────┐                              │
    │       ││────────────────────────┘           └─────┘     └───────────                   │
    │       ││──────┬─────┬─────┬─────┬─────┬─────┬─────┬─────────────────                   │
    │y$value││ 00   │01   │3F   │04   │0B   │3F   │04   │06                                  │
    │       ││──────┴─────┴─────┴─────┴─────┴─────┴─────┴─────────────────                   │
    └───────┘└───────────────────────────────────────────────────────────────────────────────┘
    |}]
;;
