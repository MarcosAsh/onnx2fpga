open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

(* A one layer model whose prediction is the top of book imbalance. *)
let config =
  {|{"locates": [2, 3], "names": ["BRVO", "CHRL"], "levels": 4, "order_bits": 6,
     "spread_shift": 6, "flow_shift": 5, "buy_threshold": 50, "sell_threshold": -50,
     "order_shares": 100,
     "model": {"format": 1, "top": "otf_top", "mode": "fixed", "latency": 3,
               "input": "x", "output": "y",
               "streams": [{"name": "x", "width": 32}, {"name": "y", "width": 8}],
               "nodes": [{"name": "fc0", "instance": "u_fc0", "kind": "otf_dense",
                          "params": {"MW": 4, "MH": 1, "ACT_BITS": 8, "WGT_BITS": 8,
                                     "ACC_BITS": 18, "OUT_BITS": 8, "MULT_BITS": 8,
                                     "SHIFT": 6, "OUT_MIN": -128, "OUT_MAX": 127,
                                     "OUT_ZP": 0, "CONSTANT": 1, "RELOADABLE": 0},
                          "memories": {"weights": {"elem_bits": 8, "lanes": 4,
                                                   "values": [0, 1, 0, 0]},
                                       "bias": {"elem_bits": 18, "lanes": 1, "values": [0]},
                                       "mult": {"elem_bits": 8, "lanes": 1, "values": [64]}},
                          "inputs": ["x"], "outputs": ["y"]}]}}|}
;;

module T = Tick_to_trade.Make (struct
    let params = Tick_to_trade.Params.of_string config
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (T.I) (T.O)

let%expect_test "stages" =
  print_s [%message (T.stages : (string * int) list) (T.latency : int)];
  [%expect
    {|
    ((T.stages
      ((moldudp64 1) (itch_parser 1) (order_book 2) (features 3) (model 3)
       (decision 1)))
     (T.latency 11))
    |}]
;;

(* A bid, an ask, an untracked add, then most of the ask trades. *)
let%expect_test "decisions from a packet" =
  Harness.run
    ~random_initial_state:`All
    ~create:T.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       Stream_bench.clear h inputs.clear;
       while not (Bits.to_bool !(o.ready)) do
         Lws.cycle h
       done;
       let packet =
         Itch_bench.packet
           [ Itch_bench.add ~order:1 ~side:'B' ~shares:100 ~price:500_000 ()
           ; Itch_bench.add ~order:2 ~side:'S' ~shares:400 ~price:500_100 ()
           ; Itch_bench.add ~locate:9 ~order:3 ~side:'S' ~shares:400 ~price:500_100 ()
           ; Itch_bench.execute ~order:2 ~shares:390 ()
           ]
       in
       let beats = List.map (Itch_bench.beats packet) ~f:Option.some in
       List.iter
         (beats @ List.init 20 ~f:(fun _ -> None))
         ~f:(fun beat ->
           Itch_bench.drive
             ~data:inputs.data
             ~keep:inputs.keep
             ~valid:inputs.valid
             ~last:inputs.last
             beat;
           Lws.cycle h;
           if Bits.to_bool !(o.decision_valid)
           then (
             let int signal = Bits.to_unsigned_int !signal in
             let action = List.nth_exn Decision.Action.all (int o.action) in
             let prediction = Bits.to_signed_int !(o.prediction) in
             let from_last = int o.t_out - int o.t_last in
             let from_first = int o.t_out - int o.t_first in
             print_s
               [%message
                 (action : Decision.Action.t)
                   (prediction : int)
                   (from_last : int)
                   (from_first : int)]));
       ());
  [%expect
    {|
    ((action Hold) (prediction 127) (from_last 11) (from_first 16))
    ((action Sell) (prediction -76) (from_last 11) (from_first 15))
    ((action Buy) (prediction 107) (from_last 11) (from_first 14))
    |}]
;;
