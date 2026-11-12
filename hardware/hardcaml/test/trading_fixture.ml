open! Core
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

let params = Tick_to_trade.Params.of_string config
