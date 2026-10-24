open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Decide = Decision.Make (struct
    let names = [ "BRVO"; "CHRL" ]
    let buy_threshold = 10
    let sell_threshold = -10
    let order_shares = 100
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Decide.I) (Decide.O)

let%expect_test "thresholds, an empty side, and the order frames" =
  Harness.run
    ~random_initial_state:`All
    ~create:Decide.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       Stream_bench.clear h inputs.clocking.clear;
       let c = inputs.context in
       List.iter
         [ 10, 1, true, true; 9, 0, true, true; -10, 1, true, true; -11, 0, false, true ]
         ~f:(fun (prediction, symbol, has_bid, has_ask) ->
           inputs.prediction.value := Bits.of_signed_int ~width:8 prediction;
           inputs.prediction.valid := Bits.vdd;
           c.symbol := Bits.of_unsigned_int ~width:3 symbol;
           c.bid := Bits.of_unsigned_int ~width:32 1_234_500;
           c.ask := Bits.of_unsigned_int ~width:32 1_234_600;
           c.has_bid := Bits.of_bool has_bid;
           c.has_ask := Bits.of_bool has_ask;
           Lws.cycle h;
           inputs.prediction.valid := Bits.gnd;
           while not (Bits.to_bool !(o.valid)) do
             Lws.cycle h
           done;
           let action =
             List.nth_exn Decision.Action.all (Bits.to_unsigned_int !(o.action))
           in
           let order =
             if Bits.to_bool !(o.order.valid)
             then Itch_bench.string_of_bits !(o.order.value) ~length:Decision.order_bytes
             else ""
           in
           print_s
             [%message (prediction : int) (action : Decision.Action.t) (order : string)]);
       ());
  [%expect
    {|
    ((prediction 10) (action Buy)
     (order
      "O\000\000\000\001B\000\000\000dCHRL    \000\000\000\000\000\018\214\1683YPNN00000000000001\000\000"))
    ((prediction 9) (action Hold) (order ""))
    ((prediction -10) (action Sell)
     (order
      "O\000\000\000\002S\000\000\000dCHRL    \000\000\000\000\000\018\214D3YPNN00000000000002\000\000"))
    ((prediction -11) (action Hold) (order ""))
    |}]
;;
