open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Harness =
  Hardcaml_test_harness.Lws_harness.Make (Book_crossing.I) (Book_crossing.O)

let%expect_test "touching prices cross only a live opposite side" =
  Harness.run ~create:Book_crossing.hierarchical (fun (h @ local) ~inputs ~outputs ->
    let o = Before_and_after_edge.before_edge outputs in
    let set signal value = signal := Bits.of_unsigned_int ~width:32 value in
    set inputs.bid.price 100;
    set inputs.offer.price 102;
    let cases =
      List.cartesian_product
        [ true; false ]
        (List.cartesian_product [ false; true ] [ 99; 100; 101; 102; 103 ])
    in
    List.iter cases ~f:(fun (live, (ask, price)) ->
      set inputs.bid.shares (Bool.to_int live);
      set inputs.offer.shares (Bool.to_int live);
      inputs.ask := Bits.of_bool ask;
      set inputs.price price;
      Lws.cycle h;
      let crosses = Bits.to_bool !(o.crosses) in
      print_s [%message (live : bool) (ask : bool) (price : int) (crosses : bool)]);
    ());
  [%expect
    {|
    ((live true) (ask false) (price 99) (crosses false))
    ((live true) (ask false) (price 100) (crosses false))
    ((live true) (ask false) (price 101) (crosses false))
    ((live true) (ask false) (price 102) (crosses true))
    ((live true) (ask false) (price 103) (crosses true))
    ((live true) (ask true) (price 99) (crosses true))
    ((live true) (ask true) (price 100) (crosses true))
    ((live true) (ask true) (price 101) (crosses false))
    ((live true) (ask true) (price 102) (crosses false))
    ((live true) (ask true) (price 103) (crosses false))
    ((live false) (ask false) (price 99) (crosses false))
    ((live false) (ask false) (price 100) (crosses false))
    ((live false) (ask false) (price 101) (crosses false))
    ((live false) (ask false) (price 102) (crosses false))
    ((live false) (ask false) (price 103) (crosses false))
    ((live false) (ask true) (price 99) (crosses false))
    ((live false) (ask true) (price 100) (crosses false))
    ((live false) (ask true) (price 101) (crosses false))
    ((live false) (ask true) (price 102) (crosses false))
    ((live false) (ask true) (price 103) (crosses false))
    |}]
;;
