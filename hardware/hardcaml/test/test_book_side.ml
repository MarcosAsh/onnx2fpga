open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Side = Book_side.Make (struct
    let levels = 3
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Side.I) (Side.O)

type op =
  | Insert of int * int
  | Remove of int * int

let word (price, shares) = Bits.of_unsigned_int ~width:64 ((shares lsl 32) lor price)

let levels bits =
  Bits.split_lsb ~part_width:64 bits
  |> List.map ~f:(fun level ->
    ( Bits.to_unsigned_int (Bits.sel_bottom level ~width:32)
    , Bits.to_unsigned_int (Bits.sel_top level ~width:32) ))
  |> List.filter ~f:(fun (_, shares) -> shares > 0)
;;

(* Applies each change to the side as the book would and prints the side after it. *)
let run ~ask ~opposite ops =
  Harness.run ~create:Side.hierarchical (fun (h @ local) ~inputs ~outputs ->
    let o = Before_and_after_edge.before_edge outputs in
    let change (c : _ Book_side.Change.t) ~valid ~price ~shares =
      c.valid := Bits.of_bool valid;
      c.price := Bits.of_unsigned_int ~width:32 price;
      c.shares := Bits.of_unsigned_int ~width:32 shares
    in
    inputs.ask := Bits.of_bool ask;
    inputs.opposite := word opposite;
    inputs.levels := Bits.zero (3 * 64);
    List.iter ops ~f:(fun op ->
      let insert, remove =
        match op with
        | Insert (price, shares) -> Some (price, shares), None
        | Remove (price, shares) -> None, Some (price, shares)
      in
      let set c = function
        | Some (price, shares) -> change c ~valid:true ~price ~shares
        | None -> change c ~valid:false ~price:0 ~shares:0
      in
      set inputs.insert insert;
      set inputs.remove remove;
      Lws.cycle h;
      let side = levels !(o.updated) in
      let accepted = Bits.to_bool !(o.accepted) in
      print_s [%message (side : (int * int) list) (accepted : bool)];
      inputs.levels := !(o.updated));
    ())
;;

let%expect_test "bids: sorted inserts, a forgotten level, partial and full removes" =
  run
    ~ask:false
    ~opposite:(105, 10)
    [ Insert (100, 5)
    ; Insert (102, 7)
    ; Insert (100, 3)
    ; Insert (101, 1)
    ; Insert (99, 4)
    ; Insert (105, 9)
    ; Insert (103, 0)
    ; Remove (101, 2)
    ; Remove (102, 3)
    ; Remove (98, 1)
    ];
  [%expect
    {|
    ((side ((100 5))) (accepted true))
    ((side ((102 7) (100 5))) (accepted true))
    ((side ((102 7) (100 8))) (accepted true))
    ((side ((102 7) (101 1) (100 8))) (accepted true))
    ((side ((102 7) (101 1) (100 8))) (accepted true))
    ((side ((102 7) (101 1) (100 8))) (accepted false))
    ((side ((102 7) (101 1) (100 8))) (accepted false))
    ((side ((102 7) (100 8))) (accepted false))
    ((side ((102 4) (100 8))) (accepted false))
    ((side ((102 4) (100 8))) (accepted false))
    |}]
;;

let%expect_test "asks keep the lowest price first, and any ask goes in with no bids" =
  run ~ask:true ~opposite:(0, 0) [ Insert (110, 2); Insert (108, 1); Insert (1, 1) ];
  [%expect
    {|
    ((side ((110 2))) (accepted true))
    ((side ((108 1) (110 2))) (accepted true))
    ((side ((1 1) (108 1) (110 2))) (accepted true))
    |}]
;;
