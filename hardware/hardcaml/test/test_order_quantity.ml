open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Harness =
  Hardcaml_test_harness.Lws_harness.Make (Order_quantity.I) (Order_quantity.O)

let%expect_test "zero, partial, full and excess reductions" =
  Harness.run ~create:Order_quantity.hierarchical (fun (h @ local) ~inputs ~outputs ->
    let o = Before_and_after_edge.before_edge outputs in
    List.iter
      [ 0, 0
      ; 0, 1
      ; 10, 0
      ; 10, 3
      ; 10, 10
      ; 10, 11
      ; 0xffff_ffff, 1
      ; 0xffff_ffff, 0xffff_ffff
      ]
      ~f:(fun (held, requested) ->
        inputs.held := Bits.of_unsigned_int ~width:32 held;
        inputs.requested := Bits.of_unsigned_int ~width:32 requested;
        Lws.cycle h;
        let executed = Bits.to_unsigned_int !(o.executed) in
        let remaining = Bits.to_unsigned_int !(o.remaining) in
        print_s
          [%message (held : int) (requested : int) (executed : int) (remaining : int)]);
    ());
  [%expect
    {|
    ((held 0) (requested 0) (executed 0) (remaining 0))
    ((held 0) (requested 1) (executed 0) (remaining 0))
    ((held 10) (requested 0) (executed 0) (remaining 10))
    ((held 10) (requested 3) (executed 3) (remaining 7))
    ((held 10) (requested 10) (executed 10) (remaining 0))
    ((held 10) (requested 11) (executed 10) (remaining 0))
    ((held 4294967295) (requested 1) (executed 1) (remaining 4294967294))
    ((held 4294967295) (requested 4294967295) (executed 4294967295)
     (remaining 0))
    |}]
;;
