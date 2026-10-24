open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

(* Bytes that come back are the bytes sent, so any framing slip shows. *)
let design =
  Design.of_string
    {|
{
 "format": 1,
 "top": "otf_top",
 "input": "x",
 "output": "y",
 "streams": [{"name": "x", "width": 16}, {"name": "x_fifo", "width": 16},
             {"name": "y", "width": 8}],
 "nodes": [
  {"name": "buffer", "instance": "u_buffer", "kind": "otf_fifo",
   "params": {"WIDTH": 16, "DEPTH": 2}, "memories": {}, "inputs": ["x"], "outputs": ["x_fifo"]},
  {"name": "split", "instance": "u_split", "kind": "otf_dwc",
   "params": {"IN_ELEMS": 2, "OUT_ELEMS": 1, "ELEM_BITS": 8, "TOTAL_ELEMS": 2},
   "memories": {}, "inputs": ["x_fifo"], "outputs": ["y"]}
 ]
}
|}
;;

let clocks_per_bit = 4

module Dut = Uart_top.Make (struct
    let design = design
    let clocks_per_bit = clocks_per_bit
    let rx_depth = 64
    let tx_depth = 64
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

type result =
  { received : int list
  ; framing_errors : int
  ; error : bool
  }
[@@deriving sexp_of]

let exchange ?random_initial_state ?(error_bit = 0.) levels =
  Harness.run
    ?random_initial_state
    ~create:Dut.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       inputs.rxd := Bits.vdd;
       Stream_bench.clear h inputs.clocking.clear;
       let receiver =
         Uart_model.Receiver.create
           ~clocks_per_bit:(Float.of_int clocks_per_bit *. (1. +. error_bit))
       in
       let cycle level =
         inputs.rxd := Bits.of_bool level;
         Lws.cycle h;
         Uart_model.Receiver.step receiver (Bits.to_bool !(o.txd))
       in
       List.iter levels ~f:cycle;
       List.iter (Uart_model.idle (40 * clocks_per_bit)) ~f:cycle;
       { received = Uart_model.Receiver.received receiver
       ; framing_errors = Uart_model.Receiver.framing_errors receiver
       ; error = Bits.to_bool !(o.error)
       })
;;

let host_break = Uart_model.break_ (24 * clocks_per_bit) @ Uart_model.idle clocks_per_bit

let%expect_test "frames sent after a break come back intact, from any power up state" =
  let rng = Random.State.make [| 5 |] in
  let sent = List.init 120 ~f:(fun _ -> Random.State.int rng 256) in
  let gap () = Random.State.int rng 2 in
  List.iter [ -0.02; 0.; 0.02 ] ~f:(fun error ->
    let host_clocks = Float.of_int clocks_per_bit *. (1. +. error) in
    let result =
      exchange
        ~random_initial_state:`All
        ~error_bit:error
        (host_break @ Uart_model.line ~clocks_per_bit:host_clocks ~gap sent)
    in
    [%test_result: int list] result.received ~expect:sent;
    [%test_result: int] result.framing_errors ~expect:0;
    [%test_result: bool] result.error ~expect:false);
  [%expect {| |}]
;;

(* A stray byte leaves half a beat that a break throws away. *)
let%expect_test "a break realigns after a stray byte" =
  let line = Uart_model.line ~clocks_per_bit:(Float.of_int clocks_per_bit) in
  let without = exchange (line [ 0xee ] @ line [ 1; 2; 3; 4 ]) in
  let with_break = exchange (line [ 0xee ] @ host_break @ line [ 1; 2; 3; 4 ]) in
  print_s [%message (without : result) (with_break : result)];
  [%expect
    {|
    ((without ((received (238 1 2 3)) (framing_errors 0) (error false)))
     (with_break ((received (1 2 3 4)) (framing_errors 0) (error false))))
    |}]
;;

let%expect_test "a low stop bit raises error until the next break" =
  let clocks = Float.of_int clocks_per_bit in
  let bad =
    Uart_model.character ~stop:false ~clocks_per_bit:clocks 0x42
    @ Uart_model.idle (4 * clocks_per_bit)
  in
  let raised = exchange (bad @ Uart_model.line ~clocks_per_bit:clocks [ 7; 8 ]) in
  let cleared =
    exchange (bad @ host_break @ Uart_model.line ~clocks_per_bit:clocks [ 7; 8 ])
  in
  print_s [%message (raised : result) (cleared : result)];
  [%expect
    {|
    ((raised ((received (7 8)) (framing_errors 0) (error true)))
     (cleared ((received (7 8)) (framing_errors 0) (error false))))
    |}]
;;
