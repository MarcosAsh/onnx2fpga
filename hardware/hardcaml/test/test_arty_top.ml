open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Dut = Arty_top.Make (struct
    let design = Design.of_string (Fixed_fixture.description ~latency:6)
    let baud = 1_000_000
    let rx_depth = 64
    let tx_depth = 64
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

let%test_unit "button reset discards a partial frame before the next prediction" =
  Harness.run
    ~random_initial_state:`All
    ~create:Dut.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       let receiver = Uart_model.Receiver.create ~clocks_per_bit:100. in
       let cycle level =
         inputs.rxd := Bits.of_bool level;
         Lws.cycle h
       in
       inputs.button := Bits.vdd;
       List.iter (Uart_model.idle 10) ~f:cycle;
       inputs.button := Bits.gnd;
       List.iter (Uart_model.idle 10) ~f:cycle;
       List.iter (Uart_model.line ~clocks_per_bit:100. [ 0xee ]) ~f:cycle;
       inputs.button := Bits.vdd;
       List.iter (Uart_model.idle 10) ~f:cycle;
       inputs.button := Bits.gnd;
       List.iter (Uart_model.idle 10) ~f:cycle;
       let x = [ 3; -7 ] in
       let levels =
         Uart_model.line ~clocks_per_bit:100. (List.map x ~f:(fun x -> x land 255))
         @ Uart_model.idle 5000
       in
       List.iter levels ~f:(fun level ->
         cycle level;
         Uart_model.Receiver.step receiver (Bits.to_bool !(o.txd)));
       [%test_result: int list]
         (Uart_model.Receiver.received receiver)
         ~expect:(List.map (Fixed_fixture.model x) ~f:(fun x -> x land 255));
       [%test_result: int] (Uart_model.Receiver.framing_errors receiver) ~expect:0;
       [%test_result: int] (Bits.to_unsigned_int !(o.led) land 2) ~expect:0)
;;
