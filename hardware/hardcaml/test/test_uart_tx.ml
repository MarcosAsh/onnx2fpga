open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

module Make_test (Config : Uart_tx.Config) = struct
  module Dut = Uart_tx.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let transmit ?random_initial_state ?print_waves_after_test ?(gap = fun () -> 0) bytes =
    Harness.run
      ?random_initial_state
      ?print_waves_after_test
      ~trace:`Ports_only
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let receiver =
           Uart_model.Receiver.create ~clocks_per_bit:(Float.of_int Config.clocks_per_bit)
         in
         let cycle () =
           Lws.cycle h;
           Uart_model.Receiver.step receiver (Bits.to_bool !(o.txd))
         in
         let cycles = ref 0 in
         List.iter bytes ~f:(fun byte ->
           inputs.byte.valid := Bits.gnd;
           List.iter (List.range 0 (gap ())) ~f:(fun _ -> cycle ());
           inputs.byte.valid := Bits.vdd;
           inputs.byte.value := Bits.of_unsigned_int ~width:8 byte;
           let sent = ref false in
           while not !sent do
             cycle ();
             incr cycles;
             sent := Bits.to_bool !(o.ready)
           done);
         inputs.byte.valid := Bits.gnd;
         List.iter (List.range 0 (12 * Config.clocks_per_bit)) ~f:(fun _ -> cycle ());
         Uart_model.Receiver.received receiver, !cycles)
  ;;
end

module Three = Make_test (struct
    let clocks_per_bit = 3
  end)

(* 0x5a goes out three cycles a bit, and the next start bit follows the stop bit directly. *)
let%expect_test "waveform of two bytes back to back" =
  let display_rules =
    [ Display_rule.port_name_matches (Glob "byte$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "ready") ~wave_format:Bit
    ; Display_rule.port_name_matches (Glob "txd") ~wave_format:Bit
    ]
  in
  let received, cycles =
    Three.transmit
      ~print_waves_after_test:(fun waves ->
        Waveform.print
          ~display_rules
          ~signals_width:11
          ~display_width:100
          ~wave_width:0
          ~start_cycle:2
          waves)
      [ 0x5a; 0xff ]
  in
  print_s [%message (received : Int.Hex.t list) (cycles : int)];
  [%expect
    {|
    ┌Signals──┐┌Waves──────────────────────────────────────────────────────────────────────────────────┐
    │byte$vali││──────────────────────────────────────────────────────────────┐                        │
    │         ││                                                              └────────────────────────│
    │         ││──┬────────────────────────────────────────────────────────────────────────────────────│
    │byte$valu││ .│FF                                                                                  │
    │         ││──┴────────────────────────────────────────────────────────────────────────────────────│
    │ready    ││──┐                                                         ┌─┐                        │
    │         ││  └─────────────────────────────────────────────────────────┘ └────────────────────────│
    │txd      ││──┐           ┌─────┐     ┌───────────┐     ┌─────┐     ┌─────┐     ┌──────────────────│
    │         ││  └───────────┘     └─────┘           └─────┘     └─────┘     └─────┘                  │
    └─────────┘└───────────────────────────────────────────────────────────────────────────────────────┘
    ((received (0x5a 0xff)) (cycles 31))
    |}]
;;

module Typical = Make_test (struct
    let clocks_per_bit = 16
  end)

(* With bytes always waiting the line never idles: one byte per ten bit times exactly. *)
let%expect_test "random bytes go out intact and back to back" =
  let rng = Random.State.make [| 1 |] in
  let sent = List.init 100 ~f:(fun _ -> Random.State.int rng 256) in
  let received, cycles = Typical.transmit ~random_initial_state:`All sent in
  [%test_result: int list] received ~expect:sent;
  [%test_result: int] cycles ~expect:(1 + (99 * 10 * 16));
  let gap () = Random.State.int rng 40 in
  let received, _ = Typical.transmit ~gap sent in
  [%test_result: int list] received ~expect:sent;
  [%expect {| |}]
;;
