open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

let debug = false

module Make_test (Config : Uart_rx.Config) = struct
  module Dut = Uart_rx.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  type result =
    { bytes : int list
    ; framing_errors : int
    ; break_cycles : int
    }
  [@@deriving sexp_of]

  let receive ?random_initial_state ?print_waves_after_test levels =
    Harness.run
      ?random_initial_state
      ?print_waves_after_test
      ~trace:`Ports_only
      ~waves_config:
        (if debug
         then Hardcaml_test_harness.Waves_config.to_test_directory ()
         else Hardcaml_test_harness.Waves_config.no_waves)
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
        let o = Before_and_after_edge.before_edge outputs in
        inputs.rxd := Bits.vdd;
        Stream_bench.clear h inputs.clocking.clear;
        let bytes = ref [] in
        let framing_errors = ref 0 in
        let break_cycles = ref 0 in
        List.iter
          (levels @ Uart_model.idle (4 * Config.clocks_per_bit))
          ~f:(fun level ->
            inputs.rxd := Bits.of_bool level;
            Lws.cycle h;
            if Bits.to_bool !(o.byte.valid)
            then bytes := Bits.to_unsigned_int !(o.byte.value) :: !bytes;
            if Bits.to_bool !(o.framing_error) then incr framing_errors;
            if Bits.to_bool !(o.line_break) then incr break_cycles);
        { bytes = List.rev !bytes
        ; framing_errors = !framing_errors
        ; break_cycles = !break_cycles
        })
  ;;
end

module Fast = Make_test (struct
    let clocks_per_bit = 4
  end)

(* A glitch start bit is ignored; 0xa5 is out a cycle after the middle of its stop bit. *)
let%expect_test "waveform of a glitch and one byte" =
  let display_rules =
    [ Display_rule.port_name_matches (Glob "rxd") ~wave_format:Bit
    ; Display_rule.port_name_matches (Glob "byte$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "framing_error") ~wave_format:Bit
    ]
  in
  let clocks_per_bit = 4. in
  let result =
    Fast.receive
      ~print_waves_after_test:(fun waves ->
        Waveform.print
          ~display_rules
          ~signals_width:13
          ~display_width:140
          ~wave_width:0
          ~start_cycle:2
          waves)
      ([ true; false; true; true; true; true ] @ Uart_model.line ~clocks_per_bit [ 0xa5 ])
  in
  print_s [%sexp (result : Fast.result)];
  [%expect
    {|
    ┌Signals────┐┌Waves────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │rxd        ││──┐ ┌───────┐       ┌───────┐       ┌───────┐               ┌───────┐       ┌────────────────────────────────────────────────│
    │           ││  └─┘       └───────┘       └───────┘       └───────────────┘       └───────┘                                                │
    │byte$valid ││                                                                                              ┌─┐                            │
    │           ││──────────────────────────────────────────────────────────────────────────────────────────────┘ └────────────────────────────│
    │           ││──────────────────────────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬──────────────────────────────────────│
    │byte$value ││ 00                           │80     │40     │A0     │50     │28     │94     │4A     │A5                                    │
    │           ││──────────────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴──────────────────────────────────────│
    │framing_err││                                                                                                                             │
    │           ││─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│
    └───────────┘└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    ((bytes (165)) (framing_errors 0) (break_cycles 0))
    |}]
;;

module Typical = Make_test (struct
    let clocks_per_bit = 16
  end)

(* Random bytes with random gaps, from hosts whose clocks are off by up to 4 percent. *)
let%expect_test "random bytes arrive intact across baud rate error" =
  List.iter [ -0.04; -0.02; 0.; 0.02; 0.04 ] ~f:(fun error ->
    let rng = Random.State.make [| Float.to_int (error *. 1000.) |] in
    let sent = List.init 200 ~f:(fun _ -> Random.State.int rng 256) in
    let gap () = if Random.State.bool rng then 0 else Random.State.int rng 3 in
    let result =
      Typical.receive
        ~random_initial_state:`All
        (Uart_model.line ~clocks_per_bit:(16. *. (1. +. error)) ~gap sent)
    in
    [%test_result: int list] result.bytes ~expect:sent;
    [%test_result: int] result.framing_errors ~expect:0;
    [%test_result: int] result.break_cycles ~expect:0);
  [%expect {| |}]
;;

(* Eight percent is past what mid bit sampling can absorb over ten bits. *)
let%expect_test "too much baud rate error shows as framing errors" =
  let sent = List.init 50 ~f:(fun k -> k * 5) in
  let result = Typical.receive (Uart_model.line ~clocks_per_bit:(16. *. 1.08) sent) in
  print_s
    [%message
      ""
        ~intact:(List.equal Int.equal result.bytes sent : bool)
        ~framing_errors:(result.framing_errors > 0 : bool)];
  [%expect {| ((intact false) (framing_errors true)) |}]
;;

let%expect_test "a low stop bit drops the byte and pulses framing_error" =
  let clocks_per_bit = 16. in
  let levels =
    List.concat
      [ Uart_model.line ~clocks_per_bit [ 0x11 ]
      ; Uart_model.character ~stop:false ~clocks_per_bit 0x3c
      ; Uart_model.idle 32
      ; Uart_model.line ~clocks_per_bit [ 0x22 ]
      ]
  in
  print_s [%sexp (Typical.receive levels : Typical.result)];
  [%expect {| ((bytes (17 34)) (framing_errors 1) (break_cycles 0)) |}]
;;

(* Zero bytes are never a break; a real break is one framing error then [line_break]. *)
let%expect_test "breaks are told apart from zero bytes" =
  let clocks_per_bit = 16. in
  let zeros =
    Typical.receive (Uart_model.line ~clocks_per_bit (List.init 20 ~f:(fun _ -> 0)))
  in
  print_s [%sexp (zeros : Typical.result)];
  let levels =
    List.concat
      [ Uart_model.break_ (30 * 16)
      ; Uart_model.idle 16
      ; Uart_model.line ~clocks_per_bit [ 0x5a ]
      ]
  in
  print_s [%sexp (Typical.receive levels : Typical.result)];
  [%expect
    {|
    ((bytes (0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)) (framing_errors 0)
     (break_cycles 0))
    ((bytes (90)) (framing_errors 1) (break_cycles 161))
    |}]
;;
