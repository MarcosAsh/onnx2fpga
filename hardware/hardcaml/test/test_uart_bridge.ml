open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

let bytes_of ~width beat =
  let bytes = (width + 7) / 8 in
  Bits.split_lsb ~part_width:8 (Bits.uresize beat ~width:(8 * bytes))
  |> List.map ~f:Bits.to_unsigned_int
;;

module Make_test (Config : Uart_bridge.Config) = struct
  module Dut = Uart_bridge.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  (* Every port moves at random, and incoming bytes cannot be stalled. *)
  let random_traffic ~seed ~beats () =
    let rng = Random.State.make [| seed |] in
    let chance duty = Random.State.int rng 100 < duty in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let beats_in = Stream_bench.beats ~rng ~width:Config.in_width beats in
         let beats_out = Stream_bench.beats ~rng ~width:Config.out_width beats in
         let bytes_in = List.concat_map beats_in ~f:(bytes_of ~width:Config.in_width) in
         let bytes_out =
           List.concat_map beats_out ~f:(bytes_of ~width:Config.out_width)
         in
         let in_bytes = List.length bytes_in / beats in
         let pending = ref bytes_in in
         let taken = ref [] in
         let offering = ref beats_out in
         let sent = ref [] in
         let overflows = ref 0 in
         let cycles = ref 0 in
         while List.length !taken < beats || List.length !sent < List.length bytes_out do
           if !cycles > 200_000 then raise_s [%message "uart_bridge stalled"];
           (match !pending with
            | byte :: rest
              when chance 30
                   && List.length bytes_in - List.length rest
                      <= (List.length !taken * in_bytes) + Config.rx_depth ->
              inputs.rx.valid := Bits.vdd;
              inputs.rx.value := Bits.of_unsigned_int ~width:8 byte;
              pending := rest
            | _ ->
              inputs.rx.valid := Bits.gnd;
              inputs.rx.value := Stream_bench.random_bits rng ~width:8);
           inputs.accel_in.ready := Bits.of_bool (chance 50);
           (match !offering with
            | beat :: _ when chance 60 ->
              inputs.accel_out.data := beat;
              inputs.accel_out.valid := Bits.vdd
            | _ ->
              inputs.accel_out.data
              := Stream_bench.random_bits rng ~width:Config.out_width;
              inputs.accel_out.valid := Bits.gnd);
           inputs.tx.ready := Bits.of_bool (chance 40);
           Lws.cycle h;
           incr cycles;
           if Bits.to_bool !(o.overflow) then incr overflows;
           if Bits.to_bool !(o.accel_in.valid) && Bits.to_bool !(inputs.accel_in.ready)
           then taken := !(o.accel_in.data) :: !taken;
           if Bits.to_bool !(inputs.accel_out.valid) && Bits.to_bool !(o.accel_out.ready)
           then offering := List.tl_exn !offering;
           if Bits.to_bool !(o.tx.valid) && Bits.to_bool !(inputs.tx.ready)
           then sent := Bits.to_unsigned_int !(o.tx.data) :: !sent
         done;
         [%test_result: Bits.t list] (List.rev !taken) ~expect:beats_in;
         [%test_result: int list] (List.rev !sent) ~expect:bytes_out;
         [%test_result: int] !overflows ~expect:0;
         !cycles)
  ;;
end

let%expect_test "beats of every shape cross intact under random traffic" =
  List.iter
    [ 8, 8; 64, 8; 12, 20; 1, 3; 24, 64 ]
    ~f:(fun (in_width, out_width) ->
      let module T =
        Make_test (struct
          let in_width = in_width
          let out_width = out_width
          let rx_depth = 16
          let tx_depth = 16
        end)
      in
      let cycles = T.random_traffic ~seed:in_width ~beats:150 () in
      print_s [%message (in_width : int) (out_width : int) (cycles : int)]);
  [%expect
    {|
    ((in_width 8) (out_width 8) (cycles 462))
    ((in_width 64) (out_width 8) (cycles 3938))
    ((in_width 12) (out_width 20) (cycles 1181))
    ((in_width 1) (out_width 3) (cycles 499))
    ((in_width 24) (out_width 64) (cycles 3130))
    |}]
;;

module Small = Make_test (struct
    let in_width = 16
    let out_width = 8
    let rx_depth = 4
    let tx_depth = 4
  end)

(* A full buffer holds at least its depth; the seventh byte overflows. *)
let%expect_test "capacity of each direction" =
  Small.Harness.run ~create:Small.Dut.hierarchical (fun (h @ local) ~inputs ~outputs ->
    let o = Before_and_after_edge.before_edge outputs in
    Stream_bench.clear h inputs.clocking.clear;
    let overflows =
      List.map (List.range 0 8) ~f:(fun byte ->
        inputs.rx.valid := Bits.vdd;
        inputs.rx.value := Bits.of_unsigned_int ~width:8 (0x10 + byte);
        Lws.cycle h;
        Bits.to_bool !(o.overflow))
    in
    inputs.rx.valid := Bits.gnd;
    inputs.accel_in.ready := Bits.vdd;
    let beats = ref [] in
    List.iter (List.range 0 8) ~f:(fun _ ->
      Lws.cycle h;
      if Bits.to_bool !(o.accel_in.valid)
      then beats := Bits.to_unsigned_int !(o.accel_in.data) :: !beats);
    let accepted = ref 0 in
    inputs.accel_out.valid := Bits.vdd;
    List.iter (List.range 0 12) ~f:(fun beat ->
      inputs.accel_out.data := Bits.of_unsigned_int ~width:8 beat;
      Lws.cycle h;
      if Bits.to_bool !(o.accel_out.ready) then incr accepted);
    print_s
      [%message
        (overflows : bool list)
          ~beats:(List.rev !beats : Int.Hex.t list)
          ~beats_accepted_while_tx_stalled:(!accepted : int)]);
  [%expect
    {|
    ((overflows (false false false false false false true true))
     (beats (0x1110 0x1312 0x1514)) (beats_accepted_while_tx_stalled 6))
    |}]
;;

let%expect_test "waveform of a two byte beat in and a byte out" =
  let display_rules =
    [ Display_rule.port_name_matches (Glob "rx$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "accel_in$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "accel_out$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "tx$*") ~wave_format:(Bit_or Hex)
    ]
  in
  Small.Harness.run
    ~create:Small.Dut.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print
        ~display_rules
        ~signals_width:15
        ~display_width:90
        ~wave_width:1
        ~start_cycle:2
        waves)
    (fun (h @ local) ~inputs ~outputs:_ ->
      Stream_bench.clear h inputs.clocking.clear;
      inputs.accel_in.ready := Bits.vdd;
      inputs.tx.ready := Bits.vdd;
      List.iter [ 0x34; 0x12 ] ~f:(fun byte ->
        inputs.rx.valid := Bits.vdd;
        inputs.rx.value := Bits.of_unsigned_int ~width:8 byte;
        Lws.cycle h;
        inputs.rx.valid := Bits.gnd;
        Lws.cycle h);
      inputs.accel_out.valid := Bits.vdd;
      inputs.accel_out.data := Bits.of_unsigned_int ~width:8 0xab;
      Lws.cycle h;
      inputs.accel_out.valid := Bits.gnd;
      Lws.cycle ~n:5 h);
  [%expect
    {|
    ┌Signals──────┐┌Waves────────────────────────────────────────────────────────────────────┐
    │rx$valid     ││────┐   ┌───┐                                                            │
    │             ││    └───┘   └───────────────────────────────                             │
    │             ││────────┬───────────────────────────────────                             │
    │rx$value     ││ 34     │12                                                              │
    │             ││────────┴───────────────────────────────────                             │
    │accel_in$read││────────────────────────────────────────────                             │
    │             ││                                                                         │
    │             ││────┬───┬───┬───┬───────────────────────────                             │
    │accel_in$data││ 00.│34.│00.│12.│0034                                                    │
    │             ││────┴───┴───┴───┴───────────────────────────                             │
    │accel_in$vali││            ┌───┐                                                        │
    │             ││────────────┘   └───────────────────────────                             │
    │             ││────────────────┬───────────────────────────                             │
    │accel_out$dat││ 00             │AB                                                      │
    │             ││────────────────┴───────────────────────────                             │
    │accel_out$val││                ┌───┐                                                    │
    │             ││────────────────┘   └───────────────────────                             │
    │accel_out$rea││────────────────────┐   ┌───────────────────                             │
    │             ││                    └───┘                                                │
    │tx$ready     ││────────────────────────────────────────────                             │
    │             ││                                                                         │
    │             ││────────────────────────┬───┬───────────────                             │
    │tx$data      ││ 00                     │AB │00                                          │
    │             ││────────────────────────┴───┴───────────────                             │
    │tx$valid     ││                        ┌───┐                                            │
    │             ││────────────────────────┘   └───────────────                             │
    └─────────────┘└─────────────────────────────────────────────────────────────────────────┘
    |}]
;;
