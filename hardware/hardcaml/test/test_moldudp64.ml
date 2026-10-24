open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml
module Harness = Hardcaml_test_harness.Lws_harness.Make (Moldudp64.I) (Moldudp64.O)

type message =
  { kind : char
  ; length : int
  ; t_first : int
  ; t_last : int
  }
[@@deriving sexp_of]

(* Beats back to back, with [now] counting cycles from the first beat. *)
let frame ?random_initial_state ?print_waves_after_test packets =
  Harness.run
    ?random_initial_state
    ?print_waves_after_test
    ~create:Moldudp64.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       Stream_bench.clear h inputs.clocking.clear;
       let received = ref [] in
       let beats = List.concat_map packets ~f:Itch_bench.beats in
       List.iteri
         (List.map beats ~f:Option.some @ List.init 3 ~f:(fun _ -> None))
         ~f:(fun cycle beat ->
           Itch_bench.drive
             ~data:inputs.beat.data
             ~keep:inputs.beat.keep
             ~valid:inputs.beat.valid
             ~last:inputs.beat.last
             beat;
           inputs.now := Bits.of_unsigned_int ~width:32 cycle;
           Lws.cycle h;
           let message = o.message in
           if Bits.to_bool !(message.valid)
           then (
             let length = Bits.to_unsigned_int !(message.length) in
             let bytes =
               Itch_bench.string_of_bits !(message.bytes) ~length:(Int.min length 40)
             in
             received
             := { kind = bytes.[0]
                ; length
                ; t_first = Bits.to_unsigned_int !(message.t_first)
                ; t_last = Bits.to_unsigned_int !(message.t_last)
                }
                :: !received));
       List.rev !received)
;;

let delete = Itch_bench.delete ~order:7 ()
let add = Itch_bench.add ~order:9 ~side:'B' ~shares:100 ~price:1000 ()

(* Blocks start on lanes 4, 2, 7, 5 and 6, so lengths and bodies straddle beats. *)
let%expect_test "messages split across beats" =
  let packet =
    Itch_bench.packet
      [ Itch_bench.system_event
      ; delete
      ; add
      ; Itch_bench.cancel ~order:9 ~shares:5 ()
      ; Itch_bench.execute ~order:9 ~shares:5 ()
      ]
  in
  print_s [%sexp (frame ~random_initial_state:`All [ packet; packet ] : message list)];
  [%expect
    {|
    (((kind S) (length 12) (t_first 2) (t_last 4))
     ((kind D) (length 19) (t_first 4) (t_last 6))
     ((kind A) (length 36) (t_first 7) (t_last 11))
     ((kind X) (length 23) (t_first 11) (t_last 14))
     ((kind E) (length 31) (t_first 15) (t_last 18))
     ((kind S) (length 12) (t_first 21) (t_last 23))
     ((kind D) (length 19) (t_first 23) (t_last 25))
     ((kind A) (length 36) (t_first 26) (t_last 30))
     ((kind X) (length 23) (t_first 30) (t_last 33))
     ((kind E) (length 31) (t_first 34) (t_last 37)))
    |}]
;;

(* Each malformed packet loses only what the spec says, and the next one frames again. *)
let%expect_test "heartbeats, end of session and malformed packets" =
  let short = Itch_bench.packet ~lengths:(List.map ~f:(fun _ -> 5)) [ delete; delete ] in
  let truncated =
    let full = Itch_bench.packet [ delete; add ] in
    String.prefix full (String.length full - 3)
  in
  let packets =
    [ Itch_bench.packet []
    ; Itch_bench.packet ~count:0xffff [ delete ]
    ; Itch_bench.packet ~count:1 [ delete; add ]
    ; short
    ; truncated
    ; Itch_bench.packet [ add ]
    ]
  in
  List.iter (frame packets) ~f:(fun m ->
    print_s [%sexp (m.kind : char), (m.length : int)]);
  [%expect {|
    (D 19)
    (D 19)
    (A 36)
    |}]
;;

let%expect_test "waveform of one delete" =
  let display_rules =
    [ Display_rule.port_name_matches (Glob "beat$valid") ~wave_format:Bit
    ; Display_rule.port_name_matches (Glob "beat$keep") ~wave_format:Hex
    ; Display_rule.port_name_matches (Glob "beat$last") ~wave_format:Bit
    ; Display_rule.port_name_matches (Glob "message$valid") ~wave_format:Bit
    ; Display_rule.port_name_matches (Glob "message$length") ~wave_format:Unsigned_int
    ; Display_rule.port_name_matches (Glob "message$t_*") ~wave_format:Unsigned_int
    ]
  in
  ignore
    (frame
       ~print_waves_after_test:(fun waves ->
         Waveform.print
           ~display_rules
           ~signals_width:15
           ~wave_width:1
           ~start_cycle:2
           waves)
       [ Itch_bench.packet [ delete ] ]
     : message list);
  [%expect
    {|
    ┌Signals──────┐┌Waves────────────────────────────────────────────────┐
    │beat$valid   ││────────────────────────┐                            │
    │             ││                        └───────────────             │
    │             ││────────────────────┬───────────────────             │
    │beat$keep    ││ FF                 │01                              │
    │             ││────────────────────┴───────────────────             │
    │beat$last    ││                    ┌───────────────────             │
    │             ││────────────────────┘                                │
    │message$valid││                        ┌───┐                        │
    │             ││────────────────────────┘   └───────────             │
    │             ││────────────────┬───────────┬───────────             │
    │message$lengt││ 0              │19         │0                       │
    │             ││────────────────┴───────────┴───────────             │
    │             ││────────────────┬───────────┬───────────             │
    │message$t_fir││ 0              │2          │0                       │
    │             ││────────────────┴───────────┴───────────             │
    │             ││────────┬───┬───┬───┬───┬───┬───┬───┬───             │
    │message$t_las││ 0      │1  │2  │3  │4  │5  │6  │7  │8               │
    │             ││────────┴───┴───┴───┴───┴───┴───┴───┴───             │
    └─────────────┘└─────────────────────────────────────────────────────┘
    |}]
;;
