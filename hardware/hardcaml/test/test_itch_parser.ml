open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Parser = Itch_parser.Make (struct
    let locates = [ 2; 7 ]
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Parser.I) (Parser.O)

type event =
  { kind : Itch_parser.Kind.t
  ; symbol : int
  ; side : int
  ; order : int
  ; new_order : int
  ; shares : int
  ; price : int
  }
[@@deriving sexp_of]

let parse messages =
  Harness.run
    ~random_initial_state:`All
    ~create:Parser.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       Stream_bench.clear h inputs.clocking.clear;
       List.iter messages ~f:(fun message ->
         let padded = String.pad_right (String.prefix message 40) ~len:40 in
         inputs.message.bytes
         := Bits.concat_lsb
              (List.map (String.to_list padded) ~f:(fun c ->
                 Bits.of_unsigned_int ~width:8 (Char.to_int c)));
         inputs.message.length := Bits.of_unsigned_int ~width:16 (String.length message);
         inputs.message.valid := Bits.vdd;
         Lws.cycle h;
         inputs.message.valid := Bits.gnd;
         Lws.cycle h;
         let e = o.event in
         let int signal =
           Bits.to_unsigned_int
             (Bits.sel_bottom !signal ~width:(Int.min 32 (Bits.width !signal)))
         in
         if Bits.to_bool !(e.valid)
         then
           print_s
             [%sexp
               ({ kind = List.nth_exn Itch_parser.Kind.all (int e.kind)
                ; symbol = int e.symbol
                ; side = int e.side
                ; order = int e.order
                ; new_order = int e.new_order
                ; shares = int e.shares
                ; price = int e.price
                }
                : event)]
         else print_endline "dropped");
       ())
;;

let%expect_test "each kind of book message, and what is dropped" =
  parse
    [ Itch_bench.add ~order:11 ~side:'S' ~shares:300 ~price:1234500 ()
    ; Itch_bench.add
        ~locate:7
        ~attribution:"NSDQ"
        ~order:12
        ~side:'B'
        ~shares:100
        ~price:99
        ()
    ; Itch_bench.execute ~order:11 ~shares:50 ()
    ; Itch_bench.cancel ~locate:7 ~order:12 ~shares:10 ()
    ; Itch_bench.delete ~order:11 ()
    ; Itch_bench.replace ~order:12 ~new_order:13 ~shares:200 ~price:101 ()
    ; Itch_bench.delete ~locate:3 ~order:11 ()
    ; Itch_bench.add ~order:11 ~side:'S' ~shares:300 ~price:1234500 () ^ "NSDQ"
    ; Itch_bench.system_event
    ];
  [%expect
    {|
    ((kind Add) (symbol 0) (side 1) (order 11) (new_order 742543958) (shares 300)
     (price 1234500))
    ((kind Add) (symbol 1) (side 0) (order 12) (new_order 1682068054)
     (shares 100) (price 99))
    ((kind Execute) (symbol 0) (side 0) (order 11) (new_order 0) (shares 50)
     (price 0))
    ((kind Cancel) (symbol 1) (side 0) (order 12) (new_order 538976288)
     (shares 10) (price 0))
    ((kind Delete) (symbol 0) (side 0) (order 11) (new_order 538976288)
     (shares 0) (price 0))
    ((kind Replace) (symbol 0) (side 0) (order 12) (new_order 13) (shares 200)
     (price 101))
    dropped
    dropped
    dropped
    |}]
;;
