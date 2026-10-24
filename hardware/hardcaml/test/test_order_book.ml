open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Book = Order_book.Make (struct
    let symbols = 2
    let levels = 2
    let order_bits = 3
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Book.I) (Book.O)

type event =
  { kind : Itch_parser.Kind.t
  ; symbol : int
  ; side : char
  ; order : int
  ; new_order : int
  ; shares : int
  ; price : int
  }

let event
  ?(symbol = 0)
  ?(side = 'B')
  ?(new_order = 0)
  ?(shares = 0)
  ?(price = 0)
  kind
  order
  =
  { kind; symbol; side; order; new_order; shares; price }
;;

let levels bits =
  Bits.split_lsb ~part_width:64 bits
  |> List.map ~f:(fun level ->
    ( Bits.to_unsigned_int (Bits.sel_bottom level ~width:32)
    , Bits.to_unsigned_int (Bits.sel_top level ~width:32) ))
  |> List.filter ~f:(fun (_, shares) -> shares > 0)
;;

(* One event every other cycle, the fastest the framer can deliver them. *)
let run events =
  Harness.run
    ~random_initial_state:`All
    ~create:Book.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       Stream_bench.clear h inputs.clocking.clear;
       while not (Bits.to_bool !(o.ready)) do
         Lws.cycle h
       done;
       let e = inputs.event in
       let set signal value =
         signal := Bits.of_unsigned_int ~width:(Bits.width !signal) value
       in
       List.iter events ~f:(fun { kind; symbol; side; order; new_order; shares; price } ->
         set e.kind (Itch_parser.Kind.code kind);
         set e.symbol symbol;
         set e.side (Bool.to_int (Char.equal side 'S'));
         set e.order order;
         set e.new_order new_order;
         set e.shares shares;
         set e.price price;
         e.valid := Bits.vdd;
         Lws.cycle h;
         e.valid := Bits.gnd;
         let b = o.book in
         while not (Bits.to_bool !(b.valid)) do
           Lws.cycle h
         done;
         let bids = levels !(b.bids) in
         let asks = levels !(b.asks) in
         let flow = Bits.to_signed_int !(b.flow) in
         print_s
           [%message
             (kind : Itch_parser.Kind.t)
               (order : int)
               (bids : (int * int) list)
               (asks : (int * int) list)
               (flow : int)]);
       ())
;;

(* 4 falls off the end, 9 collides with 1, 5 would cross, 17 replaces 1 in its slot. *)
let%expect_test "orders through the table and the levels" =
  run
    [ event Add 1 ~shares:100 ~price:10
    ; event Add 2 ~shares:50 ~price:11
    ; event Add 3 ~side:'S' ~shares:70 ~price:13
    ; event Add 4 ~shares:30 ~price:9
    ; event Add 9 ~shares:20 ~price:12
    ; event Add 5 ~side:'S' ~shares:40 ~price:11
    ; event Execute 2 ~shares:20
    ; event Replace 1 ~new_order:17 ~shares:40 ~price:12
    ; event Cancel 17 ~shares:10
    ; event Execute 3 ~shares:500
    ; event Delete 2
    ; event Execute 4 ~shares:10
    ; event Add 6 ~symbol:1 ~side:'S' ~shares:5 ~price:7
    ];
  [%expect
    {|
    ((kind Add) (order 1) (bids ((10 100))) (asks ()) (flow 0))
    ((kind Add) (order 2) (bids ((11 50) (10 100))) (asks ()) (flow 0))
    ((kind Add) (order 3) (bids ((11 50) (10 100))) (asks ((13 70))) (flow 0))
    ((kind Add) (order 4) (bids ((11 50) (10 100))) (asks ((13 70))) (flow 0))
    ((kind Add) (order 9) (bids ((11 50) (10 100))) (asks ((13 70))) (flow 0))
    ((kind Add) (order 5) (bids ((11 50) (10 100))) (asks ((13 70))) (flow 0))
    ((kind Execute) (order 2) (bids ((11 30) (10 100))) (asks ((13 70)))
     (flow -20))
    ((kind Replace) (order 1) (bids ((12 40) (11 30))) (asks ((13 70)))
     (flow -20))
    ((kind Cancel) (order 17) (bids ((12 30) (11 30))) (asks ((13 70)))
     (flow -20))
    ((kind Execute) (order 3) (bids ((12 30) (11 30))) (asks ()) (flow 53))
    ((kind Delete) (order 2) (bids ((12 30))) (asks ()) (flow 53))
    ((kind Execute) (order 4) (bids ((12 30))) (asks ()) (flow 37))
    ((kind Add) (order 6) (bids ()) (asks ((7 5))) (flow 0))
    |}]
;;
