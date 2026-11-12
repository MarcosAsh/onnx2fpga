open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Book = Order_book.Make (struct
    let symbols = 2
    let levels = 2
    let order_bits = 3
    let hash_orders = false
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
       let pending = Queue.create () in
       let cycle () =
         Lws.cycle h;
         let b = o.book in
         if Bits.to_bool !(b.valid)
         then (
           let { kind; order; _ } = Queue.dequeue_exn pending in
           let bids = levels !(b.bids) in
           let asks = levels !(b.asks) in
           let flow = Bits.to_signed_int !(b.flow) in
           print_s
             [%message
               (kind : Itch_parser.Kind.t)
                 (order : int)
                 (bids : (int * int) list)
                 (asks : (int * int) list)
                 (flow : int)])
       in
       List.iter
         events
         ~f:(fun ({ kind; symbol; side; order; new_order; shares; price } as event) ->
           Queue.enqueue pending event;
           set e.kind (Itch_parser.Kind.code kind);
           set e.symbol symbol;
           set e.side (Bool.to_int (Char.equal side 'S'));
           set e.order order;
           set e.new_order new_order;
           set e.shares shares;
           set e.price price;
           e.valid := Bits.vdd;
           cycle ();
           e.valid := Bits.gnd;
           cycle ());
       List.iter (List.init Order_book.latency ~f:Fn.id) ~f:(fun _ -> cycle ());
       assert (Queue.is_empty pending);
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

let%expect_test "rejected writes preserve live payloads" =
  run
    [ event Add 1 ~shares:100 ~price:10
    ; event Add 2 ~side:'S' ~shares:100 ~price:20
    ; event Add 9 ~shares:70 ~price:11
    ; event Execute 9 ~shares:10
    ; event Replace 9 ~new_order:17 ~shares:40 ~price:15
    ; event Replace 9 ~new_order:3 ~shares:40 ~price:15
    ; event Execute 1 ~shares:10
    ; event Replace 1 ~new_order:9 ~shares:40 ~price:20
    ; event Execute 9 ~shares:10
    ; event Add 17 ~shares:50 ~price:11
    ; event Replace 17 ~new_order:2 ~shares:60 ~price:12
    ; event Execute 2 ~shares:10
    ; event Add 25 ~shares:30 ~price:12
    ; event Replace 25 ~new_order:3 ~shares:60 ~price:20
    ; event Execute 3 ~shares:10
    ; event Add 11 ~shares:40 ~price:13
    ; event Execute 11 ~shares:5
    ];
  [%expect
    {|
    ((kind Add) (order 1) (bids ((10 100))) (asks ()) (flow 0))
    ((kind Add) (order 2) (bids ((10 100))) (asks ((20 100))) (flow 0))
    ((kind Add) (order 9) (bids ((10 100))) (asks ((20 100))) (flow 0))
    ((kind Execute) (order 9) (bids ((10 100))) (asks ((20 100))) (flow 0))
    ((kind Replace) (order 9) (bids ((10 100))) (asks ((20 100))) (flow 0))
    ((kind Replace) (order 9) (bids ((10 100))) (asks ((20 100))) (flow 0))
    ((kind Execute) (order 1) (bids ((10 90))) (asks ((20 100))) (flow -10))
    ((kind Replace) (order 1) (bids ()) (asks ((20 100))) (flow -10))
    ((kind Execute) (order 9) (bids ()) (asks ((20 100))) (flow -10))
    ((kind Add) (order 17) (bids ((11 50))) (asks ((20 100))) (flow -10))
    ((kind Replace) (order 17) (bids ()) (asks ((20 100))) (flow -10))
    ((kind Execute) (order 2) (bids ()) (asks ((20 90))) (flow 2))
    ((kind Add) (order 25) (bids ((12 30))) (asks ((20 90))) (flow 2))
    ((kind Replace) (order 25) (bids ()) (asks ((20 90))) (flow 2))
    ((kind Execute) (order 3) (bids ()) (asks ((20 90))) (flow 2))
    ((kind Add) (order 11) (bids ((13 40))) (asks ((20 90))) (flow 2))
    ((kind Execute) (order 11) (bids ((13 35))) (asks ((20 90))) (flow -3))
    |}]
;;

let%expect_test "crossing checks see the preceding event" =
  run
    [ event Add 1 ~shares:10 ~price:10
    ; event Add 2 ~side:'S' ~shares:10 ~price:10
    ; event Delete 1
    ; event Add 2 ~side:'S' ~shares:10 ~price:10
    ; event Add 3 ~shares:10 ~price:10
    ; event Delete 2
    ; event Add 3 ~shares:10 ~price:10
    ; event Add 4 ~symbol:1 ~side:'S' ~shares:10 ~price:9
    ; event Add 5 ~symbol:1 ~shares:10 ~price:9
    ; event Delete 3
    ; event Add 6 ~side:'S' ~shares:10 ~price:9
    ];
  [%expect
    {|
    ((kind Add) (order 1) (bids ((10 10))) (asks ()) (flow 0))
    ((kind Add) (order 2) (bids ((10 10))) (asks ()) (flow 0))
    ((kind Delete) (order 1) (bids ()) (asks ()) (flow 0))
    ((kind Add) (order 2) (bids ()) (asks ((10 10))) (flow 0))
    ((kind Add) (order 3) (bids ()) (asks ((10 10))) (flow 0))
    ((kind Delete) (order 2) (bids ()) (asks ()) (flow 0))
    ((kind Add) (order 3) (bids ((10 10))) (asks ()) (flow 0))
    ((kind Add) (order 4) (bids ()) (asks ((9 10))) (flow 0))
    ((kind Add) (order 5) (bids ()) (asks ((9 10))) (flow 0))
    ((kind Delete) (order 3) (bids ()) (asks ()) (flow 0))
    ((kind Add) (order 6) (bids ()) (asks ((9 10))) (flow 0))
    |}]
;;
