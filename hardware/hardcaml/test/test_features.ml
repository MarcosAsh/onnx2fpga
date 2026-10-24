open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Extract = Features.Make (struct
    let levels = 2
    let spread_shift = 6
    let flow_shift = 5
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Extract.I) (Extract.O)

let word (price, shares) =
  Bits.(of_unsigned_int ~width:32 shares @: of_unsigned_int ~width:32 price)
;;

(* Shift both until the larger fits in four bits, then look the ratio up. *)
let imbalance bid ask =
  let shift = Int.max 0 (Int.num_bits - Int.clz (bid lor ask) - 4) in
  Features.imbalance ~bid:(bid lsr shift) ~ask:(ask lsr shift)
;;

let run books =
  Harness.run
    ~random_initial_state:`All
    ~create:Extract.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       Stream_bench.clear h inputs.clocking.clear;
       let b = inputs.book in
       let results =
         List.map books ~f:(fun (bids, asks, flow) ->
           b.bids := Bits.concat_lsb (List.map bids ~f:word);
           b.asks := Bits.concat_lsb (List.map asks ~f:word);
           b.flow := Bits.of_signed_int ~width:Order_book.flow_bits flow;
           b.valid := Bits.vdd;
           Lws.cycle h;
           b.valid := Bits.gnd;
           while not (Bits.to_bool !(o.features.valid)) do
             Lws.cycle h
           done;
           Bits.split_lsb ~part_width:8 !(o.features.value)
           |> List.map ~f:Bits.to_signed_int)
       in
       results)
;;

let%expect_test "spread, imbalances and flow" =
  run
    [ [ 1_000_000, 300; 999_900, 100 ], [ 1_000_100, 100; 1_000_200, 900 ], 0
    ; [ 1_000_000, 300; 0, 0 ], [ 0, 0; 0, 0 ], 100_000
    ; [ 0, 0; 0, 0 ], [ 0, 0; 0, 0 ], -100_000
    ; [ 5_000, 1; 0, 0 ], [ 15_000, 4_000_000_000; 0, 0 ], -100
    ]
  |> List.iter ~f:(fun features -> print_s [%sexp (features : int list)]);
  [%expect
    {|
    (1 64 -54 0)
    (127 127 127 127)
    (127 0 0 -127)
    (127 -127 -127 -4)
    |}]
;;

let%expect_test "the imbalances agree with the model on random books" =
  let rng = Random.State.make [| 3 |] in
  let shares () = Random.State.int rng (1 lsl Random.State.int rng 32) in
  let books =
    List.init 300 ~f:(fun _ ->
      let bids = List.init 2 ~f:(fun i -> 1000 - i, shares ()) in
      let asks = List.init 2 ~f:(fun i -> 2000 + i, shares ()) in
      bids, asks, 0)
  in
  List.iter2_exn books (run books) ~f:(fun (bids, asks, _) features ->
    let top = imbalance (snd (List.hd_exn bids)) (snd (List.hd_exn asks)) in
    let depth =
      imbalance (List.sum (module Int) bids ~f:snd) (List.sum (module Int) asks ~f:snd)
    in
    [%test_result: int list] (List.slice features 1 3) ~expect:[ top; depth ]);
  [%expect {| |}]
;;
