open! Core
open! Hardcaml
open! Signal

let latency = 3
let count = 4

module Context = struct
  type 'a t =
    { symbol : 'a [@bits 3]
    ; bid : 'a [@bits 32]
    ; has_bid : 'a
    ; ask : 'a [@bits 32]
    ; has_ask : 'a
    ; t_first : 'a [@bits 32]
    ; t_last : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

let imbalance ~bid ~ask =
  if bid + ask = 0
  then 0
  else (
    let n = 127 * (bid - ask) in
    let q = ((2 * abs n) + bid + ask) / (2 * (bid + ask)) in
    if n < 0 then -q else q)
;;

module type Config = sig
  val levels : int
  val spread_shift : int
  val flow_shift : int
end

module Make (Config : Config) = struct
  open Config

  module Book = struct
    type 'a t =
      { valid : 'a
      ; symbol : 'a [@bits 3]
      ; bids : 'a [@bits levels * 64]
      ; asks : 'a [@bits levels * 64]
      ; flow : 'a [@bits Order_book.flow_bits]
      ; t_first : 'a [@bits 32]
      ; t_last : 'a [@bits 32]
      }
    [@@deriving hardcaml]
  end

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; book : 'a Book.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { features : 'a With_valid.t [@bits 8 * count]
      ; context : 'a Context.t
      }
    [@@deriving hardcaml]
  end

  let create scope ({ clocking; book } : Signal.t I.t) =
    let reg = Clocking.reg_no_clear clocking in
    let split bus = split_lsb ~part_width:64 bus |> List.map ~f:Book_side.Level.of_word in
    let bids = split book.bids in
    let asks = split book.asks in
    let bid = List.hd_exn bids in
    let ask = List.hd_exn asks in
    let has_bid = bid.shares <>:. 0 in
    let has_ask = ask.shares <>:. 0 in
    let byte value = of_signed_int ~width:8 value in
    let%hw spread =
      let gap = srl (ask.price -: bid.price) ~by:spread_shift in
      reg
        (mux2 (has_bid &: has_ask &: (gap <:. 127)) (sel_bottom gap ~width:8) (byte 127))
    in
    let%hw flow =
      let shifted = sra book.flow ~by:flow_shift in
      let limit value = of_signed_int ~width:(width shifted) value in
      reg
        (mux2
           (shifted >+ limit 127)
           (byte 127)
           (mux2 (shifted <+ limit (-127)) (byte (-127)) (sel_bottom shifted ~width:8)))
    in
    let depth side =
      List.map side ~f:(fun (l : _ Book_side.Level.t) ->
        uresize l.shares ~width:(32 + Int.ceil_log2 levels))
      |> reduce ~f:( +: )
    in
    let%hw top_bid = reg bid.shares in
    let%hw top_ask = reg ask.shares in
    let%hw depth_bid = reg (depth bids) in
    let%hw depth_ask = reg (depth asks) in
    (* Shift both so the larger keeps four bits; the ratio is then a 256 entry table. *)
    let normalize bid ask =
      let bits = width bid in
      let zeros = leading_zeros (bid |: ask) in
      let shift =
        mux2
          (zeros >=:. bits - 4)
          (zero (width zeros))
          (of_unsigned_int ~width:(width zeros) (bits - 4) -: zeros)
      in
      let top value = reg (sel_bottom (log_shift ~f:srl value ~by:shift) ~width:4) in
      top bid @: top ask
    in
    let table =
      List.init 256 ~f:(fun index ->
        byte (imbalance ~bid:(index lsr 4) ~ask:(index land 15)))
    in
    let%hw top_index = normalize top_bid top_ask in
    let%hw depth_index = normalize depth_bid depth_ask in
    let features =
      concat_lsb
        [ reg (reg spread)
        ; reg (mux top_index table)
        ; reg (mux depth_index table)
        ; reg (reg flow)
        ]
    in
    let context =
      { Context.symbol = book.symbol
      ; bid = bid.price
      ; has_bid
      ; ask = ask.price
      ; has_ask
      ; t_first = book.t_first
      ; t_last = book.t_last
      }
    in
    { O.features =
        { valid = Clocking.pipeline clocking ~n:latency book.valid; value = features }
    ; context = Context.map context ~f:(Clocking.pipeline_no_clear clocking ~n:latency)
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"features" create i
  ;;
end
