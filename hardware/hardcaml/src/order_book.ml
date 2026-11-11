open! Core
open! Hardcaml
open! Signal

let latency = 4
let flow_bits = 20

module type Config = sig
  val symbols : int
  val levels : int
  val order_bits : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if symbols < 1 || symbols > Itch_parser.max_symbols
    then raise_s [%message "BUG: order_book takes 1 to 8 symbols" (symbols : int)];
    if order_bits < 1 || order_bits > 20
    then
      raise_s
        [%message "BUG: order_book table must be 2 to 2^20 slots" (order_bits : int)]
  ;;

  let tag_bits = 64 - order_bits

  module Entry = struct
    type 'a t =
      { live : 'a
      ; tag : 'a [@bits tag_bits]
      ; symbol : 'a [@bits 3]
      ; side : 'a
      ; price : 'a [@bits 32]
      ; shares : 'a [@bits 32]
      }
    [@@deriving hardcaml]
  end

  module Table = Pipelined_order_table.Make (struct
      let address_bits = order_bits
      let data_bits = Entry.sum_of_port_widths
    end)

  module Side = Book_side.Make (struct
      let levels = levels
    end)

  module Snapshot = struct
    type 'a t =
      { valid : 'a
      ; symbol : 'a [@bits 3]
      ; bids : 'a [@bits levels * 64]
      ; asks : 'a [@bits levels * 64]
      ; flow : 'a [@bits flow_bits]
      ; t_first : 'a [@bits 32]
      ; t_last : 'a [@bits 32]
      }
    [@@deriving hardcaml]
  end

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; event : 'a Itch_parser.Event.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { book : 'a Snapshot.t
      ; ready : 'a
      }
    [@@deriving hardcaml]
  end

  let create scope ({ clocking; event } : Signal.t I.t) =
    let slot order = sel_bottom order ~width:order_bits in
    let tag order = sel_top order ~width:tag_bits in
    (* After a clear, port A writes every slot empty before the book takes events. *)
    let%hw sweep =
      Clocking.reg_fb clocking ~width:(order_bits + 1) ~f:(fun s ->
        mux2 (msb s) s (s +:. 1))
    in
    let%hw ready = msb sweep in
    let event_reg value =
      add_attribute
        (Clocking.reg_no_clear clocking value)
        Rtl_attribute.Vivado.Srl_style.register
    in
    let first = Itch_parser.Event.map event ~f:event_reg in
    let first = { first with valid = Clocking.reg clocking (event.valid &: ready) } in
    let current = Itch_parser.Event.map first ~f:event_reg in
    let current = { current with valid = Clocking.reg clocking first.valid } in
    let is kind = current.kind ==:. Itch_parser.Kind.code kind in
    let table_a = Entry.Of_signal.wires () in
    let table_b = Entry.Of_signal.wires () in
    let first_a = Entry.Of_signal.wires () in
    let%hw same_slot = slot current.order ==: slot current.new_order in
    let%hw hit =
      Clocking.reg_no_clear
        clocking
        (first_a.live
         &: (first_a.tag ==: tag first.order)
         &: (first_a.symbol ==: first.symbol))
    in
    let quantity =
      Order_quantity.hierarchical
        scope
        { held = first_a.shares; requested = first.shares }
    in
    let executed = Clocking.reg_no_clear clocking quantity.executed in
    let%hw reduce = is Execute |: is Cancel in
    let%hw ask = mux2 (is Add) current.side table_a.side in
    let%hw adding = is Add &: ~:(table_a.live) in
    let%hw replacing = is Replace &: hit &: (~:(table_b.live) |: same_slot) in
    let sides =
      List.init symbols ~f:(fun _ -> Signal.wire (levels * 64), Signal.wire (levels * 64))
    in
    let index = current.symbol in
    let early_bid = wire (levels * 64) in
    let early_offer = wire (levels * 64) in
    let bids = Clocking.reg_no_clear clocking early_bid in
    let asks = Clocking.reg_no_clear clocking early_offer in
    let crossing =
      Book_crossing.hierarchical
        scope
        { ask = mux2 (first.kind ==:. Itch_parser.Kind.code Add) first.side first_a.side
        ; price = first.price
        ; bid = Book_side.Level.of_word (sel_bottom early_bid ~width:64)
        ; offer = Book_side.Level.of_word (sel_bottom early_offer ~width:64)
        }
    in
    let crosses = Clocking.reg_no_clear clocking crossing.crosses in
    let update book_levels ask =
      Side.Pipelined.hierarchical
        scope
        { clocking
        ; book =
            { Side.I.levels = book_levels
            ; ask
            ; opposite = zero 64
            ; remove =
                { valid = hit &: ~:(is Add)
                ; price = table_a.price
                ; shares = mux2 reduce executed table_a.shares
                }
            ; insert =
                { valid = adding |: replacing &: ~:crosses
                ; price = current.price
                ; shares = current.shares
                }
            }
        }
    in
    let bid_update = update bids gnd in
    let ask_update = update asks vdd in
    let accepted = bid_update.accepted in
    let remaining =
      add_attribute
        (Clocking.reg_no_clear clocking quantity.remaining)
        (Rtl_attribute.Vivado.extract_reset false)
    in
    let added ~order ~side =
      { Entry.live = vdd
      ; tag = tag order
      ; symbol = current.symbol
      ; side
      ; price = current.price
      ; shares = current.shares
      }
    in
    let empty = Entry.map Entry.port_widths ~f:zero in
    let%hw stored = is Add &: accepted in
    let%hw changed = hit &: (reduce |: is Delete) in
    let%hw freed = is Replace &: hit &: ~:(same_slot &: accepted) in
    let%hw write_a = current.valid &: (stored |: changed |: freed) in
    let entry_a =
      Entry.Of_signal.mux2
        (is Add)
        (Entry.Of_signal.mux2
           adding
           (added ~order:current.order ~side:current.side)
           table_a)
        (Entry.Of_signal.mux2
           hit
           (Entry.Of_signal.mux2
              (reduce &: (remaining <>:. 0))
              { table_a with shares = remaining }
              empty)
           table_a)
    in
    let%hw write_b = current.valid &: is Replace &: accepted in
    let port ~write ~address ~data =
      { Table.Write.address; data = Entry.Of_signal.pack data; enable = write }
    in
    let table =
      Table.hierarchical
        scope
        { clock = clocking.clock
        ; initialize = ~:ready
        ; prefer_b = is Replace
        ; payload_write =
            current.valid
            &: (is Add |: reduce |: (is Replace &: (~:(table_b.live) |: same_slot)))
        ; read_a = slot event.order
        ; read_b = slot event.new_order
        ; write_a =
            port
              ~write:(write_a |: ~:ready)
              ~address:(mux2 ready (slot current.order) (lsbs sweep))
              ~data:(Entry.Of_signal.mux2 ready entry_a empty)
        ; write_b =
            port
              ~write:write_b
              ~address:(slot current.new_order)
              ~data:
                (Entry.Of_signal.mux2
                   (same_slot &: ~:hit)
                   table_a
                   (added ~order:current.new_order ~side:table_a.side))
        }
    in
    Entry.Of_signal.assign first_a (Entry.Of_signal.unpack table.qa_first);
    Entry.Of_signal.assign table_a (Entry.Of_signal.unpack table.qa);
    Entry.Of_signal.assign table_b (Entry.Of_signal.unpack table.qb);
    let delayed = Clocking.reg_no_clear clocking in
    let%hw updating = Clocking.reg clocking current.valid in
    let%hw update_index = delayed index in
    let%hw update_ask = delayed ask in
    let flows = List.init symbols ~f:(fun _ -> Signal.wire flow_bits) in
    let flow = mux update_index flows in
    let%hw traded = Clocking.reg clocking (current.valid &: is Execute &: hit) in
    let%hw trade_side = delayed table_a.side in
    let%hw trade_shares =
      delayed (mux2 (executed >:. 0xffff) (ones 16) (sel_bottom executed ~width:16))
    in
    let%hw flow_next =
      let amount = uresize trade_shares ~width:flow_bits in
      flow -: sra flow ~by:3 +: mux2 trade_side amount (negate amount)
    in
    List.iteri sides ~f:(fun s (bid_levels, ask_levels) ->
      let enable side = updating &: (update_index ==:. s) &: (update_ask ==: side) in
      bid_levels <-- Clocking.reg clocking ~enable:(enable gnd) bid_update.updated;
      ask_levels <-- Clocking.reg clocking ~enable:(enable vdd) ask_update.updated);
    List.iteri flows ~f:(fun s f ->
      f <-- Clocking.reg clocking ~enable:(traded &: (update_index ==:. s)) flow_next);
    let book =
      { Snapshot.valid = updating
      ; symbol = update_index
      ; bids = mux2 update_ask (delayed bids) bid_update.updated
      ; asks = mux2 update_ask ask_update.updated (delayed asks)
      ; flow = mux2 traded flow_next flow
      ; t_first = delayed current.t_first
      ; t_last = delayed current.t_last
      }
    in
    let pending = updating &: (update_index ==: first.symbol) in
    let read side = mux first.symbol (List.map sides ~f:side) in
    early_bid <-- mux2 pending book.bids (read fst);
    early_offer <-- mux2 pending book.asks (read snd);
    let registered = Snapshot.map book ~f:(Clocking.reg_no_clear clocking) in
    { O.book = { registered with valid = Clocking.reg clocking book.valid }; ready }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"order_book" create i
  ;;
end
