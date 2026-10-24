open! Core
open! Hardcaml
open! Signal

let latency = 2
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

  module Table = Ram.Dual_port.Make (struct
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
    let current = Itch_parser.Event.map event ~f:(Clocking.reg_no_clear clocking) in
    let current = { current with valid = Clocking.reg clocking (event.valid &: ready) } in
    let is kind = current.kind ==:. Itch_parser.Kind.code kind in
    let table_a = Entry.Of_signal.wires () in
    let table_b = Entry.Of_signal.wires () in
    let%hw same_slot = slot current.order ==: slot current.new_order in
    let%hw hit =
      table_a.live
      &: (table_a.tag ==: tag current.order)
      &: (table_a.symbol ==: current.symbol)
    in
    let%hw executed =
      mux2 (current.shares <: table_a.shares) current.shares table_a.shares
    in
    let%hw reduce = is Execute |: is Cancel in
    let%hw ask = mux2 (is Add) current.side table_a.side in
    let%hw adding = is Add &: ~:(table_a.live) in
    let%hw replacing = is Replace &: hit &: (~:(table_b.live) |: same_slot) in
    let sides =
      List.init symbols ~f:(fun _ -> Signal.wire (levels * 64), Signal.wire (levels * 64))
    in
    let index = current.symbol in
    let bids = mux index (List.map sides ~f:fst) in
    let asks = mux index (List.map sides ~f:snd) in
    let update =
      Side.hierarchical
        scope
        { levels = mux2 ask asks bids
        ; ask
        ; opposite = sel_bottom (mux2 ask bids asks) ~width:64
        ; remove =
            { valid = hit &: ~:(is Add)
            ; price = table_a.price
            ; shares = mux2 reduce executed table_a.shares
            }
        ; insert =
            { valid = adding |: replacing
            ; price = current.price
            ; shares = current.shares
            }
        }
    in
    let%hw remaining = table_a.shares -: executed in
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
    let%hw stored = is Add &: update.accepted in
    let%hw changed = hit &: (reduce |: is Delete) in
    let%hw freed = is Replace &: hit &: ~:(same_slot &: update.accepted) in
    let%hw write_a = current.valid &: (stored |: changed |: freed) in
    let entry_a =
      Entry.Of_signal.mux2
        (is Add)
        (added ~order:current.order ~side:current.side)
        (Entry.Of_signal.mux2
           (reduce &: (remaining <>:. 0))
           { table_a with shares = remaining }
           empty)
    in
    let%hw write_b = current.valid &: is Replace &: update.accepted in
    let port ~write ~address ~data ~read_address =
      { Table.Port.address = mux2 write address read_address
      ; data = Entry.Of_signal.pack data
      ; enable = write |: event.valid
      ; write
      }
    in
    let table =
      Table.hierarchical
        ~name:"order_table"
        scope
        { clock = clocking.clock
        ; port_a =
            port
              ~write:(write_a |: ~:ready)
              ~address:(mux2 ready (slot current.order) (lsbs sweep))
              ~data:(Entry.Of_signal.mux2 ready entry_a empty)
              ~read_address:(slot event.order)
        ; port_b =
            port
              ~write:write_b
              ~address:(slot current.new_order)
              ~data:(added ~order:current.new_order ~side:table_a.side)
              ~read_address:(slot event.new_order)
        }
    in
    Entry.Of_signal.assign table_a (Entry.Of_signal.unpack table.qa);
    Entry.Of_signal.assign table_b (Entry.Of_signal.unpack table.qb);
    let flows = List.init symbols ~f:(fun _ -> Signal.wire flow_bits) in
    let flow = mux index flows in
    let%hw traded = current.valid &: is Execute &: hit in
    let%hw flow_next =
      let capped = mux2 (executed >:. 0xffff) (ones 16) (sel_bottom executed ~width:16) in
      let amount = uresize capped ~width:flow_bits in
      flow -: sra flow ~by:3 +: mux2 table_a.side amount (negate amount)
    in
    List.iteri sides ~f:(fun s (bid_levels, ask_levels) ->
      let enable side = current.valid &: (index ==:. s) &: (ask ==: side) in
      bid_levels <-- Clocking.reg clocking ~enable:(enable gnd) update.updated;
      ask_levels <-- Clocking.reg clocking ~enable:(enable vdd) update.updated);
    List.iteri flows ~f:(fun s f ->
      f <-- Clocking.reg clocking ~enable:(traded &: (index ==:. s)) flow_next);
    let book =
      { Snapshot.valid = current.valid
      ; symbol = index
      ; bids = mux2 ask bids update.updated
      ; asks = mux2 ask update.updated asks
      ; flow = mux2 traded flow_next flow
      ; t_first = current.t_first
      ; t_last = current.t_last
      }
    in
    let registered = Snapshot.map book ~f:(Clocking.reg_no_clear clocking) in
    { O.book = { registered with valid = Clocking.reg clocking book.valid }; ready }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"order_book" create i
  ;;
end
