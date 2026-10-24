open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val in_width : int
  val out_width : int
  val rx_depth : int
  val tx_depth : int
end

let bytes_of_width width = (width + 7) / 8

module Make (Config : Config) = struct
  open Config

  let () =
    if in_width < 1 || out_width < 1
    then
      raise_s [%message "BUG: uart_bridge needs beats" (in_width : int) (out_width : int)];
    if rx_depth < 2 || tx_depth < 2
    then
      raise_s
        [%message
          "BUG: uart_bridge needs two bytes of buffer" (rx_depth : int) (tx_depth : int)]
  ;;

  let in_bytes = bytes_of_width in_width
  let out_bytes = bytes_of_width out_width

  module Beat_in = Stream.Source (struct
      let width = in_width
    end)

  module Beat_out = Stream.Source (struct
      let width = out_width
    end)

  module Byte = Stream.Source (struct
      let width = 8
    end)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; rx : 'a With_valid.t [@bits 8]
      ; accel_in : 'a Stream.Dest.t
      ; accel_out : 'a Beat_out.t
      ; tx : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { accel_in : 'a Beat_in.t
      ; accel_out : 'a Stream.Dest.t
      ; tx : 'a Byte.t
      ; overflow : 'a
      }
    [@@deriving hardcaml]
  end

  let byte_fifo scope (i : Signal.t I.t) ~name ~capacity ~wr ~d ~rd =
    Fifo.create
      ~showahead:true
      ~scope:(Scope.sub_scope scope name)
      ()
      ~capacity
      ~clock:i.clocking.clock
      ~clear:i.clocking.clear
      ~wr
      ~d
      ~rd
  ;;

  (* Bytes in, least significant first; the last byte is offered straight from the buffer. *)
  let deserialise scope (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw rx_take = wire 1 in
    let rx_fifo =
      byte_fifo
        scope
        i
        ~name:"rx_fifo"
        ~capacity:rx_depth
        ~wr:i.rx.valid
        ~d:i.rx.value
        ~rd:rx_take
    in
    let%hw rx_have = ~:(rx_fifo.empty) in
    let%hw overflow = i.rx.valid &: rx_fifo.full in
    if in_bytes = 1
    then (
      rx_take <-- (rx_have &: i.accel_in.ready);
      let%hw beat = sel_bottom rx_fifo.q ~width:in_width in
      { Beat_in.data = beat; valid = rx_have }, overflow)
    else (
      let count_bits = num_bits_to_represent (in_bytes - 1) in
      let%hw filled = wire count_bits in
      let%hw last = filled ==:. in_bytes - 1 in
      rx_take <-- (rx_have &: (~:last |: i.accel_in.ready));
      filled
      <-- reg spec (mux2 rx_take (mux2 last (zero count_bits) (filled +:. 1)) filled);
      let%hw collected = wire (8 * (in_bytes - 1)) in
      let shifted = List.tl_exn (split_lsb ~part_width:8 collected) @ [ rx_fifo.q ] in
      collected <-- reg spec ~enable:(rx_take &: ~:last) (concat_lsb shifted);
      let%hw beat = sel_bottom (rx_fifo.q @: collected) ~width:in_width in
      { Beat_in.data = beat; valid = rx_have &: last }, overflow)
  ;;

  (* Beats in, bytes out; a beat waits for the previous one's last byte. *)
  let serialise scope (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw remaining = wire (num_bits_to_represent out_bytes) in
    let%hw holding = remaining <>:. 0 in
    let%hw load = i.accel_out.valid &: ~:holding in
    let%hw tx_push = wire 1 in
    let%hw pending = wire (8 * out_bytes) in
    remaining
    <-- reg
          spec
          (mux2 load (of_unsigned_int ~width:(width remaining) out_bytes)
           @@ mux2 tx_push (remaining -:. 1) remaining);
    let shifted = List.tl_exn (split_lsb ~part_width:8 pending) @ [ zero 8 ] in
    pending
    <-- reg
          spec
          (mux2 load (uresize i.accel_out.data ~width:(8 * out_bytes))
           @@ mux2 tx_push (concat_lsb shifted) pending);
    let%hw tx_pop = wire 1 in
    let tx_fifo =
      byte_fifo
        scope
        i
        ~name:"tx_fifo"
        ~capacity:tx_depth
        ~wr:tx_push
        ~d:(sel_bottom pending ~width:8)
        ~rd:tx_pop
    in
    tx_push <-- (holding &: ~:(tx_fifo.full));
    let%hw tx_have = ~:(tx_fifo.empty) in
    tx_pop <-- (tx_have &: i.tx.ready);
    { Stream.Dest.ready = ~:holding }, { Byte.data = tx_fifo.q; valid = tx_have }
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let accel_in, overflow = deserialise scope i in
    let accel_out, tx = serialise scope i in
    { O.accel_in; accel_out; tx; overflow }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"uart_bridge" create i
  ;;
end
