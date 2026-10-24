open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val clocks_per_bit : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if clocks_per_bit < 1
    then raise_s [%message "BUG: uart_tx needs a bit time" (clocks_per_bit : int)]
  ;;

  let timer_bits = num_bits_to_represent (clocks_per_bit - 1)

  (* start, eight data bits, stop *)
  let frame_bits = 10

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; byte : 'a With_valid.t [@bits 8]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { ready : 'a
      ; txd : 'a
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw bits_left = wire (num_bits_to_represent frame_bits) in
    let%hw timer = wire timer_bits in
    let%hw idle = bits_left ==:. 0 in
    let%hw bit_done = ~:idle &: (timer ==:. 0) in
    (* The last cycle of the stop bit can take the next byte, so they leave no gap. *)
    let%hw ready = idle |: (bit_done &: (bits_left ==:. 1)) in
    let%hw start = i.byte.valid &: ready in
    timer
    <-- reg
          spec
          (mux2
             (start |: bit_done)
             (of_unsigned_int ~width:timer_bits (clocks_per_bit - 1))
           @@ mux2 idle timer (timer -:. 1));
    bits_left
    <-- reg
          spec
          (mux2 start (of_unsigned_int ~width:(width bits_left) frame_bits)
           @@ mux2 bit_done (bits_left -:. 1) bits_left);
    (* Fills with the idle level so the line is a flop output and rests high. *)
    let%hw frame = wire frame_bits in
    frame
    <-- reg
          spec
          ~clear_to:(ones frame_bits)
          (mux2 start (vdd @: i.byte.value @: gnd)
           @@ mux2 bit_done (vdd @: frame.:[frame_bits - 1, 1]) frame);
    { O.ready; txd = frame.:(0) }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"uart_tx" create i
  ;;
end
