open! Core
open! Hardcaml
open! Signal

let message_bytes = 40
let header_bytes = 20
let min_length = 12
let latency = 1

module Phase = struct
  type t =
    | Header
    | Length
    | Length_low
    | Body
    | Drop
  [@@deriving enumerate, sexp_of]

  let width = 3

  let code = function
    | Header -> 0
    | Length -> 1
    | Length_low -> 2
    | Body -> 3
    | Drop -> 4
  ;;

  let const t = of_unsigned_int ~width (code t)
end

module State = struct
  type 'a t =
    { phase : 'a [@bits Phase.width]
    ; skip : 'a [@bits 16]
    ; length : 'a [@bits 16]
    ; high : 'a [@bits 8]
    ; count : 'a [@bits 16]
    ; first : 'a [@bits 32]
    ; buffer : 'a [@bits 8 * message_bytes]
    }
  [@@deriving hardcaml]
end

module Beat = struct
  type 'a t =
    { data : 'a [@bits 64]
    ; keep : 'a [@bits 8]
    ; valid : 'a
    ; last : 'a
    }
  [@@deriving hardcaml]
end

module Message = struct
  type 'a t =
    { valid : 'a
    ; length : 'a [@bits 16]
    ; bytes : 'a [@bits 8 * message_bytes]
    ; t_first : 'a [@bits 32]
    ; t_last : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

let initial =
  { State.phase = Phase.const Header
  ; skip = of_unsigned_int ~width:16 header_bytes
  ; length = zero 16
  ; high = zero 8
  ; count = zero 16
  ; first = zero 32
  ; buffer = zero (8 * message_bytes)
  }
;;

module Step = struct
  module I = struct
    type 'a t =
      { state : 'a State.t
      ; beat : 'a Beat.t
      ; now : 'a [@bits 32]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { next : 'a State.t
      ; message : 'a Message.t
      }
    [@@deriving hardcaml]
  end

  (* All lanes at once; a message is at least 12 bytes, so a beat ends at most one. *)
  let create scope ({ state; beat; now } : Signal.t I.t) =
    let lanes = split_lsb ~part_width:8 beat.data in
    let keeps = List.map (bits_lsb beat.keep) ~f:(fun keep -> keep &: beat.valid) in
    let lane index = mux index lanes in
    let kept index = mux index keeps in
    let is phase = state.phase ==: Phase.const phase in
    let%hw in_header = is Header in
    let%hw in_body = is Body in
    let%hw in_length = is Length in
    let%hw in_length_low = is Length_low in
    let%hw counting = in_header |: in_body in
    let%hw prefix_ends = counting &: (state.skip <=:. 8) in
    let%hw boundary = counting &: (state.skip <:. 8) in
    let k = sel_bottom state.skip ~width:3 in
    let%hw header_count =
      let count =
        mux2 (state.skip ==:. 1) state.high (lane (k -:. 2)) @: lane (k -:. 1)
      in
      mux2 (count ==:. 0xffff) (zero 16) count
    in
    let%hw count_after = mux2 in_header header_count state.count in
    let%hw high_here = in_length |: (boundary &: (count_after <>:. 0)) in
    let high_lane = mux2 in_length (zero 3) k in
    let%hw high_byte = mux2 in_length_low state.high (lane high_lane) in
    let%hw low_lane = mux2 in_length_low (zero 4) (uresize high_lane ~width:4 +:. 1) in
    let%hw low_here = in_length_low |: (high_here &: (low_lane <:. 8)) in
    let%hw field = high_byte @: lane (sel_bottom low_lane ~width:3) in
    let%hw start = low_here &: (field >=:. min_length) in
    let%hw body_starts = start &: (low_lane <=:. 6) in
    let%hw base = state.length -: state.skip in
    let%hw complete =
      in_body &: prefix_ends &: kept (sel_bottom (state.skip -:. 1) ~width:3)
    in
    let message_byte j =
      let offset = of_unsigned_int ~width:16 j -: base in
      let index = sel_bottom offset ~width:3 in
      let hit = in_body &: (offset <:. 8) &: (offset <: state.skip) &: kept index in
      mux2 hit (lane index) state.buffer.:[(8 * j) + 7, 8 * j]
    in
    let message = List.init message_bytes ~f:message_byte in
    let buffer_byte j byte =
      if j >= 7
      then byte
      else (
        let index = of_unsigned_int ~width:4 (j + 1) +: low_lane in
        let hit = start &: (index <:. 8) &: kept (sel_bottom index ~width:3) in
        mux2 hit (lane (sel_bottom index ~width:3)) byte)
    in
    let%hw phase =
      priority_select_with_default
        [ { valid = start; value = Phase.const Body }
        ; { valid = low_here; value = Phase.const Drop }
        ; { valid = high_here; value = Phase.const Length_low }
        ; { valid = prefix_ends &: (count_after ==:. 0); value = Phase.const Drop }
        ; { valid = prefix_ends &: ~:boundary; value = Phase.const Length }
        ]
        ~default:state.phase
    in
    let next =
      { State.phase
      ; skip =
          mux2
            start
            (field +: uresize low_lane ~width:16 -:. 7)
            (mux2 counting (mux2 prefix_ends (zero 16) (state.skip -:. 8)) state.skip)
      ; length = mux2 start field state.length
      ; high =
          mux2
            high_here
            (lane high_lane)
            (mux2
               (in_header &: (state.skip >=:. 2) &: (state.skip <=:. 9))
               (lane (k -:. 2))
               state.high)
      ; count =
          mux2
            start
            (count_after -:. 1)
            (mux2 (in_header &: prefix_ends) header_count state.count)
      ; first = mux2 (body_starts |: (in_body &: (base ==:. 0))) now state.first
      ; buffer = state.buffer
      }
    in
    (* A packet's last beat resets everything but the buffer, whose writes keep gates. *)
    let next =
      State.map3 next initial state ~f:(fun next initial state ->
        mux2 beat.valid (mux2 beat.last initial next) state)
    in
    { O.next = { next with buffer = concat_lsb (List.mapi message ~f:buffer_byte) }
    ; message =
        { valid = complete
        ; length = state.length
        ; bytes = concat_lsb message
        ; t_first = state.first
        ; t_last = now
        }
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"moldudp64_step" create i
  ;;
end

module I = struct
  type 'a t =
    { clocking : 'a Clocking.t
    ; beat : 'a Beat.t
    ; now : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { message : 'a Message.t } [@@deriving hardcaml]
end

let create scope ({ clocking; beat; now } : Signal.t I.t) =
  let state = State.Of_signal.wires () in
  let step = Step.hierarchical scope { state; beat; now } in
  State.iter3 state step.next initial ~f:(fun state next initial ->
    state <-- Clocking.reg clocking ~clear_to:initial next);
  let message = Message.map step.message ~f:(Clocking.reg_no_clear clocking) in
  { O.message = { message with valid = Clocking.reg clocking step.message.valid } }
;;

let hierarchical ?instance scope i =
  let module H = Hierarchy.In_scope (I) (O) in
  H.hierarchical ?instance ~scope ~name:"moldudp64" create i
;;
