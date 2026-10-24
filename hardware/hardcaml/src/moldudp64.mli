(** MoldUDP64 framing: UDP payload bytes in, 64 bits a beat, whole messages out. *)

open! Core
open! Hardcaml

(** Bytes of each message kept, enough for the longest ITCH message we decode. *)
val message_bytes : int

val header_bytes : int

(** The shortest ITCH 5.0 message; it keeps every block longer than a beat. *)
val min_length : int

(** Cycles from the beat holding a message's last byte to [message.valid]. *)
val latency : int

module Phase : sig
  type t =
    | Header
    | Length
    | Length_low
    | Body
    | Drop
  [@@deriving enumerate, sexp_of]

  val code : t -> int
end

(** [skip] counts header or body bytes left; [high] keeps a split length's first byte. *)
module State : sig
  type 'a t =
    { phase : 'a
    ; skip : 'a
    ; length : 'a
    ; high : 'a
    ; count : 'a
    ; first : 'a
    ; buffer : 'a
    }
  [@@deriving hardcaml]
end

(** Byte [i] in bits [8i+7:8i]; [keep] is all ones except on a packet's [last] beat. *)
module Beat : sig
  type 'a t =
    { data : 'a
    ; keep : 'a
    ; valid : 'a
    ; last : 'a
    }
  [@@deriving hardcaml]
end

(** Byte 0 in the low bits; bytes past [length] are left over from earlier messages. *)
module Message : sig
  type 'a t =
    { valid : 'a
    ; length : 'a
    ; bytes : 'a
    ; t_first : 'a
    ; t_last : 'a
    }
  [@@deriving hardcaml]
end

(** The state at the start of a packet. *)
val initial : Signal.t State.t

(** One beat of the framer as a combinational function, for the formal check. *)
module Step : sig
  module I : sig
    type 'a t =
      { state : 'a State.t
      ; beat : 'a Beat.t
      ; now : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { next : 'a State.t
      ; message : 'a Message.t
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

module I : sig
  type 'a t =
    { clocking : 'a Clocking.t
    ; beat : 'a Beat.t
    ; now : 'a
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { message : 'a Message.t } [@@deriving hardcaml]
end

(** [now] is the cycle counter, stamped on each message's first and last beat. *)
val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
