(** Decodes A, F, E, C, X, D and U for a short list of stock locates, dropping the rest. *)

open! Core
open! Hardcaml

(** Cycles from [message.valid] to [event.valid]. *)
val latency : int

val max_symbols : int

module Kind : sig
  (** E and C are both [Execute]; C's execution price does not change the book. *)
  type t =
    | Add
    | Execute
    | Cancel
    | Delete
    | Replace
  [@@deriving enumerate, sexp_of]

  val code : t -> int
end

(** [symbol] indexes [locates]; [order] is the original reference for [Replace]. *)
module Event : sig
  type 'a t =
    { valid : 'a
    ; kind : 'a
    ; symbol : 'a
    ; side : 'a
    ; order : 'a
    ; new_order : 'a
    ; shares : 'a
    ; price : 'a
    ; t_first : 'a
    ; t_last : 'a
    }
  [@@deriving hardcaml]
end

module type Config = sig
  (** One to [max_symbols] distinct stock locates. *)
  val locates : int list
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; message : 'a Moldudp64.Message.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { event : 'a Event.t } [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
