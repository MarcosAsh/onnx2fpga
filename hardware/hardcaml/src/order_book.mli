(** Top of book per symbol and side, over a direct mapped table of the orders behind it. *)

open! Core
open! Hardcaml

(** Cycles from [event.valid] to [book.valid]. *)
val latency : int

val flow_bits : int

module type Config = sig
  (** One to eight. *)
  val symbols : int

  val levels : int

  (** One to 20. *)
  val order_bits : int
end

module Make (_ : Config) : sig
  (** Levels are [Book_side] words, best in the low bits; [flow] is decayed signed volume. *)
  module Snapshot : sig
    type 'a t =
      { valid : 'a
      ; symbol : 'a
      ; bids : 'a
      ; asks : 'a
      ; flow : 'a
      ; t_first : 'a
      ; t_last : 'a
      }
    [@@deriving hardcaml]
  end

  (** Events must come at least two cycles apart, which the framer guarantees. *)
  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; event : 'a Itch_parser.Event.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { book : 'a Snapshot.t
      ; ready : 'a
      }
    [@@deriving hardcaml]
  end

  (** [ready] stays low for [2^order_bits] cycles after a clear while the table empties. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
