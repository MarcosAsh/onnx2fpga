(** Spread, top and depth imbalance and trade flow, as signed bytes, with no division. *)

open! Core
open! Hardcaml

(** Cycles from [book.valid] to [features.valid]. *)
val latency : int

val count : int

(** The best prices the decision needs, delayed to line up with the features. *)
module Context : sig
  type 'a t =
    { symbol : 'a
    ; bid : 'a
    ; has_bid : 'a
    ; ask : 'a
    ; has_ask : 'a
    ; t_first : 'a
    ; t_last : 'a
    }
  [@@deriving hardcaml]
end

(** [127 * (bid - ask) / (bid + ask)] rounded half away from zero, and 0 for 0 and 0. *)
val imbalance : bid:int -> ask:int -> int

module type Config = sig
  val levels : int
  val spread_shift : int
  val flow_shift : int
end

module Make (_ : Config) : sig
  (** The fields of [Order_book.Make(_).Snapshot]. *)
  module Book : sig
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

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; book : 'a Book.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { features : 'a With_valid.t
      ; context : 'a Context.t
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
