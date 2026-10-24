(** A threshold on the model's prediction, and the prebuilt order it sends. *)

open! Core
open! Hardcaml

(** Cycles from [prediction.valid] to [valid]. *)
val latency : int

val order_bytes : int

module Action : sig
  type t =
    | Hold
    | Buy
    | Sell
  [@@deriving enumerate, sexp_of]

  val code : t -> int
end

module type Config = sig
  (** The 8 character symbol of each tracked stock locate, in order. *)
  val names : string list

  (** Signed, and above [sell_threshold]. *)
  val buy_threshold : int

  val sell_threshold : int
  val order_shares : int
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; prediction : 'a With_valid.t
      ; context : 'a Features.Context.t
      }
    [@@deriving hardcaml]
  end

  (** [order] has byte 0 in the low bits and is valid when [action] is not [Hold]. *)
  module O : sig
    type 'a t =
      { valid : 'a
      ; action : 'a
      ; prediction : 'a
      ; order : 'a With_valid.t
      ; t_first : 'a
      ; t_last : 'a
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
