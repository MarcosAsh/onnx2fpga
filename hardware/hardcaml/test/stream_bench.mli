(** Drives streams into a unit with random gaps on every side, like the Verilator harness. *)

open! Core
open! Hardcaml

module Source : sig
  type t

  (** [ready] is the unit's output, read before the edge. *)
  val create
    :  data:Bits.t ref
    -> valid:Bits.t ref
    -> ready:Bits.t ref
    -> duty:int
    -> Bits.t list
    -> t
end

module Sink : sig
  type t

  (** [data] and [valid] are the unit's outputs, read before the edge. *)
  val create
    :  data:Bits.t ref
    -> valid:Bits.t ref
    -> ready:Bits.t ref
    -> duty:int
    -> wanted:int
    -> t

  val received : t -> Bits.t list
end

(** Runs until every sink has its beats; raises after [timeout] cycles. *)
val run
  :  Hardcaml_lws.Lws.Handler.t @ local
  -> rng:Random.State.t
  -> ?timeout:int
  -> ?each_cycle:(unit -> unit)
  -> sources:Source.t list
  -> sinks:Sink.t list
  -> unit
  -> int

(** Holds [clear] high for two cycles. *)
val clear : Hardcaml_lws.Lws.Handler.t @ local -> Bits.t ref -> unit

val random_bits : Random.State.t -> width:int -> Bits.t
val beats : rng:Random.State.t -> width:int -> int -> Bits.t list
val signed_lanes : lane_bits:int -> Bits.t -> int list
val of_signed_lanes : lane_bits:int -> int list -> Bits.t
