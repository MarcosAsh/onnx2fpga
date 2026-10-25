(** Reductions across the lanes of a vector of signed codes, a fixed latency. *)

open! Core
open! Hardcaml

module Op : sig
  (** The arg operators give a lane index; Sum rescales onto an output grid. *)
  type t =
    | Arg_max
    | Arg_min
    | Max
    | Min
    | Sum
  [@@deriving sexp_of, enumerate]
end

(** Register stages added to the unit's own, each one a cycle of latency. *)
module Stages : sig
  (** Tree: cuts spread over the tree's levels; rescale: [Otf_requant] cuts, Sum only. *)
  type t =
    { tree : int
    ; rescale : int
    }
  [@@deriving sexp_of]

  val none : t
end

module type Config = sig
  val lanes : int

  (** Width of every input code. *)
  val bits : int

  val op : Op.t

  (** On a tie the arg operators return the last lane, not the first. *)
  val last : bool

  (** Sum adds [-lanes * zero_point] to the codes before the rescale. *)
  val zero_point : int

  val mult : int
  val mult_bits : int
  val acc_bits : int
  val shift : int
  val out_zp : int
  val out_min : int
  val out_max : int
  val stages : Stages.t
end

(** Pairs neighbours level by level with [f], registering after each level in [cuts]. *)
val tree
  :  Signal.t Clocking.t
  -> cuts:int list
  -> f:(Signal.t -> Signal.t -> Signal.t)
  -> Signal.t list
  -> Signal.t

(** The levels after which [stages] registers go, spread evenly over [levels]. *)
val cuts : levels:int -> stages:int -> int list

module Make (_ : Config) : sig
  (** One plus the tree stages, and one more plus the rescale stages for Sum. *)
  val latency : int

  (** [max 1 (ceil_log2 lanes)] for the arg operators, [bits] otherwise. *)
  val out_bits : int

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; x : 'a With_valid.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { y : 'a With_valid.t } [@@deriving hardcaml]
  end

  (** [clear] only clears valid. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
