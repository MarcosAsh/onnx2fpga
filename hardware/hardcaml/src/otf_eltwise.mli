(** Elementwise operators on two or three vectors of signed codes, a fixed latency. *)

open! Core
open! Hardcaml

module Op : sig
  (** Where takes a condition first; the rest take two operands. *)
  type t =
    | Add
    | Mul
    | Less
    | Less_equal
    | Equal
    | Max
    | Min
    | And
    | Or
    | Where
  [@@deriving sexp_of, enumerate]
end

module Operand : sig
  (** An input port by position, or a code per lane; only Where takes constants. *)
  type t =
    | Port of int
    | Constant of int list
  [@@deriving sexp_of]
end

(** Register stages added to the unit's own, each one a cycle of latency. *)
module Stages : sig
  (** Product: after the multiplies; rescale: [Otf_requant] cuts, or after the tables. *)
  type t =
    { product : int
    ; rescale : int
    }
  [@@deriving sexp_of]

  val none : t
end

module type Config = sig
  val lanes : int

  (** Width of every input and output code. *)
  val bits : int

  val op : Op.t
  val operands : Operand.t list

  (** A term is [code - zero point]; a code is true when it is not its zero point. *)
  val zero_points : int list

  (** Add and the comparisons weight each term by these. *)
  val mults : int list

  (** Max, Min and Where read each port operand's output code here, most negative first. *)
  val tables : int array list

  (** The rescale multiplier after Add (1) or Mul. *)
  val mult : int

  val mult_bits : int

  (** Width of the accumulator, and of each operand into a rescale. *)
  val acc_bits : int

  val shift : int
  val out_zp : int
  val out_min : int
  val out_max : int
  val stages : Stages.t
end

module Make (_ : Config) : sig
  (** Two plus the stages for Add, Mul and the comparisons, one plus them otherwise. *)
  val latency : int

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; valid : 'a
      ; x : 'a list
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { y : 'a With_valid.t } [@@deriving hardcaml]
  end

  (** Every port must hold the same vector's data; comparisons, And and Or give 0 or 1. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
