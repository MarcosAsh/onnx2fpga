(** Softmax, LogSoftmax and L1, L2 and MAX normalisation across the lanes of a vector. *)

open! Core
open! Hardcaml

module Mode : sig
  (** Divide: numerator times the reciprocal of the sum. Log: numerator minus its log.
      Root: numerator times the reciprocal square root of the sum, the scale table read at
      the leading one's parity above the mantissa. Max: numerator times the scale table at
      the largest code, with no sum and no normalisation. *)
  type t =
    | Divide
    | Log
    | Root
    | Max
  [@@deriving sexp_of]
end

(** Register stages added to the unit's own, each one a cycle of latency. *)
module Stages : sig
  (** Max and sum: cuts in those trees; table: 0 or 1; normalize: 0 to 2, the second
      splitting the leading one search in halves; rescale: 0 to 3. *)
  type t =
    { max : int
    ; table : int
    ; sum : int
    ; normalize : int
    ; rescale : int
    }
  [@@deriving sexp_of]

  val none : t
end

module type Config = sig
  val lanes : int

  (** Width of a signed input code. *)
  val in_bits : int

  (** Tables read at [max - x] when set, else at the code counted from the most negative. *)
  val subtract_max : bool

  val mode : Mode.t

  (** Unsigned values summed across the lanes, [2 ** in_bits] of them. *)
  val sum_table : int array

  val sum_bits : int

  (** Signed numerators; empty means the sum table, and then [numer_bits = sum_bits + 1]. *)
  val numer_table : int array

  val numer_bits : int

  (** Width of the sum, which must hold [lanes] times the largest sum table entry. *)
  val total_bits : int

  (** The sum is never below [2 ** min_exp] unless [min_exp] is 0. *)
  val min_exp : int

  val mantissa_bits : int

  (** At the [mantissa_bits] bits below the sum's leading one: a reciprocal or a log; for
      Max, at the largest code counted from the most negative. *)
  val scale_table : int array

  (** Log only: added to the scale for each leading one position from [min_exp] up. *)
  val exp_table : int array

  val scale_bits : int

  (** The rescale's shift: the reciprocal's plus [total_bits - 1], or Log's fraction. *)
  val shift : int

  val out_bits : int
  val out_zp : int
  val out_min : int
  val out_max : int
  val stages : Stages.t
end

module Make (_ : Config) : sig
  (** Four cycles, one more with the max tree, plus the stages; three for Max. *)
  val latency : int

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

  (** A zero sum gives [out_zp]; [clear] only clears valid. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
