(** A frozen tree ensemble: one feature vector per cycle, a fixed latency. *)

open! Core
open! Hardcaml

(** [x.(feature) <= threshold], or [=] when [equal], on signed feature codes. *)
module Split : sig
  type t =
    { feature : int
    ; threshold : int
    ; equal : bool
    }
  [@@deriving compare, sexp_of]
end

module Tree : sig
  (** A leaf holds a signed leaf code. *)
  type t =
    | Leaf of int
    | Node of
        { split : Split.t
        ; if_true : t
        ; if_false : t
        }
  [@@deriving sexp_of]

  (** The leaf code that [x] reaches. *)
  val eval : t -> int array -> int
end

module type Config = sig
  val features : int

  (** Width of a signed feature code and of every threshold. *)
  val feature_bits : int

  val leaf_bits : int

  (** Width of the sum; the leaves plus [bias] must not overflow it. *)
  val sum_bits : int

  val trees : Tree.t list
  val bias : int

  (** Output one bit, set when the sum is not negative, instead of the sum. *)
  val decision : bool
end

(** Register stages an adder tree of fan in 4 needs for this many operands. *)
val adder_stages : int -> int

module Make (_ : Config) : sig
  (** Compare, select, then [adder_stages (trees + 1)] stages for leaves and bias. *)
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

  (** Identical splits share one comparator; [clear] only clears valid. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
