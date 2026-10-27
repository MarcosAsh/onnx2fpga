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

  (** Splits on the longest path from the root to a leaf. *)
  val depth : t -> int

  (** The leaf code that [x] reaches. *)
  val eval : t -> int array -> int
end

(** One sum: its trees' leaves plus [bias]. *)
module Score : sig
  type t =
    { trees : Tree.t list
    ; bias : int
    }
  [@@deriving sexp_of]
end

(** Register stages added to the ensemble's own, each one a cycle of latency. *)
module Stages : sig
  (** Select: cuts in the leaf select, under the depth; sum: more adder levels registered;
      pick: more levels of the compare tree over several scores registered. *)
  type t =
    { select : int
    ; sum : int
    ; pick : int
    }
  [@@deriving sexp_of]

  val none : t
end

module type Config = sig
  val features : int

  (** Width of a signed feature code and of every threshold. *)
  val feature_bits : int

  val leaf_bits : int

  (** Width of every score; no score may overflow it. *)
  val sum_bits : int

  (** With one score the output is that score; with more it is the index of the largest,
      the first of a tie, in [max 1 (ceil_log2 scores)] bits. *)
  val scores : Score.t list

  (** With one score, output one bit, set when the score is not negative, instead. *)
  val decision : bool

  val stages : Stages.t
end

(** Register stages an adder tree of fan in 4 needs for this many operands. *)
val adder_stages : int -> int

module Make (_ : Config) : sig
  (** Compare, select, [adder_stages (trees + 1)] adder stages for the score with the most
      trees, one stage to pick among several scores, and [stages]. *)
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
