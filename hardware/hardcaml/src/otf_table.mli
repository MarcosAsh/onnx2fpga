(** Constant lookup tables on signed codes, one vector per cycle, one cycle late. *)

open! Core
open! Hardcaml

module type Config = sig
  val lanes : int

  (** Width of a signed input code, 1 to 12 bits. *)
  val in_bits : int

  val out_bits : int

  (** Each holds [2 ** in_bits] signed outputs, the most negative input code first. *)
  val tables : int array list

  (** The table each lane reads, by position in [tables]. *)
  val lane_table : int list
end

module Make (_ : Config) : sig
  (** Always 1: the table, then a register. *)
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

  (** Lane [k] of [y] is its table at lane [k] of [x]; [clear] only clears valid. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
