(** Fixed point rescale shared by every compute unit, combinational. *)

open! Core
open! Hardcaml

module type Config = sig
  val acc_bits : int
  val mult_bits : int
  val shift : int
  val out_bits : int
  val out_min : int
  val out_max : int
  val out_zp : int
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { acc : 'a
      ; mult : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { out : 'a } [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

module type Pipelined_config = sig
  include Config

  (** How many of the cuts after the multiply, rounding and zero point hold a register. *)
  val stages : int
end

(** The same rescale, [stages] cycles late, with registers a DSP slice can absorb. *)
module Make_pipelined (_ : Pipelined_config) : sig
  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; acc : 'a
      ; mult : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t = { out : 'a } [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
