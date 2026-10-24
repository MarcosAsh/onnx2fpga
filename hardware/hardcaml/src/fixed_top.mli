(** A fixed latency design: its layers in a chain, one vector in and one out per cycle. *)

open! Core
open! Hardcaml

module type Config = sig
  (** Must be a [Fixed] design whose nodes form a chain from its input to its output. *)
  val design : Design.t
end

module Make (_ : Config) : sig
  (** Cycles from [features_valid] to [prediction_valid], the sum over the layers. *)
  val latency : int

  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; features : 'a
      ; features_valid : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { prediction : 'a
      ; prediction_valid : 'a
      }
    [@@deriving hardcaml]
  end

  (** [clear] is synchronous and active high; it clears the valid bits and nothing else. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** One Verilog file holding the top level, named [design.top], and every module under it. *)
val verilog : Design.t -> string
