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

  (** Words in one weight bank; the reload ports exist only when this is not 0. *)
  val words : int

  val reloadable : bool

  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; features : 'a
      ; features_valid : 'a
      ; weights_write : 'a option
      ; weights_address : 'a option
      ; weights_data : 'a option
      ; weights_commit : 'a option
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { prediction : 'a
      ; prediction_valid : 'a
      ; prediction_bank : 'a option
      ; weights_bank : 'a option
      ; weights_version : 'a option
      ; weights_crc : 'a option
      ; weights_idle_crc : 'a option
      }
    [@@deriving hardcaml]
  end

  (** [clear] clears the valid bits only; [weights_commit] makes the idle bank active. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** One Verilog file holding the top level, named [design.top], and every module under it. *)
val verilog : Design.t -> string
