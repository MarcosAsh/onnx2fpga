(** Order payloads with separate creation and deletion bits; bit zero is live. *)

open! Core
open! Hardcaml

module type Config = sig
  val address_bits : int
  val data_bits : int
end

module Make (_ : Config) : sig
  module Port : sig
    type 'a t =
      { address : 'a
      ; data : 'a
      ; enable : 'a
      ; write : 'a
      }
    [@@deriving hardcaml]
  end

  (** B writes live entries; at most one live write and no same-slot dual writes. *)
  module I : sig
    type 'a t =
      { clock : 'a
      ; initialize : 'a
      ; prefer_b : 'a
      ; payload_write : 'a
      (** Covers live writes; extra writes must leave the target dead. *)
      ; port_a : 'a Port.t
      ; port_b : 'a Port.t
      }
    [@@deriving hardcaml]
  end

  (** Payload bits are unspecified when the live bit is clear. *)
  module O : sig
    type 'a t =
      { qa : 'a
      ; qb : 'a
      }
    [@@deriving hardcaml]
  end

  (** Initialize each address on A; [prefer_b] selects the live payload writer. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
