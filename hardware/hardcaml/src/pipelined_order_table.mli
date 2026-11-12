(** Three-cycle reads include writes during the first two cycles; bit zero is live. *)

open! Core
open! Hardcaml

module type Config = sig
  val address_bits : int
  val data_bits : int
  val use_ultra : bool
end

module Make (_ : Config) : sig
  module Write : sig
    type 'a t =
      { address : 'a
      ; data : 'a
      ; enable : 'a
      }
    [@@deriving hardcaml]
  end

  (** B writes live entries; at most one live write and no same-slot dual writes. *)
  module I : sig
    type 'a t =
      { clock : 'a
      ; initialize : 'a
      (** Clear the slot at [write_a.address], ignoring write enables. *)
      ; prefer_b : 'a (** Select the payload writer; write A otherwise. *)
      ; payload_write : 'a
      ; write_a : 'a Write.t
      ; write_b : 'a Write.t
      ; read_a : 'a
      ; read_b : 'a
      }
    [@@deriving hardcaml]
  end

  (** [_first] outputs arrive after two cycles; dead payloads are unspecified. *)
  module O : sig
    type 'a t =
      { qa : 'a
      ; qb : 'a
      ; qa_first : 'a
      ; qb_first : 'a
      }
    [@@deriving hardcaml]
  end

  (** Payload writes cover live writes; extra writes preserve live payloads. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
