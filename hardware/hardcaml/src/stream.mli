(** Valid/ready handshake; a dropped [valid] must re-offer the same beat. *)

open! Core
open! Hardcaml

(** Flows from consumer to producer. *)
module Dest : sig
  type 'a t = { ready : 'a } [@@deriving hardcaml]
end

module type Width = sig
  val width : int
end

(** Flows from producer to consumer. *)
module Source (_ : Width) : sig
  type 'a t =
    { data : 'a
    ; valid : 'a
    }
  [@@deriving hardcaml]
end
