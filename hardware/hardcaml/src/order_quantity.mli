(** Cap a requested reduction at the held quantity and return the remainder. *)

open! Core
open! Hardcaml

module I : sig
  type 'a t =
    { held : 'a
    ; requested : 'a
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { executed : 'a
    ; remaining : 'a
    }
  [@@deriving hardcaml]
end

val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
