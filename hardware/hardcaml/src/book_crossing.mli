(** Compare both best prices before selecting the order's side. *)

open! Core
open! Hardcaml

module I : sig
  type 'a t =
    { ask : 'a
    ; price : 'a
    ; bid : 'a Book_side.Level.t
    ; offer : 'a Book_side.Level.t
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t = { crosses : 'a } [@@deriving hardcaml]
end

val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
