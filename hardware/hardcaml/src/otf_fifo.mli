(** First word fall through buffer; a beat always spends at least one cycle inside. *)

open! Core
open! Hardcaml

module type Config = sig
  val width : int
  val depth : int
end

module Make (_ : Config) : sig
  module Beat : sig
    type 'a t =
      { data : 'a
      ; valid : 'a
      }
    [@@deriving hardcaml]
  end

  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; s : 'a Beat.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Beat.t
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
