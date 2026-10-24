(** Fork one stream to [fanout] consumers; a beat is held until every branch has taken it. *)

open! Core
open! Hardcaml

module type Config = sig
  val width : int
  val fanout : int
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
      ; m : 'a Stream.Dest.t list
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Beat.t list
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
