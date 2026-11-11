(** One side of one symbol's book, best level first. *)

open! Core
open! Hardcaml

module Level : sig
  type 'a t =
    { price : 'a
    ; shares : 'a
    }
  [@@deriving hardcaml]

  val of_word : Signal.t -> Signal.t t
end

module Change : sig
  type 'a t =
    { valid : 'a
    ; price : 'a
    ; shares : 'a
    }
  [@@deriving hardcaml]
end

module type Config = sig
  val levels : int
end

module Make (_ : Config) : sig
  (** [opposite] is the best level of the other side. *)
  module I : sig
    type 'a t =
      { levels : 'a
      ; ask : 'a
      ; opposite : 'a
      ; remove : 'a Change.t
      ; insert : 'a Change.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { updated : 'a
      ; accepted : 'a
      }
    [@@deriving hardcaml]
  end

  (** [remove] then [insert]; an insert with no shares or that would cross is refused. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t

  (** [updated] is delayed one cycle; [accepted] remains combinational for table writes. *)
  module Pipelined : sig
    module I : sig
      type 'a t =
        { clocking : 'a Clocking.t
        ; book : 'a I.t
        }
      [@@deriving hardcaml]
    end

    module O = O

    val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
  end
end
