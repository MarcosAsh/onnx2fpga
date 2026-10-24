(** 8N1 transmitter; [byte] is taken on [valid] and [ready], [txd] is a flop output. *)

open! Core
open! Hardcaml

module type Config = sig
  val clocks_per_bit : int
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; byte : 'a With_valid.t
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { ready : 'a
      ; txd : 'a
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
