(** A compiled accelerator behind a UART; the host sends a break first, then frames. *)

open! Core
open! Hardcaml

module type Config = sig
  val design : Design.t
  val clocks_per_bit : int
  val rx_depth : int
  val tx_depth : int
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clocking : 'a Clocking.t
      ; rxd : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { txd : 'a
      ; error : 'a
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
