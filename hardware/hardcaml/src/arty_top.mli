(** A UART accelerator on the Arty A7 100 MHz oscillator. *)

open! Core
open! Hardcaml

module type Config = sig
  val design : Design.t
  val baud : int
  val rx_depth : int
  val tx_depth : int
end

val mhz : int

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clk : 'a
      ; button : 'a
      ; rxd : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { txd : 'a
      ; led : 'a
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** One Verilog file with [arty_top] and everything under it. *)
val verilog : design:Design.t -> baud:int -> rx_depth:int -> tx_depth:int -> string
