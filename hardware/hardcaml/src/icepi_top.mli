(** A compiled accelerator on an Icepi Zero, serial via the FT231X. *)

open! Core
open! Hardcaml

module type Config = sig
  val design : Design.t

  (** The PLL output, a divisor of 480 from 4 to 480. *)
  val mhz : int

  (** Must come within 2 percent of [mhz] divided by a whole number of clocks. *)
  val baud : int

  val rx_depth : int
  val tx_depth : int
end

(** Clocks per UART bit, rounded. Raises when that misses [baud] by more than 2 percent. *)
val clocks_per_bit : mhz:int -> baud:int -> int

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clk : 'a
      ; button_n : 'a
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

(** One Verilog file with [icepi_top] and everything under it. *)
val verilog
  :  design:Design.t
  -> mhz:int
  -> baud:int
  -> rx_depth:int
  -> tx_depth:int
  -> string
