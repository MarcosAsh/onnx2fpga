open! Core
open! Hardcaml

(** Returns the 322.265625 MHz clock and its synchronized active-high reset. *)
val create : Scope.t -> pl_clk0:Signal.t -> pl_resetn0:Signal.t -> Signal.t * Signal.t
