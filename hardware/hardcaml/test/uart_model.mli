(** The far end of a serial line, one level per clock; [error] skews its bit period. *)

open! Core

(** One 8N1 character; [~stop:false] makes a framing error. *)
val character : ?stop:bool -> clocks_per_bit:float -> int -> bool list

(** The characters back to back, each followed by [gap ()] idle bit times. *)
val line : clocks_per_bit:float -> ?gap:(unit -> int) -> int list -> bool list

val idle : int -> bool list
val break_ : int -> bool list

(** Samples a line the way a UART does, in the middle of each of its own bit periods. *)
module Receiver : sig
  type t

  val create : clocks_per_bit:float -> t

  (** The level on the line for one cycle. *)
  val step : t -> bool -> unit

  val received : t -> int list
  val framing_errors : t -> int
end
