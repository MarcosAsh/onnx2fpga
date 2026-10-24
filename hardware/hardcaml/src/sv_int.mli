(** SystemVerilog 32 bit signed [int] semantics. *)

open! Core
open! Hardcaml

(** Raises unless [value] fits a SystemVerilog [int]. *)
val check_int : string -> int -> unit

(** [int32 v] is the parameter [v] as a 32 bit signal. *)
val int32 : int -> Signal.t

(** [cast ~width v] is [width'(v)] for an [int] parameter [v]: truncated or sign extended. *)
val cast : width:int -> int -> Signal.t

(** [of_unsigned x] is [32'($signed({1'b0, x}))]. *)
val of_unsigned : Signal.t -> Signal.t

(** [mul x v] is the wrapping 32 bit product, as in SystemVerilog. *)
val mul : Signal.t -> int -> Signal.t

(** [x < 0 ? 0 : x] on a 32 bit signal. *)
val clamp_at_zero : Signal.t -> Signal.t

(** [value == count - 1] at the counter's own width. *)
val is_last : Signal.t -> count:int -> Signal.t

(** A loop counter from 0 to [count - 1] that steps on [advance]. *)
val wrapping_counter : Signal.Reg_spec.t -> advance:Signal.t -> count:int -> Signal.t

(** A mux over constants, not a [rom], because Hardcaml compares memories by size alone. *)
val table : address:Signal.t -> Bits.t list -> Signal.t
