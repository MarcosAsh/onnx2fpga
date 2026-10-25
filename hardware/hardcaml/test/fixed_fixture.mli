(** A two layer fixed latency design with 2 int8 features, and its model. *)

open! Core
open! Hardcaml

(** design.json text stating [latency]; the units really have 6. *)
val description : latency:int -> string

(** The answer for two int8 features. *)
val model : int list -> int list

(** The two layers' answer gated by a table of the features, latency 8. *)
val gated : string

val gated_model : int list -> int list

(** The same design with both layers reloadable, its latency 6. *)
val reloadable : string

(** The answer once [reloaded_image] is in the active bank. *)
val reloaded_model : int list -> int list

(** The bank as compiled, one 32 bit word per value. *)
val compiled_image : Bits.t list

(** Other weights, biases and multipliers for the same layers, in bank order. *)
val reloaded_image : Bits.t list
