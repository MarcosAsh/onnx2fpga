(** A two layer fixed latency design with 2 int8 features, and its model. *)

open! Core

(** design.json text stating [latency]; the units really have 6. *)
val description : latency:int -> string

(** The answer for two int8 features. *)
val model : int list -> int list
