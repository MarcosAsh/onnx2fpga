(** Drives a fixed latency unit one vector per cycle and checks every answer. *)

open! Core
open! Hardcaml

(** Answers must arrive exactly [latency] cycles after their inputs; returns how many did. *)
val check
  :  Hardcaml_lws.Lws.Handler.t @ local
  -> x:Bits.t ref With_valid.t
  -> y:Bits.t ref With_valid.t
  -> latency:int
  -> expected:(Bits.t -> Bits.t)
  -> (bool * Bits.t) list
  -> int
