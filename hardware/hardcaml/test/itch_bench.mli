(** ITCH 5.0 messages and MoldUDP64 packets for the tick to trade tests. *)

open! Core
open! Hardcaml

val system_event : string

(** An F message when [attribution] is given, otherwise an A. *)
val add
  :  ?locate:int
  -> ?attribution:string
  -> order:int
  -> side:char
  -> shares:int
  -> price:int
  -> unit
  -> string

val execute : ?locate:int -> order:int -> shares:int -> unit -> string
val cancel : ?locate:int -> order:int -> shares:int -> unit -> string
val delete : ?locate:int -> order:int -> unit -> string

val replace
  :  ?locate:int
  -> order:int
  -> new_order:int
  -> shares:int
  -> price:int
  -> unit
  -> string

(** [count] overrides the header's message count and [lengths] the block lengths. *)
val packet : ?count:int -> ?lengths:(string list -> int list) -> string list -> string

type beat =
  { data : Bits.t
  ; keep : Bits.t
  ; last : bool
  }

val beats : string -> beat list

(** Sets the beat's ports, or drops [valid] for [None]. *)
val drive
  :  data:Bits.t ref
  -> keep:Bits.t ref
  -> valid:Bits.t ref
  -> last:Bits.t ref
  -> beat option
  -> unit

(** The first [length] bytes of a little endian byte vector. *)
val string_of_bits : Bits.t -> length:int -> string
