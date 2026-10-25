(** Which of two weight banks is active, the commit that flips it, a version and CRCs. *)

open! Core
open! Hardcaml

(** Width of a word address into the 32 KiB bank window, 8192 words. *)
val address_bits : int

(** One word for the units' banks: [data] goes to word [address] of bank [bank]. *)
module Write : sig
  type 'a t =
    { write : 'a
    ; bank : 'a
    ; address : 'a
    ; data : 'a
    }
  [@@deriving hardcaml]
end

(** zlib's CRC-32 state after one more 32 bit word, bit 0 first, on signals or bits. *)
val crc_step : (module Comb.S with type t = 'a) -> 'a -> 'a -> 'a

(** [zlib.crc32] of the words as little endian bytes. *)
val crc : Bits.t list -> int

module type Config = sig
  (** The compiled bank, a 32 bit word per value, which both banks start with. *)
  val image : Bits.t list
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; write : 'a
      ; address : 'a
      ; data : 'a
      ; commit : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { bank : 'a
      ; version : 'a
      ; crc : 'a
      ; idle_crc : 'a
      }
    [@@deriving hardcaml]
  end

  (** [commit] flips [bank]; a bank's CRC covers its writes since the last one to word 0. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end
