(** The netlist the Python compiler writes to design.json. *)

open! Core
open! Hardcaml

module Memory : sig
  (** Word order, [lanes] elements per word, masked to [elem_bits] bits. *)
  type t =
    { elem_bits : int
    ; lanes : int
    ; values : int list
    }
  [@@deriving sexp_of]

  (** One word per [lanes] values, element 0 in the least significant bits. *)
  val words : t -> Bits.t list
end

module Node : sig
  type t =
    { name : string
    ; instance : string
    ; kind : string
    ; params : (string * int) list
    ; memories : (string * Memory.t) list
    ; inputs : string list
    ; outputs : string list
    }
  [@@deriving sexp_of]

  (** Raises when the node has no such parameter and there is no [default]. *)
  val param : ?default:int -> t -> string -> int

  val words : t -> string -> Bits.t list
end

module Stream_width : sig
  type t =
    { name : string
    ; width : int
    }
  [@@deriving sexp_of]
end

module Mode : sig
  (** A fixed design states the cycles from [features_valid] to [prediction_valid]. *)
  type t =
    | Streaming
    | Fixed of { latency : int }
  [@@deriving sexp_of]
end

type t =
  { top : string
  ; mode : Mode.t
  ; input : string
  ; output : string
  ; streams : Stream_width.t list
  ; nodes : Node.t list
  }
[@@deriving sexp_of]

(** Raises on anything malformed, naming the field. *)
val of_string : string -> t

val of_json : Jsonaf.t -> t
val width : t -> string -> int
