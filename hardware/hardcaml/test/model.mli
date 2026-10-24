(** Software models the units are checked against. Everything is plain integers. *)

open! Core
open! Hardcaml

(** Two's complement wrap to [bits] bits, as a signed value. *)
val wrap : bits:int -> int -> int

(** Matches {!Onnx2fpga_hardcaml.Otf_requant} exactly, including its working width. *)
val requant
  :  acc_bits:int
  -> mult_bits:int
  -> shift:int
  -> out_min:int
  -> out_max:int
  -> out_zp:int
  -> acc:int
  -> mult:int
  -> int

(** The compiler's [Requantizer.apply] in unbounded integers. *)
val python_requant
  :  shift:int
  -> out_min:int
  -> out_max:int
  -> out_zp:int
  -> acc:int
  -> mult:int
  -> int

(** Each value masked to [bits] bits, as a memory word. *)
val words_of_ints : bits:int -> int list -> Bits.t list

val random_ints : Random.State.t -> bits:int -> int -> int list

(** Every element of every beat, element 0 of beat 0 first, each [bits] wide. *)
val elements : bits:int -> Bits.t list -> int list

val beats_of_elements : bits:int -> per_beat:int -> int list -> Bits.t list

module Window : sig
  type t =
    { ifm_h : int
    ; ifm_w : int
    ; ifm_ch : int
    ; kh : int
    ; kw : int
    ; stride : int * int
    ; pad : int * int
    ; dilation : int * int
    }

  val output_extent : t -> int * int

  (** Every window in (row, column, kernel row, kernel column, channel) order. *)
  val unfold : t -> pad_value:int -> int list -> int list
end
