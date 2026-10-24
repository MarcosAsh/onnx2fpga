(** Window geometries shared by the two sliding window tests. *)

open! Core
open! Hardcaml

type geometry =
  { name : string
  ; ifm : int * int
  ; channels : int
  ; kernel : int * int
  ; stride : int * int
  ; pad : int * int
  ; dilation : int * int
  ; simd : int
  }

(** [geometry name ifm channels kernel stride pad dilation simd]. *)
val geometry
  :  string
  -> int * int
  -> int
  -> int * int
  -> int * int
  -> int * int
  -> int * int
  -> int
  -> geometry

(** The eighteen geometries tests/test_sliding_window.py sweeps. *)
val all : geometry list

val window : geometry -> Model.Window.t

(** Elements per input beat, at most one pixel. *)
val sub_elems : geometry -> int

(** Kernel columns per output beat. *)
val par : geometry -> int
