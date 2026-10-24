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

let geometry name ifm channels kernel stride pad dilation simd =
  { name; ifm; channels; kernel; stride; pad; dilation; simd }
;;

(* The eighteen geometries of tests/test_sliding_window.py, with symmetric padding. *)
let all =
  [ geometry "basic" (8, 8) 4 (3, 3) (1, 1) (0, 0) (1, 1) 1
  ; geometry "folded" (8, 8) 4 (3, 3) (1, 1) (0, 0) (1, 1) 4
  ; geometry "single_chan" (12, 12) 1 (3, 3) (1, 1) (0, 0) (1, 1) 1
  ; geometry "same_pad" (7, 7) 4 (5, 5) (1, 1) (2, 2) (1, 1) 4
  ; geometry "stride_pad" (9, 9) 2 (3, 3) (2, 2) (1, 1) (1, 1) 2
  ; geometry "pool_window" (10, 10) 4 (2, 2) (2, 2) (0, 0) (1, 1) 2
  ; geometry "dilated" (8, 8) 2 (3, 3) (1, 1) (0, 0) (2, 2) 2
  ; geometry "pointwise" (6, 6) 8 (1, 1) (1, 1) (0, 0) (1, 1) 8
  ; geometry "kernel_eq_ifm" (5, 5) 1 (5, 5) (1, 1) (0, 0) (1, 1) 1
  ; geometry "non_square" (6, 10) 2 (3, 3) (1, 1) (1, 1) (1, 1) 2
  ; geometry "par_gray" (12, 12) 1 (3, 3) (1, 1) (0, 0) (1, 1) 3
  ; geometry "par_padded" (9, 9) 1 (3, 3) (1, 1) (1, 1) (1, 1) 3
  ; geometry "par_nonsquare" (6, 10) 2 (3, 3) (1, 1) (1, 1) (1, 1) 6
  ; geometry "par_stride" (11, 11) 2 (3, 3) (2, 2) (1, 1) (1, 1) 6
  ; geometry "par_dilated" (10, 10) 1 (3, 3) (1, 1) (0, 0) (2, 2) 3
  ; geometry "par_pool" (10, 10) 4 (2, 2) (2, 2) (0, 0) (1, 1) 8
  ; geometry "par_k5" (9, 9) 1 (5, 5) (1, 1) (2, 2) (1, 1) 5
  ; geometry "par_partial" (8, 8) 2 (4, 4) (1, 1) (0, 0) (1, 1) 4
  ]
;;

let window t : Model.Window.t =
  { ifm_h = fst t.ifm
  ; ifm_w = snd t.ifm
  ; ifm_ch = t.channels
  ; kh = fst t.kernel
  ; kw = snd t.kernel
  ; stride = t.stride
  ; pad = t.pad
  ; dilation = t.dilation
  }
;;

let sub_elems t = Int.min t.simd t.channels
let par t = t.simd / sub_elems t
