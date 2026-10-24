open! Core
open! Hardcaml

let wrap ~bits value =
  let shifted = value lsl (Int.num_bits - bits) in
  shifted asr (Int.num_bits - bits)
;;

let requant ~acc_bits ~mult_bits ~shift ~out_min ~out_max ~out_zp ~acc ~mult =
  let prod_bits = acc_bits + mult_bits + 1 in
  let wrap = wrap ~bits:prod_bits in
  let product = wrap (acc * mult) in
  let shifted =
    if shift >= acc_bits + mult_bits
    then 0
    else if shift = 0
    then product
    else wrap (product + (1 lsl (shift - 1))) asr shift
  in
  let rounded = wrap (shifted + out_zp) in
  Int.clamp_exn rounded ~min:out_min ~max:out_max
;;

let python_requant ~shift ~out_min ~out_max ~out_zp ~acc ~mult =
  let scaled = acc * mult in
  let scaled = if shift > 0 then (scaled + (1 lsl (shift - 1))) asr shift else scaled in
  Int.clamp_exn (scaled + out_zp) ~min:out_min ~max:out_max
;;

let signed_words ~bits words = List.map words ~f:(fun word -> wrap ~bits word)

let words_of_ints ~bits values =
  List.map values ~f:(fun value ->
    Bits.of_unsigned_int ~width:bits (value land ((1 lsl bits) - 1)))
;;

let random_ints rng ~bits count =
  List.init count ~f:(fun _ -> wrap ~bits (Random.State.bits rng))
;;

let elements ~bits beats =
  List.concat_map beats ~f:(Stream_bench.signed_lanes ~lane_bits:bits)
;;

let beats_of_elements ~bits ~per_beat elements =
  List.chunks_of elements ~length:per_beat
  |> List.map ~f:(Stream_bench.of_signed_lanes ~lane_bits:bits)
;;

module Window = struct
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

  let output_extent
    { ifm_h; ifm_w; kh; kw; stride = sh, sw; pad = pt, pl; dilation = dh, dw; _ }
    =
    let extent size k s p d = ((size + (2 * p) - (d * (k - 1)) - 1) / s) + 1 in
    extent ifm_h kh sh pt dh, extent ifm_w kw sw pl dw
  ;;

  (* im2col in (row, col, ky, kx, channel) order, reading the image channel last. *)
  let unfold t ~pad_value image =
    let { ifm_h; ifm_w; ifm_ch; kh; kw; stride = sh, sw; pad = pt, pl; dilation = dh, dw }
      =
      t
    in
    let ofm_h, ofm_w = output_extent t in
    let image = Array.of_list image in
    List.concat_map (List.range 0 ofm_h) ~f:(fun oh ->
      List.concat_map (List.range 0 ofm_w) ~f:(fun ow ->
        List.concat_map (List.range 0 kh) ~f:(fun ky ->
          List.concat_map (List.range 0 kw) ~f:(fun kx ->
            let row = (oh * sh) + (ky * dh) - pt in
            let col = (ow * sw) + (kx * dw) - pl in
            List.init ifm_ch ~f:(fun c ->
              if row >= 0 && row < ifm_h && col >= 0 && col < ifm_w
              then image.((((row * ifm_w) + col) * ifm_ch) + c)
              else pad_value)))))
  ;;
end
