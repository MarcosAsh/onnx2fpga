open! Core
open! Hardcaml
open! Signal

let check_int name value =
  if Option.is_none (Int32.of_int value)
  then
    raise_s
      [%message "design.json: parameter does not fit a 32 bit int" name (value : int)]
;;

let int32 value = of_signed_int ~width:32 value
let cast ~width value = sresize (int32 value) ~width
let of_unsigned x = uresize x ~width:32
let mul x value = sel_bottom (x *+ int32 value) ~width:32
let clamp_at_zero x = mux2 (msb x) (zero 32) x
let is_last value ~count = value ==:. count - 1

let wrapping_counter spec ~advance ~count =
  reg_fb spec ~enable:advance ~width:(address_bits_for count) ~f:(fun d ->
    if count = 1 then zero (width d) else mod_counter ~max:(count - 1) d)
;;

let table ~address words =
  match List.map words ~f:(fun word -> of_constant (Bits.to_constant word)) with
  | [ word ] -> word
  | words -> mux address words
;;
