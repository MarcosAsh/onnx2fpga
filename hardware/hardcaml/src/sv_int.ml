open! Core
open! Hardcaml
open! Signal

let check_int name value =
  if (value < Int32.(to_int_exn min_value)) || value > Int32.(to_int_exn max_value)
  then raise_s [%message "BUG: parameter does not fit a 32 bit int" name (value : int)]
;;

let int32 value = of_signed_int ~width:32 value
let cast ~width value = sresize (int32 value) ~width
let of_unsigned x = uresize x ~width:32
let mul x value = sel_bottom (x *+ int32 value) ~width:32
let is_negative x = msb x
let clamp_at_zero x = mux2 (is_negative x) (zero 32) x
let counter_width count = address_bits_for count
let is_last value ~count = value ==:. count - 1

let wrapping_counter spec ~advance ~count =
  reg_fb spec ~enable:advance ~width:(counter_width count) ~f:(fun d ->
    if count = 1 then zero (width d) else mod_counter ~max:(count - 1) d)
;;

let table ~address words =
  match words with
  | [] -> raise_s [%message "BUG: a table needs at least one word"]
  | [ word ] -> of_constant (Bits.to_constant word)
  | words ->
    mux address (List.map words ~f:(fun word -> of_constant (Bits.to_constant word)))
;;
