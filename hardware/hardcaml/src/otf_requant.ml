open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val acc_bits : int
  val mult_bits : int
  val shift : int
  val out_bits : int
  val out_min : int
  val out_max : int
  val out_zp : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if shift < 0
    then raise_s [%message "BUG: otf_requant shift is negative" (shift : int)]
  ;;

  let prod_bits = acc_bits + mult_bits + 1

  (* Below shift = acc_bits + mult_bits the datapath is exact. *)
  let quotient_is_zero = shift >= acc_bits + mult_bits

  module I = struct
    type 'a t =
      { acc : 'a [@bits acc_bits]
      ; mult : 'a [@bits mult_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { out : 'a [@bits out_bits] } [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let%hw product = sresize (i.acc *+ i.mult) ~width:prod_bits in
    let%hw shifted =
      if quotient_is_zero
      then zero prod_bits
      else if shift > 0
      then (
        let round_add = sll (one prod_bits) ~by:(shift - 1) in
        sra (product +: round_add) ~by:shift)
      else product
    in
    let%hw rounded = shifted +: Sv_int.cast ~width:prod_bits out_zp in
    let lo_limit = Sv_int.cast ~width:prod_bits out_min in
    let hi_limit = Sv_int.cast ~width:prod_bits out_max in
    let%hw clamped =
      mux2 (rounded <+ lo_limit) lo_limit @@ mux2 (rounded >+ hi_limit) hi_limit rounded
    in
    { O.out = sresize clamped ~width:out_bits }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_requant" create i
  ;;
end
