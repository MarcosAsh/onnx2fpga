open! Core
open! Hardcaml
open! Signal

module I = struct
  type 'a t =
    { held : 'a [@bits 32]
    ; requested : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { executed : 'a [@bits 32]
    ; remaining : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

let create scope ({ held; requested } : Signal.t I.t) =
  let%hw low = ue held.:[15, 0] -: ue requested.:[15, 0] in
  let%hw high_a = ue held.:[31, 16] in
  let%hw high_b = ue requested.:[31, 16] in
  (* Preserve the two parallel high-half carry chains. *)
  let keep value = add_attribute value (Rtl_attribute.Vivado.keep true) in
  let%hw high_borrow = keep (high_a +: ~:high_b) in
  let%hw high_no_borrow = keep (high_a -: high_b) in
  let%hw high = mux2 (msb low) high_borrow high_no_borrow in
  let%hw underflow = msb high in
  let%hw difference = lsbs high @: lsbs low in
  { O.executed = mux2 underflow held requested
  ; remaining = mux2 underflow (zero 32) difference
  }
;;

let hierarchical ?instance scope i =
  let module H = Hierarchy.In_scope (I) (O) in
  H.hierarchical ?instance ~scope ~name:"order_quantity" create i
;;
