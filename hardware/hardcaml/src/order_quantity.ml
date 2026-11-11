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
  let%hw difference = ue held -: ue requested in
  let%hw underflow = msb difference in
  { O.executed = mux2 underflow held requested
  ; remaining = mux2 underflow (zero 32) (lsbs difference)
  }
;;

let hierarchical ?instance scope i =
  let module H = Hierarchy.In_scope (I) (O) in
  H.hierarchical ?instance ~scope ~name:"order_quantity" create i
;;
