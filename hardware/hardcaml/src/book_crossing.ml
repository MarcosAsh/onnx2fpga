open! Core
open! Hardcaml
open! Signal

module I = struct
  type 'a t =
    { ask : 'a
    ; price : 'a [@bits 32]
    ; bid : 'a Book_side.Level.t
    ; offer : 'a Book_side.Level.t
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { crosses : 'a } [@@deriving hardcaml]
end

let create scope ({ ask; price; bid; offer } : Signal.t I.t) =
  let%hw crosses_bid = bid.shares <>:. 0 &: (price <=: bid.price) in
  let%hw crosses_offer = offer.shares <>:. 0 &: (price >=: offer.price) in
  { O.crosses = mux2 ask crosses_bid crosses_offer }
;;

let hierarchical ?instance scope i =
  let module H = Hierarchy.In_scope (I) (O) in
  H.hierarchical ?instance ~scope ~name:"book_crossing" create i
;;
