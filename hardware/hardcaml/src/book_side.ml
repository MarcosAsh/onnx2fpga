open! Core
open! Hardcaml
open! Signal

module Level = struct
  type 'a t =
    { price : 'a [@bits 32]
    ; shares : 'a [@bits 32]
    }
  [@@deriving hardcaml]

  let empty = { price = zero 32; shares = zero 32 }

  let of_word word =
    { price = sel_bottom word ~width:32; shares = sel_top word ~width:32 }
  ;;

  let to_word { price; shares } = shares @: price
end

module Change = struct
  type 'a t =
    { valid : 'a
    ; price : 'a [@bits 32]
    ; shares : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

module type Config = sig
  val levels : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if levels < 1 then raise_s [%message "BUG: book_side needs a level" (levels : int)]
  ;;

  module I = struct
    type 'a t =
      { levels : 'a [@bits levels * 64]
      ; ask : 'a
      ; opposite : 'a [@bits 64]
      ; remove : 'a Change.t
      ; insert : 'a Change.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { updated : 'a [@bits levels * 64]
      ; accepted : 'a
      }
    [@@deriving hardcaml]
  end

  let live (level : _ Level.t) = level.shares <>:. 0
  let better ~ask a b = mux2 ask (a <: b) (a >: b)
  let pick valid a b = Level.map2 a b ~f:(mux2 valid)

  let prefix_or bits =
    List.folding_map bits ~init:gnd ~f:(fun acc bit -> acc |: bit, acc |: bit)
  ;;

  (* The level at [price] loses [shares], and leaves the book when that empties it. *)
  let remove levels (change : _ Change.t) =
    let found =
      List.map levels ~f:(fun l -> change.valid &: live l &: (l.price ==: change.price))
    in
    let emptied =
      List.map2_exn found levels ~f:(fun found l -> found &: (change.shares >=: l.shares))
    in
    let shifted = List.tl_exn levels @ [ Level.empty ] in
    List.map3_exn
      (List.zip_exn levels shifted)
      found
      (prefix_or emptied)
      ~f:(fun (level, next) found gone ->
        pick
          gone
          next
          (pick found { level with shares = level.shares -: change.shares } level))
  ;;

  (* More shares at a held price, or a new level in place; one past the end is lost. *)
  let insert ~ask levels (change : _ Change.t) =
    let saturating a b =
      let sum = ue a +: ue b in
      mux2 (msb sum) (ones 32) (lsbs sum)
    in
    let matched = List.map levels ~f:(fun l -> live l &: (l.price ==: change.price)) in
    let any_match = reduce matched ~f:( |: ) in
    let ahead =
      List.map levels ~f:(fun l -> live l &: better ~ask l.price change.price)
    in
    let ahead_previous = vdd :: List.drop_last_exn ahead in
    let previous = Level.empty :: List.drop_last_exn levels in
    let added = { Level.price = change.price; shares = change.shares } in
    List.map3_exn
      (List.zip_exn levels previous)
      (List.zip_exn ahead ahead_previous)
      matched
      ~f:(fun (level, previous) (ahead, ahead_previous) matched ->
        pick
          (change.valid &: matched)
          { level with shares = saturating level.shares change.shares }
          (pick
             (change.valid &: ~:any_match &: ~:ahead)
             (pick ahead_previous added previous)
             level))
  ;;

  let create scope (i : Signal.t I.t) =
    let levels = split_lsb ~part_width:64 i.levels |> List.map ~f:Level.of_word in
    let opposite = Level.of_word i.opposite in
    let%hw crosses =
      live opposite
      &: mux2 i.ask (i.insert.price <=: opposite.price) (i.insert.price >=: opposite.price)
    in
    let%hw accepted = i.insert.valid &: (i.insert.shares <>:. 0) &: ~:crosses in
    let updated =
      insert ~ask:i.ask (remove levels i.remove) { i.insert with valid = accepted }
    in
    { O.updated = concat_lsb (List.map updated ~f:Level.to_word); accepted }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"book_side" create i
  ;;
end
