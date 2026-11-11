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
    let gone = prefix_or emptied in
    let updated =
      List.map3_exn
        (List.zip_exn levels shifted)
        found
        gone
        ~f:(fun (level, next) found gone ->
          pick
            gone
            next
            (pick found { level with shares = level.shares -: change.shares } level))
    in
    updated, gone
  ;;

  (* More shares at a held price, or a new level in place; one past the end is lost. *)
  let insert ~matched ~ahead levels (change : _ Change.t) =
    let saturating a b =
      let sum = ue a +: ue b in
      mux2 (msb sum) (ones 32) (lsbs sum)
    in
    let any_match = reduce matched ~f:( |: ) in
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

  let update scope ~register (i : Signal.t I.t) =
    let levels = split_lsb ~part_width:64 i.levels |> List.map ~f:Level.of_word in
    let opposite = Level.of_word i.opposite in
    let%hw crosses =
      live opposite
      &: mux2 i.ask (i.insert.price <=: opposite.price) (i.insert.price >=: opposite.price)
    in
    let%hw accepted = i.insert.valid &: (i.insert.shares <>:. 0) &: ~:crosses in
    let matched = List.map levels ~f:(fun l -> live l &: (l.price ==: i.insert.price)) in
    let ahead =
      List.map levels ~f:(fun l -> live l &: better ~ask:i.ask l.price i.insert.price)
    in
    let levels, gone = remove levels i.remove in
    let shift flags =
      List.map3_exn
        gone
        flags
        (List.tl_exn flags @ [ gnd ])
        ~f:(fun gone here next -> mux2 gone next here)
    in
    let updated =
      insert
        ~matched:(List.map (shift matched) ~f:register)
        ~ahead:(List.map (shift ahead) ~f:register)
        (List.map levels ~f:(Level.map ~f:register))
        (Change.map { i.insert with valid = accepted } ~f:register)
    in
    { O.updated = concat_lsb (List.map updated ~f:Level.to_word); accepted }
  ;;

  let create scope i = update scope ~register:Fn.id i

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"book_side" create i
  ;;

  module Pipelined = struct
    module I = struct
      type 'a t =
        { clocking : 'a Clocking.t
        ; book : 'a I.t
        }
      [@@deriving hardcaml]
    end

    module O = O

    let create scope ({ clocking; book } : Signal.t I.t) =
      update scope ~register:(Clocking.reg_no_clear clocking) book
    ;;

    let hierarchical ?instance scope i =
      let module H = Hierarchy.In_scope (I) (O) in
      H.hierarchical ?instance ~scope ~name:"book_side_pipelined" create i
    ;;
  end
end
