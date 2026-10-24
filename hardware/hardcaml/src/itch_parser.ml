open! Core
open! Hardcaml
open! Signal

let latency = 1
let max_symbols = 8

module Kind = struct
  type t =
    | Add
    | Execute
    | Cancel
    | Delete
    | Replace
  [@@deriving enumerate, sexp_of]

  let width = 3

  let code = function
    | Add -> 0
    | Execute -> 1
    | Cancel -> 2
    | Delete -> 3
    | Replace -> 4
  ;;
end

module Event = struct
  type 'a t =
    { valid : 'a
    ; kind : 'a [@bits Kind.width]
    ; symbol : 'a [@bits 3]
    ; side : 'a
    ; order : 'a [@bits 64]
    ; new_order : 'a [@bits 64]
    ; shares : 'a [@bits 32]
    ; price : 'a [@bits 32]
    ; t_first : 'a [@bits 32]
    ; t_last : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

(* Type, length, and the offsets of shares and price; ITCH 5.0 section 1.3. *)
let layouts =
  [ 'A', 36, Kind.Add, 20, 32
  ; 'F', 40, Add, 20, 32
  ; 'E', 31, Execute, 19, 0
  ; 'C', 36, Execute, 19, 0
  ; 'X', 23, Cancel, 19, 0
  ; 'D', 19, Delete, 0, 0
  ; 'U', 35, Replace, 27, 31
  ]
;;

module type Config = sig
  val locates : int list
end

module Make (Config : Config) = struct
  open Config

  let () =
    if List.is_empty locates || List.length locates > max_symbols
    then
      raise_s
        [%message
          "BUG: itch_parser takes 1 to 8 stock locates" (List.length locates : int)];
    if List.contains_dup locates ~compare:Int.compare
    then raise_s [%message "BUG: itch_parser stock locates repeat" (locates : int list)]
  ;;

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; message : 'a Moldudp64.Message.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { event : 'a Event.t } [@@deriving hardcaml]
  end

  let create scope ({ clocking; message } : Signal.t I.t) =
    let byte i = message.bytes.:[(8 * i) + 7, 8 * i] in
    let field offset bytes =
      concat_msb (List.init bytes ~f:(fun i -> byte (offset + i)))
    in
    let%hw locate = field 1 2 in
    let%hw_list tracked = List.map locates ~f:(fun l -> locate ==:. l) in
    let%hw symbol =
      onehot_select
        (List.mapi tracked ~f:(fun index valid ->
           { With_valid.valid; value = of_unsigned_int ~width:3 index }))
    in
    let matches =
      List.map layouts ~f:(fun (code, length, kind, shares, price) ->
        byte 0 ==: of_char code &: (message.length ==:. length), (kind, shares, price))
    in
    let pick ~f =
      onehot_select
        (List.map matches ~f:(fun (valid, layout) ->
           { With_valid.valid; value = f layout }))
    in
    let%hw known = List.map matches ~f:fst |> reduce ~f:( |: ) in
    let event =
      { Event.valid = message.valid &: known &: List.reduce_exn tracked ~f:( |: )
      ; kind =
          pick ~f:(fun (kind, _, _) -> of_unsigned_int ~width:Kind.width (Kind.code kind))
      ; symbol
      ; side = byte 19 ==: of_char 'S'
      ; order = field 11 8
      ; new_order = field 19 8
      ; shares =
          pick ~f:(fun (_, offset, _) -> if offset = 0 then zero 32 else field offset 4)
      ; price =
          pick ~f:(fun (_, _, offset) -> if offset = 0 then zero 32 else field offset 4)
      ; t_first = message.t_first
      ; t_last = message.t_last
      }
    in
    let registered = Event.map event ~f:(Clocking.reg_no_clear clocking) in
    { O.event = { registered with valid = Clocking.reg clocking event.valid } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"itch_parser" create i
  ;;
end
