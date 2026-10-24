open! Core
open! Hardcaml
open! Signal

let latency = 1
let order_bytes = 47

module Action = struct
  type t =
    | Hold
    | Buy
    | Sell
  [@@deriving enumerate, sexp_of]

  let code = function
    | Hold -> 0
    | Buy -> 1
    | Sell -> 2
  ;;
end

module type Config = sig
  val names : string list
  val buy_threshold : int
  val sell_threshold : int
  val order_shares : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if List.exists names ~f:(fun name -> String.is_empty name || String.length name > 8)
    then
      raise_s
        [%message
          "BUG: decision symbol names are 1 to 8 characters" (names : string list)];
    if sell_threshold >= buy_threshold
    then
      raise_s
        [%message
          "BUG: decision sell threshold must be below the buy threshold"
            (buy_threshold : int)
            (sell_threshold : int)]
  ;;

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; prediction : 'a With_valid.t [@bits 8]
      ; context : 'a Features.Context.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { valid : 'a
      ; action : 'a [@bits 2]
      ; prediction : 'a [@bits 8]
      ; order : 'a With_valid.t [@bits 8 * order_bytes]
      ; t_first : 'a [@bits 32]
      ; t_last : 'a [@bits 32]
      }
    [@@deriving hardcaml]
  end

  let big_endian value ~bytes = split_msb ~part_width:8 (uresize value ~width:(8 * bytes))
  let text string = List.map (String.to_list string) ~f:of_char

  let hex nibble =
    mux2
      (nibble <:. 10)
      (uresize nibble ~width:8 +:. Char.to_int '0')
      (uresize nibble ~width:8 +:. (Char.to_int 'A' - 10))
  ;;

  (* An OUCH 5.0 Enter Order, immediate or cancel, numbered by [token]. *)
  let frame ~token ~sell ~price ~symbol =
    let names =
      List.map names ~f:(fun name -> concat_msb (text (String.pad_right name ~len:8)))
    in
    [ text "O"
    ; big_endian token ~bytes:4
    ; [ mux2 sell (of_char 'S') (of_char 'B') ]
    ; big_endian (of_unsigned_int ~width:32 order_shares) ~bytes:4
    ; split_msb ~part_width:8 (mux symbol names)
    ; big_endian price ~bytes:8
    ; text "3YPNN000000"
    ; List.map (split_msb ~part_width:4 token) ~f:hex
    ; big_endian (zero 16) ~bytes:2
    ]
    |> List.concat
    |> concat_lsb
  ;;

  let create scope ({ clocking; prediction; context } : Signal.t I.t) =
    let threshold value = of_signed_int ~width:8 value in
    let%hw buy =
      prediction.valid &: context.has_ask &: (prediction.value >=+ threshold buy_threshold)
    in
    let%hw sell =
      prediction.valid
      &: context.has_bid
      &: (prediction.value <=+ threshold sell_threshold)
    in
    let%hw token =
      Clocking.reg_fb clocking ~enable:(buy |: sell) ~width:32 ~f:(fun t -> t +:. 1)
    in
    let action =
      mux2
        buy
        (of_unsigned_int ~width:2 (Action.code Buy))
        (mux2 sell (of_unsigned_int ~width:2 (Action.code Sell)) (zero 2))
    in
    let reg = Clocking.reg_no_clear clocking in
    { O.valid = Clocking.reg clocking prediction.valid
    ; action = reg action
    ; prediction = reg prediction.value
    ; order =
        { valid = Clocking.reg clocking (buy |: sell)
        ; value =
            reg
              (frame
                 ~token:(token +:. 1)
                 ~sell
                 ~price:(mux2 buy context.ask context.bid)
                 ~symbol:context.symbol)
        }
    ; t_first = reg context.t_first
    ; t_last = reg context.t_last
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"decision" create i
  ;;
end
