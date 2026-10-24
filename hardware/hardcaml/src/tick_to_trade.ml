open! Core
open! Hardcaml
open! Signal

module Params = struct
  type t =
    { locates : int list
    ; names : string list
    ; levels : int
    ; order_bits : int
    ; spread_shift : int
    ; flow_shift : int
    ; buy_threshold : int
    ; sell_threshold : int
    ; order_shares : int
    ; model : Design.t
    }
  [@@deriving sexp_of]

  let of_string text =
    let fail what = raise_s [%message "tick_to_trade config" what] in
    let field name =
      match Jsonaf.of_string text with
      | `Object fields ->
        (match List.Assoc.find fields name ~equal:String.equal with
         | Some value -> value
         | None -> fail ("missing field " ^ name))
      | _ -> fail "expected an object"
    in
    let number = function
      | `Number text ->
        (match Int.of_string_opt text with
         | Some value -> value
         | None -> fail ("not an integer: " ^ text))
      | _ -> fail "expected an integer"
    in
    let list name ~f =
      match field name with
      | `Array items -> List.map items ~f
      | _ -> fail ("expected an array for " ^ name)
    in
    let int name = number (field name) in
    { locates = list "locates" ~f:number
    ; names =
        list "names" ~f:(function
          | `String name -> name
          | _ -> fail "expected a string in names")
    ; levels = int "levels"
    ; order_bits = int "order_bits"
    ; spread_shift = int "spread_shift"
    ; flow_shift = int "flow_shift"
    ; buy_threshold = int "buy_threshold"
    ; sell_threshold = int "sell_threshold"
    ; order_shares = int "order_shares"
    ; model = Design.of_json (field "model")
    }
  ;;
end

module type Config = sig
  val params : Params.t
end

module Make (Config : Config) = struct
  open Config

  let symbols = List.length params.locates

  module Parser = Itch_parser.Make (struct
      let locates = params.locates
    end)

  module Book = Order_book.Make (struct
      let symbols = symbols
      let levels = params.levels
      let order_bits = params.order_bits
    end)

  module Extract = Features.Make (struct
      let levels = params.levels
      let spread_shift = params.spread_shift
      let flow_shift = params.flow_shift
    end)

  module Model = Fixed_top.Make (struct
      let design = params.model
    end)

  module Decide = Decision.Make (struct
      let names = params.names
      let buy_threshold = params.buy_threshold
      let sell_threshold = params.sell_threshold
      let order_shares = params.order_shares
    end)

  let () =
    let model = params.model in
    let inputs = Design.width model model.input in
    let outputs = Design.width model model.output in
    if List.length params.names <> symbols
    then raise_s [%message "tick_to_trade config: one name per stock locate"];
    if inputs <> 8 * Features.count || outputs <> 8
    then
      raise_s
        [%message
          "tick_to_trade config: the model must take four int8 features and give one int8"
            (inputs : int)
            (outputs : int)]
  ;;

  let stages =
    [ "moldudp64", Moldudp64.latency
    ; "itch_parser", Itch_parser.latency
    ; "order_book", Order_book.latency
    ; "features", Features.latency
    ; "model", Model.latency
    ; "decision", Decision.latency
    ]
  ;;

  let latency = List.sum (module Int) stages ~f:snd

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; data : 'a [@bits 64]
      ; keep : 'a [@bits 8]
      ; valid : 'a
      ; last : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { ready : 'a
      ; book_valid : 'a
      ; book_symbol : 'a [@bits 3]
      ; book_bids : 'a [@bits params.levels * 64]
      ; book_asks : 'a [@bits params.levels * 64]
      ; book_flow : 'a [@bits Order_book.flow_bits]
      ; features_valid : 'a
      ; features : 'a [@bits 8 * Features.count]
      ; decision_valid : 'a
      ; action : 'a [@bits 2]
      ; prediction : 'a [@bits 8]
      ; order_valid : 'a
      ; order : 'a [@bits 8 * Decision.order_bytes]
      ; t_first : 'a [@bits 32]
      ; t_last : 'a [@bits 32]
      ; t_out : 'a [@bits 32]
      }
    [@@deriving hardcaml]
  end

  let create scope (i : Signal.t I.t) =
    let clocking = { Clocking.clock = i.clock; clear = i.clear } in
    let%hw now = Clocking.reg_fb clocking ~width:32 ~f:(fun t -> t +:. 1) in
    let framer =
      Moldudp64.hierarchical
        scope
        { clocking
        ; beat = { data = i.data; keep = i.keep; valid = i.valid; last = i.last }
        ; now
        }
    in
    let parser = Parser.hierarchical scope { clocking; message = framer.message } in
    let book = Book.hierarchical scope { clocking; event = parser.event } in
    let snapshot = book.book in
    let features =
      Extract.hierarchical
        scope
        { clocking
        ; book =
            { valid = snapshot.valid
            ; symbol = snapshot.symbol
            ; bids = snapshot.bids
            ; asks = snapshot.asks
            ; flow = snapshot.flow
            ; t_first = snapshot.t_first
            ; t_last = snapshot.t_last
            }
        }
    in
    let model =
      Model.hierarchical
        scope
        { clock = i.clock
        ; clear = i.clear
        ; features = features.features.value
        ; features_valid = features.features.valid
        }
    in
    let decision =
      Decide.hierarchical
        scope
        { clocking
        ; prediction = { valid = model.prediction_valid; value = model.prediction }
        ; context =
            Features.Context.map
              features.context
              ~f:(Clocking.pipeline_no_clear clocking ~n:Model.latency)
        }
    in
    { O.ready = book.ready
    ; book_valid = snapshot.valid
    ; book_symbol = snapshot.symbol
    ; book_bids = snapshot.bids
    ; book_asks = snapshot.asks
    ; book_flow = snapshot.flow
    ; features_valid = features.features.valid
    ; features = features.features.value
    ; decision_valid = decision.valid
    ; action = decision.action
    ; prediction = decision.prediction
    ; order_valid = decision.order.valid
    ; order = decision.order.value
    ; t_first = decision.t_first
    ; t_last = decision.t_last
    ; t_out = now
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"tick_to_trade" create i
  ;;
end

let verilog ?unit params =
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit =
    match unit with
    | None ->
      let module T =
        Make (struct
          let params = params
        end)
      in
      let module C = Circuit.With_interface (T.I) (T.O) in
      C.create_exn ~name:"tick_to_trade" (T.create scope)
    | Some "moldudp64_step" ->
      let module C = Circuit.With_interface (Moldudp64.Step.I) (Moldudp64.Step.O) in
      C.create_exn ~name:"dut" (Moldudp64.Step.hierarchical scope)
    | Some "book_side" ->
      let module Side =
        Book_side.Make (struct
          let levels = params.Params.levels
        end)
      in
      let module C = Circuit.With_interface (Side.I) (Side.O) in
      C.create_exn ~name:"dut" (Side.hierarchical scope)
    | Some unit -> raise_s [%message "tick_to_trade: no unit called" unit]
  in
  Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  |> Rtl.full_hierarchy
  |> Rope.to_string
;;
