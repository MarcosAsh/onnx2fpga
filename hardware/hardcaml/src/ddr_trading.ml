open! Core
open! Hardcaml
open! Signal

module State = struct
  type t =
    | Idle
    | Clearing
    | Waiting
    | Launch
    | Running
    | Draining
  [@@deriving enumerate, sexp_of, compare ~localize]
end

module Make (Config : Tick_to_trade.Config) = struct
  module Trading = Tick_to_trade.Make (Config)

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; start : 'a
      ; abort_request : 'a
      ; base : 'a [@bits 40]
      ; records : 'a [@bits 32]
      ; arready : 'a
      ; rvalid : 'a
      ; rdata : 'a [@bits 128]
      ; rresp : 'a [@bits 2]
      ; rlast : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { replay : 'a Ddr_replay.O.t
      ; trading : 'a Trading.O.t
      ; busy : 'a
      ; done_ : 'a
      ; error : 'a
      ; aborted : 'a
      ; decisions : 'a [@bits 32]
      ; orders : 'a [@bits 32]
      ; cycles : 'a [@bits 32]
      }
    [@@deriving hardcaml]
  end

  let create scope (i : Signal.t I.t) =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let%hw.Always.State_machine sm = Always.State_machine.create (module State) spec in
    let%hw accepted = sm.is Idle &: i.start &: ~:(i.abort_request) in
    let%hw base = reg spec ~enable:accepted i.base in
    let%hw records = reg spec ~enable:accepted i.records in
    let%hw trading_clear = i.clear |: sm.is Clearing in
    let%hw ready = wire 1 in
    let replay =
      Ddr_replay.hierarchical
        scope
        { clock = i.clock
        ; clear = i.clear
        ; start = sm.is Launch
        ; abort_request = i.abort_request
        ; base
        ; records
        ; ready
        ; arready = i.arready
        ; rvalid = i.rvalid
        ; rdata = i.rdata
        ; rresp = i.rresp
        ; rlast = i.rlast
        }
    in
    let trading =
      Trading.hierarchical
        scope
        { clock = i.clock
        ; clear = trading_clear
        ; data = replay.data
        ; keep = replay.keep
        ; valid = replay.valid
        ; last = replay.last
        }
    in
    ready <-- trading.ready;
    let%hw_var done_ = Always.Variable.reg spec ~width:1 in
    let%hw_var error = Always.Variable.reg spec ~width:1 in
    let%hw_var aborted = Always.Variable.reg spec ~width:1 in
    let%hw_var drain =
      Always.Variable.reg spec ~width:(num_bits_to_represent (Trading.latency + 2))
    in
    let count_spec = Reg_spec.create ~clock:i.clock ~clear:(i.clear |: accepted) () in
    let%hw decisions =
      reg_fb count_spec ~width:32 ~f:(fun n ->
        n +: uresize trading.decision_valid ~width:32)
    in
    let%hw orders =
      reg_fb count_spec ~width:32 ~f:(fun n -> n +: uresize trading.order_valid ~width:32)
    in
    let%hw cycles =
      reg_fb count_spec ~width:32 ~f:(fun n -> n +: uresize ~:(sm.is Idle) ~width:32)
    in
    let finish = Always.[ done_ <--. 1; sm.set_next Idle ] in
    let cancel = Always.([ aborted <--. 1 ] @ finish) in
    Always.(
      compile
        [ sm.switch
            [ ( Idle
              , [ when_
                    accepted
                    [ done_ <--. 0; error <--. 0; aborted <--. 0; sm.set_next Clearing ]
                ] )
            ; Clearing, [ if_ i.abort_request cancel [ sm.set_next Waiting ] ]
            ; ( Waiting
              , [ if_
                    i.abort_request
                    cancel
                    [ when_ trading.ready [ sm.set_next Launch ] ]
                ] )
            ; Launch, [ if_ i.abort_request cancel [ sm.set_next Running ] ]
            ; ( Running
              , [ when_
                    replay.done_
                    [ error <-- replay.error
                    ; aborted <-- replay.aborted
                    ; drain <--. Trading.latency + 2
                    ; sm.set_next Draining
                    ]
                ] )
            ; ( Draining
              , [ if_ (drain.value ==:. 0) finish [ drain <-- drain.value -:. 1 ] ] )
            ]
        ]);
    { O.replay
    ; trading
    ; busy = ~:(sm.is Idle)
    ; done_ = done_.value
    ; error = error.value
    ; aborted = aborted.value
    ; decisions
    ; orders
    ; cycles
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"ddr_trading" create i
  ;;
end

let verilog params =
  let module T =
    Make (struct
      let params = params
    end)
  in
  let module C = Circuit.With_interface (T.I) (T.O) in
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit = C.create_exn ~name:"ddr_trading" (T.create scope) in
  Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  |> Rtl.full_hierarchy
  |> Rope.to_string
;;
