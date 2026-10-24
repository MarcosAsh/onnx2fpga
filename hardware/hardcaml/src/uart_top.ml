open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val design : Design.t
  val clocks_per_bit : int
  val rx_depth : int
  val tx_depth : int
end

module Make (Config : Config) = struct
  open Config

  let in_width = Design.width design design.input
  let out_width = Design.width design design.output

  module Rx = Uart_rx.Make (Config)
  module Tx = Uart_tx.Make (Config)

  module Bridge = Uart_bridge.Make (struct
      let in_width = in_width
      let out_width = out_width
      let rx_depth = rx_depth
      let tx_depth = tx_depth
    end)

  module Accelerator = Top.Make (Config)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; rxd : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { txd : 'a
      ; error : 'a
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let rx = Rx.hierarchical scope { clocking = i.clocking; rxd = i.rxd } in
    (* A break clears half assembled beats, buffered bytes and every unit's state. *)
    let%hw run_clear = i.clocking.clear |: rx.line_break in
    let run_clocking = { i.clocking with clear = run_clear } in
    let%hw accel_ready = wire 1 in
    let%hw accel_data = wire out_width in
    let%hw accel_valid = wire 1 in
    let%hw tx_ready = wire 1 in
    let bridge =
      Bridge.hierarchical
        scope
        { clocking = run_clocking
        ; rx = rx.byte
        ; accel_in = { ready = accel_ready }
        ; accel_out = { data = accel_data; valid = accel_valid }
        ; tx = { ready = tx_ready }
        }
    in
    let accelerator =
      Accelerator.hierarchical
        scope
        { clk = i.clocking.clock
        ; rst_n = ~:run_clear
        ; s_tdata = bridge.accel_in.data
        ; s_tvalid = bridge.accel_in.valid
        ; m_tready = bridge.accel_out.ready
        }
    in
    accel_ready <-- accelerator.s_tready;
    accel_data <-- accelerator.m_tdata;
    accel_valid <-- accelerator.m_tvalid;
    let tx =
      Tx.hierarchical
        scope
        { clocking = i.clocking
        ; byte = { valid = bridge.tx.valid; value = bridge.tx.data }
        }
    in
    tx_ready <-- tx.ready;
    let%hw error =
      reg_fb (Clocking.to_spec run_clocking) ~width:1 ~f:(fun error ->
        error |: bridge.overflow |: rx.framing_error)
    in
    { O.txd = tx.txd; error }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"uart_top" create i
  ;;
end
