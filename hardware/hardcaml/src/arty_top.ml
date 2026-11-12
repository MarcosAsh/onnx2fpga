open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val design : Design.t
  val baud : int
  val rx_depth : int
  val tx_depth : int
end

let mhz = 100

module Make (Config : Config) = struct
  module Link = Uart_top.Make (struct
      include Config

      let clocks_per_bit = Icepi_top.clocks_per_bit ~mhz ~baud
    end)

  module I = struct
    type 'a t =
      { clk : 'a
      ; button : 'a
      ; rxd : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { txd : 'a
      ; led : 'a [@bits 2]
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Reg_spec.create ~clock:i.clk () in
    let synchronizer signal =
      add_attribute signal (Rtl_attribute.create "ASYNC_REG" ~value:(String "TRUE"))
    in
    let%hw reset_meta = synchronizer (reg spec ~initialize_to:Bits.vdd i.button) in
    let%hw clear = synchronizer (reg spec ~initialize_to:Bits.vdd reset_meta) in
    let clocking = { Clocking.clock = i.clk; clear } in
    let link = Link.hierarchical scope { clocking; rxd = i.rxd } in
    let%hw heartbeat =
      reg_fb (Clocking.to_spec clocking) ~width:27 ~f:(fun count -> count +:. 1)
    in
    { O.txd = link.txd; led = link.error @: msb heartbeat }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"arty_top" create i
  ;;
end

let verilog ~design ~baud ~rx_depth ~tx_depth =
  let module T =
    Make (struct
      let design = design
      let baud = baud
      let rx_depth = rx_depth
      let tx_depth = tx_depth
    end)
  in
  let module C = Circuit.With_interface (T.I) (T.O) in
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit = C.create_exn ~name:"arty_top" (T.create scope) in
  Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  |> Rtl.full_hierarchy
  |> Rope.to_string
;;
