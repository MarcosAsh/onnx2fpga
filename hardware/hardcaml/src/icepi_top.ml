open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val design : Design.t
  val mhz : int
  val baud : int
  val rx_depth : int
  val tx_depth : int
end

let oscillator_mhz = 50

(* ecppll -i 50 -o 48 --highres, as the Icepi Zero's own gateware has it. *)
let vco_mhz = 480

let clocks_per_bit ~mhz ~baud =
  let clocks =
    Float.iround_nearest_exn (Float.of_int (mhz * 1_000_000) /. Float.of_int baud)
  in
  let error =
    Float.abs ((Float.of_int (clocks * baud) /. Float.of_int (mhz * 1_000_000)) -. 1.)
  in
  if Float.(error > 0.02)
  then
    raise_s
      [%message
        "the clock cannot make this baud rate within 2 percent" (mhz : int) (baud : int)];
  clocks
;;

module Make (Config : Config) = struct
  open Config

  let () =
    if mhz < 1 || vco_mhz % mhz <> 0 || vco_mhz / mhz > 128
    then
      raise_s
        [%message
          "the PLL makes clocks that divide its 480 MHz VCO, from 4 MHz up" (mhz : int)]
  ;;

  module Link = Uart_top.Make (struct
      let design = design
      let clocks_per_bit = clocks_per_bit ~mhz ~baud
      let rx_depth = rx_depth
      let tx_depth = tx_depth
    end)

  module I = struct
    type 'a t =
      { clk : 'a
      ; button_n : 'a
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

  let pll scope ~clk =
    let%hw pll_feedback = wire 1 in
    let int name value = Parameter.create ~name ~value:(Int value) in
    let string name value = Parameter.create ~name ~value:(String value) in
    let attribute name value = Rtl_attribute.create name ~value:(String value) in
    let pll =
      Instantiation.create
        ()
        ~name:"EHXPLLL"
        ~instance:"pll"
        ~parameters:
          [ string "PLLRST_ENA" "DISABLED"
          ; string "INTFB_WAKE" "DISABLED"
          ; string "STDBY_ENABLE" "DISABLED"
          ; string "DPHASE_SOURCE" "DISABLED"
          ; string "OUTDIVIDER_MUXA" "DIVA"
          ; string "OUTDIVIDER_MUXB" "DIVB"
          ; string "OUTDIVIDER_MUXC" "DIVC"
          ; string "OUTDIVIDER_MUXD" "DIVD"
          ; int "CLKI_DIV" 5
          ; string "CLKOP_ENABLE" "ENABLED"
          ; int "CLKOP_DIV" 48
          ; int "CLKOP_CPHASE" 9
          ; int "CLKOP_FPHASE" 0
          ; string "CLKOS_ENABLE" "ENABLED"
          ; int "CLKOS_DIV" (vco_mhz / mhz)
          ; int "CLKOS_CPHASE" 0
          ; int "CLKOS_FPHASE" 0
          ; string "FEEDBK_PATH" "CLKOP"
          ; int "CLKFB_DIV" 1
          ]
        ~attributes:
          [ attribute "FREQUENCY_PIN_CLKI" (Int.to_string oscillator_mhz)
          ; attribute "FREQUENCY_PIN_CLKOS" (Int.to_string mhz)
          ; attribute "ICP_CURRENT" "12"
          ; attribute "LPF_RESISTOR" "8"
          ; attribute "MFG_ENABLE_FILTEROPAMP" "1"
          ; attribute "MFG_GMCREF_SEL" "2"
          ]
        ~inputs:
          [ "RST", gnd
          ; "STDBY", gnd
          ; "CLKI", clk
          ; "CLKFB", pll_feedback
          ; "PHASESEL0", gnd
          ; "PHASESEL1", gnd
          ; "PHASEDIR", vdd
          ; "PHASESTEP", vdd
          ; "PHASELOADREG", vdd
          ; "PLLWAKESYNC", gnd
          ; "ENCLKOP", gnd
          ]
        ~outputs:[ "CLKOP", 1; "CLKOS", 1; "LOCK", 1 ]
    in
    pll_feedback <-- Instantiation.output pll "CLKOP";
    Instantiation.output pll "CLKOS", Instantiation.output pll "LOCK"
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let clock, locked = pll scope ~clk:i.clk in
    let%hw clock in
    let%hw locked in
    (* The one reset synchroniser; it powers up asserted. *)
    let spec_no_clear = Reg_spec.create ~clock () in
    let%hw reset_request = ~:(i.button_n) |: ~:locked in
    let%hw reset_meta = reg spec_no_clear ~initialize_to:Bits.vdd reset_request in
    let%hw clear = reg spec_no_clear ~initialize_to:Bits.vdd reset_meta in
    let clocking = { Clocking.clock; clear } in
    let link = Link.hierarchical scope { clocking; rxd = i.rxd } in
    (* A blink every second or two, so a loaded bitstream is visibly alive. *)
    let%hw heartbeat =
      reg_fb
        (Clocking.to_spec clocking)
        ~width:(Int.ceil_log2 (mhz * 1_000_000))
        ~f:(fun count -> count +:. 1)
    in
    { O.txd = link.txd; led = link.error @: msb heartbeat }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"icepi_top" create i
  ;;
end

let verilog ~design ~mhz ~baud ~rx_depth ~tx_depth =
  let module T =
    Make (struct
      let design = design
      let mhz = mhz
      let baud = baud
      let rx_depth = rx_depth
      let tx_depth = tx_depth
    end)
  in
  let module C = Circuit.With_interface (T.I) (T.O) in
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit = C.create_exn ~name:"icepi_top" (T.create scope) in
  Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  |> Rtl.full_hierarchy
  |> Rope.to_string
;;
