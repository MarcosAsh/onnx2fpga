open! Core
open! Hardcaml
open! Signal

(* 100 MHz * 103.125 / 8 is a 1289.0625 MHz VCO, and / 4 gives 322.265625 MHz. *)
let mmcm_multiply = 103.125
let mmcm_divide = 8
let mmcm_output_divide = 4.0

let mmcm scope ~pl_clk0 ~pl_resetn0 =
  let%hw mmcm_feedback = wire 1 in
  let real name value = Parameter.create ~name ~value:(Real value) in
  let int name value = Parameter.create ~name ~value:(Int value) in
  let mmcm =
    Instantiation.create
      ()
      ~name:"MMCME4_BASE"
      ~instance:"otf_mmcm"
      ~parameters:
        [ real "CLKIN1_PERIOD" 10.0
        ; real "CLKFBOUT_MULT_F" mmcm_multiply
        ; int "DIVCLK_DIVIDE" mmcm_divide
        ; real "CLKOUT0_DIVIDE_F" mmcm_output_divide
        ]
      ~inputs:
        [ "CLKIN1", pl_clk0
        ; "CLKFBIN", mmcm_feedback
        ; "RST", ~:pl_resetn0
        ; "PWRDWN", gnd
        ]
      ~outputs:[ "CLKOUT0", 1; "CLKFBOUT", 1; "LOCKED", 1 ]
  in
  mmcm_feedback <-- Instantiation.output mmcm "CLKFBOUT";
  let buffer =
    Instantiation.create
      ()
      ~name:"BUFG"
      ~instance:"otf_bufg"
      ~inputs:[ "I", Instantiation.output mmcm "CLKOUT0" ]
      ~outputs:[ "O", 1 ]
  in
  Instantiation.output buffer "O", Instantiation.output mmcm "LOCKED"
;;

let create scope ~pl_clk0 ~pl_resetn0 =
  let clock, locked = mmcm scope ~pl_clk0 ~pl_resetn0 in
  let%hw clock in
  let%hw locked in
  (* The one reset synchroniser; it powers up asserted. *)
  let spec_no_clear = Reg_spec.create ~clock () in
  let%hw reset_request = ~:pl_resetn0 |: ~:locked in
  let%hw reset_meta = reg spec_no_clear ~initialize_to:Bits.vdd reset_request in
  let%hw clear = reg spec_no_clear ~initialize_to:Bits.vdd reset_meta in
  clock, clear
;;
