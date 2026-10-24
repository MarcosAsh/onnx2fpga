open! Core
open! Hardcaml
open Onnx2fpga_hardcaml

(* The PLL is a vendor primitive, so cyclesim cannot run the top. *)

let%expect_test "clocks per bit, and the rates that cannot be made" =
  List.iter
    [ 20, 1_000_000; 24, 1_000_000; 48, 3_000_000; 20, 115_200; 20, 3_000_000 ]
    ~f:(fun (mhz, baud) ->
      print_s
        [%message
          ""
            (mhz : int)
            (baud : int)
            ~clocks_per_bit:
              (Or_error.try_with (fun () -> Icepi_top.clocks_per_bit ~mhz ~baud)
               : int Or_error.t)]);
  [%expect
    {|
    ((mhz 20) (baud 1000000) (clocks_per_bit (Ok 20)))
    ((mhz 24) (baud 1000000) (clocks_per_bit (Ok 24)))
    ((mhz 48) (baud 3000000) (clocks_per_bit (Ok 16)))
    ((mhz 20) (baud 115200) (clocks_per_bit (Ok 174)))
    ((mhz 20) (baud 3000000)
     (clocks_per_bit
      (Error
       ("the clock cannot make this baud rate within 2 percent" (mhz 20)
        (baud 3000000)))))
    |}]
;;

let design =
  Design.of_string
    {|{"format": 1, "top": "otf_top", "input": "x", "output": "y",
       "streams": [{"name": "x", "width": 8}, {"name": "y", "width": 8}],
       "nodes": [{"name": "b", "instance": "u_b", "kind": "otf_fifo",
                  "params": {"WIDTH": 8, "DEPTH": 2}, "memories": {},
                  "inputs": ["x"], "outputs": ["y"]}]}|}
;;

let%expect_test "the board top's ports and its PLL" =
  let verilog =
    Icepi_top.verilog ~design ~mhz:20 ~baud:1_000_000 ~rx_depth:64 ~tx_depth:64
  in
  let lines = String.split_lines verilog |> List.map ~f:String.strip in
  let between ~first ~last =
    List.drop_while lines ~f:(fun line -> not (String.equal line first))
    |> List.take_while ~f:(fun line -> not (String.is_prefix line ~prefix:last))
  in
  List.iter (between ~first:"module icepi_top (" ~last:");") ~f:print_endline;
  List.iter
    (List.filter lines ~f:(fun line ->
       List.exists
         [ ".CLKI_DIV"; ".CLKOP_DIV"; ".CLKOS_DIV"; ".FEEDBK_PATH" ]
         ~f:(fun p -> String.is_substring line ~substring:p)))
    ~f:print_endline;
  [%expect
    {|
    module icepi_top (
    clk,
    button_n,
    rxd,
    txd,
    led
    .CLKI_DIV(5),
    .CLKOP_DIV(48),
    .CLKOS_DIV(24),
    .FEEDBK_PATH("CLKOP"),
    |}]
;;

let%expect_test "a clock the PLL cannot make is refused" =
  print_s
    [%message
      ""
        ~_:
          (Or_error.try_with (fun () ->
             ignore
               (Icepi_top.verilog
                  ~design
                  ~mhz:25
                  ~baud:1_000_000
                  ~rx_depth:64
                  ~tx_depth:64
                : string))
           : unit Or_error.t)];
  [%expect
    {|
    (Error
     ("the PLL makes clocks that divide its 480 MHz VCO, from 4 MHz up" (mhz 25)))
    |}]
;;
