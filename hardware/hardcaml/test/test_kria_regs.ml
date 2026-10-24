open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Regs = Kria_regs.Make (struct
    let design = Design.of_string (Fixed_fixture.description ~latency:6)
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Regs.I) (Regs.O)

let%expect_test "the ARM writes features, starts, and reads the prediction back" =
  let rng = Random.State.make [| 5 |] in
  Harness.run
    ~random_initial_state:`All
    ~create:Regs.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       let handshake ready =
         let waited = ref 0 in
         Lws.cycle h;
         while not (Bits.to_bool !ready) do
           incr waited;
           if !waited > 50 then failwith "no handshake";
           Lws.cycle h
         done
       in
       let write address data =
         inputs.s_axi_awaddr := Bits.of_unsigned_int ~width:16 address;
         inputs.s_axi_wdata := Bits.of_unsigned_int ~width:32 data;
         inputs.s_axi_awvalid := Bits.vdd;
         inputs.s_axi_wvalid := Bits.vdd;
         handshake o.s_axi_awready;
         inputs.s_axi_awvalid := Bits.gnd;
         inputs.s_axi_wvalid := Bits.gnd;
         handshake o.s_axi_bvalid
       in
       let read address =
         inputs.s_axi_araddr := Bits.of_unsigned_int ~width:16 address;
         inputs.s_axi_arvalid := Bits.vdd;
         handshake o.s_axi_arready;
         inputs.s_axi_arvalid := Bits.gnd;
         handshake o.s_axi_rvalid;
         Bits.to_unsigned_int !(o.s_axi_rdata)
       in
       inputs.s_axi_bready := Bits.vdd;
       inputs.s_axi_rready := Bits.vdd;
       Stream_bench.clear h inputs.clear;
       let cycles =
         List.map (List.range 0 20) ~f:(fun _ ->
           let x = Model.random_ints rng ~bits:8 2 in
           write
             Kria_regs.Offset.features
             (Bits.to_unsigned_int (Stream_bench.of_signed_lanes ~lane_bits:8 x));
           write Kria_regs.Offset.control 1;
           let polls = ref 0 in
           while read Kria_regs.Offset.status land 1 = 0 do
             incr polls;
             if !polls > 20 then failwith "no prediction"
           done;
           let prediction =
             Bits.of_unsigned_int ~width:32 (read Kria_regs.Offset.prediction)
           in
           [%test_result: int list]
             (Stream_bench.signed_lanes
                ~lane_bits:8
                (Bits.sel_bottom prediction ~width:16))
             ~expect:(Fixed_fixture.model x);
           read Kria_regs.Offset.cycles)
       in
       write Kria_regs.Offset.weights 0xdeadbeef;
       print_s
         [%message
           ""
             ~id:(read Kria_regs.Offset.id : Int.Hex.t)
             ~latency:(read Kria_regs.Offset.latency : int)
             ~feature_bits:(read Kria_regs.Offset.feature_bits : int)
             ~prediction_bits:(read Kria_regs.Offset.prediction_bits : int)
             ~cycles:(List.dedup_and_sort cycles ~compare:Int.compare : int list)
             ~weights:(read Kria_regs.Offset.weights : Int.Hex.t)]);
  [%expect
    {|
    ((id 0x4f544631) (latency 6) (feature_bits 16) (prediction_bits 16)
     (cycles (6)) (weights 0x0))
    |}]
;;

let%expect_test "the shell's ports and its MMCM" =
  let lines =
    Kria_shell.verilog (Design.of_string (Fixed_fixture.description ~latency:6))
    |> String.split_lines
    |> List.map ~f:String.strip
  in
  List.drop_while lines ~f:(fun line -> not (String.equal line "module kria_shell ("))
  |> List.take_while ~f:(fun line -> not (String.equal line ");"))
  |> List.iter ~f:print_endline;
  List.filter lines ~f:(fun line ->
    List.exists [ "MMCME4_BASE"; "_DIVIDE"; "_MULT_F"; "BUFG" ] ~f:(fun p ->
      String.is_substring line ~substring:p))
  |> List.iter ~f:print_endline;
  [%expect
    {|
    module kria_shell (
    pl_clk0,
    pl_resetn0,
    s_axi_awaddr,
    s_axi_awvalid,
    s_axi_wdata,
    s_axi_wvalid,
    s_axi_bready,
    s_axi_araddr,
    s_axi_arvalid,
    s_axi_rready,
    aclk,
    aresetn,
    s_axi_awready,
    s_axi_wready,
    s_axi_bresp,
    s_axi_bvalid,
    s_axi_arready,
    s_axi_rdata,
    s_axi_rresp,
    s_axi_rvalid
    MMCME4_BASE
    .CLKFBOUT_MULT_F(103.125000),
    .DIVCLK_DIVIDE(8),
    .CLKOUT0_DIVIDE_F(4.000000) )
    BUFG
    |}]
;;
