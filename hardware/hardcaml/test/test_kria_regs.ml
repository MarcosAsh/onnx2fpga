open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

(* Runs [f] with AXI4-Lite [write] and [read] on the register file of [design]. *)
module Bench (Config : Kria_regs.Config) = struct
  module Regs = Kria_regs.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Regs.I) (Regs.O)

  let run f =
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
         f ~write ~read [@nontail])
  ;;
end

(* One inference on random int8 features, checked against [model]; returns its cycles. *)
let infer ~rng ~write ~read model =
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
  let prediction = Bits.of_unsigned_int ~width:32 (read Kria_regs.Offset.prediction) in
  [%test_result: int list]
    (Stream_bench.signed_lanes ~lane_bits:8 (Bits.sel_bottom prediction ~width:16))
    ~expect:(model x);
  read Kria_regs.Offset.cycles
;;

module Frozen = Bench (struct
    let design = Design.of_string (Fixed_fixture.description ~latency:6)
  end)

let%expect_test "the ARM writes features, starts, and reads the prediction back" =
  let rng = Random.State.make [| 5 |] in
  Frozen.run (fun ~write ~read ->
    let cycles =
      List.map (List.range 0 20) ~f:(fun _ -> infer ~rng ~write ~read Fixed_fixture.model)
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
          ~weights:(read Kria_regs.Offset.weights : Int.Hex.t)
          ~weights_words:(read Kria_regs.Offset.weights_words : int)]);
  [%expect
    {|
    ((id 0x4f544631) (latency 6) (feature_bits 16) (prediction_bits 16)
     (cycles (6)) (weights 0x0) (weights_words 0))
    |}]
;;

module Reloadable = Bench (struct
    let design = Design.of_string Fixed_fixture.reloadable
  end)

let%expect_test "the ARM writes the idle bank, checks its CRC, commits and predicts" =
  let rng = Random.State.make [| 6 |] in
  Reloadable.run (fun ~write ~read ->
    let state () =
      [ read Kria_regs.Offset.weights_bank
      ; read Kria_regs.Offset.weights_version
      ; read Kria_regs.Offset.weights_crc
      ; read Kria_regs.Offset.weights_idle_crc
      ]
    in
    let before = state () in
    List.iteri Fixed_fixture.reloaded_image ~f:(fun index word ->
      write (Kria_regs.Offset.weights + (4 * index)) (Bits.to_unsigned_int word);
      ignore (infer ~rng ~write ~read Fixed_fixture.model : int));
    let written = state () in
    write Kria_regs.Offset.weights_commit 1;
    let committed = state () in
    let cycles =
      List.map (List.range 0 10) ~f:(fun _ ->
        infer ~rng ~write ~read Fixed_fixture.reloaded_model)
    in
    print_s
      [%message
        ""
          ~words:(read Kria_regs.Offset.weights_words : int)
          (before : Int.Hex.t list)
          (written : Int.Hex.t list)
          (committed : Int.Hex.t list)
          ~cycles:(List.dedup_and_sort cycles ~compare:Int.compare : int list)]);
  [%expect
    {|
    ((words 22) (before (0x0 0x0 0xd6d2b031 0xd6d2b031))
     (written (0x0 0x0 0xd6d2b031 0xa0dac7ed))
     (committed (0x1 0x1 0xa0dac7ed 0xd6d2b031)) (cycles (6)))
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
