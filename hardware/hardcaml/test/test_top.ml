open! Core
open! Hardcaml
open Onnx2fpga_hardcaml

(* A fork that rejoins, as in a residual block, with one branch buffered deeper. *)
let residual =
  {|
{
 "format": 1,
 "top": "otf_top",
 "input": "x",
 "output": "y",
 "streams": [
  {"name": "x", "width": 8}, {"name": "x_fifo", "width": 8}, {"name": "h", "width": 8},
  {"name": "h_br0", "width": 8}, {"name": "h_br1", "width": 8},
  {"name": "h_br0_fifo", "width": 8}, {"name": "h_br1_fifo", "width": 8},
  {"name": "y", "width": 8}
 ],
 "nodes": [
  {"name": "x_fifo_unit", "instance": "u_x_fifo", "kind": "otf_fifo",
   "params": {"WIDTH": 8, "DEPTH": 2}, "memories": {}, "inputs": ["x"], "outputs": ["x_fifo"]},
  {"name": "scale", "instance": "u_scale", "kind": "otf_act",
   "params": {"CHANNELS": 2, "PE": 1, "IN_BITS": 8, "OUT_BITS": 8, "MULT_BITS": 4,
              "SHIFT": 1, "OUT_MIN": -128, "OUT_MAX": 127, "IN_ZP": 0, "OUT_ZP": 0},
   "memories": {"mult": {"elem_bits": 4, "lanes": 1, "values": [3, 13]}},
   "inputs": ["x_fifo"], "outputs": ["h"]},
  {"name": "fork", "instance": "u_fork", "kind": "otf_dup",
   "params": {"WIDTH": 8, "FANOUT": 2}, "memories": {},
   "inputs": ["h"], "outputs": ["h_br0", "h_br1"]},
  {"name": "short", "instance": "u_short", "kind": "otf_fifo",
   "params": {"WIDTH": 8, "DEPTH": 2}, "memories": {}, "inputs": ["h_br0"], "outputs": ["h_br0_fifo"]},
  {"name": "long", "instance": "u_long", "kind": "otf_fifo",
   "params": {"WIDTH": 8, "DEPTH": 6}, "memories": {}, "inputs": ["h_br1"], "outputs": ["h_br1_fifo"]},
  {"name": "join", "instance": "u_join", "kind": "otf_add",
   "params": {"PE": 1, "IN_BITS": 8, "OUT_BITS": 8, "ACC_BITS": 12, "LEFT_MULT": 2,
              "RIGHT_MULT": -1, "SHIFT": 0, "OUT_MIN": -128, "OUT_MAX": 127, "BIAS": 1,
              "OUT_ZP": 0},
   "memories": {}, "inputs": ["h_br0_fifo", "h_br1_fifo"], "outputs": ["y"]}
 ]
}
|}
;;

let design = Design.of_string residual

module Residual = Top.Make (struct
    let design = design
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Residual.I) (Residual.O)

(* channel c of x scaled by mult[c] / 2, rounded; then 2a - b + 1 with a = b. *)
let expected inputs =
  List.mapi inputs ~f:(fun index x ->
    let mult = if index % 2 = 0 then 3 else -3 in
    let h =
      Model.requant
        ~acc_bits:10
        ~mult_bits:4
        ~shift:1
        ~out_min:(-128)
        ~out_max:127
        ~out_zp:0
        ~acc:x
        ~mult
    in
    Int.clamp_exn (h + 1) ~min:(-128) ~max:127)
;;

let%expect_test "a design description elaborates into a working accelerator" =
  let rng = Random.State.make [| 3 |] in
  Harness.run
    ~random_initial_state:`All
    ~create:Residual.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       inputs.rst_n := Bits.gnd;
       Hardcaml_lws.Lws.cycle ~n:2 h;
       inputs.rst_n := Bits.vdd;
       let values = Model.random_ints rng ~bits:8 400 in
       let sent = Model.beats_of_elements ~bits:8 ~per_beat:1 values in
       let source =
         Stream_bench.Source.create
           ~data:inputs.s_tdata
           ~valid:inputs.s_tvalid
           ~ready:o.s_tready
           ~duty:70
           sent
       in
       let sink =
         Stream_bench.Sink.create
           ~data:o.m_tdata
           ~valid:o.m_tvalid
           ~ready:inputs.m_tready
           ~duty:60
           ~wanted:(List.length sent)
       in
       let cycles = Stream_bench.run h ~rng ~sources:[ source ] ~sinks:[ sink ] () in
       [%test_result: int list]
         (Model.elements ~bits:8 (Stream_bench.Sink.received sink))
         ~expect:(expected values);
       print_s [%message (cycles : int)]);
  [%expect {| (cycles 733) |}]
;;

let%expect_test "the top level keeps the harness's port names" =
  let verilog = Top.verilog design in
  let lines = String.split_lines verilog in
  let header =
    List.drop_while lines ~f:(fun line -> not (String.equal line "module otf_top ("))
    |> List.take_while ~f:(fun line -> not (String.equal line ");"))
  in
  List.iter header ~f:print_endline;
  let instances =
    List.filter_map lines ~f:(fun line ->
      let line = String.strip line in
      Option.some_if (String.is_prefix line ~prefix:"u_") line)
  in
  print_s [%message (instances : string list)];
  [%expect
    {|
    module otf_top (
        clk,
        rst_n,
        s_tdata,
        s_tvalid,
        m_tready,
        s_tready,
        m_tdata,
        m_tvalid
    (instances (u_long u_join u_short u_fork u_scale u_x_fifo))
    |}]
;;

let mvau_node ~instance ~weights ~input ~output =
  sprintf
    {|{"name": "%s", "instance": "%s", "kind": "otf_mvau",
       "params": {"MW": 2, "MH": 2, "SIMD": 2, "PE": 1, "ACT_BITS": 8, "WGT_BITS": 8,
                  "ACC_BITS": 18, "OUT_BITS": 8, "MULT_BITS": 4, "SHIFT": 2,
                  "OUT_MIN": -128, "OUT_MAX": 127, "OUT_ZP": 0, "BLOCK_RAM": 0},
       "memories": {"weights": {"elem_bits": 8, "lanes": 2, "values": [%s]},
                    "bias": {"elem_bits": 18, "lanes": 1, "values": [0, 0]},
                    "mult": {"elem_bits": 4, "lanes": 1, "values": [1, 1]}},
       "inputs": ["%s"], "outputs": ["%s"]}|}
    instance
    instance
    weights
    input
    output
;;

(* Matvec units differing only in weights need a module each, hence no ROMs. *)
let%expect_test "units differing only in their constants are not shared" =
  let design =
    Design.of_string
      (sprintf
         {|{"format": 1, "top": "otf_top", "input": "x", "output": "z",
            "streams": [{"name": "x", "width": 16}, {"name": "y", "width": 8},
                        {"name": "y2", "width": 16}, {"name": "z", "width": 8}],
            "nodes": [%s, %s, %s]}|}
         (mvau_node ~instance:"u_first" ~weights:"1, 2, 3, 4" ~input:"x" ~output:"y")
         {|{"name": "dwc", "instance": "u_dwc", "kind": "otf_dwc",
            "params": {"IN_ELEMS": 1, "OUT_ELEMS": 2, "ELEM_BITS": 8, "TOTAL_ELEMS": 2},
            "memories": {}, "inputs": ["y"], "outputs": ["y2"]}|}
         (mvau_node ~instance:"u_second" ~weights:"5, 6, 7, 8" ~input:"y2" ~output:"z"))
  in
  let modules =
    String.split_lines (Top.verilog design)
    |> List.filter ~f:(String.is_prefix ~prefix:"module otf_mvau")
  in
  print_s [%message (modules : string list)];
  [%expect {| (modules ("module otf_mvau (" "module otf_mvau_1 (")) |}]
;;

let%expect_test "malformed descriptions are refused with the reason" =
  List.iter
    [ {|{"format": 2}|}
    ; {|{"format": 1, "top": "t", "input": "x", "output": "y", "streams": [], "nodes": [{"name": "n"}]}|}
    ]
    ~f:(fun text ->
      print_s
        [%message
          ""
            ~_:
              (Or_error.try_with (fun () -> Design.of_string text)
               |> Result.map ~f:(fun _ -> ())
               : unit Or_error.t)]);
  [%expect
    {|
    (Error ("design.json: unsupported format" (version 2) (format 1)))
    (Error
     (design.json "missing field params" (json (Object ((name (String n)))))))
    |}]
;;
