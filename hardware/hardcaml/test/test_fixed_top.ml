open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

let dense ~name ~input ~output ~mw ~mh ~weights ~bias ~mult ~out_min =
  let ints values = String.concat ~sep:", " (List.map values ~f:Int.to_string) in
  sprintf
    {|{"name": "%s", "instance": "u_%s", "kind": "otf_dense",
       "params": {"MW": %d, "MH": %d, "ACT_BITS": 8, "WGT_BITS": 8, "ACC_BITS": 18,
                  "OUT_BITS": 8, "MULT_BITS": 8, "SHIFT": 6, "OUT_MIN": %d,
                  "OUT_MAX": 127, "OUT_ZP": 0},
       "memories": {"weights": {"elem_bits": 8, "lanes": %d, "values": [%s]},
                    "bias": {"elem_bits": 18, "lanes": 1, "values": [%s]},
                    "mult": {"elem_bits": 8, "lanes": 1, "values": [%s]}},
       "inputs": ["%s"], "outputs": ["%s"]}|}
    name
    name
    mw
    mh
    out_min
    mw
    (ints (List.map weights ~f:(fun w -> w land 0xff)))
    (ints (List.map bias ~f:(fun b -> b land 0x3ffff)))
    (ints mult)
    input
    output
;;

(* Weights are listed channel by channel, so [w0] is [c][k] flattened. *)
let w0 = [ 3; -7; 12; 5; -1; 9 ]
let w1 = [ -4; 6; 2; 11; -8; 1 ]

let description ~latency =
  sprintf
    {|{"format": 1, "top": "otf_top", "mode": "fixed", "latency": %d,
       "input": "x", "output": "y",
       "streams": [{"name": "x", "width": 16}, {"name": "h", "width": 24},
                   {"name": "y", "width": 16}],
       "nodes": [%s, %s]}|}
    latency
    (dense
       ~name:"fc0"
       ~input:"x"
       ~output:"h"
       ~mw:2
       ~mh:3
       ~weights:w0
       ~bias:[ 100; -50; 7 ]
       ~mult:[ 40; 90; 127 ]
       ~out_min:0)
    (dense
       ~name:"fc1"
       ~input:"h"
       ~output:"y"
       ~mw:3
       ~mh:2
       ~weights:w1
       ~bias:[ -3; 20 ]
       ~mult:[ 70; 33 ]
       ~out_min:(-128))
;;

let layer x ~weights ~bias ~mult ~out_min =
  List.mapi bias ~f:(fun c b ->
    let dot =
      List.foldi x ~init:b ~f:(fun k sum a ->
        sum + (a * List.nth_exn weights ((c * List.length x) + k)))
    in
    Model.requant
      ~acc_bits:18
      ~mult_bits:8
      ~shift:6
      ~out_min
      ~out_max:127
      ~out_zp:0
      ~acc:dot
      ~mult:(List.nth_exn mult c))
;;

let model x =
  layer x ~weights:w0 ~bias:[ 100; -50; 7 ] ~mult:[ 40; 90; 127 ] ~out_min:0
  |> layer ~weights:w1 ~bias:[ -3; 20 ] ~mult:[ 70; 33 ] ~out_min:(-128)
;;

module Mlp = Fixed_top.Make (struct
    let design = Design.of_string (description ~latency:6)
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Mlp.I) (Mlp.O)

let%expect_test "two layers answer exactly six cycles after the input" =
  let rng = Random.State.make [| 11 |] in
  Harness.run ~create:Mlp.hierarchical (fun (h @ local) ~inputs ~outputs ->
    let o = Before_and_after_edge.before_edge outputs in
    inputs.clear := Bits.vdd;
    Lws.cycle h;
    inputs.clear := Bits.gnd;
    let sent = Queue.create () in
    List.iter (List.range 0 500) ~f:(fun _ ->
      let x = Model.random_ints rng ~bits:8 2 in
      inputs.features := Stream_bench.of_signed_lanes ~lane_bits:8 x;
      inputs.features_valid := Bits.vdd;
      Lws.cycle h;
      Queue.enqueue sent x;
      if Queue.length sent > Mlp.latency
      then (
        let x = Queue.dequeue_exn sent in
        assert (Bits.to_bool !(o.prediction_valid));
        [%test_result: int list]
          (Stream_bench.signed_lanes ~lane_bits:8 !(o.prediction))
          ~expect:(model x))
      else assert (not (Bits.to_bool !(o.prediction_valid))));
    print_s [%message (Mlp.latency : int)]);
  [%expect {| (Mlp.latency 6) |}]
;;

let%expect_test "the ports are the ones the spec names" =
  Fixed_top.verilog (Design.of_string (description ~latency:6))
  |> String.split_lines
  |> List.drop_while ~f:(fun line -> not (String.equal line "module otf_top ("))
  |> List.take_while ~f:(fun line -> not (String.equal line ");"))
  |> List.iter ~f:print_endline;
  [%expect
    {|
    module otf_top (
        clock,
        clear,
        features,
        features_valid,
        prediction,
        prediction_valid
    |}]
;;

let%expect_test "a stated latency the units do not have is refused" =
  let design = Design.of_string (description ~latency:5) in
  Or_error.try_with (fun () -> Fixed_top.verilog design)
  |> Result.map ~f:(fun _ -> ())
  |> [%sexp_of: unit Or_error.t]
  |> print_s;
  [%expect
    {|
    (Error
     ("design.json: the stated latency is not the sum over the units"
      (stated_latency 5) (latency 6)))
    |}]
;;
