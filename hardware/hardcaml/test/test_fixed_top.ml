open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Mlp = Fixed_top.Make (struct
    let design = Design.of_string (Fixed_fixture.description ~latency:6)
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
          ~expect:(Fixed_fixture.model x))
      else assert (not (Bits.to_bool !(o.prediction_valid))));
    print_s [%message (Mlp.latency : int)]);
  [%expect {| (Mlp.latency 6) |}]
;;

let%expect_test "the ports are the ones the spec names" =
  Fixed_top.verilog (Design.of_string (Fixed_fixture.description ~latency:6))
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
  let design = Design.of_string (Fixed_fixture.description ~latency:5) in
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
