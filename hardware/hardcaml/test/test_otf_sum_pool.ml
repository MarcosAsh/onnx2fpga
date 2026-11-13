open! Core
open! Hardcaml
open Onnx2fpga_hardcaml

module Make_test (Config : Otf_sum_pool.Config) = struct
  open Config
  module Dut = Otf_sum_pool.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let nf_count = channels / pe

  (* Each window is [window] passes over the channel groups. *)
  let expected beats =
    List.chunks_of beats ~length:(window * nf_count)
    |> List.concat_map ~f:(fun passes ->
      List.init nf_count ~f:(fun nf ->
        List.filteri passes ~f:(fun index _ -> index % nf_count = nf)
        |> List.map ~f:(Stream_bench.signed_lanes ~lane_bits:data_bits)
        |> List.reduce_exn ~f:(List.map2_exn ~f:( + ))
        |> Stream_bench.of_signed_lanes ~lane_bits:out_bits))
  ;;

  let random_traffic ~seed ~windows ~in_duty ~out_duty () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         inputs.s.valid := Bits.vdd;
         inputs.s.data := Bits.ones (pe * data_bits);
         inputs.m.ready := Bits.gnd;
         Hardcaml_lws.Lws.cycle ~n:(Int.max 1 ((window - 1) * nf_count)) h;
         inputs.s.valid := Bits.gnd;
         Stream_bench.clear h inputs.clocking.clear;
         let sent =
           let extremes =
             List.concat_map
               [ -(1 lsl (data_bits - 1)); (1 lsl (data_bits - 1)) - 1 ]
               ~f:(fun value ->
                 List.init (window * nf_count) ~f:(fun _ ->
                   Stream_bench.of_signed_lanes
                     ~lane_bits:data_bits
                     (List.init pe ~f:(fun _ -> value))))
           in
           extremes
           @ Stream_bench.beats
               ~rng
               ~width:(pe * data_bits)
               ((windows - 2) * window * nf_count)
         in
         let source =
           Stream_bench.Source.create
             ~data:inputs.s.data
             ~valid:inputs.s.valid
             ~ready:o.s.ready
             ~duty:in_duty
             sent
         in
         let sink =
           Stream_bench.Sink.create
             ~data:o.m.data
             ~valid:o.m.valid
             ~ready:inputs.m.ready
             ~duty:out_duty
             ~wanted:(windows * nf_count)
         in
         let cycles = Stream_bench.run h ~rng ~sources:[ source ] ~sinks:[ sink ] () in
         [%test_result: Bits.t list]
           (Stream_bench.Sink.received sink)
           ~expect:(expected sent);
         cycles)
  ;;
end

let%expect_test "channel sums after reset under backpressure" =
  List.iter
    [ 4, 1, 15; 6, 3, 1; 4, 4, 64; 3, 1, 3 ]
    ~f:(fun (channels, pe, window) ->
      let module T =
        Make_test (struct
          let channels = channels
          let pe = pe
          let window = window
          let data_bits = 8
          let out_bits = data_bits + Int.ceil_log2 window
        end)
      in
      List.iter
        [ 100, 100; 50, 50; 20, 90; 90, 20 ]
        ~f:(fun (in_duty, out_duty) ->
          ignore (T.random_traffic ~seed:channels ~windows:20 ~in_duty ~out_duty () : int)));
  [%expect {| |}]
;;
