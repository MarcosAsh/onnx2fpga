open! Core
open! Hardcaml
open Onnx2fpga_hardcaml

module Make_test (Config : Otf_pool.Config) = struct
  open Config
  module Dut = Otf_pool.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let nf_count = channels / pe

  (* Each window is [window] passes over the channel groups. *)
  let expected beats =
    List.chunks_of beats ~length:(window * nf_count)
    |> List.concat_map ~f:(fun passes ->
      List.init nf_count ~f:(fun nf ->
        List.filteri passes ~f:(fun index _ -> index % nf_count = nf)
        |> List.map ~f:(Stream_bench.signed_lanes ~lane_bits:data_bits)
        |> List.reduce_exn ~f:(List.map2_exn ~f:Int.max)
        |> Stream_bench.of_signed_lanes ~lane_bits:data_bits))
  ;;

  let random_traffic ~seed ~windows ~in_duty ~out_duty () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let sent =
           Stream_bench.beats ~rng ~width:(pe * data_bits) (windows * window * nf_count)
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

let%expect_test "max over windows under random backpressure" =
  List.iter
    [ 4, 1, 4, 8, -128
    ; 8, 2, 9, 8, -128
    ; 6, 3, 1, 8, -128
    ; 4, 4, 2, 8, -128
    ; 3, 1, 3, 4, -8
    ]
    ~f:(fun (channels, pe, window, data_bits, data_min) ->
      let module T =
        Make_test (struct
          let channels = channels
          let pe = pe
          let window = window
          let data_bits = data_bits
          let data_min = data_min
        end)
      in
      let cycles =
        List.map
          [ 100, 100; 50, 50; 20, 90; 90, 20 ]
          ~f:(fun (in_duty, out_duty) ->
            T.random_traffic ~seed:channels ~windows:20 ~in_duty ~out_duty ())
      in
      print_s [%message (channels : int) (pe : int) (window : int) (cycles : int list)]);
  [%expect
    {|
    ((channels 4) (pe 1) (window 4) (cycles (320 795 1644 670)))
    ((channels 8) (pe 2) (window 9) (cycles (720 1588 3683 1158)))
    ((channels 6) (pe 3) (window 1) (cycles (40 176 236 188)))
    ((channels 4) (pe 4) (window 2) (cycles (40 131 192 115)))
    ((channels 3) (pe 1) (window 3) (cycles (180 474 983 425)))
    |}]
;;
