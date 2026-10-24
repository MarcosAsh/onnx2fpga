open! Core
open! Hardcaml
open Onnx2fpga_hardcaml

let pad_value = 5

module Make_test (G : sig
    val geometry : Window_bench.geometry
  end) =
struct
  let g = G.geometry
  let window = Window_bench.window g
  let ofm_h, ofm_w = Model.Window.output_extent window

  module Dut = Otf_swu.Make (struct
      let ifm_h = window.ifm_h
      let ifm_w = window.ifm_w
      let ifm_ch = window.ifm_ch
      let ofm_h = ofm_h
      let ofm_w = ofm_w
      let kh = window.kh
      let kw = window.kw
      let stride_h = fst window.stride
      let stride_w = snd window.stride
      let pad_t = fst window.pad
      let pad_l = snd window.pad
      let dil_h = fst window.dilation
      let dil_w = snd window.dilation
      let simd = g.simd
      let data_bits = 8
      let pad_value = pad_value
    end)

  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let random_traffic ?random_initial_state ~frames ~in_duty ~out_duty () =
    let rng = Random.State.make [| String.hash g.name |] in
    Harness.run
      ?random_initial_state
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let pixels = window.ifm_h * window.ifm_w * window.ifm_ch in
         let images =
           List.init frames ~f:(fun _ -> Model.random_ints rng ~bits:8 pixels)
         in
         let sent =
           List.concat_map images ~f:(Model.beats_of_elements ~bits:8 ~per_beat:g.simd)
         in
         let unfolded =
           List.concat_map images ~f:(Model.Window.unfold window ~pad_value)
         in
         let expected = Model.beats_of_elements ~bits:8 ~per_beat:g.simd unfolded in
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
             ~wanted:(List.length expected)
         in
         let cycles = Stream_bench.run h ~rng ~sources:[ source ] ~sinks:[ sink ] () in
         [%test_result: Bits.t list] (Stream_bench.Sink.received sink) ~expect:expected;
         cycles)
  ;;
end

(* At full rate a frame costs capture plus replay. *)
let%expect_test "every channel folded geometry against im2col, two frames" =
  List.iter Window_bench.all ~f:(fun geometry ->
    if Window_bench.par geometry = 1
    then
      let module T =
        Make_test (struct
          let geometry = geometry
        end)
      in
      let cycles =
        List.map
          [ 100, 100; 40, 40; 100, 25; 25, 100 ]
          ~f:(fun (in_duty, out_duty) -> T.random_traffic ~frames:2 ~in_duty ~out_duty ())
      in
      print_s [%message geometry.name (cycles : int list)]);
  [%expect
    {|
    (basic (cycles (3105 7912 11171 4601)))
    (folded (cycles (777 1939 2786 1126)))
    (single_chan (cycles (2089 5301 7720 2974)))
    (same_pad (cycles (2549 6548 9787 2849)))
    (stride_pad (cycles (613 1484 1859 1054)))
    (pool_window (cycles (801 2091 2035 2062)))
    (dilated (cycles (417 1022 1199 776)))
    (pointwise (cycles (145 320 350 376)))
    (kernel_eq_ifm (cycles (101 250 280 235)))
    (non_square (cycles (1201 3045 4642 1553)))
    |}]
;;

let%expect_test "random initial state is overwritten by the clear" =
  let module T =
    Make_test (struct
      let geometry = List.nth_exn Window_bench.all 4
    end)
  in
  let cycles =
    T.random_traffic ~random_initial_state:`All ~frames:2 ~in_duty:60 ~out_duty:60 ()
  in
  print_s [%message (cycles : int)];
  [%expect {| (cycles 1036) |}]
;;
