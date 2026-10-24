open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

let pad_value = -3

module Make_test (G : sig
    val geometry : Window_bench.geometry
  end) =
struct
  let g = G.geometry
  let window = Window_bench.window g
  let ofm_h, ofm_w = Model.Window.output_extent window
  let simd = Window_bench.sub_elems g

  module Dut = Otf_swu_line.Make (struct
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
      let simd = simd
      let par = Window_bench.par g
      let data_bits = 8
      let pad_value = pad_value
    end)

  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  (* Frames back to back, so the free running counters cross a frame boundary. *)
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
           List.concat_map images ~f:(Model.beats_of_elements ~bits:8 ~per_beat:simd)
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

(* A frame costs the larger of capture and replay, not their sum. *)
let%expect_test "every geometry against im2col, two frames, four backpressure settings" =
  List.iter Window_bench.all ~f:(fun geometry ->
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
    (basic (cycles (2637 6622 10679 2945)))
    (folded (cycles (660 1681 2591 745)))
    (single_chan (cycles (1820 4655 7372 1896)))
    (same_pad (cycles (2451 6280 9694 2499)))
    (stride_pad (cycles (454 1132 1720 633)))
    (pool_window (cycles (419 1069 1565 1642)))
    (dilated (cycles (320 739 1032 527)))
    (pointwise (cycles (74 177 247 314)))
    (kernel_eq_ifm (cycles (52 121 202 190)))
    (non_square (cycles (1085 2782 4533 1103)))
    (par_gray (cycles (626 1535 2412 1279)))
    (par_padded (cycles (496 1242 2081 649)))
    (par_nonsquare (cycles (371 908 1574 542)))
    (par_stride (cycles (257 622 889 1051)))
    (par_dilated (cycles (284 714 984 862)))
    (par_pool (cycles (202 495 376 836)))
    (par_k5 (cycles (828 2059 3129 873)))
    (par_partial (cycles (422 1071 1655 613)))
    |}]
;;

let%expect_test "random initial state is overwritten by the clear" =
  let module T =
    Make_test (struct
      let geometry = List.nth_exn Window_bench.all 12
    end)
  in
  let cycles =
    T.random_traffic ~random_initial_state:`All ~frames:2 ~in_duty:60 ~out_duty:60 ()
  in
  print_s [%message (cycles : int)];
  [%expect {| (cycles 640) |}]
;;

module Padded = Make_test (struct
    let geometry =
      Window_bench.geometry "tiny_padded" (3, 4) 1 (3, 3) (1, 1) (1, 1) (1, 1) 3
    ;;
  end)

(* The first window is emitted after two pixels of row 1, and its top row is all padding. *)
let%expect_test "waveform of the first windows of a padded image" =
  let display_rules =
    [ Display_rule.port_name_matches (Glob "s$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "m$*") ~wave_format:(Bit_or Hex)
    ]
  in
  Padded.Harness.run
    ~create:Padded.Dut.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print
        ~display_rules
        ~signals_width:9
        ~display_width:120
        ~wave_width:3
        waves)
    (fun (h @ local) ~inputs ~outputs:_ ->
      Stream_bench.clear h inputs.clocking.clear;
      inputs.m.ready := Bits.vdd;
      inputs.s.valid := Bits.vdd;
      List.iter (List.range 1 9) ~f:(fun pixel ->
        Bits.(inputs.s.data <--. pixel);
        Lws.cycle h);
      inputs.s.valid := Bits.gnd;
      Lws.cycle ~n:4 h);
  [%expect
    {|
    ┌Signals┐┌Waves────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │       ││────────────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬────────────────────────────────────│
    │s$data ││ 00             │01     │02     │03     │04     │05     │06     │07     │08                                  │
    │       ││────────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴────────────────────────────────────│
    │s$valid││                ┌───────────────────────────────────────────────────────────────┐                            │
    │       ││────────────────┘                                                               └────────────────────────────│
    │s$ready││─────────────────────────────────────────────────────────────────────────────────────────────────────────────│
    │       ││                                                                                                             │
    │m$ready││                ┌────────────────────────────────────────────────────────────────────────────────────────────│
    │       ││────────────────┘                                                                                            │
    │       ││────────────────────────┬───────────────┬───────────────────────────────┬───────┬───────┬───────┬───────┬────│
    │m$data ││ 000000                 │FDFDFD         │0201FD                         │0605FD │FDFDFD │030201 │070605 │FDFD│
    │       ││────────────────────────┴───────────────┴───────────────────────────────┴───────┴───────┴───────┴───────┴────│
    │m$valid││                        ┌───────┐       ┌───────┐                       ┌────────────────────────────────────│
    │       ││────────────────────────┘       └───────┘       └───────────────────────┘                                    │
    └───────┘└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]
;;
