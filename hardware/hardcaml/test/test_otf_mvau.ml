open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

module type Params = sig
  include Otf_mvau.Config

  val weight_values : int list
  val bias_values : int list
  val mult_values : int list
end

module Make_test (Config : Params) = struct
  open Config
  module Dut = Otf_mvau.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let sf_count = mw / simd
  let nf_count = mh / pe

  (* Lane p's word for (nf, sf) holds w[sf * simd + k][c] in element k. *)
  let expected beats =
    let weights = Array.of_list weight_values in
    let bias = Array.of_list bias_values in
    let mults = Array.of_list mult_values in
    List.chunks_of beats ~length:sf_count
    |> List.concat_map ~f:(fun vector ->
      let x = Array.of_list (Model.elements ~bits:act_bits vector) in
      List.init nf_count ~f:(fun nf ->
        List.init pe ~f:(fun p ->
          let lane = (p * nf_count) + nf in
          let dot =
            List.sum (module Int) (List.range 0 mw) ~f:(fun k ->
              let word = (p * sf_count * nf_count) + (nf * sf_count) + (k / simd) in
              x.(k) * weights.((word * simd) + (k % simd)))
          in
          Model.requant
            ~acc_bits
            ~mult_bits
            ~shift
            ~out_min
            ~out_max
            ~out_zp
            ~acc:(Model.wrap ~bits:acc_bits (bias.(lane) + dot))
            ~mult:mults.(lane))
        |> Stream_bench.of_signed_lanes ~lane_bits:out_bits))
  ;;

  let random_traffic ~seed ~vectors ~in_duty ~out_duty () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let sent =
           Stream_bench.beats ~rng ~width:(simd * act_bits) (vectors * sf_count)
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
             ~wanted:(vectors * nf_count)
         in
         let cycles = Stream_bench.run h ~rng ~sources:[ source ] ~sinks:[ sink ] () in
         [%test_result: Bits.t list]
           (Stream_bench.Sink.received sink)
           ~expect:(expected sent);
         cycles)
  ;;
end

let config
  ?(acc_bits = 20)
  ?(shift = 20)
  ?(out_min = -128)
  ?(out_max = 127)
  ?(out_zp = 0)
  ?(mult_bits = 18)
  ?(act_bits = 8)
  ?(wgt_bits = 8)
  ?(out_bits = 8)
  ?(block_ram = false)
  ~mw
  ~mh
  ~simd
  ~pe
  seed
  : (module Params)
  =
  let rng = Random.State.make [| seed |] in
  let weight_values = Model.random_ints rng ~bits:wgt_bits (mw * mh) in
  let bias_values = Model.random_ints rng ~bits:(Int.min acc_bits 16) mh in
  let mult_values = Model.random_ints rng ~bits:mult_bits mh in
  (module struct
    let mw = mw
    let mh = mh
    let simd = simd
    let pe = pe
    let act_bits = act_bits
    let wgt_bits = wgt_bits
    let acc_bits = acc_bits
    let out_bits = out_bits
    let mult_bits = mult_bits
    let shift = shift
    let out_min = out_min
    let out_max = out_max
    let out_zp = out_zp
    let block_ram = block_ram
    let weight_values = weight_values
    let bias_values = bias_values
    let mult_values = mult_values

    (* Block RAM holds one word per step with every lane's weights side by side. *)
    let weights =
      let words =
        List.chunks_of weight_values ~length:simd
        |> List.map ~f:(Stream_bench.of_signed_lanes ~lane_bits:wgt_bits)
      in
      if block_ram
      then
        List.chunks_of words ~length:(mw / simd * (mh / pe))
        |> List.transpose_exn
        |> List.map ~f:Bits.concat_lsb
      else words
    ;;

    let bias = Model.words_of_ints ~bits:acc_bits bias_values
    let mult = Model.words_of_ints ~bits:mult_bits mult_values
  end)
;;

(* At full rate a vector costs exactly (mw / simd) * (mh / pe) cycles once primed. *)
let%expect_test "matrix vector products under random backpressure" =
  List.iter
    [ "residual", config ~mw:16 ~mh:16 ~simd:4 ~pe:1 ~acc_bits:19 ~shift:24 1
    ; "mlp first layer", config ~mw:32 ~mh:64 ~simd:16 ~pe:2 2
    ; "one synapse fold", config ~mw:9 ~mh:4 ~simd:9 ~pe:1 ~acc_bits:18 3
    ; "fully serial", config ~mw:8 ~mh:8 ~simd:1 ~pe:1 4
    ; "odd folds", config ~mw:12 ~mh:10 ~simd:3 ~pe:5 ~out_zp:7 ~out_min:0 5
    ; "one word", config ~mw:4 ~mh:4 ~simd:4 ~pe:4 ~shift:0 ~acc_bits:12 6
    ; "wrapping", config ~mw:6 ~mh:9 ~simd:2 ~pe:3 ~shift:6 ~acc_bits:11 7
    ; ( "narrow"
      , config
          ~mw:4
          ~mh:6
          ~simd:2
          ~pe:2
          ~act_bits:4
          ~wgt_bits:3
          ~acc_bits:9
          ~mult_bits:5
          ~shift:3
          ~out_bits:4
          ~out_min:(-8)
          ~out_max:7
          8 )
    ; ( "mlp first layer in block RAM"
      , config ~mw:32 ~mh:64 ~simd:16 ~pe:2 ~block_ram:true 2 )
    ; "fully serial in block RAM", config ~mw:8 ~mh:8 ~simd:1 ~pe:1 ~block_ram:true 4
    ; ( "odd folds in block RAM"
      , config ~mw:12 ~mh:10 ~simd:3 ~pe:5 ~out_zp:7 ~out_min:0 ~block_ram:true 5 )
    ; ( "one synapse fold in block RAM"
      , config ~mw:9 ~mh:4 ~simd:9 ~pe:1 ~acc_bits:18 ~block_ram:true 3 )
    ]
    ~f:(fun (name, (module Config)) ->
      let module T = Make_test (Config) in
      let cycles =
        List.map
          [ 100, 100; 50, 50; 20, 90; 90, 20 ]
          ~f:(fun (in_duty, out_duty) ->
            T.random_traffic ~seed:5 ~vectors:12 ~in_duty ~out_duty ())
      in
      print_s
        [%message
          name ~cycles_per_vector:(T.sf_count * T.nf_count : int) (cycles : int list)]);
  [%expect
    {|
    (residual (cycles_per_vector 64) (cycles (769 827 917 1166)))
    ("mlp first layer" (cycles_per_vector 64) (cycles (769 984 862 1902)))
    ("one synapse fold" (cycles_per_vector 4) (cycles (49 110 99 268)))
    ("fully serial" (cycles_per_vector 64) (cycles (769 849 1216 883)))
    ("odd folds" (cycles_per_vector 8) (cycles (97 152 305 181)))
    ("one word" (cycles_per_vector 1) (cycles (13 40 96 65)))
    (wrapping (cycles_per_vector 9) (cycles (109 158 240 239)))
    (narrow (cycles_per_vector 6) (cycles (73 138 193 233)))
    ("mlp first layer in block RAM" (cycles_per_vector 64)
     (cycles (770 970 864 1929)))
    ("fully serial in block RAM" (cycles_per_vector 64)
     (cycles (770 850 1217 883)))
    ("odd folds in block RAM" (cycles_per_vector 8) (cycles (98 147 306 181)))
    ("one synapse fold in block RAM" (cycles_per_vector 4)
     (cycles (50 110 100 250)))
    |}]
;;

(* Two synapse and two neuron folds; the vector is replayed from the buffer on the second. *)
let waveform ~block_ram =
  let (module Config) = config ~mw:4 ~mh:4 ~simd:2 ~pe:2 ~shift:26 ~block_ram 10 in
  let module T = Make_test (Config) in
  let display_rules =
    [ Display_rule.port_name_matches (Glob "s$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "m$*") ~wave_format:(Bit_or Hex)
    ]
  in
  T.Harness.run
    ~create:T.Dut.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print
        ~display_rules
        ~signals_width:9
        ~display_width:100
        ~wave_width:2
        waves)
    (fun (h @ local) ~inputs ~outputs ->
      let o = Before_and_after_edge.before_edge outputs in
      Stream_bench.clear h inputs.clocking.clear;
      inputs.s.valid := Bits.vdd;
      List.iteri [ 0x0201; 0x0403; 0x0605; 0x0807 ] ~f:(fun index data ->
        Bits.(inputs.s.data <--. data);
        let taken = ref false in
        while not !taken do
          Lws.cycle h;
          taken := Bits.to_bool !(o.s.ready);
          if index = 2 then inputs.m.ready := Bits.vdd
        done);
      inputs.s.valid := Bits.gnd;
      Lws.cycle ~n:6 h)
;;

let%expect_test "waveform of a vector taken once and replayed" =
  waveform ~block_ram:false;
  [%expect
    {|
    ┌Signals┐┌Waves────────────────────────────────────────────────────────────────────────────────────┐
    │       ││────────────┬─────┬─────┬─────────────────┬──────────────────────────────────────────────│
    │s$data ││ 0000       │0201 │0403 │0605             │0807                                          │
    │       ││────────────┴─────┴─────┴─────────────────┴──────────────────────────────────────────────│
    │s$valid││            ┌───────────────────────────────────┐                                        │
    │       ││────────────┘                                   └────────────────────────────────────────│
    │s$ready││────────────────────────┐           ┌───────────┐           ┌────────────────────────────│
    │       ││                        └───────────┘           └───────────┘                            │
    │m$ready││                              ┌──────────────────────────────────────────────────────────│
    │       ││──────────────────────────────┘                                                          │
    │       ││────────────────────────┬───────────┬───────────┬───────────┬────────────────────────────│
    │m$data ││ 0000                   │EAE5       │FE04       │EAE6       │FF05                        │
    │       ││────────────────────────┴───────────┴───────────┴───────────┴────────────────────────────│
    │m$valid││                        ┌─────────────────┐     ┌─────┐     ┌─────┐                      │
    │       ││────────────────────────┘                 └─────┘     └─────┘     └──────────────────────│
    └───────┘└─────────────────────────────────────────────────────────────────────────────────────────┘
    |}]
;;

(* Block RAM moves every result one cycle later; the input side is unchanged. *)
let%expect_test "waveform with the weights in block RAM" =
  waveform ~block_ram:true;
  [%expect
    {|
    ┌Signals┐┌Waves────────────────────────────────────────────────────────────────────────────────────┐
    │       ││────────────┬─────┬─────┬─────────────────┬──────────────────────────────────────────────│
    │s$data ││ 0000       │0201 │0403 │0605             │0807                                          │
    │       ││────────────┴─────┴─────┴─────────────────┴──────────────────────────────────────────────│
    │s$valid││            ┌───────────────────────────────────┐                                        │
    │       ││────────────┘                                   └────────────────────────────────────────│
    │s$ready││────────────────────────┐           ┌───────────┐           ┌────────────────────────────│
    │       ││                        └───────────┘           └───────────┘                            │
    │m$ready││                              ┌──────────────────────────────────────────────────────────│
    │       ││──────────────────────────────┘                                                          │
    │       ││──────────────────────────────┬───────────┬───────────┬───────────┬──────────────────────│
    │m$data ││ 0000                         │EAE5       │FE04       │EAE6       │FF05                  │
    │       ││──────────────────────────────┴───────────┴───────────┴───────────┴──────────────────────│
    │m$valid││                              ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐                │
    │       ││──────────────────────────────┘     └─────┘     └─────┘     └─────┘     └────────────────│
    └───────┘└─────────────────────────────────────────────────────────────────────────────────────────┘
    |}]
;;
