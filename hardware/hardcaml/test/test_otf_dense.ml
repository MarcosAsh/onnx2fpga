open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

module type Params = sig
  include Otf_dense.Config

  (* Indexed [c][k], the weight from input k to output channel c. *)
  val weight_values : int array array
  val bias_values : int array
  val mult_values : int array
end

let config
  ?(act_bits = 8)
  ?(wgt_bits = 8)
  ?(acc_bits = 20)
  ?(out_bits = 8)
  ?(mult_bits = 18)
  ?(shift = 20)
  ?(out_min = -128)
  ?(out_max = 127)
  ?(out_zp = 0)
  ?(arithmetic = `Multiplier)
  ~mw
  ~mh
  seed
  : (module Params)
  =
  let rng = Random.State.make [| seed |] in
  let weight_values =
    Array.init mh ~f:(fun _ -> Array.of_list (Model.random_ints rng ~bits:wgt_bits mw))
  in
  let bias_values = Array.of_list (Model.random_ints rng ~bits:(acc_bits - 4) mh) in
  let mult_values = Array.of_list (Model.random_ints rng ~bits:mult_bits mh) in
  (module struct
    let mw = mw
    let mh = mh
    let act_bits = act_bits
    let wgt_bits = wgt_bits
    let acc_bits = acc_bits
    let out_bits = out_bits
    let mult_bits = mult_bits
    let shift = shift
    let out_min = out_min
    let out_max = out_max
    let out_zp = out_zp
    let arithmetic = arithmetic
    let weight_values = weight_values
    let bias_values = bias_values
    let mult_values = mult_values

    let weights =
      Array.to_list weight_values
      |> List.map ~f:(fun row ->
        Stream_bench.of_signed_lanes ~lane_bits:wgt_bits (Array.to_list row))
    ;;

    let bias = Model.words_of_ints ~bits:acc_bits (Array.to_list bias_values)
    let mult = Model.words_of_ints ~bits:mult_bits (Array.to_list mult_values)
  end)
;;

module Make_test (Config : Params) = struct
  open Config
  module Dut = Otf_dense.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let expected x =
    List.init mh ~f:(fun c ->
      let dot =
        List.foldi x ~init:bias_values.(c) ~f:(fun k sum a ->
          sum + (a * weight_values.(c).(k)))
      in
      Model.requant
        ~acc_bits
        ~mult_bits
        ~shift
        ~out_min
        ~out_max
        ~out_zp
        ~acc:(Model.wrap ~bits:acc_bits dot)
        ~mult:mult_values.(c))
    |> Stream_bench.of_signed_lanes ~lane_bits:out_bits
  ;;

  (* A vector every cycle, valid on [duty] percent, each due exactly [latency] later. *)
  let run ~seed ~cycles ~duty () =
    let rng = Random.State.make [| seed |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         inputs.clocking.clear := Bits.vdd;
         Lws.cycle h;
         inputs.clocking.clear := Bits.gnd;
         let in_flight = Queue.create () in
         let checked = ref 0 in
         List.iter (List.range 0 cycles) ~f:(fun cycle ->
           let x = Model.random_ints rng ~bits:act_bits mw in
           let valid = Random.State.int rng 100 < duty in
           inputs.x.value := Stream_bench.of_signed_lanes ~lane_bits:act_bits x;
           inputs.x.valid := Bits.of_bool valid;
           Lws.cycle h;
           Queue.enqueue in_flight (valid, x);
           let got_valid = Bits.to_bool !(o.y.valid) in
           if Queue.length in_flight <= Dut.latency
           then (
             if got_valid then raise_s [%message "valid before any input" (cycle : int)])
           else (
             let sent_valid, sent = Queue.dequeue_exn in_flight in
             if Bool.( <> ) got_valid sent_valid
             then raise_s [%message "valid is not the input delayed" (cycle : int)];
             if sent_valid
             then (
               [%test_result: Bits.t] !(o.y.value) ~expect:(expected sent);
               incr checked)));
         !checked)
  ;;
end

let%expect_test "every vector comes out exactly three cycles later" =
  let cases arithmetic =
    [ "mlp first layer", config ~arithmetic ~mw:32 ~mh:64 1
    ; "relu at a zero point", config ~arithmetic ~mw:16 ~mh:8 ~out_zp:(-5) ~out_min:(-5) 2
    ; "one by one", config ~arithmetic ~mw:1 ~mh:1 ~acc_bits:17 ~shift:4 3
    ; "no shift", config ~arithmetic ~mw:4 ~mh:3 ~acc_bits:18 ~shift:0 ~mult_bits:3 4
    ; "quotient always zero", config ~arithmetic ~mw:3 ~mh:2 ~acc_bits:18 ~shift:36 5
    ; ( "narrow"
      , config
          ~arithmetic
          ~mw:6
          ~mh:5
          ~act_bits:4
          ~wgt_bits:3
          ~acc_bits:10
          ~mult_bits:5
          ~shift:3
          ~out_bits:4
          ~out_min:(-8)
          ~out_max:7
          6 )
    ]
  in
  List.iter [ `Multiplier; `Constant ] ~f:(fun arithmetic ->
    List.iter (cases arithmetic) ~f:(fun (name, (module Config : Params)) ->
      let module T = Make_test (Config) in
      let checked =
        List.map [ 100; 60; 5 ] ~f:(fun duty -> T.run ~seed:7 ~cycles:300 ~duty ())
      in
      print_s
        [%message
          name
            (arithmetic : [ `Multiplier | `Constant ])
            ~latency:(T.Dut.latency : int)
            (checked : int list)]));
  [%expect
    {|
    ("mlp first layer" (arithmetic Multiplier) (latency 3)
     (checked (297 174 16)))
    ("relu at a zero point" (arithmetic Multiplier) (latency 3)
     (checked (297 177 13)))
    ("one by one" (arithmetic Multiplier) (latency 3) (checked (297 173 10)))
    ("no shift" (arithmetic Multiplier) (latency 3) (checked (297 188 14)))
    ("quotient always zero" (arithmetic Multiplier) (latency 3)
     (checked (297 187 12)))
    (narrow (arithmetic Multiplier) (latency 3) (checked (297 185 13)))
    ("mlp first layer" (arithmetic Constant) (latency 3) (checked (297 174 16)))
    ("relu at a zero point" (arithmetic Constant) (latency 3)
     (checked (297 177 13)))
    ("one by one" (arithmetic Constant) (latency 3) (checked (297 173 10)))
    ("no shift" (arithmetic Constant) (latency 3) (checked (297 188 14)))
    ("quotient always zero" (arithmetic Constant) (latency 3)
     (checked (297 187 12)))
    (narrow (arithmetic Constant) (latency 3) (checked (297 185 13)))
    |}]
;;

let%expect_test "constant arithmetic on the mlp first layer needs no multiplier per \
                 weight"
  =
  List.iter [ `Multiplier; `Constant ] ~f:(fun arithmetic ->
    let (module Config) = config ~arithmetic ~mw:32 ~mh:64 1 in
    let module Dut = Otf_dense.Make (Config) in
    let module C = Circuit.With_interface (Dut.I) (Dut.O) in
    let scope = Scope.create ~flatten_design:true () in
    let circuit = C.create_exn ~name:"otf_dense" (Dut.hierarchical scope) in
    let wanted = [ "adders"; "subtractors"; "signed_multipliers"; "registers" ] in
    let utilization =
      match [%sexp_of: Circuit_utilization.t] (Circuit_utilization.create circuit) with
      | List (_ :: fields) ->
        List.filter fields ~f:(function
          | List (Atom field :: _) -> List.mem wanted field ~equal:String.equal
          | _ -> false)
      | _ -> []
    in
    print_s
      [%message (arithmetic : [ `Multiplier | `Constant ]) (utilization : Sexp.t list)]);
  [%expect
    {|
    ((arithmetic Multiplier)
     (utilization
      ((adders ((count 1084) (total_bits 22896) (max_instance_bits 39)))
       (subtractors ((count 1021) (total_bits 20420) (max_instance_bits 20)))
       (signed_multipliers
        ((count 2105) (total_bits 37129) (max_instance_bits 38)))
       (registers ((count 2172) (total_bits 34451))))))
    ((arithmetic Constant)
     (utilization
      ((adders ((count 1667) (total_bits 32224) (max_instance_bits 39)))
       (subtractors ((count 1613) (total_bits 29892) (max_instance_bits 20)))
       (signed_multipliers ((count 64) (total_bits 2432) (max_instance_bits 38)))
       (registers ((count 1308) (total_bits 17786))))))
    |}]
;;

let%expect_test "waveform of back to back vectors and a gap" =
  let (module Config) = config ~mw:2 ~mh:2 ~acc_bits:17 ~mult_bits:4 ~shift:6 9 in
  let module T = Make_test (Config) in
  let display_rules =
    [ Display_rule.port_name_matches (Glob "x$*") ~wave_format:(Bit_or Hex)
    ; Display_rule.port_name_matches (Glob "y$*") ~wave_format:(Bit_or Hex)
    ]
  in
  T.Harness.run
    ~create:T.Dut.hierarchical
    ~trace:`Ports_only
    ~print_waves_after_test:(fun waves ->
      Waveform.print ~display_rules ~signals_width:9 ~display_width:90 ~wave_width:2 waves)
    (fun (h @ local) ~inputs ~outputs:_ ->
      inputs.clocking.clear := Bits.vdd;
      Lws.cycle h;
      inputs.clocking.clear := Bits.gnd;
      List.iter [ Some 0x0102; Some 0x7f80; None; Some 0x0303 ] ~f:(fun x ->
        inputs.x.valid := Bits.of_bool (Option.is_some x);
        Bits.(inputs.x.value <--. Option.value x ~default:0);
        Lws.cycle h);
      inputs.x.valid := Bits.gnd;
      Lws.cycle ~n:4 h);
  [%expect
    {|
    ┌Signals┐┌Waves──────────────────────────────────────────────────────────────────────────┐
    │x$valid││      ┌───────────┐     ┌─────┐                                                │
    │       ││──────┘           └─────┘     └─────────────────────────────                   │
    │       ││──────┬─────┬─────┬─────┬───────────────────────────────────                   │
    │x$value││ 0000 │0102 │7F80 │0000 │0303                                                  │
    │       ││──────┴─────┴─────┴─────┴───────────────────────────────────                   │
    │y$valid││                        ┌───────────┐     ┌─────┐                              │
    │       ││────────────────────────┘           └─────┘     └───────────                   │
    │       ││────────────┬───────────┬─────┬─────┬─────┬─────────────────                   │
    │y$value││ 0000       │9270       │956A │807F │9270 │976B                                │
    │       ││────────────┴───────────┴─────┴─────┴─────┴─────────────────                   │
    └───────┘└───────────────────────────────────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "a reloadable layer uses the bank each vector was tagged with" =
  let (module Compiled) = config ~mw:8 ~mh:5 21 in
  let (module Retrained) = config ~mw:8 ~mh:5 22 in
  let module Old = Make_test (Compiled) in
  let module New = Make_test (Retrained) in
  let module Dut =
    Otf_dense.Make_reloadable (struct
      include Compiled

      let base = 4
    end)
  in
  let module Next =
    Otf_dense.Make_reloadable (struct
      include Retrained

      let base = 4
    end)
  in
  let module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O) in
  let rng = Random.State.make [| 3 |] in
  Harness.run
    ~random_initial_state:`All
    ~create:Dut.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       let store ~bank address data =
         inputs.store.write := Bits.vdd;
         inputs.store.bank := Bits.of_unsigned_int ~width:1 bank;
         inputs.store.address
         := Bits.of_unsigned_int ~width:Weight_banks.address_bits address;
         inputs.store.data := data;
         Lws.cycle h;
         inputs.store.write := Bits.gnd
       in
       Stream_bench.clear h inputs.clocking.clear;
       (* Words just outside the layer belong to other layers and must not land. *)
       store ~bank:1 3 (Bits.ones 32);
       List.iteri Next.image ~f:(fun index word -> store ~bank:1 (4 + index) word);
       store ~bank:1 (4 + Dut.words) (Bits.ones 32);
       let in_flight = Queue.create () in
       let answered = Array.create ~len:2 0 in
       List.iter (List.range 0 300) ~f:(fun _ ->
         let x = Model.random_ints rng ~bits:8 8 in
         let bank = Random.State.int rng 2 in
         let valid = Random.State.int rng 100 < 80 in
         inputs.x.value := Stream_bench.of_signed_lanes ~lane_bits:8 x;
         inputs.x.valid := Bits.of_bool valid;
         inputs.x_bank := Bits.of_unsigned_int ~width:1 bank;
         Lws.cycle h;
         Queue.enqueue in_flight (valid, bank, x);
         if Queue.length in_flight > Dut.latency
         then (
           let valid, bank, x = Queue.dequeue_exn in_flight in
           [%test_result: int] (Bits.to_unsigned_int !(o.y_bank)) ~expect:bank;
           [%test_result: bool] (Bits.to_bool !(o.y.valid)) ~expect:valid;
           if valid
           then (
             [%test_result: Bits.t]
               !(o.y.value)
               ~expect:(if bank = 0 then Old.expected x else New.expected x);
             answered.(bank) <- answered.(bank) + 1)));
       print_s [%message (Dut.words : int) (answered : int array)]);
  [%expect {| ((Dut.words 50) (answered (127 110))) |}]
;;
