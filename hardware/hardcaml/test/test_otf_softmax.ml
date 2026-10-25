open! Core
open! Hardcaml
open Onnx2fpga_hardcaml

let signed_bits values =
  1
  + List.fold values ~init:1 ~f:(fun bits v ->
    Int.max bits (Int.num_bits - Int.clz (Int.abs v)))
;;

let round x = Float.iround_nearest_exn x

(* The L1 configuration's input zero point. *)
let zero_point = 3

type kind =
  | Softmax
  | Log
  | L1

(* Tables as the compiler builds them: 12 bit exponentials, then a reciprocal or a log. *)
let config ?(stages = Otf_softmax.Stages.none) ~lanes kind : (module Otf_softmax.Config) =
  let size = 256 in
  let scale = 0.06 in
  let log = 64. /. 0.08 in
  let mantissa_bits =
    match kind with
    | L1 -> 6
    | Softmax | Log -> 10
  in
  let mantissas =
    List.init (1 lsl mantissa_bits) ~f:(fun m ->
      1. +. ((Float.of_int m +. 0.5) /. Float.of_int (1 lsl mantissa_bits)))
  in
  let exponential =
    Array.init size ~f:(fun d -> round (4096. *. Float.exp (-.scale *. Float.of_int d)))
  in
  let sum_table, numer_table, min_exp =
    match kind with
    | L1 ->
      ( Array.init size ~f:(fun u -> Int.abs (u - 128 - zero_point))
      , Array.init size ~f:(fun u -> u - 128 - zero_point)
      , 0 )
    | Log ->
      ( exponential
      , Array.init size ~f:(fun d -> round (-.Float.of_int d *. scale *. log))
      , 12 )
    | Softmax -> exponential, [||], 12
  in
  let largest = Array.fold sum_table ~init:1 ~f:Int.max in
  let total_bits = Int.num_bits - Int.clz (lanes * largest) in
  let scale_table, exp_table =
    match kind with
    | Log ->
      ( List.map mantissas ~f:(fun m -> round (Float.log m *. log))
      , List.init (total_bits - min_exp) ~f:(fun j ->
          round (Float.of_int j *. Float.log 2. *. log)) )
    | Softmax | L1 -> List.map mantissas ~f:(fun m -> round (127. *. 1024. /. m)), []
  in
  let sum_bits = Int.num_bits - Int.clz largest in
  (module struct
    let lanes = lanes
    let in_bits = 8

    let subtract_max =
      match kind with
      | L1 -> false
      | Softmax | Log -> true
    ;;

    let mode =
      match kind with
      | Log -> Otf_softmax.Mode.Log
      | Softmax | L1 -> Divide
    ;;

    let sum_table = sum_table
    let sum_bits = sum_bits
    let numer_table = numer_table

    let numer_bits =
      if Array.is_empty numer_table
      then sum_bits + 1
      else signed_bits (Array.to_list numer_table)
    ;;

    let total_bits = total_bits
    let min_exp = min_exp
    let mantissa_bits = mantissa_bits
    let scale_table = Array.of_list scale_table
    let exp_table = Array.of_list exp_table

    let scale_bits =
      match mode with
      | Log -> signed_bits [ Array.last_exn scale_table + Array.last_exn exp_table ]
      | Divide -> 18
    ;;

    let shift =
      match mode with
      | Log -> 6
      | Divide -> 10 + total_bits - 1
    ;;

    let out_bits = 8

    let out_zp =
      match mode with
      | Log -> 127
      | Divide -> 0
    ;;

    let out_min = -128
    let out_max = 127
    let stages = stages
  end)
;;

module Make_test (Config : Otf_softmax.Config) = struct
  open Config
  module Dut = Otf_softmax.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let mantissa s k =
    let mask = (1 lsl mantissa_bits) - 1 in
    if k >= mantissa_bits
    then (s lsr (k - mantissa_bits)) land mask
    else (s lsl (mantissa_bits - k)) land mask
  ;;

  (* The compiler's SoftmaxUnit.execute. *)
  let expected x =
    let codes = Stream_bench.signed_lanes ~lane_bits:in_bits x in
    let top = List.fold codes ~init:Int.min_value ~f:Int.max in
    let index code = if subtract_max then top - code else code + (1 lsl (in_bits - 1)) in
    let a = List.map codes ~f:(fun code -> sum_table.(index code)) in
    let b =
      if Array.is_empty numer_table
      then a
      else List.map codes ~f:(fun code -> numer_table.(index code))
    in
    let s = List.sum (module Int) a ~f:Fn.id in
    let k = if s = 0 then min_exp else Int.max min_exp (Int.floor_log2 s) in
    let m = mantissa s k in
    let rescale acc mult =
      Model.python_requant ~shift ~out_min ~out_max ~out_zp ~acc ~mult
    in
    List.map b ~f:(fun b ->
      match mode with
      | Divide ->
        rescale (b lsl (total_bits - 1 - k)) (if s = 0 then 0 else scale_table.(m))
      | Log -> rescale (b - scale_table.(m) - exp_table.(k - min_exp)) 1)
    |> Stream_bench.of_signed_lanes ~lane_bits:out_bits
  ;;

  (* Codes within [spread] of the zero point, after a vector of it, whose L1 sum is 0. *)
  let run ~duty ~spread =
    let rng = Random.State.make [| duty; spread |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         Fixed_bench.check
           h
           ~x:inputs.x
           ~y:o.y
           ~latency:Dut.latency
           ~expected
           (( true
            , Stream_bench.of_signed_lanes
                ~lane_bits:in_bits
                (List.init lanes ~f:(Fn.const zero_point)) )
            :: List.init 199 ~f:(fun _ ->
              ( Random.State.int rng 100 < duty
              , Stream_bench.of_signed_lanes
                  ~lane_bits:in_bits
                  (List.init lanes ~f:(fun _ ->
                     Int.clamp_exn
                       (zero_point + Random.State.int_incl rng (-spread) spread)
                       ~min:(-128)
                       ~max:127)) ))))
  ;;
end

let%expect_test "softmax, logsoftmax and L1 match the compiler's model" =
  List.iter
    [ ( "softmax"
      , 5
      , Softmax
      , { Otf_softmax.Stages.max = 2; table = 1; sum = 2; normalize = 1; rescale = 3 } )
    ; ( "softmax, two lanes"
      , 2
      , Softmax
      , { max = 0; table = 1; sum = 0; normalize = 1; rescale = 2 } )
    ; "logsoftmax", 6, Log, { max = 2; table = 1; sum = 2; normalize = 1; rescale = 3 }
    ; "l1", 4, L1, { max = 0; table = 1; sum = 1; normalize = 1; rescale = 3 }
    ]
    ~f:(fun (name, lanes, kind, most) ->
      let runs =
        List.map [ Otf_softmax.Stages.none; most ] ~f:(fun stages ->
          let (module Config) = config ~stages ~lanes kind in
          let module T = Make_test (Config) in
          ( T.Dut.latency
          , List.map
              [ 100, 128; 40, 128; 100, 1 ]
              ~f:(fun (duty, spread) -> T.run ~duty ~spread) ))
      in
      let latencies, checked = List.unzip runs in
      print_s
        [%message name (latencies : int list) ~checked:(List.hd_exn checked : int list)]);
  [%expect
    {|
    (softmax (latencies (5 14)) (checked (200 78 200)))
    ("softmax, two lanes" (latencies (5 9)) (checked (200 75 200)))
    (logsoftmax (latencies (5 14)) (checked (200 81 200)))
    (l1 (latencies (4 10)) (checked (200 73 200)))
    |}]
;;
