open! Core
open! Hardcaml
open Hardcaml_lws
open! Hardcaml_waveterm
open Onnx2fpga_hardcaml

let lanes = 4
let bits = 8
let shift = 17

(* Multipliers near what the compiler picks for scales within a factor of four. *)
let near_one rng = (1 lsl 15) + Random.State.int rng (3 lsl 15)

let config
  ?(operands = [ Otf_eltwise.Operand.Port 0; Port 1 ])
  ?(stages = Otf_eltwise.Stages.none)
  ~op
  seed
  : (module Otf_eltwise.Config)
  =
  let rng = Random.State.make [| seed |] in
  let count = List.length operands in
  let wide = bits + 1 in
  let mults, mult, acc_bits =
    match (op : Otf_eltwise.Op.t) with
    | Add -> List.init count ~f:(fun k -> (if k = 1 then -1 else 1) * near_one rng), 1, 28
    | Less | Less_equal | Equal ->
      let m = near_one rng in
      [ m; -m ], 1, 28
    | Mul -> [ 1; 1 ], 256 + Random.State.int rng 768, 2 * wide
    | Max | Min | Where | And | Or -> List.init count ~f:(fun _ -> near_one rng), 1, wide
  in
  (module struct
    let lanes = lanes
    let bits = bits
    let op = op
    let operands = operands

    (* Comparisons share a zero point, so Equal finds something; truth values sit at 0. *)
    let zero_points =
      let shared = Random.State.int_incl rng (-20) 20 in
      List.mapi operands ~f:(fun k operand ->
        match op, operand with
        | (_, Constant _ | (And | Or), _ | Where, Port _) when k = 0 -> 0
        | (Less | Less_equal | Equal), _ -> shared
        | _ -> Random.State.int_incl rng (-20) 20)
    ;;

    let mults = mults

    let tables =
      List.mapi operands ~f:(fun k operand ->
        match op, operand with
        | (Max | Min), Port _ -> Array.of_list (Model.random_ints rng ~bits (1 lsl bits))
        | Where, Port _ when k > 0 ->
          Array.of_list (Model.random_ints rng ~bits (1 lsl bits))
        | _ -> [||])
    ;;

    let mult = mult
    let mult_bits = 18
    let acc_bits = acc_bits
    let shift = shift
    let out_zp = Random.State.int_incl rng (-10) 10
    let out_min = -128
    let out_max = 127
    let stages = stages
  end)
;;

module Make_test (Config : Otf_eltwise.Config) = struct
  open Config
  module Dut = Otf_eltwise.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  let rescale acc m = Model.python_requant ~shift ~out_min ~out_max ~out_zp ~acc ~mult:m
  let looked_up k code = (List.nth_exn tables k).(code + (1 lsl (bits - 1)))

  (* The compiler's EltwiseUnit.execute for one lane. *)
  let expected ports k =
    let codes =
      List.map operands ~f:(function
        | Port index -> List.nth_exn (List.nth_exn ports index) k
        | Constant codes -> List.nth_exn codes k)
    in
    let terms = List.map2_exn codes zero_points ~f:( - ) in
    let truth = List.map2_exn codes zero_points ~f:( <> ) in
    let weighted =
      List.sum (module Int) (List.zip_exn terms mults) ~f:(fun (t, m) -> t * m)
    in
    let bool b = Bool.to_int b in
    match op, terms, truth, operands with
    | Add, _, _, _ -> rescale weighted mult
    | Mul, [ a; b ], _, _ -> rescale (a * b) mult
    | Less, _, _, _ -> bool (weighted < 0)
    | Less_equal, _, _, _ -> bool (weighted <= 0)
    | Equal, _, _, _ -> bool (weighted = 0)
    | Max, _, _, _ ->
      Int.max (looked_up 0 (List.nth_exn codes 0)) (looked_up 1 (List.nth_exn codes 1))
    | Min, _, _, _ ->
      Int.min (looked_up 0 (List.nth_exn codes 0)) (looked_up 1 (List.nth_exn codes 1))
    | And, _, [ a; b ], _ -> bool (a && b)
    | Or, _, [ a; b ], _ -> bool (a || b)
    | Where, _, condition :: _, _ ->
      let k = if condition then 1 else 2 in
      (match List.nth_exn operands k with
       | Port _ -> looked_up k (List.nth_exn codes k)
       | Constant _ -> List.nth_exn codes k)
    | _ -> raise_s [%message "model has no case" (op : Otf_eltwise.Op.t)]
  ;;

  (* A random vector per port each cycle, codes within [spread] of zero. *)
  let run ~duty ~spread ~cycles =
    let rng = Random.State.make [| duty; spread |] in
    Harness.run
      ~random_initial_state:`All
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let o = Before_and_after_edge.before_edge outputs in
         Stream_bench.clear h inputs.clocking.clear;
         let in_flight = Queue.create () in
         let checked = ref 0 in
         let seen = Hash_set.create (module Int) in
         let cycle (valid, vectors) =
           List.iter2_exn inputs.x vectors ~f:(fun port vector ->
             port := Stream_bench.of_signed_lanes ~lane_bits:bits vector);
           inputs.valid := Bits.of_bool valid;
           Lws.cycle h;
           Queue.enqueue in_flight (valid, vectors);
           let got_valid = Bits.to_bool !(o.y.valid) in
           if Queue.length in_flight <= Dut.latency
           then (if got_valid then raise_s [%message "valid before any input"])
           else (
             let sent_valid, sent = Queue.dequeue_exn in_flight in
             if Bool.( <> ) got_valid sent_valid
             then raise_s [%message "valid is not the input delayed"];
             if sent_valid
             then (
               let got = Stream_bench.signed_lanes ~lane_bits:bits !(o.y.value) in
               [%test_result: int list] got ~expect:(List.init lanes ~f:(expected sent));
               List.iter got ~f:(Hash_set.add seen);
               incr checked))
         in
         let vectors () =
           List.map inputs.x ~f:(fun _ ->
             List.init lanes ~f:(fun _ -> Random.State.int_incl rng (-spread) spread))
         in
         List.iter (List.init cycles ~f:Fn.id) ~f:(fun _ ->
           cycle (Random.State.int rng 100 < duty, vectors ()));
         List.iter (List.init Dut.latency ~f:Fn.id) ~f:(fun _ ->
           cycle (false, vectors ()));
         !checked, Hash_set.length seen)
  ;;
end

let codes = [ 5; -128; 127; 0 ]

let%expect_test "every operator matches the compiler's model, with and without stages" =
  List.iter
    [ "add", Otf_eltwise.Op.Add, [], { Otf_eltwise.Stages.product = 1; rescale = 2 }
    ; "mul", Mul, [], { product = 1; rescale = 3 }
    ; "square", Mul, [ Otf_eltwise.Operand.Port 0; Port 0 ], { product = 0; rescale = 1 }
    ; "less", Less, [], { product = 1; rescale = 0 }
    ; "less or equal", Less_equal, [], { product = 1; rescale = 0 }
    ; "equal", Equal, [], { product = 1; rescale = 0 }
    ; "max", Max, [], { product = 0; rescale = 1 }
    ; "min", Min, [], { product = 0; rescale = 1 }
    ; "and", And, [], Otf_eltwise.Stages.none
    ; "or", Or, [], Otf_eltwise.Stages.none
    ; "where", Where, [ Port 0; Port 1; Port 2 ], { product = 0; rescale = 1 }
    ; ( "where, constant condition"
      , Where
      , [ Constant [ 1; 0; 0; 1 ]; Port 0; Port 1 ]
      , { product = 0; rescale = 1 } )
    ; ( "where, constant branch"
      , Where
      , [ Port 1; Constant codes; Port 0 ]
      , Otf_eltwise.Stages.none )
    ]
    ~f:(fun (name, op, operands, most) ->
      let operands = if List.is_empty operands then None else Some operands in
      let latencies, runs =
        List.map [ Otf_eltwise.Stages.none; most ] ~f:(fun stages ->
          let (module Config) = config ?operands ~stages ~op 3 in
          let module T = Make_test (Config) in
          ( T.Dut.latency
          , List.map
              [ 100, 127; 40, 127; 100, 3 ]
              ~f:(fun (duty, spread) -> T.run ~duty ~spread ~cycles:150) ))
        |> List.unzip
      in
      let checked, distinct = List.unzip (List.hd_exn runs) in
      if not ([%equal: (int * int) list] (List.hd_exn runs) (List.last_exn runs))
      then raise_s [%message "stages changed what was checked" name];
      print_s
        [%message name (latencies : int list) (checked : int list) (distinct : int list)]);
  [%expect
    {|
    (add (latencies (2 5)) (checked (150 53 150)) (distinct (215 143 9)))
    (mul (latencies (2 6)) (checked (150 53 150)) (distinct (131 84 1)))
    (square (latencies (2 3)) (checked (150 75 150)) (distinct (81 79 1)))
    (less (latencies (2 3)) (checked (150 53 150)) (distinct (2 2 2)))
    ("less or equal" (latencies (2 3)) (checked (150 53 150)) (distinct (2 2 2)))
    (equal (latencies (2 3)) (checked (150 53 150)) (distinct (2 2 2)))
    (max (latencies (1 2)) (checked (150 53 150)) (distinct (160 122 13)))
    (min (latencies (1 2)) (checked (150 53 150)) (distinct (171 114 13)))
    (and (latencies (1 1)) (checked (150 53 150)) (distinct (2 2 2)))
    (or (latencies (1 1)) (checked (150 53 150)) (distinct (1 1 2)))
    (where (latencies (1 2)) (checked (150 67 150)) (distinct (156 126 14)))
    ("where, constant condition" (latencies (1 2)) (checked (150 53 150))
     (distinct (186 120 14)))
    ("where, constant branch" (latencies (1 1)) (checked (150 53 150))
     (distinct (7 4 11)))
    |}]
;;
