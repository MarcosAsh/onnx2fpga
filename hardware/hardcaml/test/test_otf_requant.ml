open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Make_test (Config : Otf_requant.Config) = struct
  module Dut = Otf_requant.Make (Config)
  module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

  (* Raises on a model mismatch; counts the pairs where the compiler disagrees. *)
  let check ?random_initial_state ?samples () =
    let open Config in
    let range bits = List.range (-(1 lsl (bits - 1))) (1 lsl (bits - 1)) in
    let pairs =
      match samples with
      | None -> List.cartesian_product (range acc_bits) (range mult_bits)
      | Some count ->
        let rng = Random.State.make [| acc_bits; mult_bits; shift |] in
        List.init count ~f:(fun _ ->
          ( Model.wrap ~bits:acc_bits (Random.State.bits rng)
          , Model.wrap ~bits:mult_bits (Random.State.bits rng) ))
    in
    Harness.run
      ?random_initial_state
      ~create:Dut.hierarchical
      (fun (h @ local) ~inputs ~outputs ->
         let out = (Before_and_after_edge.before_edge outputs).out in
         let disagreements = ref [] in
         List.iter pairs ~f:(fun (acc, mult) ->
           inputs.acc := Bits.of_signed_int ~width:acc_bits acc;
           inputs.mult := Bits.of_signed_int ~width:mult_bits mult;
           Lws.cycle h;
           let expect =
             Model.requant
               ~acc_bits
               ~mult_bits
               ~shift
               ~out_min
               ~out_max
               ~out_zp
               ~acc
               ~mult
           in
           let got = Bits.to_signed_int !out in
           if got <> expect
           then
             raise_s
               [%message
                 "requant differs from its model"
                   (acc : int)
                   (mult : int)
                   (got : int)
                   (expect : int)];
           if got <> Model.python_requant ~shift ~out_min ~out_max ~out_zp ~acc ~mult
           then disagreements := (acc, mult) :: !disagreements);
         List.length pairs, List.length !disagreements)
  ;;
end

(* Shifts 11 and 13 once disagreed with the compiler on 552 and 472 of these 1024 pairs. *)
let%expect_test "exhaustive at small widths, against the model and the compiler" =
  List.iter
    [ 0, -8, 7, 0
    ; 1, -8, 7, 0
    ; 3, -8, 7, 2
    ; 5, 0, 7, -3
    ; 9, -8, 7, 0
    ; 10, -8, 7, 0
    ; 11, -8, 7, 0
    ; 11, -8, 7, 3
    ; 13, -8, 7, 0
    ; 13, 0, 7, -2
    ; 62, -8, 7, 0
    ]
    ~f:(fun (shift, out_min, out_max, out_zp) ->
      let module T =
        Make_test (struct
          let acc_bits = 6
          let mult_bits = 4
          let shift = shift
          let out_bits = 4
          let out_min = out_min
          let out_max = out_max
          let out_zp = out_zp
        end)
      in
      let checked, differ_from_compiler = T.check () in
      print_s
        [%message
          (shift : int) (out_zp : int) (checked : int) (differ_from_compiler : int)]);
  [%expect
    {|
    ((shift 0) (out_zp 0) (checked 1024) (differ_from_compiler 0))
    ((shift 1) (out_zp 0) (checked 1024) (differ_from_compiler 0))
    ((shift 3) (out_zp 2) (checked 1024) (differ_from_compiler 0))
    ((shift 5) (out_zp -3) (checked 1024) (differ_from_compiler 0))
    ((shift 9) (out_zp 0) (checked 1024) (differ_from_compiler 0))
    ((shift 10) (out_zp 0) (checked 1024) (differ_from_compiler 0))
    ((shift 11) (out_zp 0) (checked 1024) (differ_from_compiler 0))
    ((shift 11) (out_zp 3) (checked 1024) (differ_from_compiler 0))
    ((shift 13) (out_zp 0) (checked 1024) (differ_from_compiler 0))
    ((shift 13) (out_zp -2) (checked 1024) (differ_from_compiler 0))
    ((shift 62) (out_zp 0) (checked 1024) (differ_from_compiler 0))
    |}]
;;

let%expect_test "random pairs at the widths a compiled matvec uses" =
  List.iter
    [ 20, 18, 25
    ; 19, 18, 24
    ; 27, 2, 17
    ; 32, 18, 0
    ; 20, 18, 37
    ; 20, 18, 38
    ; 20, 18, 39
    ; 19, 2, 62
    ]
    ~f:(fun (acc_bits, mult_bits, shift) ->
      let module T =
        Make_test (struct
          let acc_bits = acc_bits
          let mult_bits = mult_bits
          let shift = shift
          let out_bits = 8
          let out_min = -128
          let out_max = 127
          let out_zp = 5
        end)
      in
      let checked, differ_from_compiler =
        T.check ~random_initial_state:`All ~samples:2000 ()
      in
      print_s
        [%message
          (acc_bits : int) (shift : int) (checked : int) (differ_from_compiler : int)]);
  [%expect
    {|
    ((acc_bits 20) (shift 25) (checked 2000) (differ_from_compiler 0))
    ((acc_bits 19) (shift 24) (checked 2000) (differ_from_compiler 0))
    ((acc_bits 27) (shift 17) (checked 2000) (differ_from_compiler 0))
    ((acc_bits 32) (shift 0) (checked 2000) (differ_from_compiler 0))
    ((acc_bits 20) (shift 37) (checked 2000) (differ_from_compiler 0))
    ((acc_bits 20) (shift 38) (checked 2000) (differ_from_compiler 0))
    ((acc_bits 20) (shift 39) (checked 2000) (differ_from_compiler 0))
    ((acc_bits 19) (shift 62) (checked 2000) (differ_from_compiler 0))
    |}]
;;
