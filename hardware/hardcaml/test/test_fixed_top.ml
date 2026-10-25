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

module Reloadable = Fixed_top.Make (struct
    let design = Design.of_string Fixed_fixture.reloadable
  end)

module Reload_harness =
  Hardcaml_test_harness.Lws_harness.Make (Reloadable.I) (Reloadable.O)

let image_crc bank =
  Weight_banks.crc
    (if bank = 0 then Fixed_fixture.compiled_image else Fixed_fixture.reloaded_image)
;;

let model bank = if bank = 0 then Fixed_fixture.model else Fixed_fixture.reloaded_model

(* A vector a cycle, one of [words] written a cycle, a commit on each of [commits]. *)
let stream ~rng ~cycles ~words ~commits =
  let checked = Array.create ~len:2 0 in
  Reload_harness.run ~create:Reloadable.hierarchical (fun (h @ local) ~inputs ~outputs ->
    let o = Before_and_after_edge.before_edge outputs in
    let get = Option.value_exn in
    let words = Queue.of_list words in
    let bank = ref 0 in
    let in_flight = Queue.create () in
    Stream_bench.clear h inputs.clear;
    List.iter (List.range 0 cycles) ~f:(fun cycle ->
      let commit = Set.mem commits cycle in
      (match Queue.dequeue words with
       | Some (address, data) ->
         get inputs.weights_write := Bits.vdd;
         get inputs.weights_address
         := Bits.of_unsigned_int ~width:Weight_banks.address_bits address;
         get inputs.weights_data := data
       | None -> get inputs.weights_write := Bits.gnd);
      get inputs.weights_commit := Bits.of_bool commit;
      let x = Model.random_ints rng ~bits:8 2 in
      inputs.features := Stream_bench.of_signed_lanes ~lane_bits:8 x;
      inputs.features_valid := Bits.vdd;
      Lws.cycle h;
      [%test_result: int] (Bits.to_unsigned_int !(get o.weights_bank)) ~expect:!bank;
      Queue.enqueue in_flight (!bank, x);
      if commit then bank := 1 - !bank;
      if Queue.length in_flight > Reloadable.latency
      then (
        let tag, x = Queue.dequeue_exn in_flight in
        [%test_result: int] (Bits.to_unsigned_int !(get o.prediction_bank)) ~expect:tag;
        [%test_result: int list]
          (Stream_bench.signed_lanes ~lane_bits:8 !(o.prediction))
          ~expect:(model tag x);
        checked.(tag) <- checked.(tag) + 1));
    let read field = Bits.to_unsigned_int !(get field) in
    read o.weights_version, read o.weights_crc, read o.weights_idle_crc, !bank, checked)
;;

let%expect_test "every answer uses the bank that was active when its features arrived" =
  let rng = Random.State.make [| 4 |] in
  let commits =
    Set.of_list
      (module Int)
      (60 :: 61 :: List.init 40 ~f:(fun _ -> 100 + Random.State.int rng 300))
  in
  let version, crc, idle_crc, bank, checked =
    stream
      ~rng
      ~cycles:450
      ~words:
        (List.mapi Fixed_fixture.reloaded_image ~f:(fun address word -> address, word))
      ~commits
  in
  [%test_result: int] version ~expect:(Set.length commits);
  [%test_result: int] crc ~expect:(image_crc bank);
  [%test_result: int] idle_crc ~expect:(image_crc (1 - bank));
  print_s
    [%message
      (Reloadable.words : int)
        (version : int)
        (checked : int array)
        ~compiled_crc:(image_crc 0 : Int.Hex.t)
        ~reloaded_crc:(image_crc 1 : Int.Hex.t)];
  [%expect
    {|
    ((Reloadable.words 22) (version 41) (checked (266 178))
     (compiled_crc 0xd6d2b031) (reloaded_crc 0xa0dac7ed))
    |}]
;;

let%expect_test "a half written bank is never visible and its CRC gives it away" =
  let words = List.mapi Fixed_fixture.reloaded_image ~f:Tuple2.create in
  let half = List.take words 11 in
  List.iter
    [ half; half @ words ]
    ~f:(fun words ->
      let version, crc, idle_crc, bank, checked =
        stream
          ~rng:(Random.State.make [| 8 |])
          ~cycles:100
          ~words
          ~commits:(Set.empty (module Int))
      in
      print_s
        [%message
          (version : int)
            (bank : int)
            (checked : int array)
            ~crc_is_compiled:(crc = image_crc 0 : bool)
            ~idle_crc_is_reloaded:(idle_crc = image_crc 1 : bool)
            ~idle_crc_is_the_half:
              (idle_crc = Weight_banks.crc (List.map half ~f:snd) : bool)]);
  [%expect
    {|
    ((version 0) (bank 0) (checked (94 0)) (crc_is_compiled true)
     (idle_crc_is_reloaded false) (idle_crc_is_the_half true))
    ((version 0) (bank 0) (checked (94 0)) (crc_is_compiled true)
     (idle_crc_is_reloaded true) (idle_crc_is_the_half false))
    |}]
;;

let%expect_test "the CRC is zlib's" =
  print_s
    [%message
      ""
        ~crc32_of_12345678:
          (Weight_banks.crc
             [ Bits.of_unsigned_int ~width:32 0x34333231
             ; Bits.of_unsigned_int ~width:32 0x38373635
             ]
           : Int.Hex.t)];
  [%expect {| (crc32_of_12345678 0x9ae0daaf) |}]
;;

let%expect_test "a reloadable design adds the reload ports" =
  Fixed_top.verilog (Design.of_string Fixed_fixture.reloadable)
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
        weights_write,
        weights_address,
        weights_data,
        weights_commit,
        prediction,
        prediction_valid,
        prediction_bank,
        weights_bank,
        weights_version,
        weights_crc,
        weights_idle_crc
    |}]
;;
