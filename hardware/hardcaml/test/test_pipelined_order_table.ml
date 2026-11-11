open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Table = Pipelined_order_table.Make (struct
    let address_bits = 2
    let data_bits = 9
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Table.I) (Table.O)

type action =
  | Idle
  | Write of int * int
  | Speculate of int * int

let%expect_test "forwarded reads agree with a flat memory" =
  Harness.run
    ~random_initial_state:`All
    ~create:Table.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.after_edge outputs in
       let memory = Array.create ~len:4 0 in
       let pending = ref (0, 0) in
       let drive (port : _ Table.Write.t) = function
         | Idle -> port.enable := Bits.gnd
         | (Write (address, data) | Speculate (address, data)) as action ->
           port.address := Bits.of_unsigned_int ~width:2 address;
           port.data := Bits.of_unsigned_int ~width:9 data;
           port.enable
           := Bits.of_bool
                (match action with
                 | Write _ -> true
                 | _ -> false)
       in
       let payload = function
         | Write (_, data) -> data land 1 <> 0
         | Speculate _ -> true
         | Idle -> false
       in
       let cycle ?(check = true) ~initialize ~prefer_b a b read_a read_b =
         inputs.initialize := Bits.of_bool initialize;
         inputs.prefer_b := Bits.of_bool prefer_b;
         inputs.payload_write := Bits.of_bool (payload (if prefer_b then b else a));
         inputs.read_a := Bits.of_unsigned_int ~width:2 read_a;
         inputs.read_b := Bits.of_unsigned_int ~width:2 read_b;
         drive inputs.write_a a;
         drive inputs.write_b b;
         let write = function
           | Write (address, data) -> memory.(address) <- data
           | Idle | Speculate _ -> ()
         in
         write a;
         write b;
         let expected = !pending in
         pending := memory.(read_a), memory.(read_b);
         Lws.cycle h;
         if check
         then (
           let canonical value = if value land 1 = 0 then 0 else value in
           let check output expected =
             [%test_result: int]
               (canonical (Bits.to_unsigned_int !output))
               ~expect:(canonical expected)
           in
           check o.qa (fst expected);
           check o.qb (snd expected);
           check o.qa_first (fst !pending);
           check o.qb_first (snd !pending))
       in
       let initialize () =
         List.iter (List.range 0 4) ~f:(fun address ->
           cycle
             ~check:false
             ~initialize:true
             ~prefer_b:false
             (Write (address, 0))
             Idle
             0
             0);
         cycle ~check:false ~initialize:false ~prefer_b:false Idle Idle 0 0
       in
       initialize ();
       let rng = Random.State.make [| 41 |] in
       List.iter (List.range 0 2000) ~f:(fun _ ->
         let a = Random.State.int rng 4 in
         let b = Random.State.int rng 4 in
         let live () = (Random.State.int rng 256 lsl 1) lor 1 in
         let prefer_b, first, second =
           match Random.State.int rng 8 with
           | 0 -> false, Write (a, live ()), Idle
           | 1 -> true, Write (a, 0), Write ((a + 1) mod 4, live ())
           | 2 -> true, Idle, Write (b, live ())
           | 3 -> false, Write (a, 0), Idle
           | 4 when memory.(a) land 1 = 0 -> false, Speculate (a, live ()), Idle
           | 5 -> true, Write (a, 0), Speculate (a, live ())
           | 6 -> false, Speculate (a, memory.(a)), Idle
           | _ -> false, Idle, Idle
         in
         cycle ~initialize:false ~prefer_b first second a b);
       initialize ();
       List.iter (List.range 0 5) ~f:(fun i ->
         cycle ~initialize:false ~prefer_b:false Idle Idle (i mod 4) (i mod 4));
       ());
  [%expect {| |}]
;;
