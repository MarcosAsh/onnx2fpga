open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Table = Order_table.Make (struct
    let address_bits = 2
    let data_bits = 9
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Table.I) (Table.O)

type access =
  | Idle
  | Read of int
  | Write of int * int
  | Speculate of int * int

let%expect_test "table updates and reset agree with a flat memory" =
  Harness.run
    ~random_initial_state:`All
    ~create:Table.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.after_edge outputs in
       let memory = Array.create ~len:4 0 in
       let expected_a = ref None in
       let expected_b = ref None in
       let drive (port : _ Table.Port.t) = function
         | Speculate (address, data) ->
           port.address := Bits.of_unsigned_int ~width:2 address;
           port.data := Bits.of_unsigned_int ~width:9 data;
           port.enable := Bits.gnd;
           port.write := Bits.gnd
         | Idle ->
           port.enable := Bits.gnd;
           port.write := Bits.gnd
         | Read address ->
           port.address := Bits.of_unsigned_int ~width:2 address;
           port.enable := Bits.vdd;
           port.write := Bits.gnd
         | Write (address, data) ->
           port.address := Bits.of_unsigned_int ~width:2 address;
           port.data := Bits.of_unsigned_int ~width:9 data;
           port.enable := Bits.vdd;
           port.write := Bits.vdd
       in
       let cycle ~initialize ~prefer_b a b =
         inputs.initialize := Bits.of_bool initialize;
         inputs.prefer_b := Bits.of_bool prefer_b;
         let payload_write = function
           | Write (_, data) -> data land 1 <> 0
           | Speculate _ -> true
           | Idle | Read _ -> false
         in
         inputs.payload_write := Bits.of_bool (payload_write (if prefer_b then b else a));
         drive inputs.port_a a;
         drive inputs.port_b b;
         let read expected = function
           | Read address -> expected := Some memory.(address)
           | Idle | Write _ | Speculate _ -> ()
         in
         read expected_a a;
         read expected_b b;
         let write = function
           | Write (address, data) -> memory.(address) <- data
           | Idle | Read _ | Speculate _ -> ()
         in
         write a;
         write b;
         Lws.cycle h;
         let check output expected =
           let canonical value = if value land 1 = 0 then 0 else value in
           Option.iter !expected ~f:(fun expected ->
             [%test_result: int]
               (canonical (Bits.to_unsigned_int !output))
               ~expect:(canonical expected))
         in
         check o.qa expected_a;
         check o.qb expected_b
       in
       let initialize () =
         List.iter (List.range 0 4) ~f:(fun address ->
           cycle ~initialize:true ~prefer_b:false (Write (address, 0)) Idle);
         ()
       in
       initialize ();
       let rng = Random.State.make [| 17 |] in
       List.iter (List.range 0 2000) ~f:(fun _ ->
         let a = Random.State.int rng 4 in
         let b = Random.State.int rng 4 in
         let live () = (Random.State.int rng 256 lsl 1) lor 1 in
         let operation = Random.State.int rng 7 in
         let prefer_b, a, b =
           match operation with
           | 0 -> false, Write (a, live ()), Read b
           | 1 -> true, Write (a, 0), Write ((a + 1) mod 4, live ())
           | 2 -> true, Read a, Write (b, live ())
           | 3 -> false, Write (a, 0), Read b
           | 4 when memory.(a) land 1 = 0 -> false, Speculate (a, live ()), Read a
           | 5 -> true, Write (a, 0), Speculate (a, live ())
           | _ -> false, Read a, Read b
         in
         cycle ~initialize:false ~prefer_b a b);
       initialize ();
       List.iter (List.range 0 4) ~f:(fun address ->
         cycle ~initialize:false ~prefer_b:false (Read address) (Read address));
       ());
  [%expect {| |}]
;;
