open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Dut = Ddr_regs.Make (struct
    let params = Trading_fixture.params
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

let%test_unit "control registers honor channel skew, byte strobes and backpressure" =
  Harness.run
    ~random_initial_state:`All
    ~create:Dut.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       inputs.clear := Bits.vdd;
       inputs.s_axi_awvalid := Bits.gnd;
       inputs.s_axi_wvalid := Bits.gnd;
       inputs.s_axi_bready := Bits.gnd;
       inputs.s_axi_arvalid := Bits.gnd;
       inputs.s_axi_rready := Bits.gnd;
       inputs.arready := Bits.gnd;
       inputs.rvalid := Bits.gnd;
       inputs.rdata := Bits.zero 128;
       inputs.rresp := Bits.zero 2;
       inputs.rlast := Bits.gnd;
       Lws.cycle h;
       inputs.clear := Bits.gnd;
       let write ?(skew = 0) ?(strobe = 15) address data =
         inputs.s_axi_awaddr := Bits.of_int_trunc ~width:16 address;
         inputs.s_axi_wdata := Bits.of_int_trunc ~width:32 data;
         inputs.s_axi_wstrb := Bits.of_int_trunc ~width:4 strobe;
         inputs.s_axi_bready := Bits.gnd;
         let aw = ref false
         and w = ref false
         and response = ref None in
         let cycle = ref 0 in
         while Option.is_none !response do
           if !cycle > 50 then failwith "write timed out";
           inputs.s_axi_awvalid := Bits.of_bool ((not !aw) && !cycle >= Int.max 0 skew);
           inputs.s_axi_wvalid := Bits.of_bool ((not !w) && !cycle >= Int.max 0 (-skew));
           Lws.cycle h;
           if Bits.to_bool !(inputs.s_axi_awvalid) && Bits.to_bool !(o.s_axi_awready)
           then aw := true;
           if Bits.to_bool !(inputs.s_axi_wvalid) && Bits.to_bool !(o.s_axi_wready)
           then w := true;
           if Bits.to_bool !(o.s_axi_bvalid)
           then response := Some (Bits.to_int_trunc !(o.s_axi_bresp));
           incr cycle
         done;
         inputs.s_axi_awvalid := Bits.gnd;
         inputs.s_axi_wvalid := Bits.gnd;
         List.iter (List.range 0 4) ~f:(fun _ ->
           Lws.cycle h;
           assert (Bits.to_bool !(o.s_axi_bvalid));
           assert (not (Bits.to_bool !(o.s_axi_awready)));
           assert (not (Bits.to_bool !(o.s_axi_wready))));
         inputs.s_axi_bready := Bits.vdd;
         Lws.cycle h;
         Option.value_exn !response
       in
       let read address =
         inputs.s_axi_araddr := Bits.of_int_trunc ~width:16 address;
         inputs.s_axi_arvalid := Bits.vdd;
         inputs.s_axi_rready := Bits.gnd;
         Lws.cycle h;
         assert (Bits.to_bool !(o.s_axi_arready));
         inputs.s_axi_arvalid := Bits.gnd;
         Lws.cycle h;
         assert (Bits.to_bool !(o.s_axi_rvalid));
         let data = Bits.to_int_trunc !(o.s_axi_rdata) in
         let response = Bits.to_int_trunc !(o.s_axi_rresp) in
         List.iter (List.range 0 4) ~f:(fun _ ->
           inputs.s_axi_araddr := Bits.of_int_trunc ~width:16 0xffff;
           Lws.cycle h;
           assert (Bits.to_bool !(o.s_axi_rvalid));
           [%test_result: int] (Bits.to_int_trunc !(o.s_axi_rdata)) ~expect:data);
         inputs.s_axi_rready := Bits.vdd;
         Lws.cycle h;
         data, response
       in
       [%test_result: int * int] (read 0) ~expect:(0x4f544432, 0);
       [%test_result: int] (write ~skew:(-4) 0x20 0xaabbccdd) ~expect:0;
       [%test_result: int] (write ~skew:5 ~strobe:5 0x20 0x11223344) ~expect:0;
       [%test_result: int * int] (read 0x20) ~expect:(0xaa22cc44, 0);
       [%test_result: int] (write 0x20 0x1000) ~expect:0;
       [%test_result: int] (write 0x28 0) ~expect:0;
       [%test_result: int] (write 0x10 1) ~expect:0;
       List.iter (List.range 0 150) ~f:(fun _ -> Lws.cycle h);
       [%test_result: int * int] (read 0x14) ~expect:(2, 0);
       [%test_result: int] (write 0x24 0x100) ~expect:0;
       [%test_result: int] (write 0x10 1) ~expect:2;
       [%test_result: int * int] (read 0x14) ~expect:(2, 0);
       [%test_result: int] (write 0x42 1) ~expect:2;
       [%test_result: int * int] (read 0x42) ~expect:(0, 3);
       [%test_result: int] (write 0x24 0) ~expect:0;
       [%test_result: int] (write 0x28 1) ~expect:0;
       [%test_result: int] (write 0x10 1) ~expect:0;
       [%test_result: int] (write 0x10 1) ~expect:2;
       List.iter (List.range 0 150) ~f:(fun _ -> Lws.cycle h);
       assert (Bits.to_bool !(o.pipeline.replay.arvalid));
       [%test_result: int] (write 0x10 2) ~expect:0;
       assert (Bits.to_bool !(o.pipeline.busy));
       inputs.arready := Bits.vdd;
       Lws.cycle h;
       inputs.rvalid := Bits.vdd;
       inputs.rlast := Bits.vdd;
       Lws.cycle h;
       assert (Bits.to_bool !(o.pipeline.replay.rready));
       inputs.rvalid := Bits.gnd;
       List.iter (List.range 0 50) ~f:(fun _ -> Lws.cycle h);
       [%test_result: int * int] (read 0x14) ~expect:(10, 0);
       let packet =
         Itch_bench.packet
           [ Itch_bench.add ~order:1 ~side:'B' ~shares:100 ~price:500_000 ()
           ; Itch_bench.add ~order:2 ~side:'S' ~shares:400 ~price:500_100 ()
           ; Itch_bench.add ~locate:9 ~order:3 ~side:'S' ~shares:400 ~price:500_100 ()
           ; Itch_bench.execute ~order:2 ~shares:390 ()
           ]
       in
       let beats = Itch_bench.beats packet in
       [%test_result: int] (write 0x28 (List.length beats)) ~expect:0;
       [%test_result: int] (write 0x10 1) ~expect:0;
       let left = ref 0
       and index = ref 0 in
       List.iter (List.range 0 600) ~f:(fun _ ->
         inputs.rvalid := Bits.of_bool (!left > 0);
         inputs.rlast := Bits.of_bool (!left = 1);
         if !left > 0
         then (
           let beat = List.nth_exn beats !index in
           inputs.rdata
           := Bits.concat_lsb
                [ beat.data; beat.keep; Bits.of_bool beat.last; Bits.zero 55 ]);
         Lws.cycle h;
         if Bits.to_bool !(o.pipeline.replay.arvalid)
         then left := Bits.to_int_trunc !(o.pipeline.replay.arlen) + 1;
         if Bits.to_bool !(inputs.rvalid) && Bits.to_bool !(o.pipeline.replay.rready)
         then (
           decr left;
           incr index));
       [%test_result: int] !index ~expect:(List.length beats);
       [%test_result: int * int] (read 0x14) ~expect:(2, 0);
       [%test_result: int * int] (read 0x34) ~expect:(3, 0);
       [%test_result: int * int] (read 0x38) ~expect:(2, 0);
       [%test_result: int * int] (read 0x44) ~expect:(0x56b, 0);
       [%test_result: int] (fst (read 0x50) - fst (read 0x4c)) ~expect:14;
       [%test_result: int] (fst (read 0x100) land 255) ~expect:(Char.to_int 'O'))
;;
