open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml

module Dut = Fixed_stream.Make (struct
    let design = Design.of_string (Fixed_fixture.description ~latency:6)
  end)

module Harness = Hardcaml_test_harness.Lws_harness.Make (Dut.I) (Dut.O)

let%test_unit "random stalls preserve results and reset discards pending work" =
  Harness.run
    ~random_initial_state:`All
    ~create:Dut.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       let rng = Random.State.make [| 513 |] in
       let pending = Queue.create () in
       let held = ref None in
       let checked = ref 0 in
       inputs.rst_n := Bits.gnd;
       inputs.s_tvalid := Bits.gnd;
       inputs.m_tready := Bits.gnd;
       Lws.cycle h;
       List.iter (List.range 0 3000) ~f:(fun cycle ->
         let reset = cycle = 1500 || cycle = 1503 in
         let valid = cycle < 2900 && Random.State.bool rng in
         let ready = cycle > 2900 || (cycle % 100 > 50 && Random.State.bool rng) in
         let x = Model.random_ints rng ~bits:8 2 in
         inputs.rst_n := Bits.of_bool (not reset);
         inputs.s_tvalid := Bits.of_bool valid;
         inputs.s_tdata := Stream_bench.of_signed_lanes ~lane_bits:8 x;
         inputs.m_tready := Bits.of_bool ready;
         Lws.cycle h;
         if reset
         then (
           Queue.clear pending;
           held := None;
           [%test_result: bool] (Bits.to_bool !(o.m_tvalid)) ~expect:false)
         else (
           Option.iter !held ~f:(fun previous ->
             [%test_result: bool] (Bits.to_bool !(o.m_tvalid)) ~expect:true;
             [%test_result: Bits.t] !(o.m_tdata) ~expect:previous);
           if valid && Bits.to_bool !(o.s_tready)
           then Queue.enqueue pending (Fixed_fixture.model x);
           if Bits.to_bool !(o.m_tvalid)
           then (
             [%test_result: int list]
               (Stream_bench.signed_lanes ~lane_bits:8 !(o.m_tdata))
               ~expect:(Queue.peek_exn pending);
             if ready
             then (
               ignore (Queue.dequeue_exn pending : int list);
               incr checked));
           held
           := if Bits.to_bool !(o.m_tvalid) && not ready then Some !(o.m_tdata) else None));
       [%test_result: int] (Queue.length pending) ~expect:0;
       assert (!checked > 50))
;;
