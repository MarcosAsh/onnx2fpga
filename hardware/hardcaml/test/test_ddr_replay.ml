open! Core
open! Hardcaml
open Hardcaml_lws
open Onnx2fpga_hardcaml
module Harness = Hardcaml_test_harness.Lws_harness.Make (Ddr_replay.I) (Ddr_replay.O)

let record index =
  let last = index % 7 = 6 || index = 52 in
  let keep = if last then 7 else 255 in
  Bits.concat_msb
    [ Bits.zero 55
    ; Bits.of_bool last
    ; Bits.of_int_trunc ~width:8 keep
    ; Bits.of_int_trunc ~width:64 (index + 100)
    ]
;;

let%test_unit "DDR records survive address, response and consumer stalls" =
  Harness.run
    ~random_initial_state:`All
    ~create:Ddr_replay.hierarchical
    (fun (h @ local) ~inputs ~outputs ->
       let o = Before_and_after_edge.before_edge outputs in
       let rng = Random.State.make [| 871 |] in
       inputs.clear := Bits.vdd;
       inputs.start := Bits.gnd;
       inputs.abort := Bits.gnd;
       inputs.ready := Bits.vdd;
       inputs.arready := Bits.gnd;
       inputs.rvalid := Bits.gnd;
       inputs.rlast := Bits.gnd;
       inputs.rresp := Bits.zero 2;
       inputs.rdata := Bits.zero 128;
       inputs.base := Bits.of_int_trunc ~width:40 0xff00;
       inputs.records := Bits.of_int_trunc ~width:32 53;
       Lws.cycle h;
       let left = ref 0 in
       let index = ref 0 in
       let sent = ref 0 in
       let addresses = ref 0 in
       let presenting = ref false in
       let held_address = ref None in
       List.iter (List.range 0 1000) ~f:(fun cycle ->
         inputs.clear := Bits.gnd;
         inputs.start := Bits.of_bool (cycle = 0);
         inputs.ready := Bits.of_bool (cycle = 0 || Random.State.bool rng);
         inputs.arready := Bits.of_bool (Random.State.bool rng);
         presenting := !left > 0 && (!presenting || Random.State.bool rng);
         inputs.rvalid := Bits.of_bool !presenting;
         inputs.rlast := Bits.of_bool (!left = 1);
         inputs.rdata := record !index;
         Lws.cycle h;
         Option.iter !held_address ~f:(fun (addr, len) ->
           assert (Bits.to_bool !(o.arvalid));
           [%test_result: int] (Bits.to_int_trunc !(o.araddr)) ~expect:addr;
           [%test_result: int] (Bits.to_int_trunc !(o.arlen)) ~expect:len);
         held_address
         := if Bits.to_bool !(o.arvalid) && not (Bits.to_bool !(inputs.arready))
            then Some (Bits.to_int_trunc !(o.araddr), Bits.to_int_trunc !(o.arlen))
            else None;
         if Bits.to_bool !(o.arvalid) && Bits.to_bool !(inputs.arready)
         then (
           assert (!left = 0);
           [%test_result: int]
             (Bits.to_int_trunc !(o.araddr))
             ~expect:(0xff00 + (!index * 16));
           [%test_result: int]
             (Bits.to_int_trunc !(o.arlen))
             ~expect:(Int.min 16 (53 - !index) - 1);
           [%test_result: int] (Bits.to_int_trunc !(o.arsize)) ~expect:4;
           left := Bits.to_int_trunc !(o.arlen) + 1;
           incr addresses);
         if Bits.to_bool !(o.valid)
         then (
           [%test_result: int] (Bits.to_int_trunc !(o.data)) ~expect:(!sent + 100);
           [%test_result: Bits.t]
             !(o.keep)
             ~expect:(Bits.select (record !sent) ~high:71 ~low:64);
           incr sent);
         if !presenting && Bits.to_bool !(o.rready)
         then (
           incr index;
           decr left;
           presenting := false));
       [%test_result: int] !sent ~expect:53;
       [%test_result: int] !addresses ~expect:4;
       [%test_result: int] (Bits.to_int_trunc !(o.delivered)) ~expect:53;
       assert (Bits.to_bool !(o.done_));
       assert (not (Bits.to_bool !(o.busy)));
       assert (not (Bits.to_bool !(o.error))))
;;

let%test_unit "errors and abort drain a burst before another start" =
  List.iter
    [ "response"; "keep"; "early_last"; "late_last"; "abort"; "abort_data" ]
    ~f:(fun fault ->
      Harness.run
        ~random_initial_state:`All
        ~create:Ddr_replay.hierarchical
        (fun (h @ local) ~inputs ~outputs ->
           let o = Before_and_after_edge.before_edge outputs in
           inputs.clear := Bits.vdd;
           inputs.start := Bits.gnd;
           inputs.abort := Bits.gnd;
           inputs.ready := Bits.vdd;
           inputs.arready := Bits.gnd;
           inputs.rvalid := Bits.gnd;
           inputs.rlast := Bits.gnd;
           inputs.rresp := Bits.zero 2;
           inputs.rdata := Bits.zero 128;
           inputs.base := Bits.of_int_trunc ~width:40 0x1000;
           inputs.records := Bits.of_int_trunc ~width:32 32;
           Lws.cycle h;
           let active = ref false in
           let index = ref 0 in
           let requests = ref 0 in
           let seen = ref 0 in
           let final =
             if String.equal fault "early_last"
             then 3
             else if String.equal fault "late_last"
             then 16
             else 15
           in
           List.iter (List.range 0 100) ~f:(fun cycle ->
             inputs.clear := Bits.gnd;
             inputs.start := Bits.of_bool (cycle = 0);
             inputs.abort
             := Bits.of_bool
                  ((String.equal fault "abort" && cycle = 3)
                   || (String.equal fault "abort_data" && cycle = 10));
             inputs.arready := Bits.of_bool (cycle >= 5);
             inputs.rvalid := Bits.of_bool !active;
             inputs.rlast := Bits.of_bool (!index = final);
             inputs.rresp
             := Bits.of_int_trunc
                  ~width:2
                  (if String.equal fault "response" && !index = 3 then 2 else 0);
             inputs.rdata
             := if String.equal fault "keep" && !index = 3
                then Bits.zero 128
                else record !index;
             Lws.cycle h;
             if Bits.to_bool !(o.arvalid) && Bits.to_bool !(inputs.arready)
             then (
               incr requests;
               active := true);
             if Bits.to_bool !(o.valid) then incr seen;
             if Bits.to_bool !(inputs.rvalid) && Bits.to_bool !(o.rready)
             then (
               if !index = final then active := false;
               incr index));
           [%test_result: int] !requests ~expect:1;
           [%test_result: int] !index ~expect:(final + 1);
           [%test_result: int]
             !seen
             ~expect:
               (if String.equal fault "abort"
                then 0
                else if String.equal fault "abort_data"
                then 4
                else if String.equal fault "late_last"
                then 15
                else 3);
           assert (Bits.to_bool !(o.done_));
           assert (not (Bits.to_bool !(o.busy)));
           [%test_result: bool]
             (Bits.to_bool !(o.error))
             ~expect:(not (String.is_prefix fault ~prefix:"abort"));
           [%test_result: bool]
             (Bits.to_bool !(o.aborted))
             ~expect:(String.is_prefix fault ~prefix:"abort");
           inputs.records := Bits.of_int_trunc ~width:32 1;
           inputs.rdata := record 52;
           inputs.rresp := Bits.zero 2;
           inputs.rlast := Bits.vdd;
           let restarted = ref 0 in
           List.iter (List.range 0 6) ~f:(fun cycle ->
             inputs.start := Bits.of_bool (cycle = 0);
             inputs.rvalid := Bits.of_bool (cycle = 2);
             Lws.cycle h;
             if Bits.to_bool !(o.valid) then incr restarted);
           [%test_result: int] !restarted ~expect:1;
           assert (Bits.to_bool !(o.done_));
           assert (not (Bits.to_bool !(o.error)));
           assert (not (Bits.to_bool !(o.aborted)))))
;;

let%test_unit "empty and invalid replay requests never issue AXI reads" =
  List.iter
    [ 0x1000, 0, false; 0x1001, 32, true; 0xffffffff00, 32, true ]
    ~f:(fun (base, count, error) ->
      Harness.run
        ~random_initial_state:`All
        ~create:Ddr_replay.hierarchical
        (fun (h @ local) ~inputs ~outputs ->
           let o = Before_and_after_edge.before_edge outputs in
           inputs.clear := Bits.vdd;
           inputs.start := Bits.gnd;
           inputs.abort := Bits.gnd;
           inputs.ready := Bits.vdd;
           inputs.arready := Bits.vdd;
           inputs.rvalid := Bits.gnd;
           inputs.rlast := Bits.gnd;
           inputs.rresp := Bits.zero 2;
           inputs.rdata := Bits.zero 128;
           inputs.base := Bits.of_int_trunc ~width:40 base;
           inputs.records := Bits.of_int_trunc ~width:32 count;
           Lws.cycle h;
           List.iter (List.range 0 10) ~f:(fun cycle ->
             inputs.clear := Bits.gnd;
             inputs.start := Bits.of_bool (cycle = 0);
             Lws.cycle h;
             assert (not (Bits.to_bool !(o.arvalid))));
           assert (Bits.to_bool !(o.done_));
           assert (not (Bits.to_bool !(o.busy)));
           [%test_result: bool] (Bits.to_bool !(o.error)) ~expect:error))
;;
