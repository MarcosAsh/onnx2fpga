open! Core
open! Hardcaml
open Hardcaml_lws

let check
  (h @ local)
  ~(x : Bits.t ref With_valid.t)
  ~(y : Bits.t ref With_valid.t)
  ~latency
  ~expected
  stimuli
  =
  let in_flight = Queue.create () in
  let checked = ref 0 in
  let idle = List.init latency ~f:(fun _ -> false, Bits.zero (Bits.width !(x.value))) in
  List.iter (stimuli @ idle) ~f:(fun (valid, value) ->
    x.value := value;
    x.valid := Bits.of_bool valid;
    Lws.cycle h;
    Queue.enqueue in_flight (valid, value);
    let got_valid = Bits.to_bool !(y.valid) in
    if Queue.length in_flight <= latency
    then (if got_valid then raise_s [%message "valid before any input"])
    else (
      let sent_valid, sent = Queue.dequeue_exn in_flight in
      if Bool.( <> ) got_valid sent_valid
      then raise_s [%message "valid is not the input delayed"];
      if sent_valid
      then (
        [%test_result: Bits.Hex.t] !(y.value) ~expect:(expected sent);
        incr checked)));
  !checked
;;
