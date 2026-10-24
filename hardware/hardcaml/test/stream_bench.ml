open! Core
open! Hardcaml
open Hardcaml_lws

module Source = struct
  type t =
    { data : Bits.t ref
    ; valid : Bits.t ref
    ; ready : Bits.t ref
    ; duty : int
    ; mutable pending : Bits.t list
    ; mutable sent : Bits.t list
    }

  let create ~data ~valid ~ready ~duty beats =
    { data; valid; ready; duty; pending = beats; sent = [] }
  ;;
end

module Sink = struct
  type t =
    { data : Bits.t ref
    ; valid : Bits.t ref
    ; ready : Bits.t ref
    ; duty : int
    ; wanted : int
    ; mutable received : Bits.t list
    }

  let create ~data ~valid ~ready ~duty ~wanted =
    { data; valid; ready; duty; wanted; received = [] }
  ;;

  let received t = List.rev t.received
end

let chance rng duty = Random.State.int rng 100 < duty

let random_bits rng ~width =
  Bits.concat_lsb
    (List.init
       ((width + 29) / 30)
       ~f:(fun _ -> Bits.of_unsigned_int ~width:30 (Random.State.bits rng)))
  |> Bits.sel_bottom ~width
;;

let run
  (h @ local)
  ~rng
  ?(timeout = 100_000)
  ?(each_cycle = fun () -> ())
  ~(sources : Source.t list)
  ~(sinks : Sink.t list)
  ()
  =
  let cycles = ref 0 in
  let finished () =
    List.for_all sinks ~f:(fun sink -> List.length sink.received >= sink.wanted)
  in
  while not (finished ()) do
    if !cycles >= timeout
    then
      raise_s
        [%message
          "stream bench stalled"
            (timeout : int)
            ~received:
              (List.map sinks ~f:(fun sink -> List.length sink.received) : int list)];
    List.iter sources ~f:(fun source ->
      let width = Bits.width !(source.data) in
      match source.pending with
      | beat :: _ when chance rng source.duty ->
        source.data := beat;
        source.valid := Bits.vdd
      | _ ->
        (* Garbage on an idle bus, so a unit that samples it without valid shows up. *)
        source.data := random_bits rng ~width;
        source.valid := Bits.gnd);
    List.iter sinks ~f:(fun sink -> sink.ready := Bits.of_bool (chance rng sink.duty));
    Lws.cycle h;
    incr cycles;
    List.iter sources ~f:(fun source ->
      if Bits.to_bool !(source.valid) && Bits.to_bool !(source.ready)
      then (
        source.sent <- List.hd_exn source.pending :: source.sent;
        source.pending <- List.tl_exn source.pending));
    List.iter sinks ~f:(fun sink ->
      if Bits.to_bool !(sink.valid) && Bits.to_bool !(sink.ready)
      then sink.received <- !(sink.data) :: sink.received);
    each_cycle ()
  done;
  List.iter sources ~f:(fun source -> source.valid := Bits.gnd);
  List.iter sinks ~f:(fun sink -> sink.ready := Bits.gnd);
  !cycles
;;

let clear (h @ local) (clear : Bits.t ref) =
  clear := Bits.vdd;
  Lws.cycle ~n:2 h;
  clear := Bits.gnd
;;

let beats ~rng ~width count = List.init count ~f:(fun _ -> random_bits rng ~width)

let signed_lanes ~lane_bits beat =
  List.map (Bits.split_lsb ~part_width:lane_bits beat) ~f:Bits.to_signed_int
;;

let of_signed_lanes ~lane_bits values =
  Bits.concat_lsb (List.map values ~f:(Bits.of_signed_int ~width:lane_bits))
;;
