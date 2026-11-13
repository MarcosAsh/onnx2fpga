open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val channels : int
  val pe : int
  val window : int
  val data_bits : int
  val out_bits : int
end

module Make (Config : Config) = struct
  open Config

  let () =
    if pe < 1
       || channels < 1
       || data_bits < 1
       || channels % pe <> 0
       || window < 1
       || out_bits < data_bits + Int.ceil_log2 window
    then
      raise_s
        [%message
          "BUG: invalid sum-pool configuration"
            (channels : int)
            (pe : int)
            (window : int)
            (data_bits : int)
            (out_bits : int)]
  ;;

  let nf_count = channels / pe

  module In = Stream.Source (struct
      let width = pe * data_bits
    end)

  module Out = Stream.Source (struct
      let width = pe * out_bits
    end)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; s : 'a In.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Out.t
      }
    [@@deriving hardcaml]
  end

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw nf = wire (address_bits_for nf_count) in
    let%hw w = wire (address_bits_for window) in
    let%hw first_pass = w ==:. 0 in
    let%hw last_pass = Sv_int.is_last w ~count:window in
    let%hw s_ready = mux2 last_pass i.m.ready vdd in
    let%hw fire = i.s.valid &: s_ready in
    let%hw merged = wire (pe * out_bits) in
    let running =
      multiport_memory
        ~name:"running_sum"
        nf_count
        ~write_ports:
          [| { write_clock = i.clocking.clock
             ; write_address = nf
             ; write_enable = fire &: ~:last_pass &: ~:(i.clocking.clear)
             ; write_data = merged
             }
          |]
        ~read_addresses:[| nf |]
    in
    let%hw running = running.(0) in
    merged
    <-- concat_lsb
          (List.map2_exn
             (split_lsb ~part_width:data_bits i.s.data)
             (split_lsb ~part_width:out_bits running)
             ~f:(fun incoming held ->
               let held = mux2 first_pass (zero out_bits) held in
               sresize incoming ~width:out_bits +: held));
    let%hw window_done = fire &: Sv_int.is_last nf ~count:nf_count in
    nf <-- Sv_int.wrapping_counter spec ~advance:fire ~count:nf_count;
    w <-- Sv_int.wrapping_counter spec ~advance:window_done ~count:window;
    { O.s = { ready = s_ready }; m = { data = merged; valid = i.s.valid &: last_pass } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_sum_pool" create i
  ;;
end
