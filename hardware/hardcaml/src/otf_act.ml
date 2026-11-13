open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val channels : int
  val pe : int
  val in_bits : int
  val out_bits : int
  val mult_bits : int
  val shift : int
  val out_min : int
  val out_max : int
  val in_zp : int
  val out_zp : int
  val mult : Bits.t list
end

module Make (Config : Config) = struct
  open Config

  let nf_count = channels / pe

  let () =
    if pe < 1 || channels % pe <> 0
    then
      raise_s
        [%message "BUG: otf_act pe must divide channels" (channels : int) (pe : int)];
    if List.length mult <> pe * nf_count
       || List.exists mult ~f:(fun word -> Bits.width word <> mult_bits)
    then raise_s [%message "BUG: otf_act needs one mult_bits word per channel"]
  ;;

  module In = Stream.Source (struct
      let width = pe * in_bits
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

  module Requant = Otf_requant.Make (struct
      let acc_bits = in_bits + 2 + mult_bits
      let mult_bits = 2
      let shift = shift
      let out_bits = out_bits
      let out_min = out_min
      let out_max = out_max
      let out_zp = out_zp
    end)

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw valid = wire 1 in
    let%hw go = ~:valid |: i.m.ready in
    let%hw product_valid = reg spec ~enable:go i.s.valid in
    valid <-- reg spec ~enable:go product_valid;
    let%hw fire = i.s.valid &: go in
    let%hw nf = Sv_int.wrapping_counter spec ~advance:fire ~count:nf_count in
    let lanes = split_lsb ~part_width:in_bits i.s.data in
    let tables = List.chunks_of mult ~length:nf_count in
    let data =
      List.map2_exn lanes tables ~f:(fun lane table ->
        let acc =
          sresize lane ~width:(in_bits + 2) -: Sv_int.cast ~width:(in_bits + 2) in_zp
        in
        let mult = Sv_int.table ~address:nf table in
        let product =
          reg (Clocking.to_spec_no_clear i.clocking) ~enable:go (acc *+ mult)
        in
        (Requant.hierarchical scope { acc = product; mult = of_signed_int ~width:2 1 })
          .out)
    in
    let%hw out_data =
      reg (Clocking.to_spec_no_clear i.clocking) ~enable:go (concat_lsb data)
    in
    { O.s = { ready = go }; m = { data = out_data; valid } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_act" create i
  ;;
end
