open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val ifm_h : int
  val ifm_w : int
  val ifm_ch : int
  val ofm_h : int
  val ofm_w : int
  val kh : int
  val kw : int
  val stride_h : int
  val stride_w : int
  val pad_t : int
  val pad_l : int
  val dil_h : int
  val dil_w : int
  val simd : int
  val data_bits : int
  val pad_value : int
end

module Make (Config : Config) = struct
  open Config

  let cg_count = ifm_ch / simd
  let total_in = ifm_h * ifm_w * cg_count

  let () =
    if simd < 1 || ifm_ch % simd <> 0
    then
      raise_s
        [%message "BUG: otf_swu simd must divide ifm_ch" (ifm_ch : int) (simd : int)]
  ;;

  module Beat = Stream.Source (struct
      let width = simd * data_bits
    end)

  module I = struct
    type 'a t =
      { clocking : 'a Clocking.t
      ; s : 'a Beat.t
      ; m : 'a Stream.Dest.t
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s : 'a Stream.Dest.t
      ; m : 'a Beat.t
      }
    [@@deriving hardcaml]
  end

  let addr_bits = Sv_int.counter_width total_in

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let%hw filling = wire 1 in
    let%hw ovalid = wire 1 in
    let%hw out_free = ~:ovalid |: i.m.ready in
    let%hw capture = filling &: i.s.valid in
    let%hw replay = ~:filling &: out_free in
    (* Windows leave in (row, col, ky, kx, channel group) order, innermost last. *)
    let%hw cg = Sv_int.wrapping_counter spec ~advance:replay ~count:cg_count in
    let%hw kx_step = replay &: Sv_int.is_last cg ~count:cg_count in
    let%hw kx = Sv_int.wrapping_counter spec ~advance:kx_step ~count:kw in
    let%hw ky_step = kx_step &: Sv_int.is_last kx ~count:kw in
    let%hw ky = Sv_int.wrapping_counter spec ~advance:ky_step ~count:kh in
    let%hw ow_step = ky_step &: Sv_int.is_last ky ~count:kh in
    let%hw ow = Sv_int.wrapping_counter spec ~advance:ow_step ~count:ofm_w in
    let%hw oh_step = ow_step &: Sv_int.is_last ow ~count:ofm_w in
    let%hw oh = Sv_int.wrapping_counter spec ~advance:oh_step ~count:ofm_h in
    let%hw replayed = oh_step &: Sv_int.is_last oh ~count:ofm_h in
    let%hw wcount =
      reg_fb
        spec
        ~enable:capture
        ~width:(num_bits_to_represent total_in)
        ~f:(mod_counter ~max:(total_in - 1))
    in
    let%hw captured = capture &: Sv_int.is_last wcount ~count:total_in in
    filling <-- reg spec ~clear_to:vdd (mux2 filling ~:captured replayed);
    let%hw row =
      Sv_int.mul (Sv_int.of_unsigned oh) stride_h
      +: Sv_int.mul (Sv_int.of_unsigned ky) dil_h
      -: Sv_int.int32 pad_t
    in
    let%hw col =
      Sv_int.mul (Sv_int.of_unsigned ow) stride_w
      +: Sv_int.mul (Sv_int.of_unsigned kx) dil_w
      -: Sv_int.int32 pad_l
    in
    let%hw in_bounds =
      ~:(Sv_int.is_negative row)
      &: (row <+ Sv_int.int32 ifm_h)
      &: ~:(Sv_int.is_negative col)
      &: (col <+ Sv_int.int32 ifm_w)
    in
    let%hw raddr =
      Sv_int.mul (Sv_int.mul row ifm_w +: col) cg_count +: Sv_int.of_unsigned cg
    in
    let frame =
      multiport_memory
        ~name:"frame"
        total_in
        ~write_ports:
          [| { write_clock = i.clocking.clock
             ; write_address = sel_bottom wcount ~width:addr_bits
             ; write_enable = capture &: ~:(i.clocking.clear)
             ; write_data = i.s.data
             }
          |]
        ~read_addresses:[| sel_bottom raddr ~width:addr_bits |]
    in
    let%hw fetched = frame.(0) in
    let pad = repeat (Sv_int.cast ~width:data_bits pad_value) ~count:simd in
    let%hw odata = reg spec ~enable:replay (mux2 in_bounds fetched pad) in
    ovalid <-- reg spec (mux2 replay vdd (ovalid &: ~:(i.m.ready)));
    { O.s = { ready = filling }; m = { data = odata; valid = ovalid } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_swu" create i
  ;;
end
