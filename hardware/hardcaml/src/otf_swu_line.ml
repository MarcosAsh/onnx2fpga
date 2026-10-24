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
  val par : int
  val data_bits : int
  val pad_value : int
end

module Make (Config : Config) = struct
  open Config

  let cg_count = ifm_ch / simd
  let kwg_count = kw / par
  let span = ((kh - 1) * dil_h * ifm_w) + ((kw - 1) * dil_w) + 1
  let tail = Int.max 0 (((ofm_w - 1) * stride_w) - pad_l)
  let need = span + tail + 1
  let logd = if need < 2 then 1 else Int.ceil_log2 need
  let depth = 1 lsl logd
  let frame = ifm_h * ifm_w
  let word_bits = simd * data_bits
  let ad_bits = Int.ceil_log2 (depth * cg_count)

  let () =
    if simd < 1 || ifm_ch % simd <> 0
    then
      raise_s
        [%message "BUG: otf_swu_line simd must divide ifm_ch" (ifm_ch : int) (simd : int)];
    if par > 1 && cg_count <> 1
    then raise_s [%message "BUG: otf_swu_line par > 1 requires simd = ifm_ch"];
    if par < 1 || kw % par <> 0
    then raise_s [%message "BUG: otf_swu_line par must divide kw" (kw : int) (par : int)]
  ;;

  module In = Stream.Source (struct
      let width = word_bits
    end)

  module Out = Stream.Source (struct
      let width = par * word_bits
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

  (* Index into the circular buffer for pixel [pixel], channel group [cg]. *)
  let buffer_address pixel cg =
    uresize (sel_bottom pixel ~width:logd *: Sv_int.int32 cg_count) ~width:ad_bits
    +: uresize cg ~width:ad_bits
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let clear = i.clocking.clear in
    let%hw ovalid = wire 1 in
    let%hw emit = wire 1 in
    let%hw can_write = wire 1 in
    let%hw write = i.s.valid &: can_write in
    (* Writer: pixels fully written, and the channel group within the next one. *)
    let%hw wr_cg = Sv_int.wrapping_counter spec ~advance:write ~count:cg_count in
    let%hw pixel_written = write &: Sv_int.is_last wr_cg ~count:cg_count in
    let%hw wr_pixel = reg_fb spec ~enable:pixel_written ~width:32 ~f:(fun d -> d +:. 1) in
    (* Reader: (row, col, ky, kx group, channel group), innermost last. *)
    let%hw rd_cg = Sv_int.wrapping_counter spec ~advance:emit ~count:cg_count in
    let%hw kxg_step = emit &: Sv_int.is_last rd_cg ~count:cg_count in
    let%hw kxg = Sv_int.wrapping_counter spec ~advance:kxg_step ~count:kwg_count in
    let%hw ky_step = kxg_step &: Sv_int.is_last kxg ~count:kwg_count in
    let%hw ky = Sv_int.wrapping_counter spec ~advance:ky_step ~count:kh in
    let%hw ow_step = ky_step &: Sv_int.is_last ky ~count:kh in
    let%hw ow = Sv_int.wrapping_counter spec ~advance:ow_step ~count:ofm_w in
    let%hw oh_step = ow_step &: Sv_int.is_last ow ~count:ofm_w in
    let%hw oh = Sv_int.wrapping_counter spec ~advance:oh_step ~count:ofm_h in
    let%hw frame_done = oh_step &: Sv_int.is_last oh ~count:ofm_h in
    let%hw frame_base =
      reg_fb spec ~enable:frame_done ~width:32 ~f:(fun d -> d +: Sv_int.int32 frame)
    in
    let oh32 = Sv_int.of_unsigned oh in
    let ow32 = Sv_int.of_unsigned ow in
    let%hw row =
      Sv_int.mul oh32 stride_h
      +: Sv_int.mul (Sv_int.of_unsigned ky) dil_h
      -: Sv_int.int32 pad_t
    in
    let%hw col_base =
      Sv_int.mul ow32 stride_w
      +: Sv_int.mul (Sv_int.mul (Sv_int.of_unsigned kxg) par) dil_w
      -: Sv_int.int32 pad_l
    in
    let%hw row_ok = ~:(msb row) &: (row <+ Sv_int.int32 ifm_h) in
    let%hw row_offset = frame_base +: Sv_int.mul row ifm_w in
    let rd_addr = Sv_int.of_unsigned rd_cg in
    let wr_addr = buffer_address wr_pixel (uresize wr_cg ~width:32) in
    let pad = repeat (Sv_int.cast ~width:data_bits pad_value) ~count:simd in
    (* Copy j serves the column j * dil_w to the right of the beat's first one. *)
    let copies =
      List.init par ~f:(fun j ->
        let col = col_base +: Sv_int.int32 (j * dil_w) in
        let in_view = row_ok &: ~:(msb col) &: (col <+ Sv_int.int32 ifm_w) in
        let needed = row_offset +: col in
        let buffer =
          multiport_memory
            ~name:("buffer_" ^ Int.to_string j)
            (depth * cg_count)
            ~write_ports:
              [| { write_clock = i.clocking.clock
                 ; write_address = wr_addr
                 ; write_enable = write &: ~:clear
                 ; write_data = i.s.data
                 }
              |]
            ~read_addresses:[| buffer_address needed rd_addr |]
        in
        let fetched = mux2 in_view buffer.(0) pad in
        let not_yet_written = ~:(msb (needed -: wr_pixel)) in
        fetched, in_view &: not_yet_written)
    in
    let%hw read_ok = ~:(reduce ~f:( |: ) (List.map copies ~f:snd)) in
    (* Oldest pixel any remaining read can reach. *)
    let%hw oldest_here =
      frame_base
      +: Sv_int.mul
           (Sv_int.clamp_at_zero (Sv_int.mul oh32 stride_h -: Sv_int.int32 pad_t))
           ifm_w
      +: Sv_int.clamp_at_zero (Sv_int.mul ow32 stride_w -: Sv_int.int32 pad_l)
    in
    let%hw oldest_next =
      frame_base
      +: Sv_int.mul
           (Sv_int.clamp_at_zero
              (Sv_int.mul (oh32 +: Sv_int.int32 1) stride_h -: Sv_int.int32 pad_t))
           ifm_w
      +: Sv_int.int32 (if pad_l > 0 then 0 else -pad_l)
    in
    let%hw last_out_row = Sv_int.is_last oh ~count:ofm_h in
    let%hw oldest =
      mux2 (~:last_out_row &: msb (oldest_next -: oldest_here)) oldest_next oldest_here
    in
    can_write <-- (wr_pixel -: oldest <+ Sv_int.int32 depth);
    let%hw out_free = ~:ovalid |: i.m.ready in
    emit <-- (read_ok &: out_free);
    let%hw odata = reg spec ~enable:emit (concat_lsb (List.map copies ~f:fst)) in
    ovalid <-- reg spec (mux2 emit vdd (ovalid &: ~:(i.m.ready)));
    { O.s = { ready = can_write }; m = { data = odata; valid = ovalid } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_swu_line" create i
  ;;
end
