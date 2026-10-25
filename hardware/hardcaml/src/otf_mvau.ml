open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val mw : int
  val mh : int
  val simd : int
  val pe : int
  val act_bits : int
  val wgt_bits : int
  val acc_bits : int
  val out_bits : int
  val mult_bits : int
  val shift : int
  val out_min : int
  val out_max : int
  val out_zp : int
  val block_ram : bool
  val weights : Bits.t list
  val bias : Bits.t list
  val mult : Bits.t list
end

module Make (Config : Config) = struct
  open Config

  let sf_count = mw / simd
  let nf_count = mh / pe

  let () =
    let check name words ~count ~width =
      if List.length words <> count
         || List.exists words ~f:(fun word -> Bits.width word <> width)
      then
        raise_s
          [%message
            "BUG: otf_mvau memory has the wrong shape" name (count : int) (width : int)]
    in
    if simd < 1 || pe < 1 || mw % simd <> 0 || mh % pe <> 0
    then
      raise_s
        [%message
          "BUG: otf_mvau folding must divide the matrix"
            (mw : int)
            (mh : int)
            (simd : int)
            (pe : int)];
    if block_ram
    then check "weights" weights ~count:(sf_count * nf_count) ~width:(pe * simd * wgt_bits)
    else check "weights" weights ~count:(pe * sf_count * nf_count) ~width:(simd * wgt_bits);
    check "bias" bias ~count:(pe * nf_count) ~width:acc_bits;
    check "mult" mult ~count:(pe * nf_count) ~width:mult_bits
  ;;

  module In = Stream.Source (struct
      let width = simd * act_bits
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
      let acc_bits = acc_bits
      let mult_bits = mult_bits
      let shift = shift
      let out_bits = out_bits
      let out_min = out_min
      let out_max = out_max
      let out_zp = out_zp
    end)

  let w_addr_bits = address_bits_for (sf_count * nf_count)

  (* The step being multiplied: issued this cycle, or with block RAM issued last cycle. *)
  type mac =
    { valid : Signal.t
    ; first : Signal.t
    ; last : Signal.t
    ; nf : Signal.t
    ; act : Signal.t
    ; weights : Signal.t list
    }

  (* Lower case: yosys ignores Hardcaml's RAM_STYLE, and Vivado takes either. *)
  let ram_style_block =
    Rtl_attribute.create "ram_style" ~applies_to:[ Memories ] ~value:(String "block")
  ;;

  (* The write port never writes, so synthesis sees a ROM. *)
  let weight_rom ~clock ~address =
    (multiport_memory
       ~name:"weight_mem"
       ~attributes:[ ram_style_block ]
       ~initialize_to:(Array.of_list weights)
       (sf_count * nf_count)
       ~write_ports:
         [| { write_clock = clock
            ; write_address = zero w_addr_bits
            ; write_enable = gnd
            ; write_data = zero (pe * simd * wgt_bits)
            }
         |]
       ~read_addresses:[| address |]).(0)
  ;;

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Clocking.to_spec i.clocking in
    let spec_no_clear = Clocking.to_spec_no_clear i.clocking in
    let clear = i.clocking.clear in
    let%hw sf = wire (address_bits_for sf_count) in
    let%hw nf = wire (address_bits_for nf_count) in
    let%hw out_valid = wire 1 in
    let%hw held = wire 1 in
    let%hw last_sf = Sv_int.is_last sf ~count:sf_count in
    let%hw out_free = ~:out_valid |: i.m.ready in
    let%hw go = ~:held |: out_free in
    let%hw need_input = nf ==:. 0 in
    let%hw step = go &: (~:need_input |: i.s.valid) in
    sf <-- Sv_int.wrapping_counter spec ~advance:step ~count:sf_count;
    nf <-- Sv_int.wrapping_counter spec ~advance:(step &: last_sf) ~count:nf_count;
    (* The input vector is kept so later neuron folds replay it rather than fetch it. *)
    let act_buf =
      multiport_memory
        ~name:"act_buf"
        sf_count
        ~write_ports:
          [| { write_clock = i.clocking.clock
             ; write_address = sf
             ; write_enable = step &: need_input &: ~:clear
             ; write_data = i.s.data
             }
          |]
        ~read_addresses:[| sf |]
    in
    let%hw replayed = act_buf.(0) in
    let%hw act_word = mux2 need_input i.s.data replayed in
    let%hw waddr =
      uresize
        (nf *: of_unsigned_int ~width:(num_bits_to_represent sf_count) sf_count)
        ~width:w_addr_bits
      +: uresize sf ~width:w_addr_bits
    in
    let%hw first = sf ==:. 0 in
    let mac =
      if block_ram
      then (
        let stage d = reg spec_no_clear ~enable:go d in
        let%hw issued = reg spec ~enable:go step in
        let%hw mac_last = stage last_sf in
        let%hw mac_weights = stage (weight_rom ~clock:i.clocking.clock ~address:waddr) in
        held <-- (issued &: mac_last);
        { valid = go &: issued
        ; first = stage first
        ; last = mac_last
        ; nf = stage nf
        ; act = stage act_word
        ; weights = split_lsb ~part_width:(simd * wgt_bits) mac_weights
        })
      else (
        held <-- last_sf;
        { valid = step
        ; first
        ; last = last_sf
        ; nf
        ; act = act_word
        ; weights =
            List.chunks_of weights ~length:(sf_count * nf_count)
            |> List.map ~f:(Sv_int.table ~address:waddr)
        })
    in
    let acts = split_lsb ~part_width:act_bits mac.act in
    let lane_bias = List.chunks_of bias ~length:nf_count in
    let lane_mult = List.chunks_of mult ~length:nf_count in
    let lanes =
      List.map3_exn mac.weights lane_bias lane_mult ~f:(fun weight_word bias mult ->
        let dot =
          List.map2_exn acts (split_lsb ~part_width:wgt_bits weight_word) ~f:(fun a w ->
            sresize (a *+ w) ~width:acc_bits)
          |> tree ~arity:2 ~f:(reduce ~f:( +: ))
        in
        let acc = wire acc_bits in
        let seed = mux2 mac.first (Sv_int.table ~address:mac.nf bias) acc in
        let acc_next = seed +: dot in
        acc <-- reg spec_no_clear ~enable:mac.valid acc_next;
        let mult = Sv_int.table ~address:mac.nf mult in
        (Requant.hierarchical scope { acc = acc_next; mult }).out)
    in
    let%hw finish = mac.valid &: mac.last in
    out_valid <-- reg spec (mux2 finish vdd (out_valid &: ~:(i.m.ready)));
    let%hw out_data = reg spec_no_clear ~enable:(finish &: ~:clear) (concat_lsb lanes) in
    { O.s = { ready = need_input &: go }; m = { data = out_data; valid = out_valid } }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"otf_mvau" create i
  ;;
end
