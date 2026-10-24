open! Core
open! Hardcaml
open! Signal

module type Config = sig
  val design : Design.t
end

module Offset = struct
  let id = 0x0000
  let latency = 0x0004
  let feature_bits = 0x0008
  let prediction_bits = 0x000c
  let control = 0x0010
  let status = 0x0014
  let cycles = 0x0018
  let features = 0x1000
  let prediction = 0x2000
  let weights = 0x8000
end

let id = 0x4f544631
let window_words = 1024

module Make (Config : Config) = struct
  open Config
  module Accelerator = Fixed_top.Make (Config)

  let feature_bits = Design.width design design.input
  let prediction_bits = Design.width design design.output
  let feature_words = (feature_bits + 31) / 32
  let prediction_words = (prediction_bits + 31) / 32

  let () =
    if feature_words > window_words || prediction_words > window_words
    then
      raise_s
        [%message
          "the features or the prediction do not fit a 4 KiB register window"
            (feature_bits : int)
            (prediction_bits : int)]
  ;;

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; s_axi_awaddr : 'a [@bits 16]
      ; s_axi_awvalid : 'a
      ; s_axi_wdata : 'a [@bits 32]
      ; s_axi_wvalid : 'a
      ; s_axi_bready : 'a
      ; s_axi_araddr : 'a [@bits 16]
      ; s_axi_arvalid : 'a
      ; s_axi_rready : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { s_axi_awready : 'a
      ; s_axi_wready : 'a
      ; s_axi_bresp : 'a [@bits 2]
      ; s_axi_bvalid : 'a
      ; s_axi_arready : 'a
      ; s_axi_rdata : 'a [@bits 32]
      ; s_axi_rresp : 'a [@bits 2]
      ; s_axi_rvalid : 'a
      }
    [@@deriving hardcaml]
  end

  let word offset = offset / 4
  let word_constant index = of_unsigned_int ~width:14 index
  let constant value = of_unsigned_int ~width:32 value

  let create (scope : Scope.t) (i : Signal.t I.t) =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let%hw bvalid = wire 1 in
    let%hw write = i.s_axi_awvalid &: i.s_axi_wvalid &: ~:bvalid in
    bvalid <-- reg spec (write |: (bvalid &: ~:(i.s_axi_bready)));
    let%hw write_word = i.s_axi_awaddr.:[15, 2] in
    let writes index = write &: (write_word ==: word_constant index) in
    let feature_registers =
      List.init feature_words ~f:(fun k ->
        reg spec ~enable:(writes (word Offset.features + k)) i.s_axi_wdata)
    in
    let%hw features = sel_bottom (concat_lsb feature_registers) ~width:feature_bits in
    let%hw features_valid =
      reg spec (writes (word Offset.control) &: i.s_axi_wdata.:(0))
    in
    let accelerator =
      Accelerator.hierarchical
        scope
        { clock = i.clock; clear = i.clear; features; features_valid }
    in
    let%hw prediction =
      reg spec ~enable:accelerator.prediction_valid accelerator.prediction
    in
    let%hw finished =
      reg_fb spec ~width:1 ~f:(fun finished ->
        accelerator.prediction_valid |: (finished &: ~:features_valid))
    in
    let%hw elapsed =
      reg_fb spec ~width:32 ~f:(fun count ->
        mux2 features_valid (constant 1) (count +:. 1))
    in
    let%hw cycles = reg spec ~enable:accelerator.prediction_valid elapsed in
    let%hw rvalid = wire 1 in
    let%hw read = i.s_axi_arvalid &: ~:rvalid in
    rvalid <-- reg spec (read |: (rvalid &: ~:(i.s_axi_rready)));
    let%hw read_word = i.s_axi_araddr.:[15, 2] in
    let window offset words =
      List.mapi words ~f:(fun k value -> word_constant (word offset + k), value)
    in
    let prediction_registers =
      split_lsb ~part_width:32 (uresize prediction ~width:(32 * prediction_words))
    in
    let%hw read_data =
      cases
        ~default:(zero 32)
        read_word
        ([ word_constant (word Offset.id), constant id
         ; word_constant (word Offset.latency), constant Accelerator.latency
         ; word_constant (word Offset.feature_bits), constant feature_bits
         ; word_constant (word Offset.prediction_bits), constant prediction_bits
         ; word_constant (word Offset.status), uresize finished ~width:32
         ; word_constant (word Offset.cycles), cycles
         ]
         @ window Offset.features feature_registers
         @ window Offset.prediction prediction_registers)
    in
    let%hw rdata = reg spec ~enable:read read_data in
    { O.s_axi_awready = write
    ; s_axi_wready = write
    ; s_axi_bresp = zero 2
    ; s_axi_bvalid = bvalid
    ; s_axi_arready = ~:rvalid
    ; s_axi_rdata = rdata
    ; s_axi_rresp = zero 2
    ; s_axi_rvalid = rvalid
    }
  ;;

  let hierarchical ?instance scope i =
    let module H = Hierarchy.In_scope (I) (O) in
    H.hierarchical ?instance ~scope ~name:"kria_regs" create i
  ;;
end
