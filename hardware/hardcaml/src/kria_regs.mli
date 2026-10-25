(** A fixed latency design behind an AXI4-Lite register window, for the Kria ARM cores. *)

open! Core
open! Hardcaml

module type Config = sig
  (** A [Fixed] design, as {!Fixed_top} takes it. *)
  val design : Design.t
end

(** Byte offsets in the 64 KiB window; onnx2fpga writes the same map into manifest.json. *)
module Offset : sig
  (** Reads {!id}. *)
  val id : int

  (** Reads the design's latency in cycles. *)
  val latency : int

  (** Reads the width of the features bus. *)
  val feature_bits : int

  (** Reads the width of the prediction bus. *)
  val prediction_bits : int

  (** Writing a 1 in bit 0 raises [features_valid] for one cycle. *)
  val control : int

  (** Bit 0 is set when a prediction arrives and cleared by the next start. *)
  val status : int

  (** Cycles from the last start to its prediction, which must equal the latency. *)
  val cycles : int

  (** The features, 32 bits per word, element 0 in the low bits of the first word. *)
  val features : int

  (** The last prediction, laid out like the features; read only. *)
  val prediction : int

  (** Reads the words in one weight bank, 0 when the weights are frozen. *)
  val weights_words : int

  (** Writing a 1 in bit 0 makes the idle bank active and counts a version. *)
  val weights_commit : int

  (** Reads the active bank, 0 or 1. *)
  val weights_bank : int

  (** Reads the commits since configuration. *)
  val weights_version : int

  (** Reads zlib's CRC-32 of the active bank's writes since the last one to word 0. *)
  val weights_crc : int

  (** Reads the same CRC for the idle bank, to check before a commit. *)
  val weights_idle_crc : int

  (** To the end of the window, the idle bank, one value per word; write only. *)
  val weights : int
end

(** "OTF1" in ASCII, so software can tell the shell is there. *)
val id : int

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; s_axi_awaddr : 'a
      ; s_axi_awvalid : 'a
      ; s_axi_wdata : 'a
      ; s_axi_wvalid : 'a
      ; s_axi_bready : 'a
      ; s_axi_araddr : 'a
      ; s_axi_arvalid : 'a
      ; s_axi_rready : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { s_axi_awready : 'a
      ; s_axi_wready : 'a
      ; s_axi_bresp : 'a
      ; s_axi_bvalid : 'a
      ; s_axi_arready : 'a
      ; s_axi_rdata : 'a
      ; s_axi_rresp : 'a
      ; s_axi_rvalid : 'a
      }
    [@@deriving hardcaml]
  end

  (** One transfer at a time each way, always OKAY; write strobes are ignored. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** One Verilog file with [kria_regs] as the top, for simulating the register file alone. *)
val verilog : Design.t -> string
