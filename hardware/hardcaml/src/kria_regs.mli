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

  (** From here to the end of the window is reserved for the weight reload of spec 003. *)
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
