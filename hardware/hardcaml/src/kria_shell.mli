(** The Kria programmable logic: a 322.265625 MHz MMCM, a reset and {!Kria_regs}. *)

open! Core
open! Hardcaml

module type Config = sig
  val design : Design.t
end

module Make (_ : Config) : sig
  module I : sig
    type 'a t =
      { pl_clk0 : 'a
      ; pl_resetn0 : 'a
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
      { aclk : 'a
      ; aresetn : 'a
      ; s_axi_awready : 'a
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

  (** [pl_clk0] must be 100 MHz; [aclk] also clocks the PS's AXI port. *)
  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

(** One Verilog file with [kria_shell] and its MMCM instance [otf_mmcm]. *)
val verilog : Design.t -> string
