open! Core
open! Hardcaml

(** AXI4-Lite controls for DDR trading; address and data channels are buffered
    independently. *)
module Make (Config : Tick_to_trade.Config) : sig
  module Pipeline : module type of Ddr_trading.Make (Config)

  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; s_axi_awaddr : 'a
      ; s_axi_awvalid : 'a
      ; s_axi_wdata : 'a
      ; s_axi_wstrb : 'a
      ; s_axi_wvalid : 'a
      ; s_axi_bready : 'a
      ; s_axi_araddr : 'a
      ; s_axi_arvalid : 'a
      ; s_axi_rready : 'a
      ; arready : 'a
      ; rdata : 'a
      ; rvalid : 'a
      ; rresp : 'a
      ; rlast : 'a
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { s_axi_awready : 'a
      ; s_axi_wready : 'a
      ; s_axi_bvalid : 'a
      ; s_axi_bresp : 'a
      ; s_axi_arready : 'a
      ; s_axi_rvalid : 'a
      ; s_axi_rdata : 'a
      ; s_axi_rresp : 'a
      ; pipeline : 'a Pipeline.O.t
      }
    [@@deriving hardcaml]
  end

  val hierarchical : ?instance:string -> Scope.t -> Signal.t I.t -> Signal.t O.t
end

val verilog : Tick_to_trade.Params.t -> string
