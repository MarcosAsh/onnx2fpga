open! Core
open! Hardcaml
open! Signal

module Make (Config : Tick_to_trade.Config) = struct
  module Regs = Ddr_regs.Make (Config)

  module I = struct
    type 'a t =
      { pl_clk0 : 'a
      ; pl_resetn0 : 'a
      ; s_axi_awaddr : 'a [@bits 16]
      ; s_axi_awvalid : 'a
      ; s_axi_wdata : 'a [@bits 32]
      ; s_axi_wstrb : 'a [@bits 4]
      ; s_axi_wvalid : 'a
      ; s_axi_bready : 'a
      ; s_axi_araddr : 'a [@bits 16]
      ; s_axi_arvalid : 'a
      ; s_axi_rready : 'a
      ; arready : 'a
      ; rdata : 'a [@bits 128]
      ; rvalid : 'a
      ; rresp : 'a [@bits 2]
      ; rlast : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { aclk : 'a
      ; aresetn : 'a
      ; s_axi_awready : 'a
      ; s_axi_wready : 'a
      ; s_axi_bvalid : 'a
      ; s_axi_bresp : 'a [@bits 2]
      ; s_axi_arready : 'a
      ; s_axi_rvalid : 'a
      ; s_axi_rdata : 'a [@bits 32]
      ; s_axi_rresp : 'a [@bits 2]
      ; araddr : 'a [@bits 40]
      ; arvalid : 'a
      ; arlen : 'a [@bits 8]
      ; arsize : 'a [@bits 3]
      ; arburst : 'a [@bits 2]
      ; rready : 'a
      }
    [@@deriving hardcaml]
  end

  let create scope (i : Signal.t I.t) =
    let clock, clear =
      Kria_clock.create scope ~pl_clk0:i.pl_clk0 ~pl_resetn0:i.pl_resetn0
    in
    let regs =
      Regs.hierarchical
        scope
        { clock
        ; clear
        ; s_axi_awaddr = i.s_axi_awaddr
        ; s_axi_awvalid = i.s_axi_awvalid
        ; s_axi_wdata = i.s_axi_wdata
        ; s_axi_wstrb = i.s_axi_wstrb
        ; s_axi_wvalid = i.s_axi_wvalid
        ; s_axi_bready = i.s_axi_bready
        ; s_axi_araddr = i.s_axi_araddr
        ; s_axi_arvalid = i.s_axi_arvalid
        ; s_axi_rready = i.s_axi_rready
        ; arready = i.arready
        ; rdata = i.rdata
        ; rvalid = i.rvalid
        ; rresp = i.rresp
        ; rlast = i.rlast
        }
    in
    let replay = regs.pipeline.replay in
    { O.aclk = clock
    ; aresetn = ~:clear
    ; s_axi_awready = regs.s_axi_awready
    ; s_axi_wready = regs.s_axi_wready
    ; s_axi_bvalid = regs.s_axi_bvalid
    ; s_axi_bresp = regs.s_axi_bresp
    ; s_axi_arready = regs.s_axi_arready
    ; s_axi_rvalid = regs.s_axi_rvalid
    ; s_axi_rdata = regs.s_axi_rdata
    ; s_axi_rresp = regs.s_axi_rresp
    ; araddr = replay.araddr
    ; arvalid = replay.arvalid
    ; arlen = replay.arlen
    ; arsize = replay.arsize
    ; arburst = replay.arburst
    ; rready = replay.rready
    }
  ;;
end

let verilog params =
  let module T =
    Make (struct
      let params = params
    end)
  in
  let module C = Circuit.With_interface (T.I) (T.O) in
  let scope = Scope.create ~auto_label_hierarchical_ports:true () in
  let circuit = C.create_exn ~name:"ddr_shell" (T.create scope) in
  Rtl.create ~database:(Scope.circuit_database scope) Verilog [ circuit ]
  |> Rtl.full_hierarchy
  |> Rope.to_string
;;
