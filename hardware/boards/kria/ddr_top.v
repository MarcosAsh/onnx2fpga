// PS control and DDR reads share the trading clock and reset.
module kria_top;
    wire        pl_clk0, pl_resetn0, aclk, aresetn;
    wire [31:0] awaddr, araddr;
    wire [2:0]  awprot, arprot;
    wire [31:0] wdata, rdata;
    wire [3:0]  wstrb;
    wire [1:0]  bresp, rresp;
    wire        awvalid, awready, wvalid, wready, bvalid, bready;
    wire        arvalid, arready, rvalid, rready;

    wire [39:0] ddr_araddr;
    wire [7:0] ddr_arlen;
    wire [2:0] ddr_arsize;
    wire [1:0] ddr_arburst, ddr_rresp;
    wire [127:0] ddr_rdata;
    wire ddr_arvalid, ddr_arready, ddr_rvalid, ddr_rready, ddr_rlast;

    system_wrapper system (
        .S_DDR_araddr(ddr_araddr), .S_DDR_arlen(ddr_arlen), .S_DDR_arsize(ddr_arsize),
        .S_DDR_arburst(ddr_arburst), .S_DDR_arid(1'b0), .S_DDR_arlock(1'b0),
        .S_DDR_arcache(4'b0011), .S_DDR_arprot(3'b010), .S_DDR_arqos(4'b0),
        .S_DDR_arvalid(ddr_arvalid), .S_DDR_arready(ddr_arready),
        .S_DDR_rdata(ddr_rdata), .S_DDR_rresp(ddr_rresp), .S_DDR_rlast(ddr_rlast),
        .S_DDR_rvalid(ddr_rvalid), .S_DDR_rready(ddr_rready),
        .pl_clk0(pl_clk0), .pl_resetn0(pl_resetn0), .aclk(aclk), .aresetn(aresetn),
        .M_AXI_awaddr(awaddr), .M_AXI_awprot(awprot), .M_AXI_awvalid(awvalid),
        .M_AXI_awready(awready), .M_AXI_wdata(wdata), .M_AXI_wstrb(wstrb),
        .M_AXI_wvalid(wvalid), .M_AXI_wready(wready), .M_AXI_bresp(bresp),
        .M_AXI_bvalid(bvalid), .M_AXI_bready(bready), .M_AXI_araddr(araddr),
        .M_AXI_arprot(arprot), .M_AXI_arvalid(arvalid), .M_AXI_arready(arready),
        .M_AXI_rdata(rdata), .M_AXI_rresp(rresp), .M_AXI_rvalid(rvalid),
        .M_AXI_rready(rready));

    ddr_shell otf (
        .araddr(ddr_araddr), .arlen(ddr_arlen), .arsize(ddr_arsize), .arburst(ddr_arburst),
        .arvalid(ddr_arvalid), .arready(ddr_arready), .rdata(ddr_rdata), .rresp(ddr_rresp),
        .rlast(ddr_rlast), .rvalid(ddr_rvalid), .rready(ddr_rready), .s_axi_wstrb(wstrb),
        .pl_clk0(pl_clk0), .pl_resetn0(pl_resetn0), .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(awaddr[15:0]), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr[15:0]), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid),
        .s_axi_rready(rready));
endmodule
