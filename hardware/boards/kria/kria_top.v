// The Vivado top: the block design (PS and SmartConnect) wired to the Hardcaml kria_shell.
module kria_top;
    wire        pl_clk0, pl_resetn0, aclk, aresetn;
    wire [15:0] awaddr, araddr;
    wire [2:0]  awprot, arprot;
    wire [31:0] wdata, rdata;
    wire [3:0]  wstrb;
    wire [1:0]  bresp, rresp;
    wire        awvalid, awready, wvalid, wready, bvalid, bready;
    wire        arvalid, arready, rvalid, rready;

    system_wrapper system (
        .pl_clk0(pl_clk0), .pl_resetn0(pl_resetn0), .aclk(aclk), .aresetn(aresetn),
        .M_AXI_awaddr(awaddr), .M_AXI_awprot(awprot), .M_AXI_awvalid(awvalid),
        .M_AXI_awready(awready), .M_AXI_wdata(wdata), .M_AXI_wstrb(wstrb),
        .M_AXI_wvalid(wvalid), .M_AXI_wready(wready), .M_AXI_bresp(bresp),
        .M_AXI_bvalid(bvalid), .M_AXI_bready(bready), .M_AXI_araddr(araddr),
        .M_AXI_arprot(arprot), .M_AXI_arvalid(arvalid), .M_AXI_arready(arready),
        .M_AXI_rdata(rdata), .M_AXI_rresp(rresp), .M_AXI_rvalid(rvalid),
        .M_AXI_rready(rready));

    kria_shell otf (
        .pl_clk0(pl_clk0), .pl_resetn0(pl_resetn0), .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid),
        .s_axi_rready(rready));
endmodule
