// Port-accurate stand-ins for the block design wrapper and the two UltraScale+ primitives, for linting kria_top.
module system_wrapper (
    output pl_clk0, output pl_resetn0, input aclk, input aresetn,
    output [15:0] M_AXI_awaddr, output [2:0] M_AXI_awprot, output M_AXI_awvalid,
    input M_AXI_awready, output [31:0] M_AXI_wdata, output [3:0] M_AXI_wstrb,
    output M_AXI_wvalid, input M_AXI_wready, input [1:0] M_AXI_bresp,
    input M_AXI_bvalid, output M_AXI_bready, output [15:0] M_AXI_araddr,
    output [2:0] M_AXI_arprot, output M_AXI_arvalid, input M_AXI_arready,
    input [31:0] M_AXI_rdata, input [1:0] M_AXI_rresp, input M_AXI_rvalid,
    output M_AXI_rready);
    assign pl_clk0 = 0; assign pl_resetn0 = 0;
    assign M_AXI_awaddr = 0; assign M_AXI_awprot = 0; assign M_AXI_awvalid = 0;
    assign M_AXI_wdata = 0; assign M_AXI_wstrb = 0; assign M_AXI_wvalid = 0;
    assign M_AXI_bready = 0; assign M_AXI_araddr = 0; assign M_AXI_arprot = 0;
    assign M_AXI_arvalid = 0; assign M_AXI_rready = 0;
endmodule

module MMCME4_BASE #(parameter real CLKIN1_PERIOD = 0.0, parameter real CLKFBOUT_MULT_F = 5.0,
                     parameter integer DIVCLK_DIVIDE = 1, parameter real CLKOUT0_DIVIDE_F = 1.0)
    (input CLKIN1, input CLKFBIN, input RST, input PWRDWN,
     output CLKOUT0, output CLKFBOUT, output LOCKED);
    assign CLKOUT0 = CLKIN1; assign CLKFBOUT = CLKIN1; assign LOCKED = 1;
endmodule

module BUFG (input I, output O);
    assign O = I;
endmodule
