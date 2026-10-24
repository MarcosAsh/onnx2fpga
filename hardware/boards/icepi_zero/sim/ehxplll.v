// Stand-in ECP5 PLL for simulation: outputs follow the input and LOCK rises after 16 cycles.
module EHXPLLL #(
    parameter PLLRST_ENA = "DISABLED",
    parameter INTFB_WAKE = "DISABLED",
    parameter STDBY_ENABLE = "DISABLED",
    parameter DPHASE_SOURCE = "DISABLED",
    parameter OUTDIVIDER_MUXA = "DIVA",
    parameter OUTDIVIDER_MUXB = "DIVB",
    parameter OUTDIVIDER_MUXC = "DIVC",
    parameter OUTDIVIDER_MUXD = "DIVD",
    parameter CLKI_DIV = 1,
    parameter CLKOP_ENABLE = "ENABLED",
    parameter CLKOP_DIV = 8,
    parameter CLKOP_CPHASE = 0,
    parameter CLKOP_FPHASE = 0,
    parameter CLKOS_ENABLE = "DISABLED",
    parameter CLKOS_DIV = 8,
    parameter CLKOS_CPHASE = 0,
    parameter CLKOS_FPHASE = 0,
    parameter FEEDBK_PATH = "CLKOP",
    parameter CLKFB_DIV = 1
) (
    input  wire RST,
    input  wire STDBY,
    input  wire CLKI,
    input  wire CLKFB,
    input  wire PHASESEL0,
    input  wire PHASESEL1,
    input  wire PHASEDIR,
    input  wire PHASESTEP,
    input  wire PHASELOADREG,
    input  wire PLLWAKESYNC,
    input  wire ENCLKOP,
    output wire CLKOP,
    output wire CLKOS,
    output wire LOCK
);

  reg [4:0] settling = 5'd0;

  always @(posedge CLKI) begin
    if (!settling[4]) settling <= settling + 5'd1;
  end

  assign CLKOP = CLKI;
  assign CLKOS = CLKI;
  assign LOCK = settling[4];

endmodule
