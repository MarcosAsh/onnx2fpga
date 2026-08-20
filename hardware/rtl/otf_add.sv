// Elementwise add of two streams, each side scaled onto the output grid.
// Both producers register their valid, so the join below cannot form a
// combinational loop.
`default_nettype none

module otf_add #(
    parameter int PE       = 1,
    parameter int IN_BITS  = 8,
    parameter int OUT_BITS = 8,
    parameter int ACC_BITS = 32,
    parameter int LEFT_MULT  = 1,
    parameter int RIGHT_MULT = 1,
    parameter int SHIFT    = 0,
    parameter int OUT_MIN  = -128,
    parameter int OUT_MAX  = 127,
    parameter int BIAS     = 0,    // -za*LEFT_MULT - zb*RIGHT_MULT
    parameter int OUT_ZP   = 0
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire  [PE*IN_BITS-1:0]  a_tdata,
    input  wire                    a_tvalid,
    output logic                   a_tready,
    input  wire  [PE*IN_BITS-1:0]  b_tdata,
    input  wire                    b_tvalid,
    output logic                   b_tready,
    output logic [PE*OUT_BITS-1:0] m_tdata,
    output logic                   m_tvalid,
    input  wire                    m_tready
);
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused_clk = clk;      // combinational unit, ports kept uniform
    wire unused_rst = rst_n;
    /* verilator lint_on UNUSEDSIGNAL */

    wire both = a_tvalid & b_tvalid;
    assign m_tvalid = both;
    assign a_tready = both & m_tready;
    assign b_tready = both & m_tready;

    genvar p;
    generate
        for (p = 0; p < PE; p = p + 1) begin : g_lane
            wire signed [ACC_BITS-1:0] left =
                ACC_BITS'($signed(a_tdata[p*IN_BITS +: IN_BITS])) * ACC_BITS'(LEFT_MULT);
            wire signed [ACC_BITS-1:0] right =
                ACC_BITS'($signed(b_tdata[p*IN_BITS +: IN_BITS])) * ACC_BITS'(RIGHT_MULT);

            otf_requant #(
                .ACC_BITS(ACC_BITS), .MULT_BITS(2), .SHIFT(SHIFT),
                .OUT_BITS(OUT_BITS), .OUT_MIN(OUT_MIN), .OUT_MAX(OUT_MAX),
                .OUT_ZP(OUT_ZP)
            ) u_requant (
                .acc  (left + right + ACC_BITS'(BIAS)),
                .mult (2'sd1),
                .out  (m_tdata[p*OUT_BITS +: OUT_BITS])
            );
        end
    endgenerate
endmodule

`default_nettype wire
