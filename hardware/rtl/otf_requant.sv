// Fixed-point rescale shared by every compute unit.
// out = clamp(((acc * mult + 2**(SHIFT-1)) >>> SHIFT) + OUT_ZP, OUT_MIN, OUT_MAX)
//
// OUT_ZP is the zero point of the output tensor. It is zero for a symmetric
// quantiser and non zero for an asymmetric one, where the integer standing for
// real zero is not itself zero.
`default_nettype none

module otf_requant #(
    parameter int ACC_BITS  = 32,
    parameter int MULT_BITS = 18,
    parameter int SHIFT     = 0,
    parameter int OUT_BITS  = 8,
    parameter int OUT_MIN   = -128,
    parameter int OUT_MAX   = 127,
    parameter int OUT_ZP    = 0
) (
    input  wire signed [ACC_BITS-1:0]  acc,
    input  wire signed [MULT_BITS-1:0] mult,
    output logic signed [OUT_BITS-1:0] out
);
    localparam int PROD_BITS = ACC_BITS + MULT_BITS + 1;

    logic signed [PROD_BITS-1:0] product;
    logic signed [PROD_BITS-1:0] rounded;
    logic signed [PROD_BITS-1:0] shifted;
    logic signed [PROD_BITS-1:0] round_add;
    logic signed [PROD_BITS-1:0] lo_limit;
    logic signed [PROD_BITS-1:0] hi_limit;

    assign round_add = (SHIFT > 0) ? (PROD_BITS'(1) << (SHIFT - 1)) : '0;
    assign lo_limit  = PROD_BITS'(OUT_MIN);
    assign hi_limit  = PROD_BITS'(OUT_MAX);

    always_comb begin
        product = acc * mult;
        shifted = (SHIFT > 0) ? ((product + round_add) >>> SHIFT) : product;
        rounded = shifted + PROD_BITS'(OUT_ZP);
        if (rounded < lo_limit)      out = OUT_BITS'(lo_limit);
        else if (rounded > hi_limit) out = OUT_BITS'(hi_limit);
        else                         out = OUT_BITS'(rounded);
    end
endmodule

`default_nettype wire
