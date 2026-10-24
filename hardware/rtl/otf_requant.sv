// out = clamp(((acc * mult + 2**(SHIFT-1)) >>> SHIFT) + OUT_ZP, OUT_MIN, OUT_MAX)
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

    // Below SHIFT = ACC_BITS+MULT_BITS the product plus rounding constant fits PROD_BITS; from there the quotient is 0, so no multiplier is built.
    localparam bit QUOTIENT_IS_ZERO = SHIFT >= ACC_BITS + MULT_BITS;

    logic signed [PROD_BITS-1:0] product;
    logic signed [PROD_BITS-1:0] rounded;
    logic signed [PROD_BITS-1:0] shifted;
    logic signed [PROD_BITS-1:0] round_add;
    logic signed [PROD_BITS-1:0] lo_limit;
    logic signed [PROD_BITS-1:0] hi_limit;

    assign round_add = (SHIFT > 0 && !QUOTIENT_IS_ZERO) ? (PROD_BITS'(1) << (SHIFT - 1)) : '0;
    assign lo_limit  = PROD_BITS'(OUT_MIN);
    assign hi_limit  = PROD_BITS'(OUT_MAX);

    always_comb begin
        product = acc * mult;
        if (QUOTIENT_IS_ZERO) shifted = '0;
        else if (SHIFT > 0)   shifted = (product + round_add) >>> SHIFT;
        else                  shifted = product;
        rounded = shifted + PROD_BITS'(OUT_ZP);
        if (rounded < lo_limit)      out = OUT_BITS'(lo_limit);
        else if (rounded > hi_limit) out = OUT_BITS'(hi_limit);
        else                         out = OUT_BITS'(rounded);
    end
endmodule

`default_nettype wire
