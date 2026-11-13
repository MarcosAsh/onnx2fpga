// Per-channel rescale and clamp on a PE-folded stream.
`default_nettype none

module otf_act #(
    parameter int CHANNELS  = 8,
    parameter int PE        = 1,
    parameter int IN_BITS   = 32,
    parameter int OUT_BITS  = 8,
    parameter int MULT_BITS = 18,
    parameter int SHIFT     = 0,
    parameter int OUT_MIN   = -128,
    parameter int OUT_MAX   = 127,
    parameter int IN_ZP     = 0,
    parameter int OUT_ZP    = 0,
    parameter string MULT_FILE = ""
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire  [PE*IN_BITS-1:0]   s_tdata,
    input  wire                     s_tvalid,
    output logic                    s_tready,
    output logic [PE*OUT_BITS-1:0]  m_tdata,
    output logic                    m_tvalid,
    input  wire                     m_tready
);
    localparam int NF   = CHANNELS / PE;
    localparam int NF_W = (NF <= 1) ? 1 : $clog2(NF);

    logic [MULT_BITS-1:0] mult_mem [0:PE*NF-1];
    initial if (MULT_FILE != "") $readmemh(MULT_FILE, mult_mem);

    logic [NF_W-1:0] nf;

    logic product_valid, out_valid;
    wire go = !out_valid || m_tready;
    assign s_tready = go;
    assign m_tvalid = out_valid;

    genvar p;
    generate
        for (p = 0; p < PE; p = p + 1) begin : g_lane
            logic signed [IN_BITS+MULT_BITS+1:0] product;
            wire signed [OUT_BITS-1:0] scaled;
            wire signed [IN_BITS+1:0] centered =
                (IN_BITS + 2)'($signed(s_tdata[p*IN_BITS +: IN_BITS])) - (IN_BITS + 2)'(IN_ZP);
            always_ff @(posedge clk) begin
                if (go) begin
                    product <= centered * $signed(mult_mem[p*NF + int'(nf)]);
                    m_tdata[p*OUT_BITS +: OUT_BITS] <= scaled;
                end
            end
            otf_requant #(
                .ACC_BITS(IN_BITS + 2 + MULT_BITS), .MULT_BITS(2), .SHIFT(SHIFT),
                .OUT_BITS(OUT_BITS), .OUT_MIN(OUT_MIN), .OUT_MAX(OUT_MAX),
                .OUT_ZP(OUT_ZP)
            ) u_requant (
                .acc  (product),
                .mult (2'sd1),
                .out  (scaled)
            );
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            nf <= '0;
            product_valid <= 1'b0;
            out_valid <= 1'b0;
        end else if (go) begin
            product_valid <= s_tvalid;
            out_valid <= product_valid;
            if (s_tvalid) nf <= (nf == NF_W'(NF - 1)) ? '0 : nf + 1'b1;
        end
    end
endmodule

`default_nettype wire
