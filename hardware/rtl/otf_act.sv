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

    assign s_tready = m_tready;
    assign m_tvalid = s_tvalid;

    genvar p;
    generate
        for (p = 0; p < PE; p = p + 1) begin : g_lane
            otf_requant #(
                .ACC_BITS(IN_BITS + 2), .MULT_BITS(MULT_BITS), .SHIFT(SHIFT),
                .OUT_BITS(OUT_BITS), .OUT_MIN(OUT_MIN), .OUT_MAX(OUT_MAX),
                .OUT_ZP(OUT_ZP)
            ) u_requant (
                .acc  ((IN_BITS + 2)'($signed(s_tdata[p*IN_BITS +: IN_BITS]))
                       - (IN_BITS + 2)'(IN_ZP)),
                .mult ($signed(mult_mem[p*NF + int'(nf)])),
                .out  (m_tdata[p*OUT_BITS +: OUT_BITS])
            );
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            nf <= '0;
        end else if (s_tvalid && s_tready) begin
            nf <= (nf == NF_W'(NF - 1)) ? '0 : nf + 1'b1;
        end
    end
endmodule

`default_nettype wire
