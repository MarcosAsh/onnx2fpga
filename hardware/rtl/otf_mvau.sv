// Folded matrix-vector engine with fused bias and rescale.
//
// The bias carries more than the model's own bias term. An asymmetric input
// quantiser contributes -zx * sum_k(W[k][c]) per output channel, which is a
// constant, so it is folded in here at compile time and costs no hardware.
//
// SIMD input elements are consumed per cycle and PE output channels are
// produced in parallel, so one input vector costs (MW/SIMD)*(MH/PE) cycles.
// Weights sit in PE independent memories addressed by {neuron fold, synapse
// fold}; each word holds the SIMD weights one lane needs that cycle. When the
// output is folded (NF > 1) the input vector is held locally and replayed for
// each neuron fold rather than re-fetched upstream.
`default_nettype none

module otf_mvau #(
    parameter int MW        = 8,
    parameter int MH        = 8,
    parameter int SIMD      = 1,
    parameter int PE        = 1,
    parameter int ACT_BITS  = 8,
    parameter int WGT_BITS  = 8,
    parameter int ACC_BITS  = 24,
    parameter int OUT_BITS  = 8,
    parameter int MULT_BITS = 18,
    parameter int SHIFT     = 0,
    parameter int OUT_MIN   = -128,
    parameter int OUT_MAX   = 127,
    parameter int OUT_ZP    = 0,
    parameter string WEIGHT_FILE = "",
    parameter string BIAS_FILE   = "",
    parameter string MULT_FILE   = ""
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire  [SIMD*ACT_BITS-1:0]   s_tdata,
    input  wire                        s_tvalid,
    output logic                       s_tready,
    output logic [PE*OUT_BITS-1:0]     m_tdata,
    output logic                       m_tvalid,
    input  wire                        m_tready
);
    localparam int SF     = MW / SIMD;
    localparam int NF     = MH / PE;
    localparam int SF_W   = (SF <= 1) ? 1 : $clog2(SF);
    localparam int NF_W   = (NF <= 1) ? 1 : $clog2(NF);
    localparam int W_ADDR = (SF*NF <= 1) ? 1 : $clog2(SF*NF);

    logic [SIMD*WGT_BITS-1:0] weight_mem [0:PE*SF*NF-1];
    logic [ACC_BITS-1:0]      bias_mem   [0:PE*NF-1];
    logic [MULT_BITS-1:0]     mult_mem   [0:PE*NF-1];
    logic [SIMD*ACT_BITS-1:0] act_buf    [0:SF-1];

    initial begin
        if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weight_mem);
        if (BIAS_FILE   != "") $readmemh(BIAS_FILE,   bias_mem);
        if (MULT_FILE   != "") $readmemh(MULT_FILE,   mult_mem);
    end

    logic [SF_W-1:0] sf;
    logic [NF_W-1:0] nf;
    logic            out_valid_r;
    logic [PE*OUT_BITS-1:0] out_data_r;

    wire last_sf    = (sf == SF_W'(SF - 1));
    wire out_free   = !out_valid_r || m_tready;
    wire may_step   = !last_sf || out_free;
    wire need_input = (nf == '0);
    wire step       = may_step && (!need_input || s_tvalid);

    assign s_tready = need_input && may_step;
    assign m_tvalid = out_valid_r;
    assign m_tdata  = out_data_r;

    wire [SIMD*ACT_BITS-1:0] act_word = need_input ? s_tdata : act_buf[sf];

    logic signed [ACC_BITS-1:0] acc      [0:PE-1];
    logic signed [ACC_BITS-1:0] acc_next [0:PE-1];
    logic signed [OUT_BITS-1:0] lane_out [0:PE-1];

    genvar p;
    generate
        for (p = 0; p < PE; p = p + 1) begin : g_lane
            wire [W_ADDR-1:0] waddr = W_ADDR'(int'(nf) * SF + int'(sf));
            wire [SIMD*WGT_BITS-1:0] weight_word = weight_mem[p*SF*NF + int'(waddr)];

            logic signed [ACC_BITS-1:0] dot;
            always_comb begin
                dot = '0;
                for (int k = 0; k < SIMD; k = k + 1) begin
                    dot = dot + ACC_BITS'($signed(act_word[k*ACT_BITS +: ACT_BITS]))
                              * ACC_BITS'($signed(weight_word[k*WGT_BITS +: WGT_BITS]));
                end
            end

            wire signed [ACC_BITS-1:0] seed =
                (sf == '0) ? $signed(bias_mem[p*NF + int'(nf)]) : acc[p];
            assign acc_next[p] = seed + dot;

            otf_requant #(
                .ACC_BITS(ACC_BITS), .MULT_BITS(MULT_BITS), .SHIFT(SHIFT),
                .OUT_BITS(OUT_BITS), .OUT_MIN(OUT_MIN), .OUT_MAX(OUT_MAX),
                .OUT_ZP(OUT_ZP)
            ) u_requant (
                .acc  (acc_next[p]),
                .mult ($signed(mult_mem[p*NF + int'(nf)])),
                .out  (lane_out[p])
            );

            always_ff @(posedge clk) begin
                if (step) acc[p] <= acc_next[p];
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sf <= '0;
            nf <= '0;
            out_valid_r <= 1'b0;
        end else begin
            if (step && need_input) act_buf[sf] <= s_tdata;

            if (step) begin
                if (last_sf) begin
                    sf <= '0;
                    nf <= (nf == NF_W'(NF - 1)) ? '0 : nf + 1'b1;
                end else begin
                    sf <= sf + 1'b1;
                end
            end

            if (step && last_sf) begin
                out_valid_r <= 1'b1;
                for (int q = 0; q < PE; q = q + 1) begin
                    out_data_r[q*OUT_BITS +: OUT_BITS] <= lane_out[q];
                end
            end else if (m_tready) begin
                out_valid_r <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
