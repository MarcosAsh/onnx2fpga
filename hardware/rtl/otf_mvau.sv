// Folded matrix-vector engine; the compile-time bias also folds in -zx * sum_k(W[k][c]) for asymmetric inputs.
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
    parameter int BLOCK_RAM = 0,  // weights in a registered block RAM, one cycle more latency
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
    localparam int WORD   = SIMD*WGT_BITS;

    logic [ACC_BITS-1:0]      bias_mem   [0:PE*NF-1];
    logic [MULT_BITS-1:0]     mult_mem   [0:PE*NF-1];
    logic [SIMD*ACT_BITS-1:0] act_buf    [0:SF-1];

    initial begin
        if (BIAS_FILE   != "") $readmemh(BIAS_FILE,   bias_mem);
        if (MULT_FILE   != "") $readmemh(MULT_FILE,   mult_mem);
    end

    logic [SF_W-1:0] sf;
    logic [NF_W-1:0] nf;
    logic            out_valid_r;
    logic [PE*OUT_BITS-1:0] out_data_r;

    // The step being multiplied: the one issued this cycle, or with BLOCK_RAM the one issued last cycle.
    logic                     mac, mac_first, mac_last, held;
    logic [NF_W-1:0]          mac_nf;
    logic [SIMD*ACT_BITS-1:0] mac_act;
    logic [PE*WORD-1:0]       mac_weights;

    wire last_sf    = (sf == SF_W'(SF - 1));
    wire out_free   = !out_valid_r || m_tready;
    wire go         = !held || out_free;
    wire need_input = (nf == '0);
    wire step       = go && (!need_input || s_tvalid);

    assign s_tready = need_input && go;
    assign m_tvalid = out_valid_r;
    assign m_tdata  = out_data_r;

    wire [SIMD*ACT_BITS-1:0] act_word = need_input ? s_tdata : act_buf[sf];
    wire [W_ADDR-1:0] waddr = W_ADDR'(int'(nf) * SF + int'(sf));

    generate
        if (BLOCK_RAM != 0) begin : g_block
            // One word holds every lane's weights for a step, lane 0 in the low bits.
            (* ram_style = "block" *) logic [PE*WORD-1:0] weight_mem [0:SF*NF-1];
            logic issued;

            initial if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weight_mem);

            always_ff @(posedge clk) begin
                if (go) begin
                    mac_weights <= weight_mem[waddr];
                    mac_first   <= (sf == '0);
                    mac_last    <= last_sf;
                    mac_nf      <= nf;
                    mac_act     <= act_word;
                end
                if (!rst_n) issued <= 1'b0;
                else if (go) issued <= step;
            end

            assign mac  = go && issued;
            assign held = issued && mac_last;
        end else begin : g_lut
            logic [WORD-1:0] weight_mem [0:PE*SF*NF-1];

            initial if (WEIGHT_FILE != "") $readmemh(WEIGHT_FILE, weight_mem);

            for (genvar q = 0; q < PE; q = q + 1) begin : g_word
                assign mac_weights[q*WORD +: WORD] = weight_mem[q*SF*NF + int'(waddr)];
            end

            assign mac       = step;
            assign mac_first = (sf == '0);
            assign mac_last  = last_sf;
            assign mac_nf    = nf;
            assign mac_act   = act_word;
            assign held      = last_sf;
        end
    endgenerate

    logic signed [ACC_BITS-1:0] acc      [0:PE-1];
    logic signed [ACC_BITS-1:0] acc_next [0:PE-1];
    logic signed [OUT_BITS-1:0] lane_out [0:PE-1];

    genvar p;
    generate
        for (p = 0; p < PE; p = p + 1) begin : g_lane
            wire [WORD-1:0] weight_word = mac_weights[p*WORD +: WORD];

            logic signed [ACC_BITS-1:0] dot;
            always_comb begin
                dot = '0;
                for (int k = 0; k < SIMD; k = k + 1) begin
                    dot = dot + ACC_BITS'($signed(mac_act[k*ACT_BITS +: ACT_BITS]))
                              * ACC_BITS'($signed(weight_word[k*WGT_BITS +: WGT_BITS]));
                end
            end

            wire signed [ACC_BITS-1:0] seed =
                mac_first ? $signed(bias_mem[p*NF + int'(mac_nf)]) : acc[p];
            assign acc_next[p] = seed + dot;

            otf_requant #(
                .ACC_BITS(ACC_BITS), .MULT_BITS(MULT_BITS), .SHIFT(SHIFT),
                .OUT_BITS(OUT_BITS), .OUT_MIN(OUT_MIN), .OUT_MAX(OUT_MAX),
                .OUT_ZP(OUT_ZP)
            ) u_requant (
                .acc  (acc_next[p]),
                .mult ($signed(mult_mem[p*NF + int'(mac_nf)])),
                .out  (lane_out[p])
            );

            always_ff @(posedge clk) begin
                if (mac) acc[p] <= acc_next[p];
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

            if (mac && mac_last) begin
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
