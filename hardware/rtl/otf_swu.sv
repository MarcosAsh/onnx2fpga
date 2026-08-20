// Sliding window generator, channel-last.
//
// Frame-buffer strategy: one feature map is captured, then windows are read
// out of it in (row, col, ky, kx, channel-group) order, which is the order the
// downstream matrix-vector engine expects its input vector in. Padding is
// synthesised as PAD_VALUE, the integer that stands for real zero.
`default_nettype none

module otf_swu #(
    parameter int IFM_H     = 8,
    parameter int IFM_W     = 8,
    parameter int IFM_CH    = 4,
    parameter int OFM_H     = 6,
    parameter int OFM_W     = 6,
    parameter int KH        = 3,
    parameter int KW        = 3,
    parameter int STRIDE_H  = 1,
    parameter int STRIDE_W  = 1,
    parameter int PAD_T     = 0,
    parameter int PAD_L     = 0,
    parameter int DIL_H     = 1,
    parameter int DIL_W     = 1,
    parameter int SIMD      = 1,
    parameter int DATA_BITS = 8,
    parameter int PAD_VALUE = 0    // integer standing for real zero
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire  [SIMD*DATA_BITS-1:0] s_tdata,
    input  wire                       s_tvalid,
    output logic                      s_tready,
    output logic [SIMD*DATA_BITS-1:0] m_tdata,
    output logic                      m_tvalid,
    input  wire                       m_tready
);
    localparam int CG       = IFM_CH / SIMD;
    localparam int TOTAL_IN = IFM_H * IFM_W * CG;
    localparam int IN_W     = $clog2(TOTAL_IN + 1);
    localparam int ADDR_W   = (TOTAL_IN <= 1) ? 1 : $clog2(TOTAL_IN);
    localparam int CG_W     = (CG <= 1) ? 1 : $clog2(CG);
    localparam int KH_W     = (KH <= 1) ? 1 : $clog2(KH);
    localparam int KW_W     = (KW <= 1) ? 1 : $clog2(KW);
    localparam int OH_W     = (OFM_H <= 1) ? 1 : $clog2(OFM_H);
    localparam int OW_W     = (OFM_W <= 1) ? 1 : $clog2(OFM_W);

    logic [SIMD*DATA_BITS-1:0] frame [0:TOTAL_IN-1];

    logic            filling;
    logic [IN_W-1:0] wcount;
    logic [CG_W-1:0] cg;
    logic [KW_W-1:0] kx;
    logic [KH_W-1:0] ky;
    logic [OW_W-1:0] ow;
    logic [OH_W-1:0] oh;
    logic                      ovalid;
    logic [SIMD*DATA_BITS-1:0] odata;

    assign s_tready = filling;
    assign m_tvalid = ovalid;
    assign m_tdata  = odata;

    wire signed [31:0] row = $signed({1'b0, oh}) * STRIDE_H
                           + $signed({1'b0, ky}) * DIL_H - PAD_T;
    wire signed [31:0] col = $signed({1'b0, ow}) * STRIDE_W
                           + $signed({1'b0, kx}) * DIL_W - PAD_L;
    wire in_bounds = (row >= 0) && (row < IFM_H) && (col >= 0) && (col < IFM_W);
    wire signed [31:0] raddr = ((row * IFM_W) + col) * CG + 32'($signed({1'b0, cg}));

    wire out_free = !ovalid || m_tready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            filling <= 1'b1;
            wcount  <= '0;
            cg <= '0; kx <= '0; ky <= '0; ow <= '0; oh <= '0;
            ovalid  <= 1'b0;
            odata   <= '0;
        end else begin
            if (ovalid && m_tready) ovalid <= 1'b0;

            if (filling) begin
                if (s_tvalid) begin
                    frame[wcount[ADDR_W-1:0]] <= s_tdata;
                    if (wcount == IN_W'(TOTAL_IN - 1)) begin
                        wcount  <= '0;
                        filling <= 1'b0;
                    end else begin
                        wcount <= wcount + 1'b1;
                    end
                end
            end else if (out_free) begin
                ovalid <= 1'b1;
                odata  <= in_bounds ? frame[raddr[ADDR_W-1:0]]
                                    : {SIMD{DATA_BITS'(PAD_VALUE)}};

                if (cg != CG_W'(CG - 1)) begin
                    cg <= cg + 1'b1;
                end else begin
                    cg <= '0;
                    if (kx != KW_W'(KW - 1)) begin
                        kx <= kx + 1'b1;
                    end else begin
                        kx <= '0;
                        if (ky != KH_W'(KH - 1)) begin
                            ky <= ky + 1'b1;
                        end else begin
                            ky <= '0;
                            if (ow != OW_W'(OFM_W - 1)) begin
                                ow <= ow + 1'b1;
                            end else begin
                                ow <= '0;
                                if (oh != OH_W'(OFM_H - 1)) begin
                                    oh <= oh + 1'b1;
                                end else begin
                                    oh      <= '0;
                                    filling <= 1'b1;
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [31:0] unused_raddr = raddr;   // upper bits are bounds-checked, not addressed
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
