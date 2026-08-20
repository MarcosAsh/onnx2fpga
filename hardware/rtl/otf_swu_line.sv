// Sliding window generator, line buffered, with optional parallel windows.
//
// Holds only the pixels a window can still reach: the (KH-1) rows above the
// current one plus the current partial row. Capture and replay run in the same
// cycles, so a convolution costs its replay term alone rather than replay plus
// a full frame capture, and the memory is a span rather than an image.
//
// PAR emits PAR adjacent kernel columns per beat instead of one. A first layer
// with few input channels cannot fold on channels at all, so without this its
// replay term is the floor for the whole pipeline. Reading PAR pixels at once
// needs PAR read ports, which is done by replicating the buffer: the writer
// writes every copy, and copy j is read at an offset of j*DIL_W pixels. That
// costs PAR times the memory and keeps the read address a plain bit slice,
// which a banked layout would not.
//
// Addressing is a circular buffer over pixel indices. The span of one window
// in raster order is
//
//     SPAN = (KH-1)*DIL_H*IFM_W + (KW-1)*DIL_W + 1
//
// but the span alone is not enough to size the buffer. With top padding two
// output rows can start on the same input row, so when the reader wraps from
// the end of one output row to the start of the next it walks backwards to a
// pixel it already passed. The buffer therefore has to cover the span plus
// that backward reach, TAIL, and the oldest pixel still needed is the minimum
// over the rest of this output row and the start of the next one, not just the
// current position. Getting either of these wrong silently corrupts the first
// window of every output row on a padded, non square image.
//
// Capacity is rounded up to a power of two so addressing is bit slicing rather
// than a modulo. All pixel counters are free running and compared by signed
// difference, so they wrap harmlessly and frames overlap without a reset.
`default_nettype none

module otf_swu_line #(
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
    parameter int SIMD      = 1,   // elements per input beat, divides IFM_CH
    parameter int PAR       = 1,   // kernel columns per output beat, divides KW
    parameter int DATA_BITS = 8,
    parameter int PAD_VALUE = 0    // integer standing for real zero
) (
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire  [SIMD*DATA_BITS-1:0]     s_tdata,
    input  wire                           s_tvalid,
    output logic                          s_tready,
    output logic [PAR*SIMD*DATA_BITS-1:0] m_tdata,
    output logic                          m_tvalid,
    input  wire                           m_tready
);
    localparam int CG    = IFM_CH / SIMD;
    localparam int KWG   = KW / PAR;
    localparam int SPAN  = (KH - 1) * DIL_H * IFM_W + (KW - 1) * DIL_W + 1;
    localparam int TAIL_RAW = (OFM_W - 1) * STRIDE_W - PAD_L;
    localparam int TAIL  = (TAIL_RAW > 0) ? TAIL_RAW : 0;
    localparam int NEED  = SPAN + TAIL + 1;
    localparam int LOGD  = (NEED < 2) ? 1 : $clog2(NEED);
    localparam int DEPTH = 1 << LOGD;
    localparam int FRAME = IFM_H * IFM_W;
    localparam int WORD  = SIMD * DATA_BITS;

    localparam int CG_W  = (CG    <= 1) ? 1 : $clog2(CG);
    localparam int KWG_W = (KWG   <= 1) ? 1 : $clog2(KWG);
    localparam int KH_W  = (KH    <= 1) ? 1 : $clog2(KH);
    localparam int OH_W  = (OFM_H <= 1) ? 1 : $clog2(OFM_H);
    localparam int OW_W  = (OFM_W <= 1) ? 1 : $clog2(OFM_W);
    localparam int AD_W  = $clog2(DEPTH * CG);

    // PAR > 1 consumes whole pixels, so channel folding and column parallelism
    // are mutually exclusive by construction.
    initial begin
        if (PAR > 1 && CG != 1)
            $fatal(1, "otf_swu_line: PAR > 1 requires SIMD == IFM_CH");
        if (KW % PAR != 0)
            $fatal(1, "otf_swu_line: PAR must divide KW");
    end

    logic [WORD-1:0] buffer [0:PAR-1][0:DEPTH*CG-1];

    logic signed [31:0] wr_pixel;    // pixels fully written, free running
    logic [CG_W-1:0]    wr_cg;
    logic signed [31:0] frame_base;  // pixel index the current output frame starts at

    logic [CG_W-1:0]  rd_cg;
    logic [KWG_W-1:0] kxg;
    logic [KH_W-1:0]  ky;
    logic [OW_W-1:0]  ow;
    logic [OH_W-1:0]  oh;

    logic                             ovalid;
    logic [PAR*SIMD*DATA_BITS-1:0]    odata;

    // Row is shared by every column in the beat; each copy shifts the column.
    wire signed [31:0] row = 32'($signed({1'b0, oh})) * STRIDE_H
                           + 32'($signed({1'b0, ky})) * DIL_H - PAD_T;
    wire signed [31:0] col_base = 32'($signed({1'b0, ow})) * STRIDE_W
                                + 32'($signed({1'b0, kxg})) * PAR * DIL_W - PAD_L;
    wire row_ok = (row >= 0) && (row < IFM_H);

    logic signed [31:0]  needed  [0:PAR-1];
    logic                in_view  [0:PAR-1];
    logic [WORD-1:0]     fetched [0:PAR-1];

    genvar j;
    generate
        for (j = 0; j < PAR; j = j + 1) begin : g_copy
            wire signed [31:0] col_j = col_base + j * DIL_W;
            assign in_view[j] = row_ok && (col_j >= 0) && (col_j < IFM_W);
            assign needed[j] = frame_base + row * IFM_W + col_j;
            wire [AD_W-1:0] addr_j =
                AD_W'(needed[j][LOGD-1:0] * CG + 32'(rd_cg));
            assign fetched[j] = in_view[j] ? buffer[j][addr_j]
                                           : {SIMD{DATA_BITS'(PAD_VALUE)}};
        end
    endgenerate

    // Every copy that will actually be read has to have arrived already.
    logic read_ok;
    always_comb begin
        read_ok = 1'b1;
        for (int c = 0; c < PAR; c = c + 1) begin
            if (in_view[c] && ((needed[c] - wr_pixel) >= 32'sd0)) read_ok = 1'b0;
        end
    end

    // Oldest pixel any remaining read can still reach. Padding puts part of
    // the window outside the image, so clamp to the first real pixel. Output
    // row oh+1 restarts at the left edge, which can be behind where row oh has
    // reached, so it has to be considered too.
    wire signed [31:0] row0_raw = 32'($signed({1'b0, oh})) * STRIDE_H - PAD_T;
    wire signed [31:0] col0_raw = 32'($signed({1'b0, ow})) * STRIDE_W - PAD_L;
    wire signed [31:0] row0 = (row0_raw < 0) ? 32'sd0 : row0_raw;
    wire signed [31:0] col0 = (col0_raw < 0) ? 32'sd0 : col0_raw;

    wire signed [31:0] next_row_raw =
        (32'($signed({1'b0, oh})) + 32'sd1) * STRIDE_H - PAD_T;
    wire signed [31:0] next_row = (next_row_raw < 0) ? 32'sd0 : next_row_raw;
    wire signed [31:0] first_col = (PAD_L > 0) ? 32'sd0 : -PAD_L;

    wire signed [31:0] oldest_here = frame_base + row0 * IFM_W + col0;
    wire signed [31:0] oldest_next = frame_base + next_row * IFM_W + first_col;
    wire last_out_row = (oh == OH_W'(OFM_H - 1));
    wire signed [31:0] oldest =
        (!last_out_row && ((oldest_next - oldest_here) < 32'sd0)) ? oldest_next
                                                                 : oldest_here;

    wire can_write = (wr_pixel - oldest) < DEPTH;
    wire out_free  = !ovalid || m_tready;
    wire emit      = read_ok && out_free;

    wire [AD_W-1:0] wr_addr = AD_W'(wr_pixel[LOGD-1:0] * CG + 32'(wr_cg));

    assign s_tready = can_write;
    assign m_tvalid = ovalid;
    assign m_tdata  = odata;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_pixel   <= '0;
            wr_cg      <= '0;
            frame_base <= '0;
            rd_cg <= '0; kxg <= '0; ky <= '0; ow <= '0; oh <= '0;
            ovalid     <= 1'b0;
            odata      <= '0;
        end else begin
            if (ovalid && m_tready) ovalid <= 1'b0;

            if (s_tvalid && can_write) begin
                for (int copy = 0; copy < PAR; copy = copy + 1) begin
                    buffer[copy][wr_addr] <= s_tdata;
                end
                if (wr_cg == CG_W'(CG - 1)) begin
                    wr_cg    <= '0;
                    wr_pixel <= wr_pixel + 32'sd1;
                end else begin
                    wr_cg <= wr_cg + 1'b1;
                end
            end

            if (emit) begin
                ovalid <= 1'b1;
                for (int copy = 0; copy < PAR; copy = copy + 1) begin
                    odata[copy*WORD +: WORD] <= fetched[copy];
                end

                if (rd_cg != CG_W'(CG - 1)) begin
                    rd_cg <= rd_cg + 1'b1;
                end else begin
                    rd_cg <= '0;
                    if (kxg != KWG_W'(KWG - 1)) begin
                        kxg <= kxg + 1'b1;
                    end else begin
                        kxg <= '0;
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
                                    oh         <= '0;
                                    frame_base <= frame_base + FRAME;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
