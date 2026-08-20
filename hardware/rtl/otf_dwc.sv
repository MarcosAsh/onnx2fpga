// Repacks a stream from IN_ELEMS to OUT_ELEMS per beat. Element 0 occupies the
// least significant bits of every beat.
`default_nettype none

module otf_dwc #(
    parameter int IN_ELEMS   = 1,
    parameter int OUT_ELEMS  = 1,
    parameter int ELEM_BITS  = 8,
    parameter int TOTAL_ELEMS = 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire                            clk,    // unused when IN_ELEMS == OUT_ELEMS
    input  wire                            rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire  [IN_ELEMS*ELEM_BITS-1:0]  s_tdata,
    input  wire                            s_tvalid,
    output logic                           s_tready,
    output logic [OUT_ELEMS*ELEM_BITS-1:0] m_tdata,
    output logic                           m_tvalid,
    input  wire                            m_tready
);
    localparam int UP    = (OUT_ELEMS > IN_ELEMS) ? (OUT_ELEMS / IN_ELEMS) : 1;
    localparam int DOWN  = (IN_ELEMS > OUT_ELEMS) ? (IN_ELEMS / OUT_ELEMS) : 1;
    localparam int UP_W  = (UP <= 1) ? 1 : $clog2(UP);
    localparam int DN_W  = (DOWN <= 1) ? 1 : $clog2(DOWN);

    generate
        if (IN_ELEMS == OUT_ELEMS) begin : g_pass
            assign m_tdata  = s_tdata;
            assign m_tvalid = s_tvalid;
            assign s_tready = m_tready;
        end else if (OUT_ELEMS > IN_ELEMS) begin : g_up
            logic [OUT_ELEMS*ELEM_BITS-1:0] shreg;
            logic [UP_W-1:0] fill;
            logic full;

            assign s_tready = !full;
            assign m_tvalid = full;
            assign m_tdata  = shreg;

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    fill <= '0;
                    full <= 1'b0;
                end else begin
                    if (full && m_tready) full <= 1'b0;
                    if (s_tvalid && !full) begin
                        shreg[fill*IN_ELEMS*ELEM_BITS +: IN_ELEMS*ELEM_BITS] <= s_tdata;
                        if (fill == UP_W'(UP - 1)) begin
                            fill <= '0;
                            full <= 1'b1;
                        end else begin
                            fill <= fill + 1'b1;
                        end
                    end
                end
            end
        end else begin : g_down
            logic [IN_ELEMS*ELEM_BITS-1:0] holding;
            logic [DN_W-1:0] slot;
            logic loaded;

            assign s_tready = !loaded;
            assign m_tvalid = loaded;
            assign m_tdata  = holding[slot*OUT_ELEMS*ELEM_BITS +: OUT_ELEMS*ELEM_BITS];

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    slot   <= '0;
                    loaded <= 1'b0;
                end else if (!loaded) begin
                    if (s_tvalid) begin
                        holding <= s_tdata;
                        loaded  <= 1'b1;
                        slot    <= '0;
                    end
                end else if (m_tready) begin
                    if (slot == DN_W'(DOWN - 1)) begin
                        loaded <= 1'b0;
                        slot   <= '0;
                    end else begin
                        slot <= slot + 1'b1;
                    end
                end
            end
        end
    endgenerate
endmodule

`default_nettype wire
