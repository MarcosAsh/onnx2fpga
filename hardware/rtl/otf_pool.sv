// Max reduction over the windows produced by otf_swu.
//
// The running maximum for each channel group is held locally and the result is
// emitted during the final pass over the window, so the unit costs exactly one
// cycle per input beat with no separate drain phase.
`default_nettype none

module otf_pool #(
    parameter int CHANNELS  = 8,
    parameter int PE        = 1,
    parameter int WINDOW    = 4,
    parameter int DATA_BITS = 8,
    parameter int DATA_MIN  = -128
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire  [PE*DATA_BITS-1:0] s_tdata,
    input  wire                     s_tvalid,
    output logic                    s_tready,
    output logic [PE*DATA_BITS-1:0] m_tdata,
    output logic                    m_tvalid,
    input  wire                     m_tready
);
    localparam int NF   = CHANNELS / PE;
    localparam int NF_W = (NF <= 1) ? 1 : $clog2(NF);
    localparam int WN_W = (WINDOW <= 1) ? 1 : $clog2(WINDOW);

    logic [PE*DATA_BITS-1:0] running [0:NF-1];
    logic [NF_W-1:0] nf;
    logic [WN_W-1:0] w;

    wire first_pass = (w == '0);
    wire last_pass  = (w == WN_W'(WINDOW - 1));

    logic [PE*DATA_BITS-1:0] merged;
    always_comb begin
        for (int q = 0; q < PE; q = q + 1) begin
            logic signed [DATA_BITS-1:0] incoming;
            logic signed [DATA_BITS-1:0] held;
            incoming = $signed(s_tdata[q*DATA_BITS +: DATA_BITS]);
            held = first_pass ? DATA_BITS'(DATA_MIN)
                              : $signed(running[nf][q*DATA_BITS +: DATA_BITS]);
            merged[q*DATA_BITS +: DATA_BITS] = (incoming > held) ? incoming : held;
        end
    end

    assign m_tdata  = merged;
    assign m_tvalid = s_tvalid & last_pass;
    assign s_tready = last_pass ? m_tready : 1'b1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            nf <= '0;
            w  <= '0;
        end else if (s_tvalid && s_tready) begin
            if (!last_pass) running[nf] <= merged;
            if (nf == NF_W'(NF - 1)) begin
                nf <= '0;
                w  <= last_pass ? '0 : w + 1'b1;
            end else begin
                nf <= nf + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
