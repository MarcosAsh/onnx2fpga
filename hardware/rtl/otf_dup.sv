// Fork one stream to FANOUT consumers. Branches may accept in different
// cycles; the sent mask holds the beat until every branch has taken it.
`default_nettype none

module otf_dup #(
    parameter int WIDTH  = 8,
    parameter int FANOUT = 2
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire  [WIDTH-1:0]        s_tdata,
    input  wire                     s_tvalid,
    output logic                    s_tready,
    output logic [FANOUT*WIDTH-1:0] m_tdata,
    output logic [FANOUT-1:0]       m_tvalid,
    input  wire  [FANOUT-1:0]       m_tready
);
    logic [FANOUT-1:0] sent;

    genvar f;
    generate
        for (f = 0; f < FANOUT; f = f + 1) begin : g_branch
            assign m_tdata[f*WIDTH +: WIDTH] = s_tdata;
            assign m_tvalid[f] = s_tvalid & ~sent[f];
        end
    endgenerate

    assign s_tready = &(sent | m_tready);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sent <= '0;
        end else if (s_tvalid) begin
            sent <= s_tready ? '0 : (sent | (m_tvalid & m_tready));
        end
    end
endmodule

`default_nettype wire
