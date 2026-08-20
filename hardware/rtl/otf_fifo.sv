// First-word-fall-through stream buffer.
`default_nettype none

module otf_fifo #(
    parameter int WIDTH = 8,
    parameter int DEPTH = 2
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire  [WIDTH-1:0] s_tdata,
    input  wire              s_tvalid,
    output logic             s_tready,
    output logic [WIDTH-1:0] m_tdata,
    output logic             m_tvalid,
    input  wire              m_tready
);
    localparam int PTR_BITS = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    localparam int CNT_BITS = $clog2(DEPTH + 1);

    logic [WIDTH-1:0] mem [0:DEPTH-1];
    logic [PTR_BITS-1:0] wptr, rptr;
    logic [CNT_BITS-1:0] count;

    wire push = s_tvalid & s_tready;
    wire pop  = m_tvalid & m_tready;

    assign s_tready = (count != CNT_BITS'(DEPTH));
    assign m_tvalid = (count != '0);
    assign m_tdata  = mem[rptr];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wptr  <= '0;
            rptr  <= '0;
            count <= '0;
        end else begin
            if (push) begin
                mem[wptr] <= s_tdata;
                wptr <= (DEPTH == 1) ? '0
                      : (wptr == PTR_BITS'(DEPTH - 1)) ? '0 : wptr + 1'b1;
            end
            if (pop) begin
                rptr <= (DEPTH == 1) ? '0
                      : (rptr == PTR_BITS'(DEPTH - 1)) ? '0 : rptr + 1'b1;
            end
            case ({push, pop})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule

`default_nettype wire
