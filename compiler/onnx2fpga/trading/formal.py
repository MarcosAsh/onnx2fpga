"""SymbiYosys proofs on the generated Verilog of the framer step and the book side update."""

import os
import pathlib
import shutil
import subprocess

# A byte at a time MoldUDP64 framer over the same state as moldudp64.ml, which is the spec.
SERIAL = """
module serial (
    input [2:0] phase, input [15:0] skip, input [15:0] length, input [7:0] high,
    input [15:0] count, input [31:0] first, input [319:0] buffer,
    input [63:0] data, input [7:0] keep, input valid, input last, input [31:0] now,
    output reg [2:0] n_phase, output reg [15:0] n_skip, output reg [15:0] n_length,
    output reg [7:0] n_high, output reg [15:0] n_count, output reg [31:0] n_first,
    output reg [319:0] n_buffer, output reg emit, output reg [15:0] e_length,
    output reg [319:0] e_bytes, output reg [31:0] e_first, output reg [31:0] e_last
);
    localparam HEADER = 0, LENGTH = 1, LENGTH_LOW = 2, BODY = 3, DROP = 4;
    integer i;
    reg [7:0] b;
    reg [15:0] off, field;
    always @* begin
        n_phase = phase; n_skip = skip; n_length = length; n_high = high;
        n_count = count; n_first = first; n_buffer = buffer;
        emit = 0; e_length = 0; e_bytes = 0; e_first = 0; e_last = 0;
        for (i = 0; i < 8; i = i + 1) begin
            if (valid && keep[i]) begin
                b = data[8*i +: 8];
                case (n_phase)
                    HEADER: begin
                        if (n_skip == 2) n_high = b;
                        if (n_skip == 1) begin
                            field = {n_high, b};
                            n_count = field == 16'hffff ? 16'd0 : field;
                            n_phase = n_count == 0 ? DROP : LENGTH;
                        end
                        n_skip = n_skip - 1;
                    end
                    LENGTH: begin
                        n_high = b;
                        n_phase = LENGTH_LOW;
                    end
                    LENGTH_LOW: begin
                        field = {n_high, b};
                        if (field < 12) n_phase = DROP;
                        else begin
                            n_phase = BODY; n_skip = field; n_length = field;
                            n_count = n_count - 1;
                        end
                    end
                    BODY: begin
                        off = n_length - n_skip;
                        if (off < 40) n_buffer[8*off +: 8] = b;
                        if (off == 0) n_first = now;
                        if (n_skip == 1) begin
                            emit = 1; e_length = n_length; e_bytes = n_buffer;
                            e_first = n_first; e_last = now;
                            n_phase = n_count == 0 ? DROP : LENGTH;
                        end
                        n_skip = n_skip - 1;
                    end
                    default: ;
                endcase
            end
        end
        if (valid && last) begin
            n_phase = HEADER; n_skip = 20; n_length = 0; n_high = 0; n_count = 0; n_first = 0;
        end
    end
endmodule

module check (
    input [2:0] phase, input [15:0] skip, input [15:0] length, input [7:0] high,
    input [15:0] count, input [31:0] first, input [319:0] buffer,
    input [63:0] data, input [7:0] keep, input valid, input last, input [31:0] now
);
    localparam HEADER = 0, BODY = 3;
    wire [2:0] p_phase, s_phase;
    wire [15:0] p_skip, s_skip, p_length, s_length, p_count, s_count, p_elen, s_elen;
    wire [7:0] p_high, s_high;
    wire [31:0] p_first, s_first, p_efirst, s_efirst, p_elast, s_elast;
    wire [319:0] p_buffer, s_buffer, p_bytes, s_bytes;
    wire p_emit, s_emit;
    dut parallel (
        .state$phase(phase), .state$skip(skip), .state$length(length), .state$high(high),
        .state$count(count), .state$first(first), .state$buffer(buffer),
        .beat$data(data), .beat$keep(keep), .beat$valid(valid), .beat$last(last), .now(now),
        .next$phase(p_phase), .next$skip(p_skip), .next$length(p_length),
        .next$high(p_high), .next$count(p_count), .next$first(p_first),
        .next$buffer(p_buffer), .message$valid(p_emit), .message$length(p_elen),
        .message$bytes(p_bytes), .message$t_first(p_efirst), .message$t_last(p_elast));
    serial reference (
        phase, skip, length, high, count, first, buffer, data, keep, valid, last, now,
        s_phase, s_skip, s_length, s_high, s_count, s_first, s_buffer,
        s_emit, s_elen, s_bytes, s_efirst, s_elast);
    function invariant(input [2:0] f_phase, input [15:0] f_skip, input [15:0] f_length);
        invariant = f_phase <= 4
            && (f_phase != HEADER || (f_skip >= 1 && f_skip <= 20))
            && (f_phase != BODY || (f_length >= 12 && f_skip >= 1 && f_skip <= f_length));
    endfunction
    wire beat_ok = !valid
        || (keep != 0 && (keep & (keep + 8'd1)) == 0 && (last || keep == 8'hff));
    always @* begin
        assume (invariant(phase, skip, length) && beat_ok);
        assert (invariant(p_phase, p_skip, p_length));
        assert ({p_phase, p_skip, p_length, p_high, p_count, p_first, p_buffer}
                == {s_phase, s_skip, s_length, s_high, s_count, s_first, s_buffer});
        assert (p_emit == s_emit);
        if (s_emit) assert ({p_elen, p_bytes, p_efirst, p_elast}
                            == {s_elen, s_bytes, s_efirst, s_elast});
        // A message of 15 bytes or more never completes on the beat after another.
        if (p_emit && !last) assert (!(p_phase == BODY && p_skip <= 8 && p_length >= 15));
        cover (p_emit && p_phase == BODY);
    end
endmodule
"""

BOOK = """
module check (
    input [%(bits)d-1:0] levels, input ask, input [63:0] opposite,
    input remove_valid, input [31:0] remove_price, input [31:0] remove_shares,
    input insert_valid, input [31:0] insert_price, input [31:0] insert_shares
);
    localparam N = %(levels)d;
    wire [%(bits)d-1:0] updated;
    wire accepted;
    dut side (
        .levels(levels), .ask(ask), .opposite(opposite),
        .remove$valid(remove_valid), .remove$price(remove_price),
        .remove$shares(remove_shares), .insert$valid(insert_valid),
        .insert$price(insert_price), .insert$shares(insert_shares),
        .updated(updated), .accepted(accepted));
    function sound(input [%(bits)d-1:0] l);
        integer i;
        reg [31:0] price, shares, next_price, next_shares;
        begin
            sound = 1;
            for (i = 0; i < N; i = i + 1) begin
                price = l[64*i +: 32];
                shares = l[64*i + 32 +: 32];
                if (shares == 0 && price != 0) sound = 0;
                if (i + 1 < N) begin
                    next_price = l[64*(i+1) +: 32];
                    next_shares = l[64*(i+1) + 32 +: 32];
                    if (next_shares != 0 && shares == 0) sound = 0;
                    if (next_shares != 0 && (ask ? price >= next_price : price <= next_price))
                        sound = 0;
                end
            end
            if (l[63:32] != 0 && opposite[63:32] != 0
                && (ask ? l[31:0] <= opposite[31:0] : l[31:0] >= opposite[31:0])) sound = 0;
        end
    endfunction
    always @* begin
        assume (sound(levels));
        assert (sound(updated));
        if (accepted) assert (insert_valid && insert_shares != 0);
        cover (accepted && remove_valid && updated[64*N-1 -: 32] != 0);
    end
endmodule
"""

SBY = """[tasks]
prove
cover

[options]
prove: mode prove
cover: mode cover
depth 1

[engines]
smtbmc %(solver)s

[script]
read -formal dut.v
read -formal -sv check.sv
prep -flatten -top check

[files]
dut.v
check.sv
"""


def prove(folder, dut_verilog, check, sby, solver="bitwuzla", timeout=3600):
    """Runs the prove and cover tasks; returns (passed, log)."""
    folder = pathlib.Path(folder)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "dut.v").write_text(dut_verilog)
    (folder / "check.sv").write_text(check)
    (folder / "check.sby").write_text(SBY % {"solver": solver})
    path = os.path.dirname(sby) + os.pathsep + os.environ.get("PATH", "")
    completed = subprocess.run([sby, "-f", "check.sby"], cwd=folder, capture_output=True,
                               text=True, timeout=timeout, env=dict(os.environ, PATH=path))
    return completed.returncode == 0, completed.stdout + completed.stderr


def book_check(levels):
    return BOOK % {"levels": levels, "bits": 64 * levels}
