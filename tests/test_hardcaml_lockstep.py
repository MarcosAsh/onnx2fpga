"""Every Hardcaml unit against the SystemVerilog module it ports, compared on every output port each cycle."""

import json
import re
import shutil
import unittest

import numpy as np

from support import ROOT, Fixtures, Hdl, run
from test_sliding_window import CONFIGS, FRAME_REGRESSION, make_unit

from onnx2fpga.compile import Compiler
from onnx2fpga.p6_emit.hardcaml import HardcamlGenerator
from onnx2fpga.p6_emit.memories import HexImage

CYCLES = 24000


def mvau(mw, mh, simd, pe, acc_bits=20, shift=20, out_min=-128, out_max=127,
         out_zp=0, mult_bits=18, act_bits=8, wgt_bits=8, out_bits=8):
    sf, nf = mw // simd, mh // pe
    params = {"MW": mw, "MH": mh, "SIMD": simd, "PE": pe, "ACT_BITS": act_bits,
              "WGT_BITS": wgt_bits, "ACC_BITS": acc_bits, "OUT_BITS": out_bits,
              "MULT_BITS": mult_bits, "SHIFT": shift, "OUT_MIN": out_min,
              "OUT_MAX": out_max, "OUT_ZP": out_zp}
    memories = {"weights": (wgt_bits, simd, pe * sf * nf),
                "bias": (acc_bits, 1, pe * nf),
                "mult": (mult_bits, 1, pe * nf)}
    return "otf_mvau", params, memories


def act(channels, pe, in_bits=20, shift=16, in_zp=0, out_zp=0, out_min=-128,
        out_max=127, mult_bits=18, out_bits=8):
    params = {"CHANNELS": channels, "PE": pe, "IN_BITS": in_bits,
              "OUT_BITS": out_bits, "MULT_BITS": mult_bits, "SHIFT": shift,
              "OUT_MIN": out_min, "OUT_MAX": out_max, "IN_ZP": in_zp,
              "OUT_ZP": out_zp}
    return "otf_act", params, {"mult": (mult_bits, 1, channels)}


def add(pe, left, right, shift, bias=0, out_zp=0, acc_bits=27, out_min=-128):
    return "otf_add", {"PE": pe, "IN_BITS": 8, "OUT_BITS": 8, "ACC_BITS": acc_bits,
                       "LEFT_MULT": left, "RIGHT_MULT": right, "SHIFT": shift,
                       "OUT_MIN": out_min, "OUT_MAX": 127, "BIAS": bias,
                       "OUT_ZP": out_zp}, {}


def swu(config, strategy):
    unit = make_unit(config, strategy)
    return unit.module, unit.sv_params(None), {}


CASES = (
    [("fifo_d%d_w%d" % (d, w), "otf_fifo", {"WIDTH": w, "DEPTH": d}, {})
     for d, w in ((1, 8), (2, 8), (3, 5), (5, 16), (16, 24))]
    + [("dup_f%d" % f, "otf_dup", {"WIDTH": w, "FANOUT": f}, {})
       for f, w in ((2, 8), (3, 16))]
    + [("dwc_%d_%d" % (i, o), "otf_dwc",
        {"IN_ELEMS": i, "OUT_ELEMS": o, "ELEM_BITS": 8, "TOTAL_ELEMS": 64}, {})
       for i, o in ((1, 4), (4, 1), (2, 2), (3, 9), (2, 16), (16, 2), (6, 2))]
    + [("pool_c%d_p%d_w%d" % (c, p, w), "otf_pool",
        {"CHANNELS": c, "PE": p, "WINDOW": w, "DATA_BITS": b, "DATA_MIN": lo}, {})
       for c, p, w, b, lo in ((4, 1, 4, 8, -128), (8, 2, 9, 8, -128),
                              (6, 3, 1, 8, -128), (4, 4, 2, 8, -128),
                              (3, 1, 3, 4, -8))]
    + [("act_" + name,) + act(*args, **kwargs) for name, args, kwargs in (
        ("folded", (8, 1), {"in_zp": 3, "out_zp": -5}),
        ("pe4", (8, 4), {"shift": 0, "in_zp": -7}),
        ("full", (6, 6), {"out_min": 0, "shift": 30}),
        ("narrow", (3, 1), {"in_bits": 8, "mult_bits": 4, "shift": 2, "out_bits": 4,
                            "out_min": -8, "out_max": 7}),
        # Shifts at and past the requant working width, ACC_BITS + MULT_BITS + 1, where rounding used to overflow.
        ("past_width", (4, 2), {"in_bits": 8, "mult_bits": 4, "shift": 15,
                                "out_zp": 3}),
        ("max_shift", (4, 4), {"in_bits": 8, "mult_bits": 4, "shift": 62,
                               "out_zp": -2}),
    )]
    + [("add_" + name,) + add(*args, **kwargs) for name, args, kwargs in (
        ("residual", (1, 98284, 51356, 17), {}),
        ("biased", (2, 3000, -2500, 10), {"bias": 12345, "out_zp": 9}),
        ("unshifted", (4, 1, 1, 0), {"acc_bits": 10, "out_min": 0}),
        ("wrapping", (1, 1500, 1500, 8), {"acc_bits": 12}),
        ("past_width", (2, 3, -2, 13), {"acc_bits": 10, "out_zp": 4}),
    )]
    + [("mvau_" + name,) + mvau(*args, **kwargs) for name, args, kwargs in (
        ("residual", (16, 16, 4, 1), {"acc_bits": 19, "shift": 24}),
        # Same parameters, different memories: must not share one module.
        ("residual_twin", (16, 16, 4, 1), {"acc_bits": 19, "shift": 24}),
        ("mlp0", (32, 64, 16, 2), {}),
        ("one_fold", (9, 4, 9, 1), {"acc_bits": 18}),
        ("serial", (8, 8, 1, 1), {}),
        ("odd_folds", (12, 10, 3, 5), {"out_zp": 7, "out_min": 0}),
        ("single_word", (4, 4, 4, 4), {"shift": 0, "acc_bits": 12}),
        ("wide_pe", (6, 9, 1, 9), {"shift": 1}),
        ("three_nf", (6, 9, 2, 3), {"shift": 40, "acc_bits": 14}),
        ("at_width", (8, 4, 2, 2), {"acc_bits": 12, "mult_bits": 5, "shift": 17}),
        ("past_width", (8, 4, 2, 2), {"acc_bits": 12, "mult_bits": 5, "shift": 18,
                                      "out_zp": -3}),
        ("tiny_mult", (8, 4, 4, 1), {"acc_bits": 19, "mult_bits": 14, "shift": 62,
                                     "out_zp": -5}),
        ("narrow", (4, 6, 2, 2), {"act_bits": 4, "wgt_bits": 3, "acc_bits": 9,
                                  "mult_bits": 5, "shift": 3, "out_bits": 4,
                                  "out_min": -8, "out_max": 7}),
    )]
    + [("swu_frame_" + c[0],) + swu(c, "frame")
       for c in CONFIGS if c[0] in FRAME_REGRESSION]
    + [("swu_line_" + c[0],) + swu(c, "line") for c in CONFIGS]
)


class Pair:
    """One SystemVerilog instance, its Hardcaml twin, their stimulus and the comparison, as a lockstep block."""

    def __init__(self, index, label, kind, params, memories, folder, rng):
        self.index = index
        self.label = label
        self.kind = kind
        self.params = params
        self.instance = "hc_%d" % index
        self.images = {}
        for key, (bits, lanes, count) in memories.items():
            values = rng.integers(0, 1 << bits, (count, lanes), dtype=np.int64)
            self.images[key] = (bits, values)
            image = HexImage.packed(values, bits)
            image.write(folder / ("%s_%s.hex" % (self.instance, key)))
        self.folder = folder

    def node(self):
        return {"name": self.label, "instance": self.instance, "kind": self.kind,
                "params": self.params,
                "memories": {key: {"elem_bits": bits, "lanes": int(values.shape[1]),
                                   "values": [int(v) for v in values.reshape(-1)]}
                             for key, (bits, values) in self.images.items()},
                "inputs": [], "outputs": []}

    def widths(self):
        p = self.params
        kind = self.kind
        if kind in ("otf_fifo", "otf_dup"):
            return p["WIDTH"], p["WIDTH"]
        if kind == "otf_dwc":
            return p["IN_ELEMS"] * p["ELEM_BITS"], p["OUT_ELEMS"] * p["ELEM_BITS"]
        if kind == "otf_pool":
            return p["PE"] * p["DATA_BITS"], p["PE"] * p["DATA_BITS"]
        if kind == "otf_act":
            return p["PE"] * p["IN_BITS"], p["PE"] * p["OUT_BITS"]
        if kind == "otf_add":
            return p["PE"] * p["IN_BITS"], p["PE"] * p["OUT_BITS"]
        if kind == "otf_mvau":
            return p["SIMD"] * p["ACT_BITS"], p["PE"] * p["OUT_BITS"]
        word = p["SIMD"] * p["DATA_BITS"]
        return word, word * p.get("PAR", 1)

    def render(self):
        i, (win, wout) = self.index, self.widths()
        fanout = self.params.get("FANOUT", 1)
        params = dict(self.params)
        files = {"weights": "WEIGHT_FILE", "bias": "BIAS_FILE", "mult": "MULT_FILE"}
        overrides = ["    .%s(%d)" % (k, v) for k, v in params.items()]
        overrides += ['    .%s("%s")' % (files[key], self.folder / (
            "%s_%s.hex" % (self.instance, key))) for key in self.images]
        lines = ["  // %s" % self.label]
        inputs = ["a", "b"] if self.kind == "otf_add" else ["s"]
        for side in inputs:
            lines.append("  logic [%d:0] p%d_%s_tdata;" % (win - 1, i, side))
            lines.append("  logic p%d_%s_tvalid;" % (i, side))
            lines.append("  logic sv%d_%s_tready, hc%d_%s_tready;" % (i, side, i, side))
        lines.append("  logic [%d:0] p%d_m_tready;" % (fanout - 1, i))
        lines.append("  logic [%d:0] sv%d_m_tdata, hc%d_m_tdata;"
                     % (fanout * wout - 1, i, i))
        lines.append("  logic [%d:0] sv%d_m_tvalid, hc%d_m_tvalid;" % (fanout - 1, i, i))
        lines.append("  always_ff @(posedge clk) begin")
        for side in inputs:
            words = (win + 31) // 32
            lines.append("    p%d_%s_tdata <= %d'({%s});" % (
                i, side, win, ", ".join(["$urandom"] * words)))
            lines.append("    p%d_%s_tvalid <= ($urandom %% 100) < duty_in;" % (i, side))
        for f in range(fanout):
            lines.append("    p%d_m_tready[%d] <= ($urandom %% 100) < duty_out;" % (i, f))
        lines.append("  end")

        sv = ["    .clk(clk)", "    .rst_n(rst_n)"]
        hc = ["    .clock(clk)", "    .clear(!rst_n)"]
        for side in inputs:
            sv += ["    .%s_tdata(p%d_%s_tdata)" % (side, i, side),
                   "    .%s_tvalid(p%d_%s_tvalid)" % (side, i, side),
                   "    .%s_tready(sv%d_%s_tready)" % (side, i, side)]
            hc += ["    .%s$data(p%d_%s_tdata)" % (side, i, side),
                   "    .%s$valid(p%d_%s_tvalid)" % (side, i, side),
                   "    .%s$ready(hc%d_%s_tready)" % (side, i, side)]
        sv += ["    .m_tdata(sv%d_m_tdata)" % i, "    .m_tvalid(sv%d_m_tvalid)" % i,
               "    .m_tready(p%d_m_tready)" % i]
        if self.kind == "otf_dup":
            for f in range(fanout):
                hc += ["    .m$data_%d(hc%d_m_tdata[%d +: %d])" % (f, i, f * wout, wout),
                       "    .m$valid_%d(hc%d_m_tvalid[%d])" % (f, i, f),
                       "    .m$ready_%d(p%d_m_tready[%d])" % (f, i, f)]
        else:
            hc += ["    .m$data(hc%d_m_tdata)" % i, "    .m$valid(hc%d_m_tvalid)" % i,
                   "    .m$ready(p%d_m_tready)" % i]
        lines.append("  %s #(\n%s\n  ) sv%d (\n%s\n  );"
                     % (self.kind, ",\n".join(overrides), i, ",\n".join(sv)))
        lines.append("  %s hc%d (\n%s\n  );" % (self.instance, i, ",\n".join(hc)))

        compared = ["m_tdata", "m_tvalid"] + ["%s_tready" % side for side in inputs]
        checks = " || ".join("sv%d_%s !== hc%d_%s" % (i, name, i, name)
                             for name in compared)
        lines.append("  always @(negedge clk) if (%s) begin" % checks)
        lines.append('    $display("MISMATCH %s cycle %%0d", cycle);' % self.label)
        for name in compared:
            lines.append('    $display("  %s sv %%h hc %%h", sv%d_%s, hc%d_%s);'
                         % (name, i, name, i, name))
        lines.append("    $fatal(1);")
        lines.append("  end")
        return "\n".join(lines)


WRAPPER = """`default_nettype none
/* verilator lint_off UNUSEDSIGNAL */
module lockstep;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  int cycle = 0;
  int duty_in, duty_out;
  int duties [5] = '{100, 50, 20, 90, 10};
  initial forever #5 clk = ~clk;
  always_ff @(posedge clk) begin
    cycle <= cycle + 1;
    // A second reset lands in the middle of traffic, not just at power up.
    rst_n <= !(cycle < 4 || (cycle >= %(reset_at)d && cycle < %(reset_at)d + 3));
    duty_in  <= duties[(cycle / 1500) %% 5];
    duty_out <= duties[(cycle / 1500 + (cycle / 7500)) %% 5];
    if (cycle == %(cycles)d) begin
      $display("PASS %(count)d pairs over %%0d cycles", cycle);
      $finish;
    end
  end
%(pairs)s
endmodule
"""


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
class HardcamlLockstepTest(unittest.TestCase):
    def test_every_unit_matches_its_systemverilog_cycle_for_cycle(self):
        folder = Fixtures.build_dir("hardcaml_lockstep")
        rng = np.random.default_rng(11)
        pairs = [Pair(index, *case, folder=folder, rng=rng)
                 for index, case in enumerate(CASES)]

        design = folder / "units.json"
        design.write_text(json.dumps({
            "format": 1, "top": "unused", "input": "", "output": "", "streams": [],
            "nodes": [pair.node() for pair in pairs]}, indent=1) + "\n")
        (folder / "hc.v").write_text(_units_verilog(design))
        (folder / "lockstep.sv").write_text(WRAPPER % {
            "cycles": CYCLES, "reset_at": CYCLES // 2, "count": len(pairs),
            "pairs": "\n\n".join(pair.render() for pair in pairs)})

        library = sorted(str(p) for p in (ROOT / "hardware" / "rtl").glob("otf_*.sv"))
        built = run(["verilator", "--binary", "--timing", "-j", "0",
                     "-Wno-fatal", "-Wno-lint", "-Wno-style", "--top-module", "lockstep",
                     "-o", "Vlockstep", str(folder / "lockstep.sv"),
                     str(folder / "hc.v")] + library, cwd=folder)
        self.assertEqual(built.returncode, 0, built.stderr[-4000:])
        result = run(["./obj_dir/Vlockstep"], cwd=folder)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-2000:])
        self.assertIn("PASS %d pairs" % len(pairs), result.stdout)


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
class HardcamlTopLevelTest(unittest.TestCase):
    """Whole designs on both backends under backpressure; identical first/last output cycles and beats mean cycle for cycle."""

    MODELS = {
        "mlp": ("mlp", Fixtures.MLP_SHAPE, 64),
        "cnn": ("cnn", Fixtures.CNN_SHAPE, 400),
        "cnn_pushed": ("cnn", Fixtures.CNN_SHAPE, 25),
        "residual": ("residual", (1, 16), 64),
    }
    BACKPRESSURE = [(100, 100), (50, 50), (20, 20), (100, 10), (10, 100), (5, 5)]

    def _reports(self, name, hdl):
        factory_name, shape, target = self.MODELS[name]
        build = Fixtures.build_dir("toplevel_%s_%s" % (name, hdl))
        Compiler(device="vu9p", target_cycles=target, hdl=hdl).compile(
            getattr(Fixtures.factory(), factory_name)(), Fixtures.samples(shape), build)
        built = run(["make", "-s"], cwd=build)
        self.assertEqual(built.returncode, 0, built.stderr[-4000:])
        reports = []
        for duty in self.BACKPRESSURE:
            for seed in (1, 2, 3):
                result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                              "--expected", "golden/expected.hex",
                              "--input-duty", str(duty[0]),
                              "--output-duty", str(duty[1]), "--seed", str(seed),
                              "--dump", "dumped.hex"], cwd=build)
                reports.append((duty, seed, result.returncode, result.stdout,
                                build.joinpath("dumped.hex").read_text()))
        return reports

    def test_both_backends_report_the_same_cycles(self):
        for name in self.MODELS:
            with self.subTest(model=name):
                sv, hardcaml = self._reports(name, "sv"), self._reports(name, "hardcaml")
                for left, right in zip(sv, hardcaml):
                    self.assertEqual(left[2], 0, left[3])
                    self.assertEqual(left, right, "duty %s seed %d" % left[:2])


def _units_verilog(design):
    """The Hardcaml modules share names with the library they port, so they are renamed apart before meeting in one model."""
    completed = run([str(HardcamlGenerator().executable), "units", str(design)])
    if completed.returncode != 0:
        raise AssertionError(completed.stderr[-4000:])
    return re.sub(r"\botf_", "hcport_otf_", completed.stdout)


if __name__ == "__main__":
    unittest.main()
