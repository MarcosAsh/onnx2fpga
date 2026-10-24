"""Fixed latency mode on the example MLP: the latency, the summary, the errors, simulation and the proofs."""

import json
import os
import pathlib
import shutil
import subprocess
import unittest

import numpy as np

from support import Fixtures, Hdl

import onnx2fpga as otf
from onnx2fpga.board import IcepiZero
from onnx2fpga.errors import BudgetError, ConfigError
from onnx2fpga.p6_emit.hardcaml import HardcamlGenerator

HARDCAML = Hdl.skip_reason("hardcaml")

# After the first clear, prediction_valid must equal features_valid delayed by exactly LATENCY cycles.
CHECK = """module valid_delay_check (
    input wire clock,
    input wire clear,
    input wire [%(width)d-1:0] features,
    input wire features_valid
);
    wire prediction_valid;
    otf_top dut (.clock(clock), .clear(clear), .features(features),
                 .features_valid(features_valid), .prediction(),
                 .prediction_valid(prediction_valid));
    reg [%(latency)d-1:0] history;
    reg cleared = 1'b0;
    always @(posedge clock) begin
        history <= clear ? %(latency)d'b0 : {history[%(latency)d-2:0], features_valid};
        cleared <= cleared | clear;
    end
    always @(*) if (cleared) assert (prediction_valid == history[%(latency)d-1]);
    always @(*) cover (cleared && prediction_valid);
endmodule
"""

# Every flip flop becomes a wire, which is sound here because each path crosses the same three stages.
WIRES = """module \\$dff (CLK, D, Q);
    parameter WIDTH = 1;
    parameter CLK_POLARITY = 1'b1;
    input CLK;
    input [WIDTH-1:0] D;
    output [WIDTH-1:0] Q;
    assign Q = D;
endmodule
"""

SBY = """[tasks]
prove
cover

[options]
prove: mode prove
prove: depth %(depth)d
cover: mode cover
cover: depth %(depth)d

[engines]
smtbmc yices

[script]
read -formal top.v
read -formal check.sv
prep -flatten -top valid_delay_check
opt_clean

[files]
top.v
check.sv
"""


def mlp(device="vu9p", arithmetic=None, **options):
    acc = otf.load(Fixtures.factory().mlp(), name="mlp")
    for name, kind in (arithmetic or {}).items():
        acc.layers[name].arithmetic(kind)
    x = np.random.default_rng(5).normal(0, 1, (8, 32)).astype(np.float32)
    folder = "fixed_mlp_%s%s" % (device, "".join("_%s_%s" % item for item in
                                                 sorted((arithmetic or {}).items())))
    acc.compile(device=device, mode="fixed", calibration=x,
                out_dir=Fixtures.build_dir(folder), **options)
    return acc


def equivalent(left, right, module):
    """ABC's cec on module in two Verilog files with registers made wires; True if it is the same logic."""
    yosys, abc = IcepiZero.tool("yosys"), IcepiZero.tool("yosys-abc")
    folder = pathlib.Path(left).parent.parent / "cec"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    (folder / "wires.v").write_text(WIRES)
    for name, verilog in (("left", left), ("right", right)):
        script = ("read_verilog %s; hierarchy -top %s; proc; flatten; techmap -map wires.v; "
                  "opt_clean; techmap; opt -fast; aigmap; opt_clean; write_aiger %s.aig"
                  % (verilog, module, name))
        subprocess.run([yosys, "-q", "-p", script], cwd=folder, check=True, timeout=900)
    completed = subprocess.run([abc, "-c", "cec left.aig right.aig"], cwd=folder,
                               capture_output=True, text=True, timeout=1800)
    if "Networks are equivalent" in completed.stdout:
        return True
    if "Verification failed" in completed.stdout:
        return False
    raise AssertionError(completed.stdout[-2000:] + completed.stderr[-2000:])


def prove(build, width, latency):
    """Runs sby on the valid path; True only if the property holds and prediction_valid can rise."""
    sby = IcepiZero.tool("sby")
    folder = build / ("formal_%d" % latency)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    shutil.copyfile(build / "rtl" / "top.v", folder / "top.v")
    (folder / "check.sv").write_text(CHECK % {"width": width, "latency": latency})
    (folder / "valid.sby").write_text(SBY % {"depth": latency + 2})
    path = os.path.dirname(sby) + os.pathsep + os.environ.get("PATH", "")
    completed = subprocess.run([sby, "-f", "valid.sby"], cwd=folder, capture_output=True,
                               text=True, timeout=900, env=dict(os.environ, PATH=path))
    return completed.returncode == 0, completed.stdout


class Errors(unittest.TestCase):
    def test_sv(self):
        with self.assertRaisesRegex(ConfigError, "needs hdl='hardcaml'"):
            mlp(hdl="sv")

    def test_target(self):
        with self.assertRaisesRegex(ConfigError, "does not take target_cycles"):
            mlp(target_cycles=64)

    def test_unknown_mode(self):
        acc = otf.load(Fixtures.factory().mlp())
        with self.assertRaisesRegex(ConfigError, "'stream' or 'fixed'"):
            acc.compile(mode="latency")

    @unittest.skipIf(HARDCAML, HARDCAML)
    def test_conv(self):
        acc = otf.load(Fixtures.factory().cnn())
        with self.assertRaisesRegex(ConfigError, r"cannot build conv0 \(Conv\+Relu\)"):
            acc.compile(mode="fixed", calibration=np.zeros((1, 1, 12, 12)))

    @unittest.skipIf(HARDCAML, HARDCAML)
    def test_too_big(self):
        with self.assertRaisesRegex(BudgetError, "largest layer fc0 has 2048"):
            mlp(device="icepi25")

    def test_arithmetic_name(self):
        acc = otf.load(Fixtures.factory().mlp())
        with self.assertRaisesRegex(ConfigError, "'constant', 'multiplier' or 'auto'"):
            acc.layers["fc0"].arithmetic("shifts")

    def test_arithmetic_without_weights(self):
        acc = otf.load(Fixtures.factory().cnn())
        with self.assertRaisesRegex(ConfigError, "flat0 .* has no weights"):
            acc.layers["flat0"].arithmetic("constant")

    def test_constant_streaming(self):
        acc = otf.load(Fixtures.factory().mlp())
        acc.layers["fc1"].arithmetic("constant")
        with self.assertRaisesRegex(ConfigError, "fc1 is folded in mode='stream'"):
            acc.compile(mode="stream", calibration=np.zeros((1, 32)))

    @unittest.skipIf(HARDCAML, HARDCAML)
    def test_fold_pin(self):
        acc = otf.load(Fixtures.factory().mlp())
        acc.layers["fc0"].fold(pe=2)
        with self.assertRaisesRegex(ConfigError, "pins on fc0"):
            acc.compile(mode="fixed", calibration=np.zeros((1, 32)))


@unittest.skipIf(HARDCAML, HARDCAML)
class Mlp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acc = mlp()

    def test_latency(self):
        self.assertEqual(self.acc.latency, 9)
        self.assertEqual([self.acc.layers[name].latency for name in ("fc0", "fc1", "fc2")],
                         [3, 3, 3])
        manifest = json.loads((self.acc.out_dir / "manifest.json").read_text())
        self.assertEqual((manifest["mode"], manifest["latency"]), ("fixed", 9))
        self.assertIn("fixed latency 9 cycles", repr(self.acc))

    def test_summary(self):
        lines = []
        self.acc.summary(print_fn=lines.append)
        text = "\n".join(lines)
        self.assertRegex(text, r"fc0 \(Gemm\+Relu\)\s+32 -> 64\s+constant\s+0\s+3")
        self.assertRegex(text, r"Total\s+0\s+9")
        self.assertIn("Constant arithmetic: 3232 multipliers saved", text)
        self.assertIn("Latency: 9 cycles, 30.0 ns at 300 MHz", text)

    def test_arithmetic(self):
        acc = mlp(arithmetic={"fc1": "multiplier"})
        lines = []
        acc.summary(print_fn=lines.append)
        self.assertRegex("\n".join(lines), r"fc1 \(Gemm\+Relu\)\s+64 -> 16\s+multiplier\s+1024")
        self.assertIn("2208 multipliers saved", "\n".join(lines))
        self.assertEqual(acc.latency, 9)
        design = json.loads((acc.out_dir / "design.json").read_text())
        self.assertEqual([node["params"]["CONSTANT"] for node in design["nodes"]], [1, 0, 1])
        manifest = json.loads((acc.out_dir / "manifest.json").read_text())
        self.assertEqual([node["arithmetic"] for node in manifest["nodes"]],
                         ["constant", "multiplier", "constant"])

    def test_auto(self):
        """The sim device has as many LUT as DSP, so the multipliers are the cheaper choice there."""
        acc = mlp(device="sim")
        self.assertIn("arithmetic multiplier", repr(acc.layers["fc0"]))

    def test_board(self):
        with self.assertRaisesRegex(ConfigError, "needs mode='stream'"):
            self.acc.build(board="icepi_zero")

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_simulate(self):
        report = self.acc.simulate(sweep=True)
        self.assertTrue(report.ok, report.first.output)
        self.assertEqual((report.frames, report.latency, report.cycles), (10000, 9, 10009))
        self.assertEqual(len(report.sweep), 6)

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        build = pathlib.Path(self.acc.out_dir)
        width = 32 * 8
        ok, log = prove(build, width, 9)
        self.assertTrue(ok, log[-3000:])
        self.assertIn("successful proof by k-induction", log)
        self.assertFalse(prove(build, width, 8)[0], "a latency of 8 must not prove")

    @unittest.skipIf(IcepiZero.tool("yosys-abc") is None, "yosys-abc not found")
    def test_equivalence(self):
        """fc2, a whole layer, is the same logic with constant arithmetic as with multipliers."""
        multiplier = mlp(arithmetic={name: "multiplier" for name in ("fc0", "fc1", "fc2")})
        built = [pathlib.Path(self.acc.out_dir) / "rtl" / "top.v",
                 pathlib.Path(multiplier.out_dir) / "rtl" / "top.v"]
        self.assertTrue(equivalent(*built, "otf_dense_2"))
        design = json.loads((self.acc.out_dir / "design.json").read_text())
        design["nodes"][2]["memories"]["weights"]["values"][5] ^= 4
        wrong = pathlib.Path(self.acc.out_dir) / "wrong"
        wrong.mkdir(exist_ok=True)
        (wrong / "design.json").write_text(json.dumps(design))
        (wrong / "top.v").write_text(HardcamlGenerator().verilog(wrong / "design.json"))
        self.assertFalse(equivalent(wrong / "top.v", built[1], "otf_dense_2"))


if __name__ == "__main__":
    unittest.main()
