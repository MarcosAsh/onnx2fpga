"""Fixed latency mode on the example MLP: the latency, the summary, the errors, simulation and the proof."""

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


def mlp(device="vu9p", **options):
    acc = otf.load(Fixtures.factory().mlp(), name="mlp")
    x = np.random.default_rng(5).normal(0, 1, (8, 32)).astype(np.float32)
    acc.compile(device=device, mode="fixed", calibration=x,
                out_dir=Fixtures.build_dir("fixed_mlp_%s" % device), **options)
    return acc


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
        with self.assertRaisesRegex(ConfigError, "Hardcaml units only"):
            mlp(hdl="sv")

    def test_target(self):
        with self.assertRaisesRegex(ConfigError, "target_cycles does not apply"):
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
        with self.assertRaisesRegex(BudgetError, "largest layer fc0 alone has 2048"):
            mlp(device="icepi25")

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
        self.assertIn("fc0 (Gemm+Relu)", text)
        self.assertRegex(text, r"Total\s+3232\s+9")
        self.assertIn("Latency: 9 cycles, 30.0 ns at 300 MHz", text)

    def test_board(self):
        with self.assertRaisesRegex(ConfigError, "does not speak yet"):
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


if __name__ == "__main__":
    unittest.main()
