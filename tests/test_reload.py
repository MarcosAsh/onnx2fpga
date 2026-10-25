"""Reloadable weights on the example MLP: the errors, bank swaps in simulation, load_weights through the register file, and the tag proof."""

import json
import os
import shutil
import subprocess
import unittest

import numpy as np

from support import ROOT, Fixtures, Hdl

import onnx2fpga as otf
from onnx2fpga.board import KV260, OFFSETS, BoardError, IcepiZero
from onnx2fpga.errors import ConfigError, NotDeployedError
from onnx2fpga.simulate import SimulatedRegisters

HARDCAML = Hdl.skip_reason("hardcaml")
VERILATOR = None if shutil.which("verilator") else "verilator not on PATH"
SBY = IcepiZero.tool("sby")
X = np.random.default_rng(5).normal(0, 1, (8, 32)).astype(np.float32)

# Once LATENCY cycles have passed, the output's bank tag is the bank active LATENCY cycles before.
CHECK = """module bank_tag_check (
    input wire clock,
    input wire clear,
    input wire [255:0] features,
    input wire features_valid,
    input wire weights_write,
    input wire [12:0] weights_address,
    input wire [31:0] weights_data,
    input wire weights_commit
);
    wire prediction_bank, weights_bank;
    otf_top dut (.clock(clock), .clear(clear), .features(features),
                 .features_valid(features_valid), .weights_write(weights_write),
                 .weights_address(weights_address), .weights_data(weights_data),
                 .weights_commit(weights_commit), .prediction(), .prediction_valid(),
                 .prediction_bank(prediction_bank), .weights_bank(weights_bank),
                 .weights_version(), .weights_crc(), .weights_idle_crc());
    reg [%(latency)d-1:0] history;
    reg [7:0] age = 8'd0;
    always @(posedge clock) begin
        history <= {history[%(latency)d-2:0], weights_bank};
        if (age != 8'd255) age <= age + 8'd1;
    end
    always @(*) if (age >= %(latency)d) assert (prediction_bank == history[%(latency)d-1]);
    always @(*) cover (age >= %(latency)d && prediction_bank != history[0]);
endmodule
"""

SBY_FILE = """[tasks]
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
prep -flatten -top bank_tag_check
opt_clean

[files]
top.v
check.sv
"""


def mlp(widths=(32, 64, 16, 10), seed=7, last=1.0):
    """The example MLP's shape with other weights, the last layer's scaled by `last`."""
    factory = Fixtures.factory(seed)
    dense = factory.dense
    factory.dense = lambda fan_in, fan_out: dense(fan_in, fan_out) * (
        last if fan_out == widths[-1] else 1.0)
    return factory.mlp(widths)


class Errors(unittest.TestCase):
    def test_weights_name(self):
        with self.assertRaisesRegex(ConfigError, "'frozen' or 'reloadable', not 'hot'"):
            otf.load(mlp()).compile(mode="fixed", weights="hot")

    def test_stream(self):
        with self.assertRaisesRegex(ConfigError, "weights='reloadable' needs mode='fixed'"):
            otf.load(mlp()).compile(weights="reloadable", calibration=X)

    def test_constant(self):
        acc = otf.load(mlp())
        acc.layers["fc1"].arithmetic("constant")
        with self.assertRaisesRegex(ConfigError, r"^fc1 is set to arithmetic\('constant'\), "
                                                 r"which needs weights='frozen'$"):
            acc.compile(mode="fixed", weights="reloadable", calibration=X)

    def test_trees(self):
        acc = otf.load(ROOT / "tests" / "fixtures" / "xgb_regressor.onnx")
        with self.assertRaisesRegex(ConfigError, "does not support tree ensembles"):
            acc.compile(mode="fixed", weights="reloadable")


@unittest.skipIf(HARDCAML, HARDCAML)
class Reloadable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acc = otf.load(mlp(), name="mlp")
        cls.acc.compile(device="vu9p", mode="fixed", weights="reloadable", calibration=X,
                        out_dir=Fixtures.build_dir("reload_mlp"))
        cls.manifest = json.loads((cls.acc.out_dir / "manifest.json").read_text())

    def test_compile(self):
        self.assertEqual(self.acc.latency, 9)
        self.assertIn("arithmetic multiplier", repr(self.acc.layers["fc0"]))
        weights = self.manifest["weights"]
        self.assertEqual((weights["mode"], weights["words"]), ("reloadable", 3412))
        self.assertEqual([layer["bias"]["bits"] for layer in weights["layers"]], [22, 23, 21])
        self.assertEqual(weights["layers"][1]["mult"],
                         {"offset": 2176 + 1024 + 16, "count": 16, "bits": 18})
        design = json.loads((self.acc.out_dir / "design.json").read_text())
        self.assertEqual([node["params"]["RELOADABLE"] for node in design["nodes"]], [1, 1, 1])
        lines = []
        self.acc.summary(print_fn=lines.append)
        self.assertIn("Weights: reloadable, two banks of 3412 words, CRC 0x%08x"
                      % weights["crc"], "\n".join(lines))

    def test_registers(self):
        KV260(self.acc.out_dir).write()
        manifest = json.loads((self.acc.out_dir / "manifest.json").read_text())
        registers = manifest["registers"]["map"]
        self.assertEqual((registers["weights"]["access"], registers["weights"]["words"]),
                         ("w", 3412))
        self.assertEqual(registers["weights_words"]["value"], 3412)
        self.assertEqual([registers[name]["offset"] for name in
                          ("weights_commit", "weights_bank", "weights_version", "weights_crc",
                           "weights_idle_crc")], [0x24, 0x28, 0x2C, 0x30, 0x34])
        shell = (self.acc.out_dir / "kv260" / "kria_shell.v").read_text()
        self.assertIn("weights_idle_crc", shell)

    def test_mismatch(self):
        cases = [
            (mlp((32, 48, 16, 10)), r"^fc0 has weights shaped \(32, 64\) in the compiled "
                                    r"model but \(32, 48\) in the new one$"),
            (mlp((32, 64, 10)), "^the compiled model has 3 layers but the new one has 2$"),
            (mlp((16, 64, 16, 10)), r"^the input is shaped \(1, 32\) in the compiled model "
                                    r"but \(1, 16\) in the new one$"),
            (mlp(last=1000.0), r"^fc2 needs a rescale multiplier of \d+, past the "
                                         r"18 bits the design holds$"),
        ]
        for model, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ConfigError, message):
                    self.acc.load_weights(model)

    def test_frozen(self):
        acc = otf.load(mlp((16, 8)), name="mlp")
        acc.compile(device="vu9p", mode="fixed", calibration=X[:, :16],
                    out_dir=Fixtures.build_dir("reload_frozen"))
        self.assertEqual(json.loads((acc.out_dir / "manifest.json").read_text())["weights"],
                         {"mode": "frozen"})
        with self.assertRaisesRegex(ConfigError, "has frozen weights"):
            acc.load_weights(mlp((16, 8), seed=8))

    def test_no_registers(self):
        with self.assertRaisesRegex(NotDeployedError, "needs registers="):
            self.acc.load_weights(mlp(seed=8))

    @unittest.skipIf(VERILATOR, VERILATOR)
    def test_swap(self):
        """One commit halfway: inputs up to cycle 5000 use the compiled bank, the rest the new one."""
        report = self.acc.simulate(reload=mlp(seed=8), commits=[5000])
        self.assertTrue(report.ok, report.first.output)
        self.assertIn("answers by bank  5001 4999", report.first.output)
        golden = self.acc.out_dir / "golden"
        old, new = ((golden / name).read_text().split()
                    for name in ("x_expected.hex", "x_expected_new.hex"))
        self.assertGreater(np.mean([a != b for a, b in zip(old, new)]), 0.95)
        # Told the banks the wrong way round, the harness must fail.
        swapped = subprocess.run(
            ["./obj_dir/Votf_top", "--input", "golden/x_input.hex",
             "--expected", "golden/x_expected_new.hex",
             "--expected-new", "golden/x_expected.hex", "--latency", "9",
             "--weights", "golden/x_weights.hex", "--commits", "5000"],
            cwd=self.acc.out_dir, capture_output=True, text=True, timeout=600)
        self.assertEqual(swapped.returncode, 1, swapped.stdout)

    @unittest.skipIf(VERILATOR, VERILATOR)
    def test_random_commits(self):
        rng = np.random.default_rng(1)
        commits = sorted(set(rng.integers(3412, 10000, 40).tolist()) | {6000, 6001})
        report = self.acc.simulate(reload=mlp(seed=9), commits=commits, sweep=True)
        self.assertTrue(report.ok, [run.output for run in [report.first] + report.sweep])
        self.assertIn("commits          %d" % len(commits), report.first.output)
        with self.assertRaisesRegex(ConfigError, "commit at cycle 100 comes before"):
            self.acc.simulate(reload=mlp(seed=9), commits=[100])

    @unittest.skipIf(VERILATOR, VERILATOR)
    def test_load_weights(self):
        """Through kria_regs under Verilator: load, predict over AXI4-Lite, refuse a lost write, load back."""
        def infer(registers, codes):
            lanes = [int(v) & 0xFF for v in codes.reshape(-1)]
            for word in range(len(lanes) // 4):
                registers.write(OFFSETS["features"] + 4 * word,
                                sum(lanes[4 * word + k] << (8 * k) for k in range(4)))
            registers.write(OFFSETS["control"], 1)
            while not registers.read(OFFSETS["status"]) & 1:
                pass
            words = [registers.read(OFFSETS["prediction"] + 4 * k) for k in range(3)]
            lanes = [(words[k // 4] >> (8 * (k % 4))) & 0xFF for k in range(10)]
            return [lane - 256 if lane > 127 else lane for lane in lanes]

        class Lossy:
            def __init__(self, registers):
                self.read, self.registers = registers.read, registers

            def write(self, offset, value):
                if offset != OFFSETS["weights"] + 40:
                    self.registers.write(offset, value)

        compiled = self.acc.predict(X, quantized=True)
        with SimulatedRegisters(self.acc.out_dir) as registers:
            self.acc.load_weights(mlp(seed=8), registers=registers)
            loaded = self.acc.predict(X, quantized=True)
            self.assertGreater(np.mean(loaded != compiled), 0.8)
            for k in range(4):
                self.assertEqual(infer(registers, self.acc._quantize({"x": X[k:k + 1]})),
                                 loaded[k].tolist())
            with self.assertRaisesRegex(BoardError, "so it was not committed"):
                self.acc.load_weights(mlp(seed=10), registers=Lossy(registers))
            self.assertEqual(registers.read(OFFSETS["weights_version"]), 1)
            self.acc.load_weights(mlp(), registers=registers)
            state = [registers.read(OFFSETS[name])
                     for name in ("weights_bank", "weights_version", "weights_crc")]
            self.assertEqual(state, [0, 2, self.manifest["weights"]["crc"]])
            np.testing.assert_array_equal(self.acc.predict(X, quantized=True), compiled)

    @unittest.skipIf(SBY is None, "sby not found")
    def test_proof(self):
        ok, log = prove_bank(self.acc.out_dir, 9)
        self.assertTrue(ok, log[-3000:])
        self.assertIn("successful proof by k-induction", log)
        self.assertFalse(prove_bank(self.acc.out_dir, 8)[0], "a latency of 8 must not prove")


@unittest.skipIf(HARDCAML, HARDCAML)
class Pipelined(unittest.TestCase):
    """Reloadable at the trading clock: every stage the tag must follow."""

    @classmethod
    def setUpClass(cls):
        cls.acc = otf.load(mlp(), name="mlp")
        cls.acc.compile(device="vu9p", mode="fixed", weights="reloadable", calibration=X,
                        clock_mhz=322.265625, out_dir=Fixtures.build_dir("reload_mlp_322"))

    def test_stages(self):
        units = self.acc.result.hardware_graph.topological_order()
        self.assertTrue(any(unit.stages["product"] > 0 for unit in units))
        self.assertEqual(self.acc.latency, sum(unit.latency for unit in units))

    @unittest.skipIf(VERILATOR, VERILATOR)
    def test_swap(self):
        report = self.acc.simulate(reload=mlp(seed=8), commits=[5000])
        self.assertTrue(report.ok, report.first.output)
        self.assertIn("answers by bank  5001 4999", report.first.output)

    @unittest.skipIf(SBY is None, "sby not found")
    def test_proof(self):
        latency = self.acc.latency
        ok, log = prove_bank(self.acc.out_dir, latency)
        self.assertTrue(ok, log[-3000:])
        self.assertFalse(prove_bank(self.acc.out_dir, latency - 1)[0])


def prove_bank(out_dir, latency):
    """Runs sby on the bank tag; True only if the property holds and the tag can change."""
    folder = out_dir / ("formal_bank_%d" % latency)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    shutil.copyfile(out_dir / "rtl" / "top.v", folder / "top.v")
    (folder / "check.sv").write_text(CHECK % {"latency": latency})
    (folder / "bank.sby").write_text(SBY_FILE % {"depth": latency + 2})
    path = os.path.dirname(SBY) + os.pathsep + os.environ.get("PATH", "")
    completed = subprocess.run([SBY, "-f", "bank.sby"], cwd=folder, capture_output=True,
                               text=True, timeout=1800, env=dict(os.environ, PATH=path))
    return completed.returncode == 0, completed.stdout


if __name__ == "__main__":
    unittest.main()
