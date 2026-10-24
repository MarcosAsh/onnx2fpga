"""The synthesis driver without synthesis; the TCL itself has never run under a real Vivado."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import unittest

from support import Fixtures, ROOT

from onnx2fpga.compile import Compiler
from onnx2fpga.measure import (Characterizer, Comparison, SynthesisResult,
                               Subject, VivadoRunner, calibration_samples, render)
from onnx2fpga.p1_ingest import OnnxModel
from onnx2fpga.p3_ops.hardware_ops import MatVecUnit
from onnx2fpga.targets.device import Device
from onnx2fpga.targets.resources import Resources

CNN = ROOT / "examples" / "models" / "cnn.onnx"
FAKE_VIVADO = ROOT / "tests" / "fake_vivado"

# What the TCL writes, in the shape it writes it.
SUMMARY = {
    "top": "otf_mvau", "part": "xc7z020-clg400-1", "stage": "synth",
    "generics": "MW=9 MH=4 SIMD=3 PE=4", "period_ns": 5.0, "wns_ns": 0.842,
    "fmax_mhz": 240.15, "lut": 412, "ff": 288, "dsp": 6,
    "ramb36": 1, "ramb18": 2, "bram36": 2.0, "uram": 0,
}


def synth(*arguments, timeout=900):
    return subprocess.run([sys.executable, "-m", "onnx2fpga", "synth", str(CNN),
                           "--device", "z7020"] + list(arguments),
                          capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "compiler")})


class Results(unittest.TestCase):
    def test_summary(self):
        path = Fixtures.build_dir("characterize") / "summary.json"
        path.write_text(json.dumps(SUMMARY))
        result = SynthesisResult.load(path)
        self.assertEqual((result.measured.lut, result.measured.dsp,
                          result.measured.bram36), (412, 6, 2.0))
        self.assertEqual(result.fmax_mhz, 240.15)
        self.assertEqual(result.stage, "synth")

    def test_comparison(self):
        predicted = Resources(lut=500, ff=288, dsp=6, bram36=1.0, uram=0)
        comparison = Comparison(Subject("u", "otf_mvau", {"MW": 9}, predicted, 100),
                                SynthesisResult(SUMMARY))
        self.assertAlmostEqual(comparison.error("lut"), 500 / 412 - 1.0)
        self.assertAlmostEqual(comparison.error("dsp"), 0.0)
        self.assertAlmostEqual(comparison.error("bram36"), -0.5)
        self.assertIsNone(comparison.error("uram"))
        self.assertIn("otf_mvau", comparison.row())
        self.assertIn("slowest unit runs at 240.2 MHz", render([comparison]))


class Subjects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = OnnxModel.load(CNN)
        cls.build = Fixtures.build_dir("characterize_subjects")
        cls.result = Compiler(device="z7020", target_cycles=400).compile(
            model, calibration_samples(model), cls.build)
        cls.driver = Characterizer(VivadoRunner("xc7z020-clg400-1"), Device.get("z7020"))

    def test_subjects(self):
        subjects = self.driver.subjects(self.result.hardware_graph)
        modules = {s.module for s in subjects}
        self.assertLessEqual({"otf_mvau", "otf_swu_line"}, modules)
        keys = [(s.module, s.generic_arguments()) for s in subjects]
        self.assertEqual(len(keys), len(set(keys)))

    def test_memory_files(self):
        graph = self.result.hardware_graph
        unit = next(n for n in graph.topological_order() if isinstance(n, MatVecUnit))
        subject = Subject.from_node(unit, graph, Device.get("z7020"))
        self.assertIn("WEIGHT_FILE=mem/", subject.generic_arguments())

    def test_sources_are_relative(self):
        sources = self.driver.sources(self.build)
        self.assertTrue(sources, "no RTL found in %s" % self.build)
        for source in sources:
            self.assertFalse(pathlib.Path(source).is_absolute(), source)


class Runner(unittest.TestCase):
    def test_environment(self):
        runner = VivadoRunner("xc7z020-clg400-1", period_ns=4.0, implement=True)
        subject = Subject("u", "otf_fifo", {"WIDTH": 24, "DEPTH": 2}, None, 0)
        environment = runner.environment(subject, ["rtl/otf_fifo.sv"], "synth/u")
        self.assertEqual(environment["OTF_TOP"], "otf_fifo")
        self.assertEqual(environment["OTF_PERIOD"], "4.0")
        self.assertEqual(environment["OTF_IMPL"], "1")
        self.assertEqual(environment["OTF_GENERICS"], "DEPTH=2 WIDTH=24")
        self.assertEqual(environment["OTF_RTL"], "rtl/otf_fifo.sv")
        self.assertTrue(runner.script.exists(), runner.script)
        self.assertIn(str(runner.script), " ".join(runner.command()))

    def test_empty_generics_are_dropped(self):
        """An empty file name would make Vivado read a file called nothing, so those parameters keep their default."""
        subject = Subject("u", "otf_mvau",
                          {"MW": 9, "WEIGHT_FILE": "", "BIAS_FILE": "mem/b.hex"}, None, 0)
        self.assertNotIn("WEIGHT_FILE", subject.generic_arguments())


@unittest.skipIf(shutil.which("tclsh") is None, "tclsh not installed")
class EndToEnd(unittest.TestCase):
    """The whole measurement path against a Vivado stand in; the printed numbers are canned and mean nothing."""

    def test_measure(self):
        self.assertTrue(os.access(FAKE_VIVADO, os.X_OK), "%s is not executable" % FAKE_VIVADO)
        build = Fixtures.build_dir("characterize_e2e")
        completed = synth("--vivado", str(FAKE_VIVADO), "--out", str(build),
                          "--measurements", str(build / "measured.json"))
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("otf_mvau", completed.stdout)
        self.assertIn("slowest unit runs at", completed.stdout)
        measured = json.loads((build / "measured.json").read_text())
        self.assertTrue(measured)
        for row in measured:
            self.assertIn("predicted", row)
            self.assertGreater(row["measured"]["fmax_mhz"], 0)

    def test_missing_vivado(self):
        completed = synth("--vivado", "/nonexistent/vivado",
                          "--out", str(Fixtures.build_dir("characterize_missing")),
                          timeout=600)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("settings64.sh", completed.stderr)


if __name__ == "__main__":
    unittest.main()
