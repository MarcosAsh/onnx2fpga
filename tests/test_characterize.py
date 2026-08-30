"""The synthesis driver, minus the synthesis.

Vivado is not installed here, so the TCL script has never been executed. What
can be checked is everything around it: which units get characterised, what
environment they are launched with, and whether a report is read and compared
correctly. Those are covered below. The TCL itself has never met a real
Vivado install.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import unittest

from support import Fixtures, ROOT  # noqa: F401

from onnx2fpga.measure import (Characterizer, Comparison, SynthesisResult,
                               Subject, VivadoRunner, calibration_samples, render)

from onnx2fpga.compile import Compiler
from onnx2fpga.p1_ingest import OnnxModel
from onnx2fpga.p3_ops.hardware_ops import MatVecUnit
from onnx2fpga.targets.device import Device

# What the TCL writes, in the shape it writes it.
SUMMARY = {
    "top": "otf_mvau", "part": "xc7z020-clg400-1", "stage": "synth",
    "generics": "MW=9 MH=4 SIMD=3 PE=4", "period_ns": 5.0, "wns_ns": 0.842,
    "fmax_mhz": 240.15, "lut": 412, "ff": 288, "dsp": 6,
    "ramb36": 1, "ramb18": 2, "bram36": 2.0, "uram": 0,
}


class SynthesisResultTest(unittest.TestCase):
    def test_reads_what_the_script_writes(self):
        path = Fixtures.build_dir("characterize") / "summary.json"
        path.write_text(json.dumps(SUMMARY))
        result = SynthesisResult.load(path)
        self.assertEqual(result.measured.lut, 412)
        self.assertEqual(result.measured.dsp, 6)
        self.assertEqual(result.measured.bram36, 2.0)
        self.assertEqual(result.fmax_mhz, 240.15)
        self.assertEqual(result.stage, "synth")

    def test_half_block_rams_count_as_half(self):
        self.assertEqual(SUMMARY["bram36"],
                         SUMMARY["ramb36"] + 0.5 * SUMMARY["ramb18"])

    def test_an_unconstrained_unit_reads_back_as_no_fmax_at_all(self):
        path = Fixtures.build_dir("characterize") / "unconstrained.json"
        path.write_text(json.dumps(
            dict(SUMMARY, timed=False, wns_ns=None, fmax_mhz=None)))
        result = SynthesisResult.load(path)
        self.assertIsNone(result.fmax_mhz)
        self.assertEqual(result.measured.lut, 412)


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        from onnx2fpga.targets.resources import Resources
        predicted = Resources(lut=500, ff=288, dsp=6, bram36=1.0, uram=0)
        self.comparison = Comparison(
            Subject("u", "otf_mvau", {"MW": 9}, predicted, 100),
            SynthesisResult(SUMMARY))

    def test_error_is_signed_and_relative(self):
        self.assertAlmostEqual(self.comparison.error("lut"), 500 / 412 - 1.0)
        self.assertAlmostEqual(self.comparison.error("dsp"), 0.0)
        self.assertAlmostEqual(self.comparison.error("bram36"), -0.5)

    def test_predicting_something_that_does_not_exist_is_infinite(self):
        self.assertEqual(self.comparison.error("uram"), None)

    def test_row_and_table_render(self):
        self.assertIn("otf_mvau", self.comparison.row())
        table = render([self.comparison])
        self.assertIn("slowest unit runs at 240.2 MHz", table)


class SubjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = OnnxModel.load(ROOT / "examples" / "models" / "cnn.onnx")
        cls.result = Compiler(device="z7020", target_cycles=400).compile(
            model, calibration_samples(model), Fixtures.build_dir("characterize_subjects"))

    def test_every_unit_becomes_a_subject(self):
        driver = Characterizer(VivadoRunner("xc7z020-clg400-1"), Device.get("z7020"))
        graph = self.result.hardware_graph
        subjects = driver.subjects(graph)
        self.assertTrue(subjects)
        modules = {s.module for s in subjects}
        self.assertIn("otf_mvau", modules)
        self.assertIn("otf_swu_line", modules)

    def test_identical_units_are_characterised_once(self):
        driver = Characterizer(VivadoRunner("xc7z020-clg400-1"), Device.get("z7020"))
        subjects = driver.subjects(self.result.hardware_graph)
        keys = [(s.module, s.generic_arguments()) for s in subjects]
        self.assertEqual(len(keys), len(set(keys)))

    def test_generics_carry_the_memory_files(self):
        graph = self.result.hardware_graph
        unit = next(n for n in graph.topological_order() if isinstance(n, MatVecUnit))
        subject = Subject.from_node(unit, graph, Device.get("z7020"))
        self.assertIn("WEIGHT_FILE=mem/", subject.generic_arguments())

    def test_empty_string_generics_are_dropped(self):
        """An empty file name would make Vivado try to read a file called
        nothing, so those parameters keep their default instead."""
        subject = Subject("u", "otf_mvau",
                          {"MW": 9, "WEIGHT_FILE": "", "BIAS_FILE": "mem/b.hex"},
                          None, 0)
        self.assertNotIn("WEIGHT_FILE", subject.generic_arguments())


class EndToEndTest(unittest.TestCase):
    """The whole measurement path, against a stand-in for Vivado.

    This proves the plumbing: units are enumerated, the tool is invoked once
    per unit, each summary is read back, and the comparison renders. The
    numbers it prints are the stub's canned ones and mean nothing. Whether
    Vivado produces sensible numbers is what the first real run decides.
    """

    FAKE = ROOT / "tests" / "fake_vivado"

    @unittest.skipIf(shutil.which("tclsh") is None, "tclsh not installed")
    def test_every_unit_is_measured_and_compared(self):
        self.assertTrue(os.access(self.FAKE, os.X_OK), "%s must be executable" % self.FAKE)
        build = Fixtures.build_dir("characterize_e2e")
        model_path = ROOT / "examples" / "models" / "cnn.onnx"
        completed = subprocess.run(
            [sys.executable, "-m", "onnx2fpga", "synth", str(model_path),
             "--device", "z7020", "--vivado", str(self.FAKE),
             "--out", str(build), "--measurements", str(build / "measured.json")],
            capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTHONPATH": str(ROOT / "compiler")})
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        self.assertIn("otf_mvau", completed.stdout)
        self.assertIn("slowest unit runs at", completed.stdout)

        measured = json.loads((build / "measured.json").read_text())
        self.assertTrue(measured)
        for row in measured:
            self.assertIn("predicted", row)
            self.assertIn("measured", row)
            self.assertGreater(row["measured"]["fmax_mhz"], 0)

    @unittest.skipIf(shutil.which("tclsh") is None, "tclsh not installed")
    def test_a_missing_tool_fails_clearly_rather_than_crashing(self):
        completed = subprocess.run(
            [sys.executable, "-m", "onnx2fpga", "synth",
             str(ROOT / "examples" / "models" / "cnn.onnx"),
             "--device", "z7020", "--vivado", "/nonexistent/vivado",
             "--out", str(Fixtures.build_dir("characterize_missing"))],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": str(ROOT / "compiler")})
        self.assertEqual(completed.returncode, 2)
        self.assertIn("settings64.sh", completed.stderr)


class RunnerTest(unittest.TestCase):
    def test_environment_is_complete(self):
        runner = VivadoRunner("xc7z020-clg400-1", period_ns=4.0, implement=True)
        subject = Subject("u", "otf_fifo", {"WIDTH": 24, "DEPTH": 2}, None, 0)
        environment = runner.environment(subject, ["rtl/otf_fifo.sv"], "synth/u")
        self.assertEqual(environment["OTF_TOP"], "otf_fifo")
        self.assertEqual(environment["OTF_PERIOD"], "4.0")
        self.assertEqual(environment["OTF_IMPL"], "1")
        self.assertEqual(environment["OTF_GENERICS"], "DEPTH=2 WIDTH=24")
        self.assertEqual(environment["OTF_RTL"], "rtl/otf_fifo.sv")

    def test_the_script_it_invokes_is_found_through_the_asset_locator(self):
        runner = VivadoRunner("xc7z020-clg400-1")
        self.assertTrue(runner.script.exists(), runner.script)
        self.assertIn(str(runner.script), " ".join(runner.command()))

    def test_sources_are_relative_to_the_build_directory(self):
        """Built here rather than borrowed from another test class, so the
        result does not depend on which test ran first."""
        model = OnnxModel.load(ROOT / "examples" / "models" / "cnn.onnx")
        build = Fixtures.build_dir("characterize_sources")
        Compiler(device="z7020", target_cycles=400).compile(
            model, calibration_samples(model), build)
        driver = Characterizer(VivadoRunner("xc7z020-clg400-1"), Device.get("z7020"))
        sources = driver.sources(build)
        self.assertTrue(sources, "no RTL found in %s" % build)
        for source in sources:
            self.assertFalse(pathlib.Path(source).is_absolute(), source)


if __name__ == "__main__":
    unittest.main()
