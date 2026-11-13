"""Whole CNNs under producer and consumer stalls."""

import argparse
import shutil
import unittest

from support import Fixtures, Hdl, ROOT
from tools import report_corpus


HARDCAML = Hdl.skip_reason("hardcaml")


class CnnSimulation(unittest.TestCase):
    @unittest.skipIf(HARDCAML, HARDCAML)
    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_stalled_frames(self):
        args = argparse.Namespace(device="k26", mode="stream", clock_mhz=None,
                                  frames=16, sweep=True, inspect_only=False, implement=False,
                                  out=Fixtures.build_dir("cnn"))
        for name in ("maxpool", "residual"):
            with self.subTest(model=name):
                row = report_corpus.report_model(ROOT / "tests/fixtures/cnn" / (name + ".onnx"), args)
                self.assertEqual(row["status"], "passed", row)
                self.assertEqual(len(row["simulation"]["runs"]), 16)
                self.assertTrue(all(run["passed"] for run in row["simulation"]["runs"]))
