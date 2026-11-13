"""Whole CNNs under producer and consumer stalls."""

import argparse
import shutil
import unittest

import numpy as np

from support import Fixtures, Hdl, ROOT
from test_cnn import negative_pool
from tools import report_corpus
import onnx2fpga as otf


HARDCAML = Hdl.skip_reason("hardcaml")


class CnnSimulation(unittest.TestCase):
    @unittest.skipIf(HARDCAML, HARDCAML)
    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_negative_pool_padding(self):
        x = -np.arange(8, 0, -1, dtype=np.float32).reshape(1, 2, 2, 2)
        samples = np.concatenate([x, x * 0.5, x * 0.25])
        for hdl in ("sv", "hardcaml"):
            with self.subTest(hdl=hdl):
                acc = otf.load(negative_pool())
                acc.compile(device="k26", mode="stream", hdl=hdl, calibration=samples,
                            out_dir=Fixtures.build_dir("negative_pool_" + hdl))
                got = acc.predict(samples)
                np.testing.assert_allclose(got, samples.transpose(0, 2, 3, 1), atol=8 / 127)
                result = acc.simulate(samples, sweep=True)
                self.assertTrue(result.ok, result)

    @unittest.skipIf(HARDCAML, HARDCAML)
    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_stalled_frames(self):
        args = argparse.Namespace(device="k26", mode="stream", clock_mhz=None,
                                  frames=16, sweep=True, inspect_only=False, implement=False,
                                  out=Fixtures.build_dir("cnn"))
        for name in ("maxpool", "residual", "global_average"):
            with self.subTest(model=name):
                row = report_corpus.report_model(ROOT / "tests/fixtures/cnn" / (name + ".onnx"), args)
                self.assertEqual(row["status"], "passed", row)
                self.assertEqual(len(row["simulation"]["runs"]), 16)
                self.assertTrue(all(run["passed"] for run in row["simulation"]["runs"]))
