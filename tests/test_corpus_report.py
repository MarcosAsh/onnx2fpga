"""Corpus reports distinguish checked stages from estimates and missing evidence."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import report_corpus as report


class CorpusReport(unittest.TestCase):
    def test_regression_and_labels(self):
        expected = np.array([[1.0], [2.0], [3.0]])
        self.assertTrue(report.compare(expected + 0.001, expected, False)["passed"])
        self.assertFalse(report.compare(expected + 1, expected, False)["passed"])
        self.assertTrue(report.compare(expected, expected, True)["passed"])
        self.assertFalse(report.compare(expected + 1, expected, True)["passed"])

    def test_constant_and_nonfinite_reference(self):
        expected = np.ones((4, 1))
        result = report.compare(expected, expected, False)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["normalized_mae"])
        self.assertFalse(report.compare(expected + 0.1, expected, False)["passed"])
        self.assertFalse(report.compare(expected * np.nan, expected, False)["passed"])
        json.dumps(result, allow_nan=False)

    def test_inspection_is_not_simulation(self):
        args = argparse.Namespace(device="k26", mode="fixed", clock_mhz=322.265625,
                                  inspect_only=True)
        with patch.object(report.otf, "load", side_effect=AssertionError("compiled a model")):
            row = report.report_model(report.CORPUS / "linear_regression.onnx", args)
        self.assertEqual(row["status"], "inspected")
        self.assertEqual(row["stages"]["inspection"], "passed")
        for stage in ("compilation", "reference", "simulation", "implementation", "physical_board"):
            self.assertEqual(row["stages"][stage], "not_run")
        self.assertEqual(len(row["model_sha256"]), 64)
        self.assertNotIn("estimates", row)

    def test_failure_is_recorded(self):
        args = argparse.Namespace(device="k26", mode="stream", clock_mhz=None,
                                  inspect_only=True)
        row = report.report_model(report.CORPUS / "xgb_small.onnx", args)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["stages"]["inspection"], "blocked_or_unknown")
        self.assertEqual(row["stages"]["compilation"], "not_run")

    def test_partial_report_and_failed_exit(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(report, "metadata", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            status = report.main(["linear_regression", "xgb_small", "--inspect-only",
                                  "--mode", "stream", "--out", directory])
            self.assertEqual(status, 1)
            data = json.loads((Path(directory) / "report.json").read_text())
            self.assertTrue(data["complete"])
            self.assertEqual([row["status"] for row in data["models"]], ["inspected", "failed"])
            self.assertFalse((Path(directory) / "report.tmp").exists())
            with patch.object(report, "report_model", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    report.main(["linear_regression", "--inspect-only", "--out", directory])
            data = json.loads((Path(directory) / "report.json").read_text())
            self.assertFalse(data["complete"])
            self.assertEqual(data["models"], [])
