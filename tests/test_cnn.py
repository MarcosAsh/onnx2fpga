"""CNN provenance, independent references and current compatibility boundaries."""

import hashlib
import json
import unittest

import numpy as np

from support import ROOT
import onnx2fpga as otf


CORPUS = ROOT / "tests/fixtures/cnn"
BLOCKED = {"average": "AveragePool", "global_average": "GlobalAveragePool",
           "grouped": "grouped convolution", "depthwise": "grouped convolution"}


class Cnn(unittest.TestCase):
    def test_provenance(self):
        self.assertEqual({path.stem for path in CORPUS.glob("*.json")},
                         {"maxpool", "residual", *BLOCKED})
        for path in sorted(CORPUS.glob("*.json")):
            with self.subTest(model=path.stem):
                record = json.loads(path.read_text())
                self.assertEqual(record["training"], "none")
                for name, digest in record["sha256"].items():
                    self.assertEqual(hashlib.sha256((CORPUS / name).read_bytes()).hexdigest(), digest)
                with np.load(path.with_suffix(".npz")) as data:
                    self.assertEqual(data["x"].shape, (256, 4, 8, 8))
                    self.assertEqual(data["calibration"].shape, (512, 4, 8, 8))
                    self.assertEqual(data["y"].shape, (256, 3))
                    calibration = {row.tobytes() for row in data["calibration"]}
                    self.assertFalse(any(row.tobytes() in calibration for row in data["x"]))

    def test_inspection(self):
        for path in sorted(CORPUS.glob("*.onnx")):
            with self.subTest(model=path.stem):
                report = otf.check(path, mode="stream", device="k26")
                if path.stem in BLOCKED:
                    self.assertFalse(report.ok)
                    self.assertIn(BLOCKED[path.stem], json.dumps(report.to_dict()))
                else:
                    self.assertTrue(report.ok, report.to_dict())

    def test_ort(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        for path in sorted(CORPUS.glob("*.onnx")):
            with self.subTest(model=path.stem), np.load(path.with_suffix(".npz")) as data:
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                got = np.concatenate([session.run(["y"], {"x": row[None]})[0]
                                      for row in data["x"]])
                np.testing.assert_allclose(got, data["y"], atol=1e-6, rtol=1e-5)
