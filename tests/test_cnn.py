"""CNN provenance, independent references and current compatibility boundaries."""

import argparse
import hashlib
import json
import shutil
import unittest

import numpy as np

from support import Fixtures, Hdl, ROOT
from tools import report_corpus
import onnx2fpga as otf
from onnx2fpga.p1_ingest import ModelBuilder
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.reference.runner import GraphRunner


CORPUS = ROOT / "tests/fixtures/cnn"
BLOCKED = {"grouped": "grouped convolution", "depthwise": "grouped convolution"}


def negative_pool():
    return (ModelBuilder("negative_pool").add_input("x", (1, 2, 2, 2))
            .add_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], pads=[1, 1, 0, 0])
            .add_output("y", (1, 2, 2, 2)).build())


class Cnn(unittest.TestCase):
    def test_pool_default_stride(self):
        graph = OnnxImporter(negative_pool()).run()
        x = -np.arange(8, 0, -1, dtype=np.float32).reshape(1, 2, 2, 2)
        x = graph.layouts["x"].to_internal(x)
        got = GraphRunner(graph).run({"x": x})[graph.outputs[0]]
        np.testing.assert_array_equal(got, x)

    def test_pool_refusals(self):
        for attributes, outputs, reason in (({"ceil_mode": 1}, ["y"], "ceil_mode"),
                                            ({}, ["y", "indices"], "indices")):
            model = (ModelBuilder("pool").add_input("x", (1, 1, 3, 3))
                     .add_node("MaxPool", ["x"], outputs, kernel_shape=[2, 2], **attributes)
                     .add_output("y", (1, 1, 2, 2)).build())
            report = otf.check(model, mode="stream", device="k26")
            self.assertFalse(report.ok)
            self.assertIn(reason, json.dumps(report.to_dict()))

    def test_provenance(self):
        self.assertEqual({path.stem for path in CORPUS.glob("*.json")},
                         {"maxpool", "residual", "global_average", "average", *BLOCKED})
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


@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
@unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
class Simulation(unittest.TestCase):
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

    def test_stalled_frames(self):
        args = argparse.Namespace(device="k26", mode="stream", clock_mhz=None,
                                  frames=16, sweep=True, inspect_only=False, implement=False,
                                  out=Fixtures.build_dir("cnn"))
        for name in ("maxpool", "residual", "global_average", "average"):
            with self.subTest(model=name):
                row = report_corpus.report_model(ROOT / "tests/fixtures/cnn" / (name + ".onnx"), args)
                self.assertEqual(row["status"], "passed", row)
                self.assertEqual(len(row["simulation"]["runs"]), 16)
                self.assertTrue(all(run["passed"] for run in row["simulation"]["runs"]))
