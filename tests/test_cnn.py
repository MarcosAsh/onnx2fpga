"""CNN provenance, independent references and current compatibility boundaries."""

import argparse
import importlib.util
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
from onnx2fpga.p2_graph.datatype import INT4, INT8, QuantSpec
from onnx2fpga.p4_quantize.lower import FuseRelu, LowerToHardware
from onnx2fpga.p4_quantize.plan import QuantizationPlan
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.targets.device import Device
from onnx2fpga.reference.runner import GraphRunner


CORPUS = ROOT / "tests/fixtures/cnn"
MODELS = ("maxpool", "residual", "global_average", "average", "grouped", "depthwise", "mobile",
          "mnist")


def negative_pool():
    return (ModelBuilder("negative_pool").add_input("x", (1, 2, 2, 2))
            .add_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], pads=[1, 1, 0, 0])
            .add_output("y", (1, 2, 2, 2)).build())


def grouped_model(groups=2, multiplier=3):
    rng = np.random.default_rng(17)
    channels = groups * multiplier
    weights = rng.normal(0, 0.1, (channels, 6 // max(1, groups), 3, 2)).astype(np.float32)
    bias = rng.normal(0, 0.1, channels).astype(np.float32)
    return (ModelBuilder("grouped", opset=20).add_input("x", (1, 6, 5, 7))
            .add_initializer("w", weights).add_initializer("b", bias)
            .add_node("Conv", ["x", "w", "b"], ["y"], group=groups,
                      strides=[2, 1], pads=[1, 1, 1, 0], dilations=[1, 2])
            .add_output("y", (1, channels, 3, 6)).build())


class Cnn(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "onnxruntime not installed")
    def test_groups(self):
        import onnxruntime as ort
        x = np.random.default_rng(5).uniform(-1, 1, (1, 6, 5, 7)).astype(np.float32)
        for groups, multiplier in ((1, 3), (2, 3), (3, 2), (6, 1), (6, 2)):
            with self.subTest(groups=groups, multiplier=multiplier):
                model = grouped_model(groups, multiplier)
                graph = OnnxImporter(model).run()
                got = GraphRunner(graph).run({"x": x.transpose(0, 2, 3, 1)})["y"]
                expected = ort.InferenceSession(model.proto.encode()).run(["y"], {"x": x})[0]
                np.testing.assert_allclose(got, expected.transpose(0, 2, 3, 1), atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "onnxruntime not installed")
    def test_group_grids(self):
        import onnxruntime as ort
        codes = np.random.default_rng(19).integers(-128, 128, (1, 6, 5, 7))
        for groups in (2, 3, 6):
            model = grouped_model(groups)
            session = ort.InferenceSession(model.proto.encode())
            graph = OnnxImporter(model).run()
            plan = QuantizationPlan.from_annotations({
                "x": QuantSpec(0.125, -17), "y": QuantSpec(0.25, 11)})
            graph = LowerToHardware(plan).run(graph, CompilerContext(Device.get("k26")))
            got = GraphRunner(graph).run({"x": codes.transpose(0, 2, 3, 1)})["y"]
            real = ((codes + 17) * 0.125).astype(np.float32)
            expected = session.run(["y"], {"x": real})[0].transpose(0, 2, 3, 1)
            expected = np.clip(np.floor(expected / 0.25 + 0.5) + 11, -128, 127)
            np.testing.assert_allclose(got, expected, atol=1, rtol=0)

    def test_group_refusals(self):
        for groups in (0, 4, 7):
            report = otf.check(grouped_model(groups), device="k26", mode="stream")
            self.assertFalse(report.ok)
            self.assertIn("channels", json.dumps(report.to_dict()))

        model = (ModelBuilder("bad_groups").add_input("x", (1, 6, 5, 7))
                 .add_initializer("w", np.ones((5, 3, 1, 1), dtype=np.float32))
                 .add_node("Conv", ["x", "w"], ["y"], group=2)
                 .add_output("y", (1, 5, 5, 7)).build())
        self.assertFalse(otf.check(model, device="k26", mode="stream").ok)
        for name, count in (("x", 6), ("y", 6)):
            graph = OnnxImporter(grouped_model()).run()
            specs = {"x": QuantSpec(0.125), "y": QuantSpec(0.25)}
            specs[name] = QuantSpec(np.full(count, 0.125))
            plan = QuantizationPlan.from_annotations(specs)
            with self.assertRaisesRegex(NotImplementedError, "per-tensor"):
                LowerToHardware(plan).run(graph, CompilerContext(Device.get("k26")))

    def test_clip_fuses(self):
        rng = np.random.default_rng(3)
        model = (ModelBuilder("relu6", opset=20).add_input("x", (1, 2, 4, 4))
                 .add_initializer("w", rng.normal(0, 1, (4, 2, 3, 3)).astype(np.float32))
                 .add_node("Constant", [], ["low"], value=np.array(0, np.float32))
                 .add_node("Constant", [], ["high"], value=np.array(6, np.float32))
                 .add_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1])
                 .add_node("Clip", ["c", "low", "high"], ["y"])
                 .add_output("y", (1, 4, 4, 4)).build())
        self.assertTrue(otf.check(model, device="k26", mode="stream").ok)
        graph = FuseRelu().run(OnnxImporter(model).run(), None)
        x = rng.normal(0, 4, (1, 4, 4, 2))
        real = GraphRunner(graph).run({"x": x})["y"]
        self.assertEqual((real.min(), real.max()), (0, 6))
        plan = QuantizationPlan.from_annotations({"x": QuantSpec(0.125, -17),
                                                  "y": QuantSpec(0.25, 11)})
        graph = LowerToHardware(plan).run(graph, CompilerContext(Device.get("k26")))
        conv = next(node for node in graph.nodes if hasattr(node, "requant"))
        self.assertEqual((conv.requant.lower_clamp, conv.requant.upper_clamp), (11, 35))
        codes = np.clip(np.rint(x / 0.125) - 17, -128, 127)
        got = GraphRunner(graph).run({"x": codes})["y"]
        expected = GraphRunner(FuseRelu().run(OnnxImporter(model).run(), None)).run(
            {"x": (codes + 17) * 0.125})["y"]
        np.testing.assert_allclose((got - 11) * 0.25, expected, atol=0.25)

    def test_spatial_mean(self):
        x = np.random.default_rng(4).normal(0, 1, (1, 3, 5, 2))
        for keepdims, shape in ((1, (1, 2, 1, 1)), (0, (1, 2))):
            model = (ModelBuilder("mean", opset=18).add_input("x", (1, 2, 3, 5))
                     .add_initializer("axes", np.array([2, -1], np.int64))
                     .add_node("ReduceMean", ["x", "axes"], ["y"], keepdims=keepdims)
                     .add_output("y", shape).build())
            self.assertTrue(otf.check(model, device="k26", mode="stream").ok)
            got = GraphRunner(OnnxImporter(model).run()).run({"x": x})["y"]
            np.testing.assert_allclose(got.reshape(-1), x.mean(axis=(1, 2)).reshape(-1))

    def test_spatial_mean_bad_axes(self):
        for axes in ([2, 7], [-6, -1], [2, 2]):
            model = (ModelBuilder("mean", opset=18).add_input("x", (1, 2, 3, 5))
                     .add_initializer("axes", np.array(axes, np.int64))
                     .add_node("ReduceMean", ["x", "axes"], ["y"])
                     .add_output("y", (1, 2, 1, 1)).build())
            self.assertFalse(otf.check(model, device="k26", mode="stream").ok)

    def test_relu_then_negative_clip(self):
        model = (ModelBuilder("negative_clip", opset=10).add_input("x", (1, 2))
                 .add_initializer("w", np.eye(2, dtype=np.float32))
                 .add_node("Gemm", ["x", "w"], ["linear"])
                 .add_node("Relu", ["linear"], ["relu"])
                 .add_node("Clip", ["relu"], ["y"], min=-2.0, max=-1.0)
                 .add_output("y", (1, 2)).build())
        for dtype, expected in ((INT8, -16), (INT4, -8)):
            graph = FuseRelu().run(OnnxImporter(model).run(), None)
            plan = QuantizationPlan.from_annotations({"x": QuantSpec(0.25),
                                                      "y": QuantSpec(0.0625, dtype=dtype)})
            graph = LowerToHardware(plan).run(graph, CompilerContext(Device.get("k26")))
            unit = next(node for node in graph.nodes if hasattr(node, "requant"))
            self.assertEqual((unit.requant.lower_clamp, unit.requant.upper_clamp),
                             (expected, expected))
            got = GraphRunner(graph).run({"x": np.array([[-8, 8]])})["y"]
            np.testing.assert_array_equal(got, [[expected, expected]])
        graph = FuseRelu().run(OnnxImporter(model).run(), None)
        plan = QuantizationPlan.from_annotations({"x": QuantSpec(0.25)})
        with self.assertRaisesRegex(NotImplementedError, "output quantization grid"):
            LowerToHardware(plan).run(graph, CompilerContext(Device.get("k26")))

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
                         set(MODELS))
        for path in sorted(CORPUS.glob("*.json")):
            with self.subTest(model=path.stem):
                record = json.loads(path.read_text())
                trained = path.stem == "mnist"
                self.assertEqual(record["training"], "MNIST" if trained else "none")
                if trained:
                    self.assertTrue(record["passed"])
                    self.assertEqual(record["accuracy_samples"], 10000)
                    self.assertGreaterEqual(record["float_accuracy"], 0.95)
                    self.assertGreaterEqual(record["quantized_accuracy"], 0.945)
                    self.assertLessEqual(record["float_accuracy"] - record["quantized_accuracy"], 0.005)
                    self.assertLess(record["normalized_mae"], 0.025)
                for name, digest in record["sha256"].items():
                    self.assertEqual(hashlib.sha256((CORPUS / name).read_bytes()).hexdigest(), digest)
                with np.load(path.with_suffix(".npz")) as data:
                    shape = (1, 28, 28) if trained else (4, 8, 8)
                    self.assertEqual(data["x"].shape, (256, *shape))
                    self.assertEqual(data["calibration"].shape, (512, *shape))
                    self.assertEqual(data["y"].shape, (256, 10 if trained else 3))
                    if trained:
                        rng = np.random.default_rng(81)
                        np.testing.assert_array_equal(data["calibration_indices"],
                                                      rng.choice(60000, 512, replace=False))
                        np.testing.assert_array_equal(data["test_indices"],
                                                      rng.choice(10000, 256, replace=False))
                        self.assertEqual(data["labels"].shape, (256,))
                    calibration = {row.tobytes() for row in data["calibration"]}
                    self.assertFalse(any(row.tobytes() in calibration for row in data["x"]))

    def test_inspection(self):
        for path in sorted(CORPUS.glob("*.onnx")):
            with self.subTest(model=path.stem):
                report = otf.check(path, mode="stream", device="k26")
                self.assertTrue(report.ok, report.to_dict())

    def test_mnist_schedule(self):
        with np.load(CORPUS / "mnist.npz") as data:
            calibration = data["calibration"]
        acc = otf.load(CORPUS / "mnist.onnx")
        out = Fixtures.build_dir("mnist_schedule")
        acc.compile(device="k26", mode="stream", hdl="sv", calibration=calibration, out_dir=out)
        graph = acc.result.hardware_graph
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["cycles_per_frame"], 3920)
        self.assertEqual(max(node.cycles for node in graph.nodes), 3920)
        self.assertNotRegex((out / "rtl/top.sv").read_text(), r"[( ]/\w")

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
    def test_spatial_mean(self):
        samples = np.random.default_rng(4).normal(0, 1, (8, 2, 3, 5)).astype(np.float32)
        for keepdims in (0, 1):
            model = (ModelBuilder("mean", opset=18).add_input("x", (1, 2, 3, 5))
                     .add_initializer("axes", np.array([2, -1], np.int64))
                     .add_node("ReduceMean", ["x", "axes"], ["y"], keepdims=keepdims)
                     .add_output("y", (1, 2, 1, 1) if keepdims else (1, 2)).build())
            for hdl in ("sv", "hardcaml"):
                with self.subTest(keepdims=keepdims, hdl=hdl):
                    acc = otf.load(model)
                    acc.compile(device="k26", mode="stream", hdl=hdl, calibration=samples,
                                out_dir=Fixtures.build_dir("mean_%d_%s" % (keepdims, hdl)))
                    np.testing.assert_allclose(acc.predict(samples).reshape(8, 2),
                                                samples.mean(axis=(2, 3)), atol=0.025)
                    result = acc.simulate(samples, sweep=True, reset=True)
                    self.assertTrue(result.ok, result)

    def test_mobile_sv(self):
        with np.load(CORPUS / "mobile.npz") as data:
            acc = otf.load(CORPUS / "mobile.onnx")
            acc.compile(device="k26", mode="stream", hdl="sv", calibration=data["calibration"],
                        out_dir=Fixtures.build_dir("mobile_sv"))
            result = acc.simulate(data["x"][:16], sweep=True)
            self.assertTrue(result.ok, result)

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "onnxruntime not installed")
    def test_grouped_edges(self):
        import onnxruntime as ort
        samples = np.random.default_rng(8).uniform(-1, 1, (8, 6, 5, 7)).astype(np.float32)
        for groups, multiplier in ((2, 3), (3, 2), (6, 1), (6, 2)):
            model = grouped_model(groups, multiplier)
            session = ort.InferenceSession(model.proto.encode())
            expected = np.concatenate([session.run(["y"], {"x": row[None]})[0] for row in samples])
            for hdl in ("sv", "hardcaml"):
                with self.subTest(groups=groups, multiplier=multiplier, hdl=hdl):
                    acc = otf.load(model)
                    acc.compile(device="k26", mode="stream", hdl=hdl, calibration=samples,
                                out_dir=Fixtures.build_dir("groups_%d_%d_%s" % (groups, multiplier, hdl)))
                    got = acc.predict(samples).reshape(expected.transpose(0, 2, 3, 1).shape)
                    error = np.abs(got - expected.transpose(0, 2, 3, 1)).mean() / expected.std()
                    self.assertLess(error, 0.025)
                    result = acc.simulate(samples, sweep=True)
                    self.assertTrue(result.ok, result)

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "onnxruntime not installed")
    def test_grouped_grids(self):
        import onnxruntime as ort
        rng = np.random.default_rng(23)
        model = ModelBuilder("grouped_qdq", opset=20).add_input("x", (1, 6, 2, 2))
        constants = {
            "xs": np.float32(0.125), "xz": np.uint8(111),
            "ys": np.float32(0.25), "yz": np.uint8(139),
            "ws": np.linspace(0.01, 0.04, 6, dtype=np.float32), "wz": np.zeros(6, np.int8),
            "wq": rng.integers(-32, 33, (6, 2, 1, 2), dtype=np.int8),
            "b": rng.normal(0, 0.1, 6).astype(np.float32)}
        for name, value in constants.items():
            model.add_initializer(name, value)
        model = (model.add_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"])
                 .add_node("DequantizeLinear", ["xq", "xs", "xz"], ["xd"])
                 .add_node("DequantizeLinear", ["wq", "ws", "wz"], ["w"], axis=0)
                 .add_node("Conv", ["xd", "w", "b"], ["conv"], group=3)
                 .add_node("QuantizeLinear", ["conv", "ys", "yz"], ["yq"])
                 .add_node("DequantizeLinear", ["yq", "ys", "yz"], ["y"])
                 .add_output("y", (1, 6, 2, 1)).build())
        samples = ((rng.integers(0, 256, (8, 6, 2, 2)) - 111) * 0.125).astype(np.float32)
        session = ort.InferenceSession(model.proto.encode())
        expected = np.concatenate([session.run(["y"], {"x": row[None]})[0] for row in samples])
        for hdl in ("sv", "hardcaml"):
            with self.subTest(hdl=hdl):
                acc = otf.load(model)
                acc.compile(device="k26", mode="stream", hdl=hdl,
                            out_dir=Fixtures.build_dir("grouped_qdq_" + hdl))
                got = acc.predict(samples).reshape(8, 2, 1, 6)
                np.testing.assert_allclose(got, expected.transpose(0, 2, 3, 1), atol=0.25, rtol=0)
                result = acc.simulate(samples, sweep=True)
                self.assertTrue(result.ok, result)

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
        for name in MODELS:
            with self.subTest(model=name):
                row = report_corpus.report_model(ROOT / "tests/fixtures/cnn" / (name + ".onnx"), args)
                self.assertEqual(row["status"], "passed", row)
                self.assertEqual(len(row["simulation"]["runs"]), 16)
                self.assertLess(row["simulation"]["runs"][0]["mean_completion_interval_cycles"],
                                1.01 * row["estimates"]["initiation_interval_cycles"])
                self.assertTrue(all(run["passed"] for run in row["simulation"]["runs"]))
                for run in row["simulation"]["runs"]:
                    self.assertEqual(run["frames"], 16)
                    self.assertGreater(run["frame_latency_cycles"]["min"], 0)
                    self.assertGreater(run["mean_completion_interval_cycles"], 0)
                self.assertTrue(row["simulation"]["reset"]["passed"])
                self.assertEqual(row["simulation"]["reset"]["resets"], 1)
                self.assertEqual(row["simulation"]["reset"]["frames"], 16)
