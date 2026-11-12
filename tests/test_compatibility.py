"""Compatibility reports retain independent failures and mark blocked branches."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import onnx2fpga as otf
from onnx2fpga.cli import main
from onnx2fpga.errors import ConfigError
from onnx2fpga.p1_ingest import ModelBuilder, OnnxModel, TensorProto
from support import ROOT


def dense():
    return (ModelBuilder().add_input("x", (1, 2)).add_output("y", (1, 2))
            .add_initializer("w", np.eye(2, dtype=np.float32))
            .add_node("Gemm", ["x", "w"], ["y"]).build())


class Compatibility(unittest.TestCase):
    def test_corpus_without_tools(self):
        with patch("subprocess.run", side_effect=AssertionError("external tool launched")), \
                patch("subprocess.Popen", side_effect=AssertionError("external tool launched")):
            for path in sorted((ROOT / "tests/fixtures/corpus").glob("*.onnx")):
                with self.subTest(model=path.name):
                    model = OnnxModel.load(path)
                    before = model.proto.encode()
                    report = otf.check(model, device="k26", mode="fixed")
                    self.assertTrue(report.ok, report.render())
                    self.assertIn("routed timing", report.unchecked)
                    self.assertEqual(model.proto.encode(), before)

    def test_independent_failures_and_dependencies(self):
        model = (ModelBuilder().add_input("x", (1, 2)).add_output("end", (1, 2))
                 .add_initializer("w", np.eye(2, dtype=np.float32))
                 .add_node("Mystery", ["x"], ["a"], name="bad_op")
                 .add_node("Relu", ["a"], ["end"], name="dependent")
                 .add_node("Gemm", ["x", "w"], ["b"], name="bad_attr", transA=1)
                 .add_node("AnotherMystery", ["a"], ["c"], name="also_bad")
                 .build())
        before = model.proto.encode()
        report = otf.check(model)
        self.assertFalse(report.ok)
        self.assertEqual({f.node for f in report.findings if f.severity == "error"},
                         {"bad_op", "bad_attr", "also_bad"})
        self.assertEqual([(f.node, f.code) for f in report.findings if f.severity == "unknown"],
                         [("dependent", "dependency")])
        self.assertEqual(model.proto.encode(), before)

    def test_two_attribute_failures(self):
        model = dense()
        builder = (ModelBuilder().add_input("x", (1, 2))
                   .add_initializer("w", np.eye(2, dtype=np.float32)))
        for name in ("first", "second"):
            builder.add_node("Gemm", ["x", "w"], [name], name=name, transA=1)
            builder.add_output(name, (1, 2))
        report = otf.check(builder.build())
        self.assertEqual([f.node for f in report.findings], ["first", "second"])
        self.assertTrue(otf.check(model).ok)

    def test_domain_is_not_ignored(self):
        model = dense()
        node = model.nodes()[0]
        node.domain = "custom.vendor"
        report = otf.check(model)
        self.assertEqual(report.findings[0].code, "operator_domain")
        self.assertEqual(report.findings[0].domain, "custom.vendor")
        node.domain = "ai.onnx.ml"
        self.assertFalse(otf.check(model).ok)

    def test_node_identity_and_opset(self):
        model = (ModelBuilder(opset=19).add_input("x", (1, 2)).add_output("y", (1, 2))
                 .add_node("Missing", ["x"], ["y"]).build())
        model.nodes()[0].name = ""
        finding = otf.check(model).findings[0]
        self.assertEqual((finding.node_index, finding.node, finding.opset), (0, "", 19))

    def test_shape_assumptions(self):
        model = dense()
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value, dims[0].dim_param = 0, "batch"
        report = otf.check(model)
        self.assertTrue(report.ok)
        self.assertEqual(report.findings[0].severity, "warning")
        dims[1].dim_value, dims[1].dim_param = 0, "width"
        report = otf.check(model)
        self.assertFalse(report.ok)
        self.assertIn("input_shape", [f.code for f in report.findings])
        self.assertIn("dependency", [f.code for f in report.findings])

    def test_bad_initializer_and_input_type(self):
        model = dense()
        model.graph.initializer[0].data_type = 8
        model.graph.input[0].type.tensor_type.elem_type = 8
        report = otf.check(model)
        self.assertEqual({f.code for f in report.findings},
                         {"input_type", "initializer", "dependency"})

    def test_integer_semantics_are_unknown(self):
        model = dense()
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.INT64
        report = otf.check(model)
        self.assertFalse(report.ok)
        self.assertEqual(report.findings[0].severity, "unknown")

    def test_fixed_and_stream_restrictions(self):
        model = (ModelBuilder().add_input("x", (1, 1, 4, 4)).add_output("y", (1, 1, 3, 3))
                 .add_initializer("w", np.ones((1, 1, 2, 2), dtype=np.float32))
                 .add_node("Conv", ["x", "w"], ["y"]).build())
        self.assertTrue(otf.check(model).ok)
        self.assertEqual(otf.check(model, mode="fixed").findings[0].code, "execution_mode")
        tree = ROOT / "tests/fixtures/corpus/xgb_small.onnx"
        self.assertFalse(otf.check(tree, mode="stream").ok)
        self.assertTrue(otf.check(tree, mode="fixed").ok)

    def test_stream_mode_reports_each_node(self):
        model = (ModelBuilder().add_input("x", (1, 2)).add_output("y", (1, 2))
                 .add_node("Sigmoid", ["x"], ["s"], name="sigmoid")
                 .add_node("Tanh", ["s"], ["y"], name="tanh").build())
        report = otf.check(model)
        self.assertEqual([(f.node_index, f.node, f.code) for f in report.findings],
                         [(0, "sigmoid", "execution_mode"), (1, "tanh", "execution_mode")])
        self.assertTrue(otf.check(model, mode="fixed").ok)

    def test_deferred_output_errors(self):
        builder = ModelBuilder().add_input("x", (1, 1, 2, 2))
        for name in ("a", "b"):
            builder.add_node("Transpose", ["x"], [name], name=name, perm=[0, 1, 3, 2])
            builder.add_output(name, (1, 1, 2, 2))
        report = otf.check(builder.build())
        self.assertFalse(report.ok)
        self.assertEqual({f.node for f in report.findings}, {"a", "b"})

    def test_clock_and_configuration(self):
        self.assertFalse(otf.check(dense(), clock_mhz=100).ok)
        self.assertFalse(otf.check(dense(), device="z7020", mode="fixed", clock_mhz=100).ok)
        for options in ({"device": "missing"}, {"mode": "missing"},
                        {"clock_mhz": float("nan")}, {"clock_mhz": -1}):
            with self.subTest(options=options), self.assertRaises(ConfigError):
                otf.check(dense(), **options)

    def test_missing_and_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            self.assertEqual(otf.check(path).findings[0].code, "model_read")
            path.write_bytes(b"")
            self.assertEqual(otf.check(path).findings[0].code, "model_read")

    def test_cli_json_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            dense().save(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(["check", str(path), "--json", "--device", "k26", "--mode", "fixed"])
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(output.getvalue())["ok"])
            path.write_bytes(b"")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(["check", str(path), "--json"])
            self.assertEqual(status, 1)
            self.assertFalse(json.loads(output.getvalue())["ok"])
            self.assertEqual(list(Path(directory).iterdir()), [path])
