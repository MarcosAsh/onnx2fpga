"""Global pooling arithmetic and importer boundaries."""

import shutil
import unittest

import numpy as np

from support import Fixtures, Hdl
import onnx2fpga as otf
from onnx2fpga.p1_ingest import ModelBuilder
from onnx2fpga.p2_graph.datatype import QuantSpec
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.p4_quantize.lower import LowerToHardware
from onnx2fpga.p4_quantize.plan import QuantizationPlan
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.reference.runner import GraphRunner
from onnx2fpga.targets.device import Device


def model(shape=(1, 3, 3, 5), **attrs):
    return (ModelBuilder("global_average").add_input("x", shape)
            .add_node("GlobalAveragePool", ["x"], ["y"], **attrs)
            .add_output("y", (1, shape[1], 1, 1)).build())


class Arithmetic(unittest.TestCase):
    def test_grids(self):
        codes = np.random.default_rng(1).integers(-128, 128, (1, 3, 5, 3))
        for zero in (0, -17, 63):
            for values in (codes, np.full_like(codes, -128), np.full_like(codes, 127)):
                with self.subTest(zero=zero, constant=values.min() == values.max()):
                    graph = OnnxImporter(model()).run()
                    plan = QuantizationPlan.from_annotations({
                        "x": QuantSpec(0.125, zero), "y": QuantSpec(0.25, 11)})
                    graph = LowerToHardware(plan).run(graph, CompilerContext(Device.get("k26")))
                    got = GraphRunner(graph).run({"x": values})[graph.outputs[0]]
                    real = (values.astype(np.float64) - zero).mean(axis=(1, 2), keepdims=True) / 2
                    expected = np.clip(np.floor(real + 0.5) + 11, -128, 127)
                    np.testing.assert_allclose(got, expected, atol=1, rtol=0)

    def test_float_reference(self):
        graph = OnnxImporter(model()).run()
        x = np.random.default_rng(3).normal(size=(1, 3, 5, 3))
        got = GraphRunner(graph).run({"x": x})[graph.outputs[0]]
        np.testing.assert_allclose(got.reshape(-1), [x[0, :, :, c].sum() / 15 for c in range(3)])

    def test_refusals(self):
        for fixture, reason in ((model((1, 3, 5)), "2D"), (model(axes=[2, 3]), "attributes")):
            report = otf.check(fixture, mode="stream")
            self.assertFalse(report.ok)
            self.assertIn(reason, report.render())
        self.assertFalse(otf.check(model(), mode="fixed").ok)

    def test_channel_grids_refused(self):
        graph = OnnxImporter(model()).run()
        plan = QuantizationPlan.from_annotations({
            "x": QuantSpec([0.125, 0.25, 0.5], 0), "y": QuantSpec(0.25, 0)})
        with self.assertRaisesRegex(NotImplementedError, "per-tensor"):
            LowerToHardware(plan).run(graph, CompilerContext(Device.get("k26")))


class Simulation(unittest.TestCase):
    @unittest.skipIf(Hdl.skip_reason("hardcaml"), "Hardcaml unavailable")
    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_nonzero_grids(self):
        shape = (1, 3, 3, 5)
        builder = ModelBuilder("quantized_global").add_input("x", shape)
        for name, value in (("si", np.array(0.125, np.float32)),
                            ("so", np.array(0.25, np.float32)),
                            ("zi", np.array(111, np.uint8)), ("zo", np.array(139, np.uint8))):
            builder.add_initializer(name, value)
        fixture = (builder.add_node("QuantizeLinear", ["x", "si", "zi"], ["q"])
                   .add_node("DequantizeLinear", ["q", "si", "zi"], ["d"])
                   .add_node("GlobalAveragePool", ["d"], ["mean"])
                   .add_node("QuantizeLinear", ["mean", "so", "zo"], ["qo"])
                   .add_node("DequantizeLinear", ["qo", "so", "zo"], ["y"])
                   .add_output("y", (1, 3, 1, 1)).build())
        x = np.random.default_rng(5).integers(0, 256, (8, *shape[1:]))
        x = ((x - 111) * 0.125).astype(np.float32)
        for hdl in ("sv", "hardcaml"):
            with self.subTest(hdl=hdl):
                acc = otf.load(fixture)
                acc.compile(device="k26", mode="stream", hdl=hdl,
                            out_dir=Fixtures.build_dir("global_quantized_" + hdl))
                expected = x.mean(axis=(2, 3)).reshape(8, 1, 1, 3)
                np.testing.assert_allclose(acc.predict(x), expected, atol=0.25)
                report = acc.simulate(x, sweep=True)
                self.assertTrue(report.ok, report)

    @unittest.skipIf(Hdl.skip_reason("hardcaml"), "Hardcaml unavailable")
    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_odd_and_single_pixel(self):
        for shape in ((1, 3, 3, 5), (1, 3, 1, 1)):
            x = np.random.default_rng(7).uniform(-1, 1, (5, *shape[1:])).astype(np.float32)
            x[0], x[1], x[2] = -1, 1, 0
            for hdl in ("sv", "hardcaml"):
                with self.subTest(shape=shape, hdl=hdl):
                    acc = otf.load(model(shape))
                    acc.compile(device="k26", mode="stream", hdl=hdl, calibration=x,
                                out_dir=Fixtures.build_dir("global_%s_%s" % (shape[2], hdl)))
                    expected = x.mean(axis=(2, 3)).reshape(5, 1, 1, 3)
                    np.testing.assert_allclose(acc.predict(x), expected, atol=2 / 127)
                    report = acc.simulate(x, sweep=True)
                    self.assertTrue(report.ok, report)
