"""Reductions and softmax in fixed latency mode: the units against float, two designs simulated bit exact, the latency proof and the errors."""

import pathlib
import shutil
import unittest

import numpy as np

from support import Fixtures, Hdl
from test_fixed_latency import prove

import onnx2fpga as otf
from onnx2fpga.board import IcepiZero
from onnx2fpga.errors import ConfigError, UnsupportedOpError
from onnx2fpga.p1_ingest import ModelBuilder
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.p3_ops.reductions import ReduceUnit, SoftmaxUnit
from onnx2fpga.p4_quantize.calibrate import Calibrator
from onnx2fpga.p4_quantize.lower import LowerToHardware
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.targets.device import Device

HARDCAML = Hdl.skip_reason("hardcaml")
VERILATOR = None if shutil.which("verilator") else "verilator not on PATH"
CLOCK = 322.265625


def single(op, lanes=8, opset=13, **attrs):
    builder = ModelBuilder(op.lower(), opset=opset).add_input("x", (1, lanes))
    builder.add_node(op, ["x"], ["y"], name="op", **attrs)
    return builder.add_output("y", (1, lanes)).build()


def lowered(model, lanes=8, spread=3.0):
    """The calibrated plan and the fixed latency unit of a one node model."""
    graph = OnnxImporter(model).run()
    samples = [{"x": np.random.default_rng(seed).normal(0, spread, (1, lanes))}
               for seed in range(16)]
    plan = Calibrator(graph).plan(samples)
    hardware = LowerToHardware(plan, fixed=True).run(graph, CompilerContext(Device.get("sim")))
    (unit,) = hardware.nodes
    return plan, unit


def codes(lanes, count=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(np.rint(rng.normal(0, 50, (count, lanes))), -128, 127).astype(np.int64)


class Units(unittest.TestCase):
    def against(self, op, reference, lanes, **attrs):
        """Output codes of the unit and of the float function rounded onto the output grid."""
        plan, unit = lowered(single(op, lanes, **attrs), lanes)
        x = codes(lanes)
        got = np.concatenate([unit.execute(None, {"x": row[None]})[0] for row in x])
        real = reference(plan.scale("x") * (x - plan.zero_point("x")))
        wanted = plan.act_dtype.clamp(np.rint(real / plan.scale("y")) + plan.zero_point("y"))
        return got, wanted

    def test_softmax(self):
        """Within one code of the correctly rounded softmax, and equal on at least 98 percent."""
        def softmax(x):
            e = np.exp(x - x.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        for lanes in (3, 10, 16):
            with self.subTest(lanes=lanes):
                got, wanted = self.against("Softmax", softmax, lanes, axis=-1)
                self.assertLessEqual(np.abs(got - wanted).max(), 1)
                self.assertGreaterEqual(np.mean(got == wanted), 0.98)

    def test_log_softmax(self):
        def log_softmax(x):
            shifted = x - x.max(axis=1, keepdims=True)
            return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

        got, wanted = self.against("LogSoftmax", log_softmax, 10, axis=1)
        self.assertLessEqual(np.abs(got - wanted).max(), 1)
        self.assertGreaterEqual(np.mean(got == wanted), 0.98)

    def test_normalizer(self):
        """L1 within one code, and a vector of zeros gives zeros."""
        def l1(x):
            total = np.abs(x).sum(axis=1, keepdims=True)
            return x / np.where(total == 0, 1, total)

        got, wanted = self.against("Normalizer", l1, 6, norm="L1")
        self.assertLessEqual(np.abs(got - wanted).max(), 1)
        plan, unit = lowered(single("Normalizer", 6, norm="L1"), 6)
        zeros = np.full((1, 6), plan.zero_point("x"))
        np.testing.assert_array_equal(unit.execute(None, {"x": zeros})[0],
                                      np.full((1, 6), plan.zero_point("y")))

    def test_l2(self):
        """L2 within one code on 1, 6 and 16 lanes, and a vector of zeros gives zeros."""
        def l2(x):
            total = np.sqrt((x * x).sum(axis=1, keepdims=True))
            return x / np.where(total == 0, 1, total)

        for lanes in (1, 6, 16):
            with self.subTest(lanes=lanes):
                got, wanted = self.against("Normalizer", l2, lanes, norm="L2")
                self.assertLessEqual(np.abs(got - wanted).max(), 1)
                self.assertGreaterEqual(np.mean(got == wanted), 0.98)
        plan, unit = lowered(single("Normalizer", 6, norm="L2"), 6)
        self.assertEqual(unit.mode, "root")
        zeros = np.full((1, 6), plan.zero_point("x"))
        np.testing.assert_array_equal(unit.execute(None, {"x": zeros})[0],
                                      np.full((1, 6), plan.zero_point("y")))

    def test_max(self):
        """x over its largest value, as ONNX Runtime divides, and x itself when that is 0."""
        def by_max(x):
            top = x.max(axis=1, keepdims=True)
            return x / np.where(top == 0, 1, top)

        for lanes in (2, 6, 16):
            with self.subTest(lanes=lanes):
                got, wanted = self.against("Normalizer", by_max, lanes, norm="MAX")
                self.assertLessEqual(np.abs(got - wanted).max(), 1)
                self.assertGreaterEqual(np.mean(got == wanted), 0.98)
        plan, unit = lowered(single("Normalizer", 6, norm="MAX"), 6)
        self.assertEqual((unit.mode, unit.latency), ("max", 3))
        x = plan.zero_point("x") + np.array([[-40, -3, 0, -7, -1, -20]])
        real = plan.scale("x") * (x - plan.zero_point("x"))
        wanted = np.rint(real / plan.scale("y")) + plan.zero_point("y")
        np.testing.assert_array_equal(unit.execute(None, {"x": x})[0], wanted)

    def test_arg(self):
        """The first index of a tie, or the last with select_last_index."""
        x = np.clip(codes(10), -2, 2)
        for op, last in (("ArgMax", 0), ("ArgMax", 1), ("ArgMin", 0), ("ArgMin", 1)):
            with self.subTest(op=op, last=last):
                _, unit = lowered(single(op, 10, axis=1, select_last_index=last), 10)
                got = [unit.execute(None, {"x": row[None]})[0].item() for row in x]
                pick = np.argmax if op == "ArgMax" else np.argmin
                wanted = 9 - pick(x[:, ::-1], axis=1) if last else pick(x, axis=1)
                np.testing.assert_array_equal(got, wanted)
                self.assertEqual(unit.out_dtype.bits, 4)

    def test_reduce(self):
        """Max and Min are codes on the input grid; Sum and Mean are within a code of the real value."""
        x = codes(10)
        for op, reference in (("ReduceMax", np.max), ("ReduceMin", np.min),
                              ("ReduceSum", np.sum), ("ReduceMean", np.mean)):
            with self.subTest(op=op):
                plan, unit = lowered(single(op, 10, axes=[1], keepdims=0), 10)
                got = np.array([unit.execute(None, {"x": row[None]})[0].item() for row in x])
                real = reference(plan.scale("x") * (x - plan.zero_point("x")), axis=1)
                wanted = plan.act_dtype.clamp(np.rint(real / plan.scale("y"))
                                              + plan.zero_point("y"))
                self.assertLessEqual(np.abs(got - wanted).max(), 1 if op in ("ReduceSum",
                                                                              "ReduceMean") else 0)


class Import(unittest.TestCase):
    def test_argmax_of_softmax(self):
        """An ArgMax that is a Softmax's only reader takes the Softmax's input instead."""
        builder = ModelBuilder("label").add_input("x", (1, 5))
        builder.add_node("Softmax", ["x"], ["p"], name="sm", axis=1)
        builder.add_node("ArgMax", ["p"], ["y"], name="am", axis=1, keepdims=0)
        graph = OnnxImporter(builder.add_output("y", (1,)).build()).run()
        self.assertEqual([(n.op_type, n.inputs) for n in graph.nodes], [("ArgMax", ["x"])])

    def test_normalizer_of_softmax(self):
        builder = ModelBuilder("probabilities").add_input("x", (1, 5))
        builder.add_node("Softmax", ["x"], ["p"], name="sm", axis=1)
        builder.add_node("Normalizer", ["p"], ["y"], name="l1", norm="L1")
        graph = OnnxImporter(builder.add_output("y", (1, 5)).build()).run()
        self.assertEqual([n.op_type for n in graph.nodes], ["Softmax"])
        self.assertEqual(graph.outputs, ["p"])

    def test_axes(self):
        """Axes from the attribute up to opset 17, from a constant input from 18; none means all."""
        for opset, attrs, inputs in ((13, {"axes": [-1], "keepdims": 0}, []),
                                     (18, {}, ["axes"]), (18, {}, [])):
            with self.subTest(opset=opset, inputs=inputs):
                builder = ModelBuilder("mean", opset=opset).add_input("x", (1, 4))
                builder.add_initializer("axes", np.array([1], dtype=np.int64))
                builder.add_node("ReduceMean", ["x"] + inputs, ["y"], name="mean", **attrs)
                graph = OnnxImporter(builder.add_output("y", (1, 1)).build()).run()
                self.assertEqual(graph.tensor("y").shape, (1,) if attrs else (1, 1))

    def test_classifier(self):
        """A three class classifier with post transform NONE gives its label now, not its scores."""
        builder = ModelBuilder("classes").add_input("x", (1, 4))
        builder.add_node("LinearClassifier", ["x"], ["label", "scores"], name="lc",
                         coefficients=list(np.arange(12.0)), intercepts=[0.0, 1.0, 2.0],
                         classlabels_ints=[0, 1, 2], post_transform="NONE")
        model = builder.add_output("label", (1,), elem_type=7).add_output("scores", (1, 3))
        graph = OnnxImporter(model.build()).run()
        self.assertEqual([n.op_type for n in graph.nodes], ["Gemm", "ArgMax"])
        self.assertEqual(graph.outputs, ["label"])


class Errors(unittest.TestCase):
    def test_streaming(self):
        acc = otf.load(single("Softmax", axis=1))
        with self.assertRaisesRegex(ConfigError, r"cannot build op \(Softmax\); it needs "
                                                 r"mode='fixed'"):
            acc.compile(calibration=np.zeros((1, 8)))

    def test_refused(self):
        for model, text in ((single("ArgMax", axis=0), "works on axis 0"),
                            (single("Normalizer", norm="L3"), "has norm L3"),
                            (single("ReduceSum", noop_with_empty_axes=1), "noop_with_empty")):
            with self.subTest(text), self.assertRaisesRegex(UnsupportedOpError, text):
                OnnxImporter(model).run()

    def test_index_read(self):
        builder = ModelBuilder("index").add_input("x", (1, 4))
        builder.add_initializer("one", np.float32(1.0))
        builder.add_node("ArgMax", ["x"], ["i"], name="am", axis=1)
        builder.add_node("Add", ["i", "one"], ["y"], name="next")
        with self.assertRaisesRegex(ConfigError, "am .ArgMax. gives an index"):
            lowered(builder.add_output("y", (1, 1)).build(), 4)


def ensemble(width=6, seed=2):
    """Two softmax heads on one input, averaged, then the label."""
    rng = np.random.default_rng(seed)
    builder = ModelBuilder("ensemble").add_input("x", (1, 16))
    for head in ("a", "b"):
        builder.add_initializer("w" + head, rng.normal(0, 0.5, (16, width)).astype(np.float32))
        builder.add_initializer("b" + head, rng.normal(0, 0.2, width).astype(np.float32))
        builder.add_node("Gemm", ["x", "w" + head, "b" + head], ["h" + head], name="fc_" + head)
        builder.add_node("Softmax", ["h" + head], ["p" + head], name="softmax_" + head, axis=-1)
    builder.add_node("Add", ["pa", "pb"], ["p"], name="mix")
    builder.add_node("ArgMax", ["p"], ["y"], name="label", axis=1, keepdims=0)
    return builder.add_output("y", (1,), elem_type=7).build()


def pooled(seed=3):
    """LogSoftmax, the L1 Normalizer and every Reduce op on one layer, joined into a number."""
    rng = np.random.default_rng(seed)
    builder = ModelBuilder("pooled", opset=18).add_input("x", (1, 16))
    builder.add_initializer("w", rng.normal(0, 0.5, (16, 8)).astype(np.float32))
    builder.add_initializer("axes", np.array([1], dtype=np.int64))
    builder.add_node("Gemm", ["x", "w"], ["h"], name="fc")
    builder.add_node("LogSoftmax", ["h"], ["ls"], name="logsoftmax", axis=1)
    builder.add_node("ReduceMax", ["ls", "axes"], ["top"], name="top")
    builder.add_node("Normalizer", ["h"], ["n"], name="l1", norm="L1")
    builder.add_node("ReduceSum", ["n", "axes"], ["total"], name="total")
    builder.add_node("ReduceMin", ["h", "axes"], ["low"], name="low")
    builder.add_node("ReduceMean", ["h", "axes"], ["mean"], name="mean")
    builder.add_node("Sub", ["top", "low"], ["spread"], name="spread")
    builder.add_node("Add", ["total", "mean"], ["level"], name="level")
    builder.add_node("Add", ["spread", "level"], ["y"], name="out")
    return builder.add_output("y", (1, 1)).build()


def norms(seed=4):
    """One dense layer through an L2 and a MAX Normalizer, added lane by lane."""
    rng = np.random.default_rng(seed)
    builder = ModelBuilder("norms").add_input("x", (1, 16))
    builder.add_initializer("w", rng.normal(0, 0.5, (16, 8)).astype(np.float32))
    builder.add_node("Gemm", ["x", "w"], ["h"], name="fc")
    builder.add_node("Normalizer", ["h"], ["l2"], name="l2", norm="L2")
    builder.add_node("Normalizer", ["h"], ["top"], name="max", norm="MAX")
    builder.add_node("Add", ["l2", "top"], ["y"], name="out")
    return builder.add_output("y", (1, 8)).build()


def compiled(model, name, **options):
    acc = otf.load(model, name=name)
    x = np.random.default_rng(1).normal(0, 1, (64, 16)).astype(np.float32)
    acc.compile(device="k26", mode="fixed", calibration=x,
                out_dir=Fixtures.build_dir("reductions_" + name), **options)
    return acc


@unittest.skipIf(HARDCAML, HARDCAML)
class Designs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ensemble = compiled(ensemble(), "ensemble")
        cls.fast = compiled(ensemble(), "ensemble_fast", clock_mhz=CLOCK)
        cls.pooled = compiled(pooled(), "pooled", clock_mhz=CLOCK)
        cls.norms = compiled(norms(), "norms", clock_mhz=CLOCK)

    def test_latency(self):
        """Dense 3, softmax 5, add 2 and argmax 1; at 322 MHz every unit fits the k26 model."""
        self.assertEqual(self.ensemble.latency, 3 + 5 + 2 + 1)
        kinds = {type(unit) for unit in self.pooled.result.hardware_graph.nodes}
        self.assertTrue({ReduceUnit, SoftmaxUnit} <= kinds)
        for acc in (self.fast, self.pooled, self.norms):
            self.assertGreaterEqual(acc.result.fmax_mhz, CLOCK)
        modes = {unit.name: unit.mode for unit in self.norms.result.hardware_graph.nodes
                 if isinstance(unit, SoftmaxUnit)}
        self.assertEqual(modes, {"l2": "root", "max": "max"})

    @unittest.skipIf(VERILATOR, VERILATOR)
    def test_simulate(self):
        for acc in (self.ensemble, self.fast, self.pooled, self.norms):
            with self.subTest(acc.name):
                report = acc.simulate()
                self.assertTrue(report.ok, report.first.output[-2000:])
                self.assertEqual((report.frames, report.latency), (10000, acc.latency))

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        for acc in (self.ensemble, self.norms):
            with self.subTest(acc.name):
                build, latency = pathlib.Path(acc.out_dir), acc.latency
                ok, log = prove(build, 16 * 8, latency)
                self.assertTrue(ok, log[-3000:])
                self.assertFalse(prove(build, 16 * 8, latency - 1)[0])


if __name__ == "__main__":
    unittest.main()
