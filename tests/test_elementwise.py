"""Lookup tables and elementwise operators in fixed latency mode: the tables, the units against the integer model, the latency proof and the errors."""

import copy
import json
import math
import pathlib
import shutil
import unittest

import numpy as np

from support import Fixtures, Hdl
from test_fixed_latency import prove

import onnx2fpga as otf
from onnx2fpga import reload
from onnx2fpga.board import IcepiZero
from onnx2fpga.errors import ConfigError, UnsupportedOpError
from onnx2fpga.p1_ingest import ModelBuilder
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.p3_ops.elementwise import EltwiseUnit, TableUnit
from onnx2fpga.p4_quantize.calibrate import Calibrator
from onnx2fpga.p4_quantize.lower import LowerToHardware
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.reference.runner import GraphRunner
from onnx2fpga.targets.device import Device

HARDCAML = Hdl.skip_reason("hardcaml")
VERILATOR = None if shutil.which("verilator") else "verilator not on PATH"
WIDTH = 8
SLOPE = np.linspace(-0.3, 0.4, WIDTH).astype(np.float32)
OFFSET = np.linspace(-1, 1, WIDTH).astype(np.float32)
MASK = np.array([1, 0] * (WIDTH // 2), dtype=bool)


def selu(x):
    return 1.05070102214813232421875 * (x if x > 0 else 1.67326319217681884765625 * math.expm1(x))


def log(x):
    return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)


# Written with the math module, apart from the numpy definitions they check.
UNARY = {
    "Sigmoid": lambda x: 1 / (1 + math.exp(-x)),
    "Tanh": math.tanh,
    "HardSigmoid": lambda x: max(0.0, min(1.0, 0.2 * x + 0.5)),
    "HardSwish": lambda x: x * max(0.0, min(1.0, x / 6 + 0.5)),
    "Elu": lambda x: x if x >= 0 else math.expm1(x),
    "Selu": selu,
    "Gelu": lambda x: 0.5 * x * (1 + math.erf(x / math.sqrt(2))),
    "Softplus": lambda x: math.log1p(math.exp(x)),
    "Softsign": lambda x: x / (1 + abs(x)),
    "Exp": math.exp,
    "Log": log,
    "Sqrt": lambda x: math.sqrt(x) if x >= 0 else math.nan,
    "Reciprocal": lambda x: 1 / x if x else math.inf,
    "Erf": math.erf,
    "Mish": lambda x: x * math.tanh(math.log1p(math.exp(x))),
    "LeakyRelu": lambda x: x if x >= 0 else 0.01 * x,
    "Relu": lambda x: max(x, 0.0),
    "Neg": lambda x: -x,
    "Abs": abs,
    "Not": lambda x: float(x == 0),
}

# One live operand x and constants, each a table per lane: (op, operands, attributes, reference).
WITH_CONSTANTS = [
    ("PRelu", ["x", SLOPE], {}, lambda x, s: x if x >= 0 else s * x),
    ("Clip", ["x", np.float32(-0.5), np.float32(1.0)], {}, lambda x, lo, hi: min(max(x, lo), hi)),
    ("Gelu", ["x"], {"approximate": "tanh"}, lambda x: 0.5 * x * (1 + math.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))),
    ("LeakyRelu", ["x"], {"alpha": 0.2}, lambda x: x if x >= 0 else 0.2 * x),
    ("Add", ["x", OFFSET], {}, lambda x, c: x + c),
    ("Sub", [OFFSET, "x"], {}, lambda c, x: c - x),
    ("Mul", ["x", np.float32(-1.5)], {}, lambda x, c: x * c),
    ("Mul", ["x", "x"], {}, lambda x, y: x * y),
    ("Div", ["x", np.float32(3.0)], {}, lambda x, c: x / c),
    ("Max", ["x", np.float32(0.25)], {}, max),
    ("Min", [OFFSET, "x"], {}, min),
    ("Greater", ["x", np.float32(0.3)], {}, lambda x, c: float(x > c)),
    ("LessOrEqual", ["x", OFFSET], {}, lambda x, c: float(x <= c)),
    ("Equal", ["x", np.float32(0.0)], {}, lambda x, c: float(x == c)),
    ("And", ["x", MASK], {}, lambda x, m: float(bool(x) and bool(m))),
    ("Where", [MASK, "x", np.float32(0.7)], {}, lambda m, x, c: x if m else c),
]


def single(op, operands, attrs=None):
    """A model of one op on x, shaped (1, WIDTH), its constants as initializers."""
    builder = ModelBuilder(op.lower()).add_input("x", (1, WIDTH))
    names = []
    for index, operand in enumerate(operands):
        if isinstance(operand, str):
            names.append(operand)
        else:
            names.append("c%d" % index)
            builder.add_initializer(names[-1], np.asarray(operand))
    builder.add_node(op, names, ["y"], name="op", **(attrs or {}))
    return builder.add_output("y", (1, WIDTH)).build()


def lowered(model):
    """The float graph, the calibrated plan and the fixed latency units of a model."""
    graph = OnnxImporter(model).run()
    samples = [{"x": np.random.default_rng(seed).normal(0, 2, (1, WIDTH))} for seed in range(16)]
    plan = Calibrator(graph).plan(samples)
    hardware = LowerToHardware(plan, fixed=True).run(graph, CompilerContext(Device.get("sim")))
    return plan, hardware


def expected_codes(reference, unit, plan, constants):
    """Each lane's table from the reference: rounded, clamped, NaN at the zero point."""
    dtype, source, out = plan.act_dtype, unit.inputs[0], unit.outputs[0]
    rows, ties = [], []
    for lane in range(unit.lanes):
        row, tie = [], []
        for code in range(dtype.min, dtype.max + 1):
            x = plan.scale(source) * (code - plan.zero_point(source))
            with np.errstate(all="ignore"):
                try:
                    value = reference(*[x if c is None else c[lane] for c in constants])
                except OverflowError:
                    value = math.inf
            ratio = value / plan.scale(out)
            tie.append(not math.isnan(ratio) and abs(abs(ratio % 1) - 0.5) < 1e-6)
            if math.isnan(ratio):
                row.append(plan.zero_point(out))
            else:
                row.append(int(np.clip(np.rint(np.clip(ratio, -1e6, 1e6)) + plan.zero_point(out),
                                       dtype.min, dtype.max)))
        rows.append(row)
        ties.append(tie)
    return np.array(rows), np.array(ties)


class Tables(unittest.TestCase):
    def check(self, op, operands, attrs, reference):
        plan, hardware = lowered(single(op, operands, attrs))
        (unit,) = hardware.nodes
        self.assertIsInstance(unit, TableUnit)
        constants = [None if isinstance(o, str) else
                     np.broadcast_to(np.asarray(o, dtype=np.float64), (WIDTH,)) for o in operands]
        expected, ties = expected_codes(reference, unit, plan, constants)
        got = unit.rows()
        self.assertTrue(np.all((got == expected) | (ties & (np.abs(got - expected) <= 1))),
                        "%s differs at %d entries" % (op, np.sum(got != expected)))
        return unit

    def test_unary(self):
        for op, reference in UNARY.items():
            with self.subTest(op=op):
                unit = self.check(op, ["x"], {}, reference)
                self.assertEqual(unit.tables.shape, (1, 256))

    def test_with_constants(self):
        for op, operands, attrs, reference in WITH_CONSTANTS:
            with self.subTest(op=op, operands=len(operands)):
                self.check(op, operands, attrs, reference)

    def test_per_channel(self):
        """A slope per channel gives a table per distinct slope; a scalar constant gives one."""
        plan, hardware = lowered(single("PRelu", ["x", SLOPE]))
        self.assertEqual(hardware.nodes[0].tables.shape, (WIDTH, 256))
        plan, hardware = lowered(single("Max", ["x", np.float32(0.1)]))
        self.assertEqual(hardware.nodes[0].tables.shape, (1, 256))

    def test_booleans(self):
        """A comparison's output is 0 or 1 on a grid of scale 1, whatever calibration saw."""
        plan, hardware = lowered(single("Less", ["x", np.float32(0.0)]))
        self.assertEqual((plan.scale("y"), plan.zero_point("y")), (1.0, 0))
        self.assertEqual(sorted(set(hardware.nodes[0].rows().reshape(-1))), [0, 1])


def gate_model(seed=3):
    """fc0, a gate x * Sigmoid(x), every table op on a branch, summed in a tree of binary ops."""
    rng = np.random.default_rng(seed)
    builder = ModelBuilder("activations").add_input("x", (1, 16))
    builder.add_initializer("w0", rng.normal(0, 0.4, (16, WIDTH)).astype(np.float32))
    builder.add_initializer("b0", rng.normal(0, 0.1, WIDTH).astype(np.float32))
    builder.add_initializer("slope", SLOPE)
    builder.add_initializer("lo", np.float32(-0.5))
    builder.add_initializer("hi", np.float32(1.0))
    builder.add_node("Gemm", ["x", "w0", "b0"], ["h"], name="fc0")
    builder.add_node("Sigmoid", ["h"], ["s"], name="sigmoid")
    builder.add_node("Mul", ["h", "s"], ["g"], name="gate")
    branches = []
    for op in sorted(set(UNARY) - {"Not"}) + ["PRelu", "Clip"]:
        extra = {"PRelu": ["slope"], "Clip": ["lo", "hi"]}.get(op, [])
        source = "s" if op in ("Log", "Sqrt", "Reciprocal") else "g"
        builder.add_node(op, [source] + extra, [op.lower()], name=op.lower())
        branches.append(op.lower())
    joins = ["Add", "Sub", "Max", "Min", "Mul"]
    level = 0
    while len(branches) > 1:
        merged = []
        for index, pair in enumerate(zip(branches[::2], branches[1::2])):
            name = "j%d_%d" % (level, index)
            builder.add_node(joins[(index + level) % len(joins)], list(pair), [name], name=name)
            merged.append(name)
        branches = merged + branches[len(branches) // 2 * 2:]
        level += 1
    builder.add_node("Add", [branches[0], "h"], ["y"], name="out")
    return builder.add_output("y", (1, WIDTH)).build()


def logic_model(seed=4):
    """Every comparison, And, Or, Not and Where between two dense layers' outputs."""
    rng = np.random.default_rng(seed)
    builder = ModelBuilder("logic").add_input("x", (1, 16))
    for name in ("a", "b"):
        builder.add_initializer("w" + name, rng.normal(0, 0.4, (16, WIDTH)).astype(np.float32))
        builder.add_initializer("b" + name, rng.normal(0, 0.1, WIDTH).astype(np.float32))
        builder.add_node("Gemm", ["x", "w" + name, "b" + name], [name], name="fc_" + name)
    builder.add_initializer("mask", MASK)
    builder.add_initializer("half", np.float32(0.5))
    nodes = [("Less", ["a", "b"], "lt"), ("Greater", ["a", "b"], "gt"),
             ("LessOrEqual", ["a", "b"], "le"), ("GreaterOrEqual", ["a", "b"], "ge"),
             ("Equal", ["lt", "gt"], "eq"), ("Not", ["le"], "nle"),
             ("And", ["ge", "nle"], "both"), ("Or", ["eq", "both"], "either"),
             ("Where", ["either", "a", "b"], "w0"), ("Where", ["lt", "a", "half"], "w1"),
             ("Where", ["mask", "w0", "w1"], "w2"), ("Add", ["w2", "either"], "y")]
    for op, inputs, out in nodes:
        builder.add_node(op, inputs, [out], name=out + "_op")
    return builder.add_output("y", (1, WIDTH)).build()


def compiled(model, name, **options):
    acc = otf.load(model, name=name)
    x = np.random.default_rng(1).normal(0, 1, (64, 16)).astype(np.float32)
    acc.compile(mode="fixed", calibration=x, out_dir=Fixtures.build_dir("elementwise_" + name),
                **options)
    return acc


@unittest.skipIf(HARDCAML, HARDCAML)
class Designs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = compiled(gate_model(), "gate", device="k26", clock_mhz=322.265625)
        cls.logic = compiled(logic_model(), "logic")

    def test_latency(self):
        """The latency is the longest path, and each binary unit's earlier input waits for it."""
        graph = self.logic.result.hardware_graph
        units = {unit.name: unit for unit in graph.nodes}
        self.assertEqual(self.logic.latency, 3 + 2 + 2 + 1 + 1 + 1 + 2)
        self.assertEqual(units["w0_op"].skew, [0, 5, 5])
        design = json.loads((self.logic.out_dir / "design.json").read_text())
        kinds = {node["kind"] for node in design["nodes"]}
        self.assertEqual(kinds, {"otf_dense", "otf_table", "otf_eltwise"})
        self.assertEqual(units["nle_op"].latency, 1)

    @unittest.skipIf(VERILATOR, VERILATOR)
    def test_simulate(self):
        for acc in (self.gate, self.logic):
            with self.subTest(acc.name):
                report = acc.simulate()
                self.assertTrue(report.ok, report.first.output[-2000:])
                self.assertEqual((report.frames, report.latency, report.cycles),
                                 (10000, acc.latency, 10000 + acc.latency))

    def test_clock(self):
        """At the trading clock every unit fits the k26 model, and the elementwise ones took stages."""
        self.assertGreaterEqual(self.gate.result.fmax_mhz, 322.265625)
        units = self.gate.result.hardware_graph.nodes
        self.assertTrue(any(sum(unit.stages.values()) for unit in units
                            if isinstance(unit, EltwiseUnit)))
        lines = []
        self.gate.summary(print_fn=lines.append)
        self.assertRegex("\n".join(lines), r"gate \(Mul\)\s+8 -> 8\s+mul\s+8")

    @unittest.skipIf(VERILATOR, VERILATOR)
    def test_compose(self):
        """Tables in a row become one table, the same integer function as the chain."""
        builder = ModelBuilder("chain").add_input("x", (1, 16))
        builder.add_initializer("two", np.float32(2.0))
        builder.add_node("Sigmoid", ["x"], ["s"], name="sig")
        builder.add_node("Mul", ["s", "two"], ["m"], name="double")
        builder.add_node("Tanh", ["m"], ["y"], name="tanh")
        model = builder.add_output("y", (1, 16)).build()
        acc = compiled(model, "chain")
        (unit,) = acc.result.hardware_graph.nodes
        self.assertEqual((unit.name, unit.attrs["merged"], acc.latency), ("tanh", ["sig", "double"], 1))
        chain = LowerToHardware(acc.result.plan, fixed=True).run(
            copy.deepcopy(acc.result.float_graph), CompilerContext(Device.get("sim")))
        codes = {"x": np.tile(np.arange(-128, 128)[:, None], (1, 16))}
        np.testing.assert_array_equal(GraphRunner(chain).run(codes)["y"],
                                      GraphRunner(acc.result.hardware_graph).run(codes)["y"])
        report = acc.simulate()
        self.assertTrue(report.ok, report.first.output[-2000:])
        lines = []
        acc.summary(print_fn=lines.append)
        self.assertRegex("\n".join(lines), r"tanh \(Sigmoid\+Mul\+Tanh\)\s+16 -> 16\s+table")

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        build, latency = pathlib.Path(self.logic.out_dir), self.logic.latency
        ok, log = prove(build, 16 * 8, latency)
        self.assertTrue(ok, log[-3000:])
        self.assertFalse(prove(build, 16 * 8, latency - 1)[0])


def wide_deep_model(seed=6):
    """The raw features beside a table of them and a hidden layer, then one output."""
    rng = np.random.default_rng(seed)
    builder = ModelBuilder("wide_deep").add_input("x", (1, 16))
    builder.add_initializer("w0", rng.normal(0, 0.4, (16, WIDTH)).astype(np.float32))
    builder.add_initializer("b0", rng.normal(0, 0.1, WIDTH).astype(np.float32))
    builder.add_initializer("w1", rng.normal(0, 0.3, (16 + 16 + WIDTH, 1)).astype(np.float32))
    builder.add_node("Gemm", ["x", "w0", "b0"], ["h"], name="fc0")
    builder.add_node("Relu", ["h"], ["r"], name="relu")
    builder.add_node("Sigmoid", ["x"], ["s"], name="sig")
    builder.add_node("Concat", ["x", "s", "r"], ["c"], name="join", axis=1)
    builder.add_node("MatMul", ["c", "w1"], ["y"], name="fc1")
    return builder.add_output("y", (1, 1)).build()


@unittest.skipIf(HARDCAML, HARDCAML)
class Concat(unittest.TestCase):
    """Pieces that arrive at different times wait in registers, and each moves onto the output grid."""

    @classmethod
    def setUpClass(cls):
        cls.acc = compiled(wide_deep_model(), "wide_deep", device="k26", clock_mhz=322.265625)

    def test_latency(self):
        units = {unit.name: unit for unit in self.acc.result.hardware_graph.nodes}
        self.assertEqual(units["join"].skew, [7, 6, 0])
        self.assertEqual(self.acc.latency, units["fc0"].latency + 1 + units["fc1"].latency)
        design = json.loads((self.acc.out_dir / "design.json").read_text())
        self.assertIn("otf_concat", {node["kind"] for node in design["nodes"]})

    def test_tracks_float(self):
        x = np.random.default_rng(2).normal(0, 1, (256, 16)).astype(np.float32)
        floating = np.concatenate([self.acc._float_reference({"x": v[None]}) for v in x])
        error = self.acc.evaluate(x)["mean_abs_error"] / np.std(floating)
        self.assertLess(error, 0.03)

    @unittest.skipIf(VERILATOR, VERILATOR)
    def test_simulate(self):
        report = self.acc.simulate()
        self.assertTrue(report.ok, report.first.output[-2000:])
        self.assertEqual((report.frames, report.latency), (10000, self.acc.latency))

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        build, latency = pathlib.Path(self.acc.out_dir), self.acc.latency
        ok, log = prove(build, 16 * 8, latency)
        self.assertTrue(ok, log[-3000:])
        self.assertFalse(prove(build, 16 * 8, latency - 1)[0])

    def test_streaming(self):
        builder = ModelBuilder("join").add_input("x", (1, WIDTH))
        builder.add_initializer("w", np.ones((WIDTH, WIDTH), dtype=np.float32))
        builder.add_node("Gemm", ["x", "w"], ["h"], name="fc")
        builder.add_node("Relu", ["h"], ["r"], name="relu")
        builder.add_node("Concat", ["x", "r"], ["y"], name="join", axis=1)
        acc = otf.load(builder.add_output("y", (1, 2 * WIDTH)).build())
        with self.assertRaisesRegex(ConfigError, r"cannot build join \(Concat\); it needs "
                                                 r"mode='fixed'"):
            acc.compile(calibration=np.zeros((1, WIDTH)))


def prelu_model(seed, slope):
    rng = np.random.default_rng(seed)
    builder = ModelBuilder("prelu").add_input("x", (1, 16))
    builder.add_initializer("w0", rng.normal(0, 0.4, (16, WIDTH)).astype(np.float32))
    builder.add_initializer("w1", rng.normal(0, 0.4, (WIDTH, 4)).astype(np.float32))
    builder.add_initializer("slope", slope)
    builder.add_node("Gemm", ["x", "w0"], ["h"], name="fc0")
    builder.add_node("PRelu", ["h", "slope"], ["p"], name="prelu")
    builder.add_node("Gemm", ["p", "w1"], ["y"], name="fc1")
    return builder.add_output("y", (1, 4)).build()


@unittest.skipIf(HARDCAML, HARDCAML)
class Reload(unittest.TestCase):
    def test_frozen(self):
        """Retrained dense layers reload around a table on the grids it was built for; a new slope cannot."""
        acc = compiled(prelu_model(0, SLOPE), "prelu", weights="reloadable")
        samples = acc.settings["samples"]
        _, retrained = reload.retarget(prelu_model(1, SLOPE), acc.result, samples)
        tables = [[unit.rows() for unit in graph.nodes if isinstance(unit, TableUnit)]
                  for graph in (acc.result.hardware_graph, retrained)]
        np.testing.assert_array_equal(*tables)
        self.assertEqual(acc.latency, 3 + 1 + 3)
        with self.assertRaisesRegex(ConfigError, "prelu is frozen, and the new model changes it"):
            reload.retarget(prelu_model(1, 2 * SLOPE), acc.result, samples)


class Errors(unittest.TestCase):
    def test_streaming(self):
        acc = otf.load(single("Sigmoid", ["x"]))
        with self.assertRaisesRegex(ConfigError, r"cannot build op \(Sigmoid\); it needs "
                                                 r"mode='fixed'"):
            acc.compile(calibration=np.zeros((1, WIDTH)))

    def test_live_divisor(self):
        with self.assertRaisesRegex(UnsupportedOpError, "Div 'op' needs a constant divisor"):
            OnnxImporter(single("Div", ["x", "x"])).run()

    def test_shapes(self):
        builder = ModelBuilder("shapes").add_input("x", (1, WIDTH))
        builder.add_initializer("w", np.ones((WIDTH, 1), dtype=np.float32))
        builder.add_node("Gemm", ["x", "w"], ["h"], name="fc")
        builder.add_node("Max", ["x", "h"], ["y"], name="op")
        model = builder.add_output("y", (1, WIDTH)).build()
        with self.assertRaisesRegex(ConfigError, "live operands must have one shape"):
            lowered(model)


if __name__ == "__main__":
    unittest.main()
