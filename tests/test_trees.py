"""Tree ensembles: import against XGBoost and ONNX Runtime, the integer grid, and the compiled design."""

import pathlib
import shutil
import unittest

import numpy as np

from support import ROOT, Fixtures, Hdl
from test_fixed_latency import prove

import onnx2fpga as otf
from onnx2fpga.board import IcepiZero
from onnx2fpga.errors import BudgetError, ConfigError, UnsupportedOpError
from onnx2fpga.p1_ingest import OnnxModel
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.p3_ops.trees import LEAF, LEQ, LT
from onnx2fpga.p4_quantize.lower import LowerToHardware
from onnx2fpga.p4_quantize.trees import largest_code, leaf_scale, plan_trees
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.targets.device import Device

HARDCAML = Hdl.skip_reason("hardcaml")
FIXTURES = ROOT / "tests" / "fixtures"
EXPECTED = np.load(FIXTURES / "trees_expected.npz")


def model(name):
    return OnnxModel.load(FIXTURES / ("%s.onnx" % name))


def lowered(name):
    """The float op, the plan and the integer unit for a fixture."""
    graph = OnnxImporter(model(name)).run()
    op, plan = graph.nodes[0], plan_trees(graph)
    unit = LowerToHardware(plan).run(graph, CompilerContext(Device.get("sim"))).nodes[0]
    return op, plan, unit


def compiled(name):
    acc = otf.load(FIXTURES / ("%s.onnx" % name))
    acc.compile(device="vu9p", mode="fixed", out_dir=Fixtures.build_dir("trees_" + name))
    return acc


class Import(unittest.TestCase):
    def margin(self, name, x):
        op = OnnxImporter(model(name)).run().nodes[0]
        return op, op.margin(np.asarray(x, dtype=np.float64))

    def test_xgboost(self):
        x = EXPECTED["x"]
        op, margin = self.margin("xgb_regressor", x)
        self.assertEqual((op.ensemble.trees, op.ensemble.depth), (100, 4))
        np.testing.assert_allclose(margin, EXPECTED["regressor"], rtol=1e-5, atol=1e-3)
        op, margin = self.margin("xgb_classifier", x)
        np.testing.assert_allclose(margin, EXPECTED["classifier_margin"], atol=1e-4)
        self.assertEqual(list((margin > op.decision).astype(int)),
                         list(EXPECTED["classifier_label"]))

    def test_skl2onnx(self):
        graph = OnnxImporter(model("skl_gbc")).run()
        self.assertEqual(graph.outputs, ["label"])
        labels = graph.nodes[0].execute(graph, {"X": EXPECTED["x"]})[0].ravel()
        self.assertEqual(list(labels.astype(int)), list(EXPECTED["skl_gbc_label"]))

    def test_modes(self):
        """Opset 3 with double tensors, one tree per split mode, against ONNX Runtime."""
        expected = np.load(FIXTURES / "modes_v3.npz")
        _, margin = self.margin("modes_v3", expected["x"])
        np.testing.assert_allclose(margin, expected["y"].ravel(), atol=1e-6)

    def test_refused(self):
        softmax = model("xgb_classifier")
        node = softmax.graph.node[0]
        next(a for a in node.attribute if a.name == "post_transform").s = b"SOFTMAX"
        with self.assertRaisesRegex(UnsupportedOpError, "post_transform SOFTMAX"):
            otf.load(softmax)
        node.op_type = "TreeEnsemble"
        with self.assertRaisesRegex(UnsupportedOpError, "opset 5, which is not supported"):
            otf.load(softmax)


class Grid(unittest.TestCase):
    def test_thresholds(self):
        """For every split of every fixture, on every code of its feature, the integer test equals the float one."""
        codes = np.arange(-(1 << 15), 1 << 15)
        for name in ("xgb_regressor", "xgb_classifier", "modes_v3"):
            op, plan, unit = lowered(name)
            spec, ensemble = plan.spec(op.inputs[0]), op.ensemble
            for k in np.flatnonzero(np.isin(ensemble.mode, (LEQ, LT))):
                scale = spec.scale[ensemble.feature[k]]
                zero = spec.zero_point[ensemble.feature[k]]
                strict = ensemble.mode[k] == LT
                real = scale * (codes - zero)
                wanted = real < ensemble.threshold[k] if strict else real <= ensemble.threshold[k]
                found = largest_code(ensemble.threshold[k], scale, zero, plan.act_dtype, strict)
                np.testing.assert_array_equal(codes <= found, wanted)

    def test_leaves_on_grid(self):
        """On grid points every tree picks the leaf the float model picks, as a rounded leaf code."""
        rng = np.random.default_rng(2)
        for name in ("xgb_regressor", "xgb_classifier", "modes_v3"):
            op, plan, unit = lowered(name)
            spec = plan.spec(op.inputs[0])
            codes = rng.integers(-(1 << 15), 1 << 15, (2000, unit.features))
            grid = spec.scale * (codes - spec.zero_point)
            scale = leaf_scale(op.ensemble, plan.weight_dtype.bits)
            np.testing.assert_array_equal(unit.ensemble.leaf_values(codes),
                                          np.rint(op.ensemble.leaf_values(grid) / scale))

    def test_agreement(self):
        """On raw inputs the integer model differs only where a feature falls between a threshold and its code."""
        x = np.random.default_rng(3).normal(0, 1.5, (10000, 8))
        for name in ("xgb_regressor", "xgb_classifier"):
            op, plan, unit = lowered(name)
            codes = plan.quantize(op.inputs[0], x)
            scale = leaf_scale(op.ensemble, plan.weight_dtype.bits)
            same = np.all(unit.ensemble.leaf_values(codes)
                          == np.rint(op.ensemble.leaf_values(x) / scale), axis=1)
            self.assertGreater(same.mean(), 0.995, name)
            if unit.decision:
                labels = unit.execute(None, {op.inputs[0]: codes})[0].ravel()
                self.assertGreater(np.mean(labels == (op.margin(x) > op.decision)), 0.999)

    def test_pruned(self):
        """A split no code can take one way is cut, leaving its live side."""
        op, plan, unit = lowered("modes_v3")
        self.assertEqual(unit.ensemble.splits, 6)
        self.assertTrue(np.all(unit.ensemble.mode[unit.ensemble.roots] != LEAF))


class Errors(unittest.TestCase):
    def test_stream(self):
        acc = otf.load(FIXTURES / "xgb_regressor.onnx")
        with self.assertRaisesRegex(ConfigError, "only with mode='fixed'"):
            acc.compile(device="vu9p")

    @unittest.skipIf(HARDCAML, HARDCAML)
    def test_budget(self):
        acc = otf.load(FIXTURES / "xgb_regressor.onnx")
        with self.assertRaisesRegex(BudgetError, r"needs 20164 LUT.*100 trees of depth 4"):
            acc.compile(device="icepi25", mode="fixed")


@unittest.skipIf(HARDCAML, HARDCAML)
class Regressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acc = compiled("xgb_regressor")

    def test_latency(self):
        self.assertEqual(self.acc.latency, 6)
        lines = []
        self.acc.summary(print_fn=lines.append)
        text = "\n".join(lines)
        self.assertRegex(text, r"100/4\s+871\s+6")
        self.assertIn("Latency: 6 cycles", text)
        self.assertIn("features int16 on per feature grids", text)

    def test_evaluate(self):
        error = self.acc.evaluate(EXPECTED["x"])["mean_abs_error"]
        self.assertLess(error, 0.5)

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_simulate(self):
        report = self.acc.simulate(sweep=True)
        self.assertTrue(report.ok, report.first.output)
        self.assertEqual((report.frames, report.latency, report.cycles), (10000, 6, 10006))

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        build = pathlib.Path(self.acc.out_dir)
        ok, log = prove(build, 8 * 16, 6)
        self.assertTrue(ok, log[-3000:])
        self.assertFalse(prove(build, 8 * 16, 5)[0], "a latency of 5 must not prove")


@unittest.skipIf(HARDCAML, HARDCAML)
class Classifier(unittest.TestCase):
    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_simulate(self):
        acc = compiled("xgb_classifier")
        self.assertEqual(acc.evaluate(EXPECTED["x"], EXPECTED["classifier_label"])["agreement"],
                         1.0)
        report = acc.simulate()
        self.assertTrue(report.ok, report.first.output)
        self.assertEqual(report.latency, 6)
        answers = (acc.out_dir / "golden" / "x_expected.hex").read_text().split()
        self.assertEqual(sorted(set(answers)), ["0", "1"])


if __name__ == "__main__":
    unittest.main()
