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
# Three class classifiers from the corpus, with ONNX Runtime's labels.
MULTICLASS = ("xgb_multiclass", "lgbm_multiclass", "forest_multiclass")


def model(name):
    path = FIXTURES / ("%s.onnx" % name)
    return OnnxModel.load(path if path.exists() else FIXTURES / "corpus" / path.name)


def lowered(name):
    """The float op, the plan and the integer unit for a fixture."""
    graph = OnnxImporter(model(name)).run()
    op, plan = graph.nodes[0], plan_trees(graph)
    unit = LowerToHardware(plan).run(graph, CompilerContext(Device.get("sim"))).nodes[0]
    return op, plan, unit


def compiled(name, device="vu9p", clock_mhz=None):
    acc = otf.load(model(name))
    acc.compile(device=device, mode="fixed", clock_mhz=clock_mhz,
                out_dir=Fixtures.build_dir("trees_%s_%s_%s" % (name, device, clock_mhz)))
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

    def test_multiclass(self):
        """The label is the class with the largest score, whatever the post transform."""
        for name in MULTICLASS:
            with self.subTest(name):
                data = np.load(FIXTURES / "corpus" / ("%s.npz" % name))
                graph = OnnxImporter(model(name)).run()
                op = graph.nodes[0]
                self.assertEqual(op.ensemble.value.shape[1], 3)
                labels = op.execute(graph, {op.inputs[0]: data["x"]})[0]
                np.testing.assert_array_equal(labels, data["y"])

    def test_opset5(self):
        """TreeEnsemble keeps splits and leaves apart; the same trees give the same margin."""
        old, new = (OnnxImporter(model(name)).run().nodes[0] for name in ("xgb_small",
                                                                           "xgb_small_v5"))
        x = np.load(FIXTURES / "corpus" / "xgb_small.npz")["x"]
        np.testing.assert_allclose(new.margin(x), old.margin(x), atol=1e-5)
        self.assertEqual((new.ensemble.trees, new.ensemble.splits),
                         (old.ensemble.trees, old.ensemble.splits))
        for key, value, text in (("post_transform", 2, "post_transform 2"),
                                 ("aggregate_function", 3, "aggregate_function 3"),
                                 ("n_targets", 2, "2 targets")):
            with self.subTest(key):
                changed = model("xgb_small_v5")
                next(a for a in changed.graph.node[0].attribute if a.name == key).i = value
                with self.assertRaisesRegex(UnsupportedOpError, text):
                    otf.load(changed)
        member = model("xgb_small_v5")
        modes = next(a for a in member.graph.node[0].attribute if a.name == "nodes_modes").t
        modes.int32_data[0] = 6
        with self.assertRaisesRegex(UnsupportedOpError, "BRANCH_MEMBER"):
            otf.load(member)

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
        unscored = model("xgb_multiclass")
        node = unscored.graph.node[0]
        ids = next(a for a in node.attribute if a.name == "class_ids")
        ids.ints[:] = [min(k, 1) for k in ids.ints]
        with self.assertRaisesRegex(UnsupportedOpError, "3 classes and scores 2 of them"):
            otf.load(unscored)
        labels = next(a for a in node.attribute if a.name == "classlabels_int64s")
        labels.ints[:] = [0, 1]
        with self.assertRaisesRegex(UnsupportedOpError, "scores both of its 2 classes"):
            otf.load(unscored)


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

    def test_multiclass(self):
        """Each class sums its own copy of the trees that score it, and the sums never overflow."""
        x = np.random.default_rng(4).normal(0, 1.5, (4000, 8))
        for name in MULTICLASS:
            with self.subTest(name):
                op, plan, unit = lowered(name)
                self.assertEqual(unit.classes, 3)
                self.assertEqual(unit.out_dtype.bits, 2)
                codes = plan.quantize(op.inputs[0], x)
                labels = unit.execute(None, {op.inputs[0]: codes})[0].ravel()
                self.assertGreater(np.mean(labels == np.argmax(op.margin(x), axis=1)), 0.99)
                for score, bias in enumerate(unit.bias):
                    trees = [leaves for leaves, owner in zip(unit.ensemble.tree_leaves(),
                                                              unit.scores) if owner == score]
                    high = sum(unit.ensemble.value[leaves].max() for leaves in trees) + bias
                    low = sum(unit.ensemble.value[leaves].min() for leaves in trees) + bias
                    self.assertLessEqual(high, unit.sum_dtype.max)
                    self.assertGreaterEqual(low, unit.sum_dtype.min)

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
class Pipelined(unittest.TestCase):
    """At 500 MHz on k26 the adder tree takes every level registered, three more cycles."""

    @classmethod
    def setUpClass(cls):
        cls.acc = compiled("xgb_regressor", "k26", 500)

    def test_latency(self):
        unit = self.acc.result.hardware_graph.nodes[0]
        self.assertEqual((unit.stages, self.acc.latency), ({"select": 0, "sum": 3}, 9))
        self.assertGreaterEqual(self.acc.result.fmax_mhz, 500)

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_simulate(self):
        report = self.acc.simulate()
        self.assertTrue(report.ok, report.first.output)
        self.assertEqual((report.frames, report.latency), (10000, 9))

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        build = pathlib.Path(self.acc.out_dir)
        self.assertTrue(prove(build, 8 * 16, 9)[0])
        self.assertFalse(prove(build, 8 * 16, 8)[0], "a latency of 8 must not prove")


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


@unittest.skipIf(HARDCAML, HARDCAML)
class Multiclass(unittest.TestCase):
    """At 322 MHz on k26 the pick over three scores takes a second register."""

    @classmethod
    def setUpClass(cls):
        cls.acc = compiled("xgb_multiclass", "k26", 322.265625)

    def test_latency(self):
        unit = self.acc.result.hardware_graph.nodes[0]
        self.assertEqual((unit.stages, self.acc.latency), ({"select": 0, "sum": 0, "pick": 1}, 7))
        self.assertGreaterEqual(self.acc.result.fmax_mhz, 322.265625)

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_simulate(self):
        report = self.acc.simulate()
        self.assertTrue(report.ok, report.first.output)
        self.assertEqual((report.frames, report.latency), (10000, 7))
        answers = (self.acc.out_dir / "golden" / "x_expected.hex").read_text().split()
        self.assertEqual(sorted(set(answers)), ["0", "1", "2"])

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        build = pathlib.Path(self.acc.out_dir)
        self.assertTrue(prove(build, 8 * 16, 7)[0])
        self.assertFalse(prove(build, 8 * 16, 6)[0], "a latency of 6 must not prove")


@unittest.skipIf(HARDCAML, HARDCAML)
class Narrow(unittest.TestCase):
    """12 bit features and 8 bit leaves on the three class XGBoost model at 322 MHz."""

    @classmethod
    def setUpClass(cls):
        cls.acc = otf.load(model("xgb_multiclass"))
        cls.acc.layers[next(iter(cls.acc.layers))].quantize(weights="int8", activations="int12")
        cls.acc.compile(device="k26", mode="fixed", clock_mhz=322.265625,
                        out_dir=Fixtures.build_dir("trees_narrow"))

    def test_widths(self):
        unit = self.acc.result.hardware_graph.nodes[0]
        self.assertEqual((unit.feature_dtype.bits, unit.leaf_dtype.bits), (12, 8))
        lines = []
        self.acc.summary(print_fn=lines.append)
        self.assertIn("features int12", lines[0])
        wide = compiled("xgb_multiclass", "k26", 322.265625).result.hardware_graph.nodes[0]
        self.assertLess(unit.resources(self.acc.device).lut,
                        0.6 * wide.resources(self.acc.device).lut)

    def test_agreement(self):
        data = np.load(FIXTURES / "corpus" / "xgb_multiclass.npz")
        got = self.acc.predict(data["x"]).reshape(data["y"].shape)
        self.assertGreaterEqual(np.mean(got == data["y"]), 0.99)

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_simulate(self):
        report = self.acc.simulate()
        self.assertTrue(report.ok, report.first.output)

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        build, latency = pathlib.Path(self.acc.out_dir), self.acc.latency
        self.assertTrue(prove(build, 8 * 12, latency)[0])
        self.assertFalse(prove(build, 8 * 12, latency - 1)[0])

    def test_refused(self):
        layer = otf.load(model("xgb_multiclass")).layers
        layer = layer[next(iter(layer))]
        with self.assertRaisesRegex(ConfigError, "activations='int17' .* 'int2' to 'int16'"):
            layer.quantize(activations="int17")
        with self.assertRaisesRegex(ConfigError, "weights='int1' .* 'int2' to 'int16'"):
            layer.quantize(weights="int1")


if __name__ == "__main__":
    unittest.main()
