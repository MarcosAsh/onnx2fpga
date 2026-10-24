"""Models already in QDQ form: their grids are read and their meaning is kept."""

import shutil
import unittest

import numpy as np

from support import Fixtures, Hdl, run
from make_models import Grid, QdqModelFactory

from onnx2fpga.compile import Compiler
from onnx2fpga.p1_ingest import TensorProto
from onnx2fpga.p3_ops.hardware_ops import MatVecUnit
from onnx2fpga.reference.runner import GraphRunner

ACTIVATIONS = (("uint8", TensorProto.UINT8), ("int8", TensorProto.INT8))


class QdqImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """No calibration samples: the model states every scale."""
        cls.compiled = {}
        for label, dtype in ACTIVATIONS:
            factory = QdqModelFactory(seed=7, activation_dtype=dtype)
            cls.compiled[label] = (factory, Compiler(device="vu9p", target_cycles=64).compile(
                factory.mlp(), out_dir=Fixtures.build_dir("qdq_" + label)))

    def test_grids_come_from_the_model(self):
        for label, (_, result) in self.compiled.items():
            with self.subTest(activations=label):
                self.assertTrue(result.plan.from_model)
                self.assertTrue(any("from the model" in line for line in result.context.log))

    def test_float_model_needs_calibration(self):
        with self.assertRaises(ValueError):
            Compiler(device="vu9p").compile(
                Fixtures.factory().mlp(), out_dir=Fixtures.build_dir("qdq_nofloat"))

    def test_unsigned_activations_rebase_onto_signed_grid(self):
        result = self.compiled["uint8"][1]
        spec = result.plan.spec(result.hardware_graph.inputs[0])
        self.assertTrue(spec.rebased)
        self.assertEqual(spec.dtype.name, "INT8")
        self.assertTrue(-128 <= spec.zero_point.flat[0] <= 127)
        signed = self.compiled["int8"][1]
        self.assertFalse(signed.plan.spec(signed.hardware_graph.inputs[0]).rebased)

    def test_integer_pipeline_matches_qdq_semantics(self):
        """Checked against what the model means at runtime, not against itself."""
        for label, (factory, result) in self.compiled.items():
            with self.subTest(activations=label):
                graph, plan = result.hardware_graph, result.plan
                source, sink = graph.inputs[0], graph.outputs[0]
                values = np.random.default_rng(21).normal(0, 1, graph.tensor(source).shape)
                integers = plan.act_dtype.clamp(
                    np.rint(values / plan.scale(source)) + plan.zero_point(source))
                produced = GraphRunner(graph).run({source: integers})[sink]
                actual = (produced - plan.zero_point(sink)) * plan.scale(sink)
                step = plan.scale(sink)
                worst = float(np.max(np.abs(actual - factory.reference(values))))
                budget = 0.5 if label == "int8" else 1.5
                self.assertLessEqual(worst, budget * step,
                                     "worst error is %.2f output steps" % (worst / step))

    def test_input_zero_point_folds_into_bias(self):
        """An asymmetric input adds -zx * sum_k W[k][c] per channel to the bias."""
        graph, plan = self.compiled["uint8"][1].hardware_graph, self.compiled["uint8"][1].plan
        zero = plan.zero_point(graph.inputs[0])
        first = next(n for n in graph.topological_order() if isinstance(n, MatVecUnit))
        correction = -zero * np.sum(first.weights, axis=0)
        self.assertTrue(np.any(correction != 0))
        self.assertGreaterEqual(int(np.abs(first.bias).max()),
                                int(np.abs(correction).max()) // 2)

        def peak_bias(result):
            return max(int(np.abs(n.bias).max()) for n in result.hardware_graph.nodes
                       if isinstance(n, MatVecUnit))

        # Without the correction the biases would be the size of the symmetric model's.
        self.assertGreater(peak_bias(self.compiled["uint8"][1]),
                           10 * peak_bias(self.compiled["int8"][1]))

    def test_quantized_weights_survive_dequantize(self):
        factory, result = self.compiled["int8"]
        layers, _ = factory._reference
        units = [n for n in result.hardware_graph.topological_order()
                 if isinstance(n, MatVecUnit)]
        self.assertEqual(len(units), len(layers))
        for unit, (weights, _) in zip(units, layers):
            np.testing.assert_array_equal(unit.weights,
                                          Grid.for_weights(weights, 1).quantize(weights, 1))


class QdqSimulationCases:
    HDL = "sv"

    def test_rtl_matches_golden_model(self):
        for label, dtype in ACTIVATIONS:
            with self.subTest(activations=label):
                build = Fixtures.build_dir("qdq_sim_" + label + Hdl.suffix(self.HDL))
                Compiler(device="vu9p", target_cycles=64, hdl=self.HDL).compile(
                    QdqModelFactory(seed=7, activation_dtype=dtype).mlp(),
                    out_dir=build)
                built = run(["make", "-s"], cwd=build)
                self.assertEqual(built.returncode, 0, built.stderr[-3000:])
                for duty in ((100, 100), (40, 40), (100, 25)):
                    result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                                  "--expected", "golden/expected.hex",
                                  "--input-duty", str(duty[0]),
                                  "--output-duty", str(duty[1]), "--quiet"], cwd=build)
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class QdqSimulationTest(QdqSimulationCases, unittest.TestCase):
    HDL = "sv"


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
class HardcamlQdqSimulationTest(QdqSimulationCases, unittest.TestCase):
    HDL = "hardcaml"


if __name__ == "__main__":
    unittest.main()
