"""Importing models that were already quantized.

Two things have to be true. The compiler must read the grids the model states
rather than inventing its own, and its integer pipeline must compute what the
QDQ model means. The second check runs the model the way a runtime does, in
float between each quantize and dequantize pair, and compares.
"""

import shutil
import unittest

import numpy as np

from support import Fixtures, SingleUnitProject, run  # noqa: F401
from make_models import QdqModelFactory, QOperatorModelFactory

from onnx2fpga.compile import Compiler
from onnx2fpga.p1_ingest import TensorProto
from onnx2fpga.reference.runner import GraphRunner


class QdqImportTest(unittest.TestCase):
    ACTIVATIONS = (("uint8", TensorProto.UINT8), ("int8", TensorProto.INT8))

    def _compile(self, activation_dtype, name):
        factory = QdqModelFactory(seed=7, activation_dtype=activation_dtype)
        model = factory.mlp()
        result = Compiler(device="vu9p", target_cycles=64).compile(
            model, out_dir=Fixtures.build_dir(name))
        return factory, result

    def test_grids_come_from_the_model_not_calibration(self):
        for label, dtype in self.ACTIVATIONS:
            with self.subTest(activations=label):
                _, result = self._compile(dtype, "qdq_plan_" + label)
                self.assertTrue(result.plan.from_model)
                self.assertTrue(any("from the model" in line
                                    for line in result.context.log))

    def test_no_calibration_samples_are_required(self):
        factory = QdqModelFactory(seed=7)
        Compiler(device="vu9p", target_cycles=64).compile(
            factory.mlp(), out_dir=Fixtures.build_dir("qdq_nosamples"))

    def test_a_float_model_still_demands_calibration(self):
        with self.assertRaises(ValueError):
            Compiler(device="vu9p").compile(
                Fixtures.factory().mlp(), out_dir=Fixtures.build_dir("qdq_nofloat"))

    def test_unsigned_activations_rebase_onto_the_signed_grid(self):
        _, unsigned = self._compile(TensorProto.UINT8, "qdq_rebase_u")
        spec = unsigned.plan.spec(unsigned.hardware_graph.inputs[0])
        self.assertTrue(spec.rebased)
        self.assertEqual(spec.dtype.name, "INT8")
        self.assertTrue(-128 <= spec.zero_point.flat[0] <= 127)

        _, signed = self._compile(TensorProto.INT8, "qdq_rebase_s")
        self.assertFalse(signed.plan.spec(signed.hardware_graph.inputs[0]).rebased)

    def test_integer_pipeline_matches_the_qdq_specification(self):
        """The strong check: not self consistency, but agreement with what the
        model means when a runtime evaluates it."""
        for label, dtype in self.ACTIVATIONS:
            with self.subTest(activations=label):
                factory, result = self._compile(dtype, "qdq_semantics_" + label)
                graph, plan = result.hardware_graph, result.plan
                source, sink = graph.inputs[0], graph.outputs[0]

                rng = np.random.default_rng(21)
                values = rng.normal(0, 1, graph.tensor(source).shape)

                integers = plan.act_dtype.clamp(
                    np.rint(values / plan.scale(source)) + plan.zero_point(source))
                produced = GraphRunner(graph).run({source: integers})[sink]
                actual = (produced - plan.zero_point(sink)) * plan.scale(sink)

                expected = factory.reference(values)
                step = plan.scale(sink)
                worst = float(np.max(np.abs(actual - expected)))
                budget = 0.5 if label == "int8" else 1.5
                self.assertLessEqual(
                    worst, budget * step,
                    "worst error %.6g is %.2f output steps" % (worst, worst / step))

    def test_input_zero_point_is_folded_into_the_bias(self):
        """An asymmetric input contributes -zx * sum_k W[k][c] per channel.
        It is a constant, so it belongs in the bias; if it were dropped the
        results would be wrong by a large margin rather than a rounding step."""
        from onnx2fpga.p3_ops.hardware_ops import MatVecUnit
        _, result = self._compile(TensorProto.UINT8, "qdq_biasfold")
        graph, plan = result.hardware_graph, result.plan
        units = [n for n in graph.topological_order() if isinstance(n, MatVecUnit)]
        self.assertTrue(units)

        folded_anywhere = False
        for unit in units:
            source = unit.stream_inputs(graph)[0]
            producer = graph.producer(source)
            while producer is not None and not isinstance(producer, MatVecUnit):
                upstream = producer.stream_inputs(graph)
                producer = graph.producer(upstream[0]) if upstream else None
            zero = plan.zero_point(graph.inputs[0]) if producer is None else None
            if zero in (None, 0):
                continue
            correction = -zero * np.sum(unit.weights, axis=0)
            self.assertTrue(np.any(correction != 0))
            self.assertGreaterEqual(int(np.abs(unit.bias).max()),
                                    int(np.abs(correction).max()) // 2)
            folded_anywhere = True
        self.assertTrue(folded_anywhere, "no asymmetric input reached a matvec")

    def test_asymmetric_biases_are_far_larger_than_symmetric_ones(self):
        """A cheap guard that the correction is present at all: dropping it
        would leave the asymmetric model's biases the size of the symmetric
        model's."""
        from onnx2fpga.p3_ops.hardware_ops import MatVecUnit

        def peak_bias(dtype, name):
            _, result = self._compile(dtype, name)
            return max(int(np.abs(n.bias).max())
                       for n in result.hardware_graph.topological_order()
                       if isinstance(n, MatVecUnit))

        asymmetric = peak_bias(TensorProto.UINT8, "qdq_bias_u")
        symmetric = peak_bias(TensorProto.INT8, "qdq_bias_s")
        self.assertGreater(asymmetric, 10 * symmetric)

    def test_weights_survive_the_dequantize_requantize_round_trip(self):
        """A DequantizeLinear on an integer initializer is folded to float and
        re-derived. That has to land on the original integers."""
        factory, result = self._compile(TensorProto.INT8, "qdq_weights")
        from onnx2fpga.p3_ops.hardware_ops import MatVecUnit
        layers, _ = factory._reference
        units = [n for n in result.hardware_graph.topological_order()
                 if isinstance(n, MatVecUnit)]
        self.assertEqual(len(units), len(layers))
        from make_models import Grid
        for unit, (weights, _) in zip(units, layers):
            grid = Grid.for_weights(weights, 1)
            np.testing.assert_array_equal(unit.weights, grid.quantize(weights, 1))


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class QdqSimulationTest(unittest.TestCase):
    def test_generated_rtl_matches_the_golden_model(self):
        for label, dtype in QdqImportTest.ACTIVATIONS:
            with self.subTest(activations=label):
                build = Fixtures.build_dir("qdq_sim_" + label)
                Compiler(device="vu9p", target_cycles=64).compile(
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


class QOperatorImportTest(unittest.TestCase):
    """The other form a static quantizer emits.

    QDQ states a tensor's grid beside the tensor; QOperator folds it into the
    operator's signature. Same information, so the compiler should reach the
    same place, and these check that it does rather than that QOperator merely
    parses."""

    ACTIVATIONS = (("uint8", TensorProto.UINT8), ("int8", TensorProto.INT8))

    def _compile(self, activation_dtype, name):
        factory = QOperatorModelFactory(seed=3, activation_dtype=activation_dtype)
        model = factory.mlp()
        result = Compiler(device="vu9p", target_cycles=64).compile(
            model, out_dir=Fixtures.build_dir(name))
        return factory, result

    def test_a_qlinearmatmul_model_compiles(self):
        for label, dtype in self.ACTIVATIONS:
            with self.subTest(activations=label):
                _, result = self._compile(dtype, "qop_compile_" + label)
                self.assertTrue(result.hardware_graph.nodes)

    def test_grids_come_from_the_model_not_calibration(self):
        """The whole point of reading a quantized model: it already knows."""
        for label, dtype in self.ACTIVATIONS:
            with self.subTest(activations=label):
                _, result = self._compile(dtype, "qop_plan_" + label)
                self.assertTrue(result.plan.from_model)

    def test_no_calibration_samples_are_required(self):
        factory = QOperatorModelFactory(seed=3)
        Compiler(device="vu9p", target_cycles=64).compile(
            factory.mlp(), out_dir=Fixtures.build_dir("qop_nosamples"))

    def test_integer_pipeline_matches_the_specification(self):
        """Agreement with what the model means, not with itself."""
        for label, dtype in self.ACTIVATIONS:
            with self.subTest(activations=label):
                factory, result = self._compile(dtype, "qop_semantics_" + label)
                graph, plan = result.hardware_graph, result.plan
                source, sink = graph.inputs[0], graph.outputs[0]

                rng = np.random.default_rng(5)
                values = rng.normal(0, 1, graph.tensor(source).shape)
                integers = plan.act_dtype.clamp(
                    np.rint(values / plan.scale(source)) + plan.zero_point(source))
                produced = GraphRunner(graph).run({source: integers})[sink]
                actual = (produced - plan.zero_point(sink)) * plan.scale(sink)

                expected = factory.reference(values)
                step = plan.scale(sink)
                worst = float(np.max(np.abs(actual - expected)))
                self.assertLessEqual(
                    worst, 1.5 * step,
                    "worst error %.6g is %.2f output steps" % (worst, worst / step))

    def test_a_streamed_weight_is_refused_with_a_reason(self):
        """QLinearMatMul allows both operands to be streams. This compiler
        holds weights on chip, so it cannot take the second one as a stream,
        and should say which operator and which node rather than failing later
        somewhere that does not mention either."""
        from onnx2fpga.p1_ingest.onnx_model import ModelBuilder
        from onnx2fpga.p2_graph.onnx_importer import ImportError_, OnnxImporter

        builder = ModelBuilder("streamed_weight").add_input("x", (1, 4))
        builder.add_initializer("scale", np.float32(0.02))
        builder.add_initializer("zp", np.int8(0))
        builder.add_initializer("w_f", np.zeros((4, 3), dtype=np.float32))
        # w_i is produced by a node, so it is not an initializer the importer
        # can read the numbers out of.
        builder.add_node("QuantizeLinear", ["w_f", "scale", "zp"], ["w_i"],
                         name="quant_w")
        builder.add_node("QuantizeLinear", ["x", "scale", "zp"], ["x_i"],
                         name="quant_x")
        builder.add_node("QLinearMatMul",
                         ["x_i", "scale", "zp", "w_i", "scale", "zp",
                          "scale", "zp"], ["y_i"], name="qmm")
        builder.add_node("DequantizeLinear", ["y_i", "scale", "zp"], ["y"],
                         name="dequant")
        model = builder.add_output("y", (1, 3)).build()

        with self.assertRaises(ImportError_) as caught:
            OnnxImporter(model).run()
        message = str(caught.exception)
        self.assertIn("QLinearMatMul", message)
        self.assertIn("qmm", message)


if __name__ == "__main__":
    unittest.main()
