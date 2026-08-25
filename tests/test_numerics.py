"""The arithmetic contract: requantisation, accumulator width, layout."""

import unittest

import numpy as np

from support import Fixtures  # noqa: F401

from onnx2fpga.compile import Compiler
from onnx2fpga.p2_graph.datatype import INT8, INT16, INT32, IntType
from onnx2fpga.p2_graph.layout import LayoutAdapter
from onnx2fpga.p4_quantize.requantize import Requantizer
from onnx2fpga.reference.runner import GraphRunner


class RequantizerTest(unittest.TestCase):
    def test_matches_float_scaling_closely(self):
        rng = np.random.default_rng(0)
        reals = rng.uniform(1e-4, 0.05, 16)
        requant = Requantizer.from_real_multipliers(reals, INT8, mult_bits=18)
        accumulators = rng.integers(-30000, 30000, (64, 16))

        produced = requant.apply(accumulators)
        expected = np.clip(np.rint(accumulators * reals), -128, 127)
        self.assertLessEqual(np.max(np.abs(produced - expected)), 1)

    def test_rounds_half_up_on_both_signs(self):
        requant = Requantizer(np.asarray([1]), 1, INT8)
        np.testing.assert_array_equal(
            requant.apply(np.asarray([-3, -1, 1, 3])),
            np.asarray([-1, 0, 1, 2]))

    def test_relu_clamps_at_zero(self):
        requant = Requantizer.from_real_multipliers([0.5], INT8, relu=True)
        self.assertEqual(requant.lower_clamp, 0)
        self.assertEqual(int(requant.apply(np.asarray([-1000]))[0]), 0)

    def test_saturates_at_the_output_range(self):
        requant = Requantizer.from_real_multipliers([1.0], INT8)
        values = requant.apply(np.asarray([-10 ** 6, 10 ** 6]))
        self.assertEqual(list(values), [-128, 127])

    def test_zero_multiplier_is_safe(self):
        requant = Requantizer.from_real_multipliers([0.0, 0.0], INT8)
        np.testing.assert_array_equal(requant.apply(np.asarray([5, -5])), [0, 0])


class AccumulatorWidthTest(unittest.TestCase):
    def test_smallest_for_covers_the_range(self):
        for low, high in [(0, 1), (0, 255), (-1, 0), (-128, 127), (-5000, 5000)]:
            dtype = IntType.smallest_for(low, high)
            self.assertLessEqual(dtype.min, low)
            self.assertGreaterEqual(dtype.max, high)

    def test_narrower_than_int32_for_a_small_layer(self):
        from onnx2fpga.p3_ops.hardware_ops import MatVecUnit
        rng = np.random.default_rng(1)
        weights = rng.integers(-127, 128, (32, 16))
        unit = MatVecUnit("m", ["x"], ["y"], weights, np.zeros(16, dtype=np.int64),
                          Requantizer.from_real_multipliers([0.01], INT8),
                          INT8, INT8, INT8)
        self.assertLess(unit.accumulator_dtype.bits, INT32.bits)
        self.assertGreater(unit.accumulator_dtype.bits, 8)


class LayoutTest(unittest.TestCase):
    def test_nchw_round_trip(self):
        adapter = LayoutAdapter.for_rank(4)
        array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        internal = adapter.to_internal(array)
        self.assertEqual(internal.shape, (2, 4, 5, 3))
        np.testing.assert_array_equal(adapter.to_external(internal), array)

    def test_rank_two_is_untouched(self):
        adapter = LayoutAdapter.for_rank(2)
        array = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(adapter.to_internal(array), array)


class DatapathWidthTest(unittest.TestCase):
    """int16 activations. The width is one number for the whole graph, because
    widening an activation widens the unit that consumes it and everything
    after it — unlike the graph's output, which nothing consumes."""

    LAYERS = 3

    def float_reference(self, model, values):
        """The model evaluated the way it was written, before anyone
        quantized it. This is what the integer pipeline is approximating."""
        cursor = np.asarray(values, dtype=np.float64)
        for layer in range(self.LAYERS):
            cursor = cursor @ np.asarray(model.initializer("w%d" % layer),
                                         dtype=np.float64)
            cursor = cursor + np.asarray(model.initializer("b%d" % layer),
                                         dtype=np.float64)
            if layer < self.LAYERS - 1:
                cursor = np.maximum(cursor, 0.0)
        return cursor

    def error_at(self, bits, model, values):
        """Both widths together. They are one decision in practice: see
        test_widening_only_half_the_datapath_buys_nothing."""
        result = Compiler(device="vu9p", target_cycles=64, act_bits=bits,
                          weight_bits=bits).compile(
            model, Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("width_%d" % bits))
        graph, plan = result.hardware_graph, result.plan
        source, sink = graph.inputs[0], graph.outputs[0]
        integers = plan.act_dtype.clamp(
            np.rint(values / plan.scale(source)) + plan.zero_point(source))
        produced = GraphRunner(graph).run({source: integers})[sink]
        actual = (produced - plan.zero_point(sink)) * plan.scale(sink)
        expected = self.float_reference(model, values)
        return float(np.max(np.abs(actual - expected)))

    def test_int16_activations_compile_and_carry_the_model(self):
        model = Fixtures.factory().mlp()
        result = Compiler(device="vu9p", target_cycles=64, act_bits=16,
                          weight_bits=16).compile(
            model, Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("width_compile16"))
        self.assertEqual(result.plan.act_dtype, INT16)
        self.assertEqual(result.plan.weight_dtype, INT16)

    def test_int16_is_far_closer_to_the_float_model_than_int8(self):
        """The item's own Done-when. Two orders of magnitude, not a few
        percent: if this ever narrowed to a small factor the wider datapath
        would be costing multipliers for very little."""
        model = Fixtures.factory().mlp()
        rng = np.random.default_rng(4)
        values = rng.normal(0, 1, (1, 32))
        narrow = self.error_at(8, model, values)
        wide = self.error_at(16, model, values)
        self.assertLess(wide, narrow / 100,
                        "int8 error %.6g, int16 error %.6g" % (narrow, wide))

    def test_widening_only_half_the_datapath_buys_nothing(self):
        """Worth pinning, because it is the mistake the flags invite. On this
        model int8 weights dominate the error, so int16 activations alone buy
        about a factor of two, and int16 weights alone buy nothing at all —
        the activations they are multiplied by are still the coarse ones."""
        model = Fixtures.factory().mlp()
        rng = np.random.default_rng(4)
        values = rng.normal(0, 1, (1, 32))

        def error(act, weight):
            result = Compiler(device="vu9p", target_cycles=64, act_bits=act,
                              weight_bits=weight).compile(
                model, Fixtures.samples(Fixtures.MLP_SHAPE),
                Fixtures.build_dir("width_%d_%d" % (act, weight)))
            graph, plan = result.hardware_graph, result.plan
            source, sink = graph.inputs[0], graph.outputs[0]
            integers = plan.act_dtype.clamp(
                np.rint(values / plan.scale(source)) + plan.zero_point(source))
            produced = GraphRunner(graph).run({source: integers})[sink]
            actual = (produced - plan.zero_point(sink)) * plan.scale(sink)
            return float(np.max(np.abs(actual - self.float_reference(model, values))))

        both = error(16, 16)
        self.assertLess(both, error(16, 8) / 100)
        self.assertLess(both, error(8, 16) / 100)

    def test_the_wider_datapath_is_not_free(self):
        """It buys accuracy with multipliers, and the report should show it."""
        model = Fixtures.factory().mlp()
        narrow = Compiler(device="vu9p", target_cycles=64, act_bits=8).compile(
            model, Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("width_cost8"))
        wide = Compiler(device="vu9p", target_cycles=64, act_bits=16,
                        weight_bits=16).compile(
            model, Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("width_cost16"))
        self.assertGreater(wide.folding.total.dsp, narrow.folding.total.dsp)


if __name__ == "__main__":
    unittest.main()
