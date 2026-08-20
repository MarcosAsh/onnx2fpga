"""The arithmetic contract: requantisation, accumulator width, layout."""

import unittest

import numpy as np

from support import Fixtures  # noqa: F401

from onnx2fpga.p2_graph.datatype import INT8, INT32, IntType
from onnx2fpga.p2_graph.layout import LayoutAdapter
from onnx2fpga.p4_quantize.requantize import Requantizer


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


if __name__ == "__main__":
    unittest.main()
