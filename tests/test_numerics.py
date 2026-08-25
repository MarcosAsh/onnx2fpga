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


class PerFeatureInputTest(unittest.TestCase):
    """One scale for a whole feature vector fits the loudest column and leaves
    the quiet ones with almost no integers to sit on. A scale per column fixes
    that, and costs nothing, because it folds into the weights it multiplies:
    y_c = sum_k x_k (s_k w_kc) is still one grid per output channel."""

    SPREAD_DECADES = (-3, 1)

    def mixed_range_samples(self, count=48, features=32, seed=11):
        rng = np.random.default_rng(seed)
        spread = 10.0 ** rng.uniform(*self.SPREAD_DECADES, features)
        samples = [{"x": (rng.normal(0, 1, (1, features)) * spread)}
                   for _ in range(count)]
        probe = rng.normal(0, 1, (1, features)) * spread
        return samples, probe

    def float_reference(self, model, values):
        cursor = np.asarray(values, dtype=np.float64)
        for layer in range(3):
            cursor = cursor @ np.asarray(model.initializer("w%d" % layer),
                                         dtype=np.float64)
            cursor = cursor + np.asarray(model.initializer("b%d" % layer),
                                         dtype=np.float64)
            if layer < 2:
                cursor = np.maximum(cursor, 0.0)
        return cursor

    def compiled(self, per_feature, samples, name):
        return Compiler(device="vu9p", target_cycles=64,
                        per_feature_input=per_feature).compile(
            Fixtures.factory().mlp(), samples, Fixtures.build_dir(name))

    def error_of(self, result, model, probe):
        graph, plan = result.hardware_graph, result.plan
        source, sink = graph.inputs[0], graph.outputs[0]
        integers = plan.spec(source).quantize(probe, plan.act_dtype)
        produced = GraphRunner(graph).run({source: integers})[sink]
        actual = (produced - plan.zero_point(sink)) * plan.scale(sink)
        return float(np.max(np.abs(actual - self.float_reference(model, probe))))

    def test_it_fits_one_scale_per_feature(self):
        samples, _ = self.mixed_range_samples()
        result = self.compiled(True, samples, "pf_scales")
        spec = result.plan.spec(result.hardware_graph.inputs[0])
        self.assertTrue(spec.per_channel)
        self.assertEqual(spec.scale.size, 32)

    def test_it_closes_some_of_the_gap_on_a_mixed_range_input(self):
        """Synthetic, deliberately: the columns span four decades. Confirming
        the size of the win on a real feature set needs a real feature set,
        which this project does not have yet."""
        model = Fixtures.factory().mlp()
        samples, probe = self.mixed_range_samples()
        flat = self.error_of(self.compiled(False, samples, "pf_flat"), model, probe)
        split = self.error_of(self.compiled(True, samples, "pf_split"), model, probe)
        self.assertLess(split, flat)

    def test_it_adds_no_hardware(self):
        """The whole argument for folding into the weights. If this ever
        stopped holding, per-feature scaling would need a rescale unit and
        would stop being free.

        Not identical, though: the effective weights have a different dynamic
        range from the originals, so the accumulator can come out a bit
        narrower and the estimate with it. What must hold is that nothing was
        added — same units, same schedule, no more multipliers."""
        samples, _ = self.mixed_range_samples()
        flat = self.compiled(False, samples, "pf_cost_flat")
        split = self.compiled(True, samples, "pf_cost_split")
        self.assertEqual(split.folding.bottleneck_cycles,
                         flat.folding.bottleneck_cycles)
        self.assertLessEqual(split.folding.total.dsp, flat.folding.total.dsp)
        self.assertEqual(
            sorted(type(n).__name__ for n in split.hardware_graph.nodes),
            sorted(type(n).__name__ for n in flat.hardware_graph.nodes))

    def test_asking_for_one_scale_of_a_per_feature_tensor_is_refused(self):
        """Returning the first feature's scale would be wrong in a way nothing
        downstream could detect."""
        samples, _ = self.mixed_range_samples()
        result = self.compiled(True, samples, "pf_guard")
        with self.assertRaises(ValueError) as caught:
            result.plan.scale(result.hardware_graph.inputs[0])
        self.assertIn("per feature", str(caught.exception))

    def test_the_manifest_carries_every_scale(self):
        """A reader that only understands one number should fail on the type
        rather than silently use the wrong scale for 31 of the 32 features."""
        import json
        samples, _ = self.mixed_range_samples()
        build = Fixtures.build_dir("pf_manifest")
        Compiler(device="vu9p", target_cycles=64,
                 per_feature_input=True).compile(
            Fixtures.factory().mlp(), samples, build)
        manifest = json.loads((build / "manifest.json").read_text())
        self.assertIsInstance(manifest["input"]["scale"], list)
        self.assertEqual(len(manifest["input"]["scale"]), 32)


if __name__ == "__main__":
    unittest.main()
