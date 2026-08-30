"""Answering "will it fit, and how fast" without building anything.

The estimate is only worth having if it agrees with the build it is predicting.
Most of these check that rather than checking it produces plausible numbers.
"""

import unittest

from support import Fixtures

from onnx2fpga.compile import Compiler
from onnx2fpga.estimate import Estimate
from onnx2fpga.p5_schedule.latency import LatencyModel
from onnx2fpga.targets.device import Device
from onnx2fpga.targets.resources import Resources


class EstimateAgreesWithTheBuildTest(unittest.TestCase):
    """A pre-flight that disagreed with the compiler would be worse than none,
    because it would be believed."""

    TARGET = 64

    def test_the_latency_it_predicts_is_the_one_the_build_has(self):
        estimate = Compiler(device="vu9p", target_cycles=self.TARGET).estimate(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE))
        built = Compiler(device="vu9p", target_cycles=self.TARGET).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("estimate_agree"))
        self.assertEqual(estimate.latency_cycles,
                         LatencyModel(built.hardware_graph).cycles)

    def test_the_resources_it_predicts_are_the_ones_the_build_has(self):
        estimate = Compiler(device="vu9p", target_cycles=self.TARGET).estimate(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE))
        built = Compiler(device="vu9p", target_cycles=self.TARGET).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("estimate_agree_res"))
        self.assertEqual(estimate.folding.total.dsp, built.folding.total.dsp)
        self.assertEqual(estimate.frame_cycles, built.folding.bottleneck_cycles)

    def test_estimating_writes_nothing(self):
        """The whole point: no build directory, no RTL, no golden vectors. A
        pre-flight that left a build behind would not be one."""
        out = Fixtures.build_dir("estimate_untouched")
        before = sorted(path.name for path in out.glob("*"))
        Compiler(device="vu9p", target_cycles=self.TARGET).estimate(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE))
        self.assertEqual(sorted(path.name for path in out.glob("*")), before)


class EstimateArithmeticTest(unittest.TestCase):
    def estimate(self, device="z7020"):
        return Compiler(device=device, target_cycles=64).estimate(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE))

    def test_nanoseconds_come_from_the_device_clock(self):
        estimate = self.estimate()
        expected = 1000.0 * estimate.latency_cycles / estimate.device.fmax_mhz
        self.assertAlmostEqual(estimate.latency_ns, expected, places=6)

    def test_it_names_the_tightest_resource(self):
        estimate = self.estimate()
        used = estimate.utilisation
        self.assertEqual(estimate.tightest, max(used, key=used.get))

    def test_the_render_says_the_clock_is_assumed(self):
        """Until 0.4 replaces it with a measured one, quoting nanoseconds
        without that caveat would be the kind of number this project exists
        not to quote."""
        text = self.estimate().render()
        self.assertIn("assumed", text)
        self.assertIn("not synthesis", text)

    def test_a_design_over_budget_says_so_and_says_what_to_do(self):
        """Constructed, because the allocator will not itself produce a design
        over the limit: it stops folding first. This is the path taken when a
        model does not fit even at its cheapest folding."""
        estimate = self.estimate()
        device = Device.get("z7020")
        over = Resources(lut=10 ** 9, ff=1, dsp=1, bram36=1, uram=0)

        class Folding:
            total = over
            bottleneck_cycles = 64

        blown = Estimate(estimate.graph, Folding(), device)
        self.assertFalse(blown.fits)
        self.assertIn("DOES NOT FIT", blown.render())
        self.assertIn("larger device", blown.render())

    def test_a_design_within_budget_says_it_fits(self):
        estimate = self.estimate()
        self.assertTrue(estimate.fits)
        self.assertIn("fits on z7020", estimate.render())


class InspectTest(unittest.TestCase):
    def inspect(self, *extra):
        import contextlib, io
        from onnx2fpga.cli import CommandLine
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = CommandLine().main(
                ["inspect", "examples/models/mlp.onnx"] + list(extra))
        return code, out.getvalue()

    def test_without_a_device_it_only_describes_the_model(self):
        code, text = self.inspect()
        self.assertEqual(code, 0)
        self.assertNotIn("cycles to first output", text)

    def test_with_a_device_it_answers_fit_and_latency(self):
        code, text = self.inspect("--device", "z7020", "--target-cycles", "64")
        self.assertEqual(code, 0)
        self.assertIn("fits on z7020", text)
        self.assertIn("cycles to first output", text)
        self.assertIn("ns at an assumed", text)


if __name__ == "__main__":
    unittest.main()
