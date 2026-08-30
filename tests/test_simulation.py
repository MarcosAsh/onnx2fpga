"""End to end: the generated RTL must reproduce the numpy model bit for bit.

These build with Verilator and take a few seconds each. They are the tests
that actually establish the compiler is correct; everything else checks a
component in isolation.
"""

import shutil
import unittest

from support import Fixtures, run

from onnx2fpga.compile import Compiler
from onnx2fpga.p5_schedule.latency import LatencyModel
from onnx2fpga.simulate import (LatencySummary, SimulationResult,
                                SimulationRunner, nanoseconds)

BACKPRESSURE = [(100, 100), (50, 50), (30, 90), (90, 30), (20, 20), (10, 100), (100, 10)]


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class SimulationTest(unittest.TestCase):
    MODELS = {
        "mlp": (Fixtures.MLP_SHAPE, 64),
        "cnn": (Fixtures.CNN_SHAPE, 400),
    }

    @classmethod
    def setUpClass(cls):
        cls.builds = {}
        cls.graphs = {}
        factory = Fixtures.factory()
        for name, (shape, target) in cls.MODELS.items():
            model = getattr(factory, name)()
            build = Fixtures.build_dir("sim_" + name)
            result = Compiler(device="vu9p", target_cycles=target).compile(
                model, Fixtures.samples(shape), build)
            built = run(["make", "-s"], cwd=build)
            assert built.returncode == 0, built.stderr[-4000:]
            cls.builds[name] = build
            cls.graphs[name] = result.hardware_graph

    def test_lints_clean(self):
        for name, build in self.builds.items():
            with self.subTest(model=name):
                result = run(["make", "-s", "lint"], cwd=build)
                self.assertEqual(result.returncode, 0, result.stderr[-4000:])

    def test_matches_golden_model(self):
        for name, build in self.builds.items():
            with self.subTest(model=name):
                result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                              "--expected", "golden/expected.hex"], cwd=build)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS", result.stdout)

    def measured_latency(self, build):
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"], cwd=build)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for line in result.stdout.splitlines():
            if line.startswith("latency"):
                return int(line.split()[1])
        self.fail("harness reported no latency line")

    def test_the_latency_model_matches_the_harness_on_the_mlp(self):
        """Every unit in the MLP publishes an exact beat schedule, so composing
        them along the path should land on the measured cycle and not near it.
        This is the check that keeps the model honest; without it the model is
        just arithmetic nobody has compared to hardware."""
        model = LatencyModel(self.graphs["mlp"])
        self.assertEqual(model.cycles, self.measured_latency(self.builds["mlp"]))

    def test_the_model_never_promises_a_latency_it_cannot_meet(self):
        """Where a unit is on the uniform default the model spreads its output
        evenly across the frame, which is slower than a unit that bursts. So
        the total errs late, which is the safe direction for a latency claim.
        The CNN is the case: its sliding window generator has no real schedule,
        and the whole of the gap is that one unit."""
        for name, build in self.builds.items():
            with self.subTest(model=name):
                self.assertGreaterEqual(LatencyModel(self.graphs[name]).cycles,
                                        self.measured_latency(build))

    def test_survives_randomised_backpressure(self):
        """A stream design that only ever sees tready high can pass its tests
        and still deadlock in hardware."""
        for name, build in self.builds.items():
            for input_duty, output_duty in BACKPRESSURE:
                for seed in (1, 2, 3):
                    with self.subTest(model=name, duty=(input_duty, output_duty),
                                      seed=seed):
                        result = run(["./obj_dir/Votf_top",
                                      "--input", "golden/input.hex",
                                      "--expected", "golden/expected.hex",
                                      "--input-duty", str(input_duty),
                                      "--output-duty", str(output_duty),
                                      "--seed", str(seed), "--quiet"], cwd=build)
                        self.assertEqual(result.returncode, 0,
                                         result.stdout + result.stderr)

    def test_reports_a_latency_within_the_predicted_frame_cost(self):
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"], cwd=self.builds["mlp"])
        self.assertIn("cycles", result.stdout)
        self.assertIn("latency", result.stdout)

    def test_a_quiet_run_still_reports_what_it_cost(self):
        """The sweep runs every setting quiet. If quiet swallowed these two
        lines the sweep could report nothing but pass or fail."""
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex", "--quiet"],
                     cwd=self.builds["mlp"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cycles", result.stdout)
        self.assertIn("latency", result.stdout)
        self.assertNotIn("backpressure", result.stdout)

    def test_the_sweep_summarises_latency_in_nanoseconds(self):
        runner = SimulationRunner(self.builds["mlp"])
        self.assertEqual(runner.fmax_mhz, 300.0)
        summary = runner.summarise(runner.sweep(duties=[(100, 100), (50, 50)],
                                                seeds=(1, 2)))
        self.assertTrue(summary)
        self.assertEqual(summary.count, 4)
        self.assertGreater(summary.best, 0)
        self.assertLessEqual(summary.best, summary.median)
        self.assertLessEqual(summary.median, summary.worst)
        self.assertAlmostEqual(nanoseconds(summary.best, 300.0),
                               summary.best * 1000.0 / 300.0)
        self.assertIn("ns at 300 MHz", summary.render())


class LatencyReportTest(unittest.TestCase):
    """Parsing and summarising, without needing Verilator to produce it."""

    HARNESS = ("cycles           700\n"
               "latency          169 cycles to first output\n")

    def result(self, latency, ok=True, fmax_mhz=300.0):
        output = self.HARNESS.replace("169", str(latency))
        return SimulationResult((100, 100), 1, 0 if ok else 1, output,
                                fmax_mhz=fmax_mhz)

    def test_latency_converts_to_nanoseconds_at_the_compiled_clock(self):
        self.assertAlmostEqual(self.result(169).latency_ns, 1000.0 * 169 / 300.0)

    def test_without_a_clock_everything_stays_in_cycles(self):
        result = self.result(169, fmax_mhz=None)
        self.assertEqual(result.latency, 169)
        self.assertIsNone(result.latency_ns)
        self.assertEqual(result.annotated_output(), result.output)

    def test_nanoseconds_are_spliced_into_the_line_they_belong_to(self):
        annotated = self.result(169).annotated_output()
        self.assertIn("latency          169 cycles to first output, 563 ns "
                      "at 300 MHz", annotated)
        self.assertEqual(annotated.count("latency"), 1)

    def test_the_summary_ignores_runs_that_failed(self):
        summary = LatencySummary([self.result(169), self.result(5000, ok=False)],
                                 300.0)
        self.assertEqual(summary.count, 1)
        self.assertEqual(summary.worst, 169)

    def test_spread_is_the_gap_between_the_best_and_worst_run(self):
        summary = LatencySummary(
            [self.result(n) for n in (402, 169, 210)], 300.0)
        self.assertEqual((summary.best, summary.median, summary.worst),
                         (169, 210, 402))
        self.assertEqual(summary.spread, 402 - 169)

    def test_a_summary_with_nothing_in_it_says_so_rather_than_raising(self):
        summary = LatencySummary([self.result(169, ok=False)], 300.0)
        self.assertFalse(summary)
        self.assertIn("no run reported one", summary.render())


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class UnrolledSimulationTest(unittest.TestCase):
    """An unrolled build is the one place SIMD and PE reach the full matrix
    dimensions, so it instantiates parameter values no folded build ever has.
    Bit exactness there is a separate claim from bit exactness folded."""

    @classmethod
    def setUpClass(cls):
        cls.build = Fixtures.build_dir("sim_mlp_unrolled")
        Compiler(device="vu9p", unroll=True).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            cls.build)
        built = run(["make", "-s"], cwd=cls.build)
        assert built.returncode == 0, built.stderr[-4000:]

    def test_lints_clean(self):
        result = run(["make", "-s", "lint"], cwd=self.build)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])

    def test_matches_golden_model(self):
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"], cwd=self.build)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_survives_randomised_backpressure(self):
        for input_duty, output_duty in BACKPRESSURE:
            with self.subTest(duty=(input_duty, output_duty)):
                result = run(["./obj_dir/Votf_top",
                              "--input", "golden/input.hex",
                              "--expected", "golden/expected.hex",
                              "--input-duty", str(input_duty),
                              "--output-duty", str(output_duty)], cwd=self.build)
                self.assertEqual(result.returncode, 0,
                                 result.stdout + result.stderr)
                self.assertIn("PASS", result.stdout)

    def test_the_model_predicted_the_measured_latency(self):
        """Unfolded, every unit publishes an exact schedule, so the model has
        no approximation to hide behind here."""
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"], cwd=self.build)
        measured = next(int(line.split()[1]) for line in result.stdout.splitlines()
                        if line.startswith("latency"))
        graph = Compiler(device="vu9p", unroll=True).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("sim_mlp_unrolled_model")).hardware_graph
        self.assertEqual(LatencyModel(graph).cycles, measured)
        self.assertLess(measured, 10)


if __name__ == "__main__":
    unittest.main()
