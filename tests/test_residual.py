"""Forks and joins, and the buffers that keep them from deadlocking."""

import shutil
import unittest

import numpy as np

from support import Fixtures, Hdl, run

from onnx2fpga.compile import Compiler
from onnx2fpga.p2_graph.graph import GraphError
from onnx2fpga.p3_ops.hardware_ops import AddUnit, DuplicateUnit, StreamFifo
from onnx2fpga.p5_schedule.streams import StreamsAreSingleConsumer
from onnx2fpga.reference.runner import GraphRunner

TARGET = 64
BACKPRESSURE = [(100, 100), (50, 50), (20, 20), (100, 10), (10, 100), (5, 5)]


def compile_residual(name, samples=None, **kwargs):
    return Compiler(device="vu9p", target_cycles=TARGET, **kwargs).compile(
        Fixtures.factory().residual(),
        Fixtures.samples((1, 16)) if samples is None else samples,
        Fixtures.build_dir(name))


class ResidualStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = compile_residual("residual_structure")
        cls.graph = cls.result.hardware_graph

    def test_the_fork_becomes_a_duplicate_unit(self):
        forks = [n for n in self.graph.nodes if isinstance(n, DuplicateUnit)]
        self.assertEqual(len(forks), 1)
        self.assertEqual(len(forks[0].outputs), 2)

    def test_the_join_becomes_an_add_unit_with_two_streams(self):
        joins = [n for n in self.graph.nodes if isinstance(n, AddUnit)]
        self.assertEqual(len(joins), 1)
        self.assertEqual(len(joins[0].stream_inputs(self.graph)), 2)

    def test_no_stream_feeds_two_consumers(self):
        """Verilator accepts a doubly driven ready line silently, so it has to be checked here."""
        StreamsAreSingleConsumer().run(self.graph, self.result.context)

    def test_the_check_actually_rejects_a_bad_graph(self):
        graph = self.graph
        join = next(n for n in graph.nodes if isinstance(n, AddUnit))
        shared = join.stream_inputs(graph)[0]
        victim = next(n for n in graph.nodes
                      if isinstance(n, StreamFifo) and n is not join)
        original = list(victim.inputs)
        victim.inputs = [shared]
        try:
            with self.assertRaises(GraphError):
                StreamsAreSingleConsumer().run(graph, self.result.context)
        finally:
            victim.inputs = original

    def test_the_skip_branch_gets_the_deepest_buffer(self):
        analysis = self.result.context.artifacts["tokens"]
        peaks = {name: edge.peak for name, edge in analysis.edges.items()}
        self.assertTrue(peaks)
        self.assertGreater(max(peaks.values()), 2,
                           "a reconvergent path should need more than the minimum")


class ResidualNumericsTest(unittest.TestCase):
    """How far int8 drifts from float across a residual, set against a well calibrated model since few samples clip."""

    CALIBRATION = 256
    EVAL_SEEDS = (4, 5, 6, 7)

    @staticmethod
    def worst_drift(result, seeds, trials=20):
        graph, plan = result.hardware_graph, result.plan
        source, sink = graph.inputs[0], graph.outputs[0]
        worst = 0.0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for _ in range(trials):
                values = rng.normal(0, 1, graph.tensor(source).shape)
                integers = plan.act_dtype.clamp(
                    np.rint(values / plan.scale(source)) + plan.zero_point(source))
                produced = GraphRunner(graph).run({source: integers})[sink]
                actual = (produced - plan.zero_point(sink)) * plan.scale(sink)
                expected = GraphRunner(result.float_graph).run({source: values})[sink]
                worst = max(worst, float(np.max(np.abs(actual - expected))))
        return worst / plan.scale(sink)

    def test_a_well_calibrated_residual_tracks_float(self):
        result = compile_residual(
            "residual_numerics",
            samples=Fixtures.samples((1, 16), count=self.CALIBRATION))
        steps = self.worst_drift(result, self.EVAL_SEEDS)
        self.assertLess(steps, 16.0,
                        "int8 residual drifted %.1f output steps from float" % steps)

    def test_more_calibration_narrows_the_drift(self):
        """Guards the calibration path: if it stopped widening the fitted range as samples arrive, this ordering breaks."""
        drifts = []
        for count in (16, 256):
            result = compile_residual(
                "residual_calib_%d" % count,
                samples=Fixtures.samples((1, 16), count=count))
            drifts.append(self.worst_drift(result, self.EVAL_SEEDS, trials=10))
        self.assertLess(drifts[1], drifts[0],
                        "more calibration should not drift further: %s" % drifts)


class ResidualSimulationCases:
    HDL = "sv"

    @classmethod
    def setUpClass(cls):
        name = "residual_sim" + Hdl.suffix(cls.HDL)
        cls.build = Fixtures.build_dir(name)
        compile_residual(name, hdl=cls.HDL)
        built = run(["make", "-s"], cwd=cls.build)
        assert built.returncode == 0, built.stderr[-4000:]

    def test_duplicate_and_add_units_work_end_to_end(self):
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"], cwd=self.build)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_survives_randomised_backpressure(self):
        for input_duty, output_duty in BACKPRESSURE:
            for seed in (1, 2, 3):
                with self.subTest(duty=(input_duty, output_duty), seed=seed):
                    result = run(["./obj_dir/Votf_top",
                                  "--input", "golden/input.hex",
                                  "--expected", "golden/expected.hex",
                                  "--input-duty", str(input_duty),
                                  "--output-duty", str(output_duty),
                                  "--seed", str(seed), "--quiet"], cwd=self.build)
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)


class BufferSizingCases:
    """The negative control: without it the sizing analysis is unfalsifiable; on both backends it also checks timing."""

    HDL = "sv"

    def test_capping_every_buffer_at_the_minimum_deadlocks(self):
        name = "residual_capped" + Hdl.suffix(self.HDL)
        build = Fixtures.build_dir(name)
        compile_residual(name, fifo_cap=2, hdl=self.HDL)
        built = run(["make", "-s"], cwd=build)
        self.assertEqual(built.returncode, 0, built.stderr[-4000:])
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex",
                      "--timeout", "400000", "--quiet"], cwd=build, timeout=300)
        self.assertNotEqual(result.returncode, 0,
                            "starved buffers should deadlock, but the run passed")
        self.assertIn("stalled", result.stderr)

    def test_the_computed_sizes_do_not(self):
        name = "residual_sized" + Hdl.suffix(self.HDL)
        build = Fixtures.build_dir(name)
        compile_residual(name, hdl=self.HDL)
        built = run(["make", "-s"], cwd=build)
        self.assertEqual(built.returncode, 0, built.stderr[-4000:])
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex",
                      "--timeout", "400000"], cwd=build, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


VERILATOR = shutil.which("verilator") is not None


@unittest.skipUnless(VERILATOR, "verilator not installed")
class ResidualSimulationTest(ResidualSimulationCases, unittest.TestCase):
    HDL = "sv"


@unittest.skipUnless(VERILATOR, "verilator not installed")
@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
class HardcamlResidualSimulationTest(ResidualSimulationCases, unittest.TestCase):
    HDL = "hardcaml"


@unittest.skipUnless(VERILATOR, "verilator not installed")
class BufferSizingIsLoadBearingTest(BufferSizingCases, unittest.TestCase):
    HDL = "sv"


@unittest.skipUnless(VERILATOR, "verilator not installed")
@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
class HardcamlBufferSizingIsLoadBearingTest(BufferSizingCases, unittest.TestCase):
    HDL = "hardcaml"


if __name__ == "__main__":
    unittest.main()
