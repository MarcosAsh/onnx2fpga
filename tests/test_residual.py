"""Forks and joins, and the buffers that keep them from deadlocking.

A skip connection is the first shape where buffer sizing stops being a
performance detail. The join cannot take a beat from the short branch until
the long one catches up, so everything the short branch produced in the
meantime has to be held. Too small a buffer there fills, which stalls the
fork, which stalls the long branch, which never reaches the join. The design
stops dead rather than running slowly, which is why this file carries a
negative control as well as a positive one.
"""

import shutil
import unittest

import numpy as np

from support import Fixtures, run

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
        """Verilator accepts a doubly driven ready line without a word, so this
        has to be checked here rather than left to simulation."""
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
    """How far int8 drifts from float across a residual.

    The drift is dominated by how well calibration covered the input range, not
    by the residual itself. Calibrate on too few samples and a fresh input
    falls outside the fitted scale and clips, which is real quantiser
    behaviour rather than a compiler fault. So the threshold below is set
    against a well calibrated model, and a second test pins the relationship
    that made the first one meaningful.
    """

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
        """Guards the calibration path: if it stopped widening the fitted range
        as samples arrive, this ordering would break."""
        drifts = []
        for count in (16, 256):
            result = compile_residual(
                "residual_calib_%d" % count,
                samples=Fixtures.samples((1, 16), count=count))
            drifts.append(self.worst_drift(result, self.EVAL_SEEDS, trials=10))
        self.assertLess(drifts[1], drifts[0],
                        "more calibration should not drift further: %s" % drifts)


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class ResidualSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = Fixtures.build_dir("residual_sim")
        compile_residual("residual_sim")
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


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class BufferSizingIsLoadBearingTest(unittest.TestCase):
    """The negative control. If starving the buffers did not deadlock, the
    sizing analysis would be unfalsifiable and this file would prove nothing.
    """

    def test_capping_every_buffer_at_the_minimum_deadlocks(self):
        build = Fixtures.build_dir("residual_capped")
        compile_residual("residual_capped", fifo_cap=2)
        built = run(["make", "-s"], cwd=build)
        self.assertEqual(built.returncode, 0, built.stderr[-4000:])
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex",
                      "--timeout", "400000", "--quiet"], cwd=build, timeout=300)
        self.assertNotEqual(result.returncode, 0,
                            "starved buffers should deadlock, but the run passed")
        self.assertIn("stalled", result.stderr)

    def test_the_computed_sizes_do_not(self):
        build = Fixtures.build_dir("residual_sized")
        compile_residual("residual_sized")
        built = run(["make", "-s"], cwd=build)
        self.assertEqual(built.returncode, 0, built.stderr[-4000:])
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex",
                      "--timeout", "400000"], cwd=build, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
