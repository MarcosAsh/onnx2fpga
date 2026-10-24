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

    def test_fork_and_join_become_duplicate_and_add(self):
        forks = [n for n in self.graph.nodes if isinstance(n, DuplicateUnit)]
        self.assertEqual([len(n.outputs) for n in forks], [2])
        joins = [n for n in self.graph.nodes if isinstance(n, AddUnit)]
        self.assertEqual([len(n.stream_inputs(self.graph)) for n in joins], [2])

    def test_single_consumer_check(self):
        """Verilator accepts a doubly driven ready line silently, so it is checked here."""
        graph = self.graph
        StreamsAreSingleConsumer().run(graph, self.result.context)
        join = next(n for n in graph.nodes if isinstance(n, AddUnit))
        victim = next(n for n in graph.nodes if isinstance(n, StreamFifo))
        original = list(victim.inputs)
        victim.inputs = [join.stream_inputs(graph)[0]]
        try:
            with self.assertRaises(GraphError):
                StreamsAreSingleConsumer().run(graph, self.result.context)
        finally:
            victim.inputs = original

    def test_skip_branch_needs_a_deep_buffer(self):
        peaks = [edge.peak for edge in self.result.context.artifacts["tokens"].edges.values()]
        self.assertGreater(max(peaks), 2)


class ResidualNumericsTest(unittest.TestCase):
    """How far int8 drifts from float across a residual."""

    @staticmethod
    def worst_drift(result, seeds=(4, 5, 6, 7), trials=20):
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

    def test_drift_is_small_and_shrinks_with_calibration(self):
        """Few samples clip, so more calibration must narrow the drift."""
        drifts = [self.worst_drift(compile_residual(
            "residual_calib_%d" % count, samples=Fixtures.samples((1, 16), count=count)))
            for count in (16, 256)]
        self.assertLess(drifts[1], 16.0)
        self.assertLess(drifts[1], drifts[0])


class ResidualSimulationCases:
    HDL = "sv"

    @classmethod
    def setUpClass(cls):
        name = "residual_sim" + Hdl.suffix(cls.HDL)
        cls.build = Fixtures.build_dir(name)
        compile_residual(name, hdl=cls.HDL)
        built = run(["make", "-s"], cwd=cls.build)
        assert built.returncode == 0, built.stderr[-4000:]

    def test_passes(self):
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"], cwd=self.build)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_survives_backpressure(self):
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
    """The negative control for buffer sizing, which also checks timing on both backends."""

    HDL = "sv"

    def test_minimal_buffers_deadlock(self):
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

    def test_computed_buffers_do_not(self):
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
