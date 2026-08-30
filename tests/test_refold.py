"""Folding against latency rather than against the bottleneck stage."""

import json
import shutil
import unittest

from support import Fixtures, run

from onnx2fpga.compile import Compiler
from onnx2fpga.p2_graph.graph import GraphError
from onnx2fpga.p3_ops.hardware_ops import StreamFifo, WidthConverter
from onnx2fpga.p5_schedule.latency import LatencyModel
from onnx2fpga.p5_schedule.streams import (AlignBeatWidths, InsertFifos,
                                           InsertWidthConverters,
                                           StripStreamPlumbing)
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.targets.device import Device


def shape_of(graph):
    return [(n.name, type(n).__name__, dict(n.folding))
            for n in graph.topological_order()]


class StripIsAFixedPointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("refold_strip"))
        cls.graph = cls.result.hardware_graph

    def replumb(self, graph):
        context = CompilerContext(Device.get("vu9p"))
        for stage in (StripStreamPlumbing(), AlignBeatWidths(),
                      InsertWidthConverters(), InsertFifos()):
            graph = stage(graph, context) or graph
        return graph

    def test_the_graph_starts_with_plumbing_in_it(self):
        plumbing = [n for n in self.graph.nodes
                    if isinstance(n, (WidthConverter, StreamFifo))]
        self.assertTrue(plumbing, "nothing to strip; the test proves nothing")

    def test_strip_and_reinsert_reproduces_the_graph(self):
        before = shape_of(self.graph)
        self.assertEqual(shape_of(self.replumb(self.graph)), before)

    def test_repeated_rounds_do_not_drift(self):
        before = shape_of(self.graph)
        graph = self.graph
        for _ in range(3):
            graph = self.replumb(graph)
        self.assertEqual(shape_of(graph), before)

    @staticmethod
    def orphans(graph):
        wired = {n for node in graph.nodes for n in node.inputs + node.outputs}
        return sorted(t for t in graph.tensor_names()
                      if t not in wired and t not in graph.inputs + graph.outputs)

    def test_stripping_adds_no_orphan_tensors(self):
        """Lowering already leaves some behind -- folded weights, a fused relu.
        What matters is that a stripped converter does not add to them, since
        fresh_name scans the tensor table and a leftover renames the next one."""
        before = self.orphans(self.graph)
        self.assertEqual(self.orphans(self.replumb(self.graph)), before)

    def test_a_still_wired_tensor_refuses_to_be_removed(self):
        with self.assertRaises(GraphError):
            self.graph.remove_tensor(self.graph.outputs[0])


class RefusalTest(unittest.TestCase):
    def test_unroll_and_a_latency_target_are_refused_together(self):
        with self.assertRaises(GraphError):
            Compiler(device="vu9p", unroll=True, target_latency_ns=100.0)


class ObjectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samples = Fixtures.samples(Fixtures.MLP_SHAPE)
        model = Fixtures.factory().mlp()
        cls.throughput = Compiler(device="vu9p", target_cycles=64).compile(
            model, samples, Fixtures.build_dir("refold_throughput"))
        cls.latency = Compiler(device="vu9p", target_cycles=64,
                               target_latency_ns=100.0).compile(
            model, samples, Fixtures.build_dir("refold_latency"))

    def modelled(self, result):
        return LatencyModel(result.hardware_graph).cycles

    def test_refolding_for_latency_is_faster(self):
        self.assertLess(self.modelled(self.latency),
                        self.modelled(self.throughput))

    def test_the_report_carries_the_latency_it_optimised(self):
        self.assertEqual(self.latency.folding.latency_cycles,
                         self.modelled(self.latency))

    def test_the_throughput_build_reports_no_latency(self):
        self.assertIsNone(self.throughput.folding.latency_cycles)

    def test_the_manifest_states_a_latency(self):
        manifest = json.loads(
            (self.latency.build.root / "manifest.json").read_text())
        self.assertEqual(manifest["latency_cycles"], self.modelled(self.latency))
        self.assertIn("latency_ns", manifest)

    def test_it_does_not_pay_for_latency_in_throughput(self):
        """Latency is minimised subject to the frame the throughput phase
        reached. Without the constraint the CNN gave up 400 cycles per frame
        for two cycles of latency, silently violating --target-cycles."""
        self.assertLessEqual(self.latency.folding.bottleneck_cycles,
                             self.throughput.folding.bottleneck_cycles)

    def test_it_still_fits(self):
        used = self.latency.folding.total.utilisation(Device.get("vu9p").budget)
        self.assertLessEqual(max(used.values()), 0.80)


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class MeasuredTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samples = Fixtures.samples(Fixtures.MLP_SHAPE)
        model = Fixtures.factory().mlp()
        cls.builds, cls.graphs = {}, {}
        for name, kwargs in (("throughput", {}),
                             ("latency", {"target_latency_ns": 100.0})):
            build = Fixtures.build_dir("refold_sim_" + name)
            result = Compiler(device="vu9p", target_cycles=64,
                              **kwargs).compile(model, samples, build)
            built = run(["make", "-s"], cwd=build)
            assert built.returncode == 0, built.stderr[-4000:]
            cls.builds[name] = build
            cls.graphs[name] = result.hardware_graph

    def measured(self, name):
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"],
                     cwd=self.builds[name])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for line in result.stdout.splitlines():
            if line.startswith("latency"):
                return int(line.split()[1])
        self.fail("harness reported no latency line")

    def test_it_is_still_bit_exact(self):
        for name, build in self.builds.items():
            with self.subTest(model=name):
                result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                              "--expected", "golden/expected.hex"], cwd=build)
                self.assertIn("PASS", result.stdout)

    def test_the_latency_build_is_measurably_faster(self):
        self.assertLess(self.measured("latency"), self.measured("throughput"))

    def test_the_model_did_not_promise_what_the_hardware_cannot_meet(self):
        for name in self.builds:
            with self.subTest(model=name):
                self.assertEqual(LatencyModel(self.graphs[name]).cycles,
                                 self.measured(name))
