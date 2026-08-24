"""End-to-end latency in the IR: what each unit contributes to first output.

`cycles` is the initiation interval and says nothing about latency. These pin
the distinction, and pin the places where the model is known to understate.
"""

import unittest

from support import Fixtures

from onnx2fpga.compile import Compiler
from onnx2fpga.p3_ops.hardware_ops import (MatVecUnit, PoolUnit, StreamingNode,
                                           StreamFifo, WidthConverter)


class MlpLatencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_latency"))
        cls.graph = cls.result.hardware_graph

    def units(self, kind):
        return [n for n in self.graph.nodes if isinstance(n, kind)]

    def test_every_unit_reports_a_latency(self):
        streaming = [n for n in self.graph.nodes if isinstance(n, StreamingNode)]
        self.assertTrue(streaming)
        for node in streaming:
            latency = node.latency(self.graph)
            self.assertIsInstance(latency, int, node.name)
            self.assertGreaterEqual(latency, 0, node.name)

    def test_matvec_is_late_by_its_synapse_fold(self):
        """It cannot emit until the last partial product is accumulated, so
        folding the synapses deeper costs latency even though it saves DSPs.
        Plus the one cycle its output register costs."""
        units = self.units(MatVecUnit)
        self.assertTrue(units)
        for unit in units:
            self.assertEqual(unit.latency(self.graph),
                             unit.synapse_fold - 1 + unit.REGISTER_STAGES,
                             unit.name)

    def test_latency_is_not_the_initiation_interval(self):
        """The whole point of the item. If these were equal there would be
        nothing to model."""
        matvec = self.units(MatVecUnit)[0]
        self.assertNotEqual(matvec.latency(self.graph), matvec.cycles)
        self.assertLess(matvec.latency(self.graph), matvec.cycles)

    def test_a_fifo_still_costs_its_register(self):
        """It reorders nothing, but it writes on a clock edge and reads the
        cell combinationally, so a beat through it is one cycle late."""
        for fifo in self.units(StreamFifo):
            self.assertEqual(fifo.latency(self.graph), fifo.REGISTER_STAGES,
                             fifo.name)

    def test_a_widening_converter_waits_for_a_full_wide_beat(self):
        widening = [c for c in self.units(WidthConverter)
                    if c.out_elems > c.in_elems]
        self.assertTrue(widening)
        for conv in widening:
            self.assertEqual(conv.latency(self.graph),
                             conv.ratio - 1 + conv.REGISTER_STAGES, conv.name)

    def test_a_narrowing_converter_can_emit_immediately(self):
        for conv in self.units(WidthConverter):
            if conv.in_elems > conv.out_elems:
                self.assertEqual(conv.latency(self.graph),
                                 conv.REGISTER_STAGES, conv.name)


class CnnLatencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = Compiler(device="vu9p", target_cycles=400).compile(
            Fixtures.factory().cnn(), Fixtures.samples(Fixtures.CNN_SHAPE),
            Fixtures.build_dir("cnn_latency"))
        cls.graph = cls.result.hardware_graph

    def test_pooling_waits_for_its_whole_window(self):
        pools = [n for n in self.graph.nodes if isinstance(n, PoolUnit)]
        self.assertTrue(pools)
        for pool in pools:
            folds = pool.channels // pool.pe
            self.assertEqual(pool.latency(self.graph),
                             (pool.window_size - 1) * folds + pool.REGISTER_STAGES,
                             pool.name)


class UniformDefaultTest(unittest.TestCase):
    """A unit that has not stated a beat schedule inherits the uniform
    approximation, so it looks instantaneous and reports only its register
    cycle. That is very nearly right for the elementwise units and wrong for
    the sliding window generator, which buffers rows before it can emit
    anything. Pinned deliberately: the day the generator reports a real
    schedule this test should be the one that notices."""

    def test_the_uniform_default_reports_only_its_register(self):
        result = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_uniform"))
        graph = result.hardware_graph
        defaults = [n for n in graph.nodes
                    if isinstance(n, StreamingNode)
                    and type(n).output_schedule is StreamingNode.output_schedule]
        self.assertTrue(defaults)
        for node in defaults:
            self.assertEqual(node.latency(graph), node.REGISTER_STAGES, node.name)


if __name__ == "__main__":
    unittest.main()
