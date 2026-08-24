"""Stages 2 to 6: import, lowering, folding, stream insertion, emission."""

import json
import unittest

import numpy as np

from support import Fixtures

from onnx2fpga.compile import Compiler
from onnx2fpga.p2_graph.graph import GraphError
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.p3_ops.hardware_ops import (MatVecUnit, PoolUnit, SlidingWindowUnit,
                                           StreamFifo, WidthConverter)
from onnx2fpga.p3_ops.onnx_ops import ConvOp, GemmOp, ReluOp
from onnx2fpga.p4_quantize.lower import FuseRelu
from onnx2fpga.p5_schedule.latency import graph_latency
from onnx2fpga.p5_schedule.streams import InsertWidthConverters, compatible_widths
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.targets.device import Device


class ImportTest(unittest.TestCase):
    def test_conv_model_becomes_channel_last(self):
        graph = OnnxImporter(Fixtures.factory().cnn()).run()
        self.assertEqual(graph.tensor("x").shape, (1, 12, 12, 1))
        conv = next(n for n in graph.nodes if isinstance(n, ConvOp))
        self.assertEqual(graph.tensor(conv.inputs[1]).shape, (3, 3, 1, 4))
        self.assertEqual(graph.tensor("c0").shape, (1, 10, 10, 4))

    def test_relu_fuses_into_its_producer(self):
        graph = OnnxImporter(Fixtures.factory().mlp()).run()
        self.assertEqual(sum(isinstance(n, ReluOp) for n in graph.nodes), 2)
        FuseRelu().run(graph, CompilerContext(Device.get("sim")))
        self.assertEqual(sum(isinstance(n, ReluOp) for n in graph.nodes), 0)
        fused = [n for n in graph.nodes if isinstance(n, GemmOp) and n.attrs.get("relu")]
        self.assertEqual(len(fused), 2)
        self.assertEqual(graph.outputs, ["y"])


class LoweringTest(unittest.TestCase):
    def setUp(self):
        self.result = Compiler(device="vu9p", target_cycles=400).compile(
            Fixtures.factory().cnn(), Fixtures.samples(Fixtures.CNN_SHAPE),
            Fixtures.build_dir("cnn"))
        self.graph = self.result.hardware_graph

    def test_conv_becomes_window_plus_matvec(self):
        kinds = [type(n) for n in self.graph.topological_order()]
        self.assertIn(SlidingWindowUnit, kinds)
        self.assertIn(MatVecUnit, kinds)
        self.assertIn(PoolUnit, kinds)

    def test_every_node_is_a_hardware_unit(self):
        for node in self.graph.nodes:
            self.assertTrue(hasattr(node, "module"), node)
            self.assertIsNotNone(node.module, node)

    def test_stream_widths_agree_across_every_edge(self):
        for node in self.graph.topological_order():
            for slot, name in enumerate(node.stream_inputs(self.graph)):
                producer = self.graph.producer(name)
                if producer is None:
                    continue
                supplied = producer.output_beat_elems(self.graph)[
                    producer.outputs.index(name)]
                self.assertEqual(supplied, node.input_beat_elems(self.graph)[slot],
                                 "width mismatch on %s into %s" % (name, node.name))

    def test_every_internal_edge_is_buffered(self):
        for node in self.graph.topological_order():
            if isinstance(node, StreamFifo):
                continue
            for name in node.stream_inputs(self.graph):
                producer = self.graph.producer(name)
                if producer is not None:
                    self.assertIsInstance(producer, (StreamFifo, WidthConverter),
                                          "unbuffered edge into %s" % node.name)


class FoldingTest(unittest.TestCase):
    def test_meeting_a_target_balances_every_stage(self):
        target = 128
        result = Compiler(device="vu9p", target_cycles=target).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_target"))
        self.assertLessEqual(result.folding.bottleneck_cycles, target)

    def test_a_tighter_target_costs_more_and_runs_faster(self):
        loose = Compiler(device="vu9p", target_cycles=512).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_loose"))
        tight = Compiler(device="vu9p", target_cycles=32).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_tight"))
        self.assertLess(tight.folding.bottleneck_cycles,
                        loose.folding.bottleneck_cycles)
        self.assertGreater(tight.folding.total.dsp, loose.folding.total.dsp)

    def test_never_exceeds_the_utilisation_limit(self):
        result = Compiler(device="z7020", utilisation_limit=0.5).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_small"))
        budget = Device.get("z7020").budget
        self.assertLessEqual(result.folding.total.worst_utilisation(budget), 0.5)


class BeatWidthTest(unittest.TestCase):
    """Neighbouring units must meet at an integer ratio, or the converter
    between them throttles the edge to their common divisor."""

    TARGETS = (400, 200, 100, 50, 25)

    def test_every_edge_converts_in_one_step(self):
        for target in self.TARGETS:
            result = Compiler(device="vu9p", target_cycles=target).compile(
                Fixtures.factory().cnn(), Fixtures.samples(Fixtures.CNN_SHAPE),
                Fixtures.build_dir("align_%d" % target))
            graph = result.hardware_graph
            for node in graph.topological_order():
                for slot, name in enumerate(node.stream_inputs(graph)):
                    producer = graph.producer(name)
                    if producer is None:
                        continue
                    supplied = producer.output_beat_elems(graph)[
                        producer.outputs.index(name)]
                    wanted = node.input_beat_elems(graph)[slot]
                    with self.subTest(target=target, edge=name):
                        self.assertTrue(
                            compatible_widths(supplied, wanted),
                            "%d -> %d into %s" % (supplied, wanted, node.name))

    def test_coprime_widths_fall_back_to_a_two_step_chain(self):
        self.assertEqual(InsertWidthConverters._stages(4, 2), [(4, 2)])
        self.assertEqual(InsertWidthConverters._stages(2, 8), [(2, 8)])
        self.assertEqual(InsertWidthConverters._stages(2, 5), [(2, 1), (1, 5)])
        self.assertEqual(InsertWidthConverters._stages(6, 4), [(6, 2), (2, 4)])

    def test_alignment_never_spends_more_resources(self):
        loose = Compiler(device="vu9p", target_cycles=200).compile(
            Fixtures.factory().cnn(), Fixtures.samples(Fixtures.CNN_SHAPE),
            Fixtures.build_dir("align_budget"))
        budget = Device.get("vu9p").budget
        self.assertLessEqual(loose.folding.total.worst_utilisation(budget), 0.80)


class EmissionTest(unittest.TestCase):
    def setUp(self):
        self.build = Fixtures.build_dir("emit")
        self.result = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE), self.build)

    def test_writes_a_complete_project(self):
        for relative in ("rtl/top.sv", "Makefile", "manifest.json",
                         "golden/input.hex", "golden/expected.hex"):
            self.assertTrue((self.build / relative).exists(), relative)
        self.assertTrue(list((self.build / "mem").glob("*.hex")))
        self.assertTrue(list((self.build / "rtl").glob("otf_*.sv")))

    def test_manifest_describes_the_ports(self):
        manifest = json.loads((self.build / "manifest.json").read_text())
        self.assertEqual(manifest["input"]["dtype"], "INT8")
        self.assertEqual(manifest["input"]["shape"], [1, 32])
        self.assertEqual(manifest["output"]["shape"], [1, 10])
        self.assertGreater(manifest["input"]["scale"], 0)
        self.assertEqual(manifest["cycles_per_frame"], 64)

    def test_golden_vectors_have_the_right_beat_count(self):
        manifest = json.loads((self.build / "manifest.json").read_text())
        stimulus = (self.build / "golden/input.hex").read_text().split()
        expected = (self.build / "golden/expected.hex").read_text().split()
        self.assertEqual(len(stimulus), manifest["input"]["beats"])
        self.assertEqual(len(expected), manifest["output"]["beats"])

    def test_weight_memory_has_one_word_per_lane_and_address(self):
        graph = self.result.hardware_graph
        unit = next(n for n in graph.nodes if isinstance(n, MatVecUnit))
        words = (self.build / ("mem/%s_weights.hex" % unit.name)).read_text().split()
        self.assertEqual(len(words),
                         unit.pe * unit.neuron_fold * unit.synapse_fold)
        self.assertEqual(len(words[0]) * 4, unit.simd * unit.weight_dtype.bits)


class UnrollTest(unittest.TestCase):
    """Fully unrolled: every unit at its widest folding, no balancing.

    The balanced allocator equalises stages because throughput cannot use a
    stage faster than its slowest neighbour. Minimising the wait for one answer
    inverts that, and on a small model against a large device the pipeline
    depth becomes the layer count."""

    @classmethod
    def setUpClass(cls):
        cls.result = Compiler(device="vu9p", unroll=True).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_unrolled"))
        cls.graph = cls.result.hardware_graph

    def matvecs(self):
        return [n for n in self.graph.nodes if isinstance(n, MatVecUnit)]

    def test_every_matvec_is_fully_parallel(self):
        units = self.matvecs()
        self.assertTrue(units)
        for unit in units:
            self.assertEqual(unit.simd, unit.mw, unit.name)
            self.assertEqual(unit.pe, unit.mh, unit.name)
            self.assertEqual(unit.synapse_fold, 1, unit.name)
            self.assertEqual(unit.neuron_fold, 1, unit.name)

    def test_a_layer_costs_a_cycle_per_vector(self):
        for unit in self.matvecs():
            self.assertEqual(unit.cycles, unit.vectors, unit.name)

    def test_the_pipeline_is_single_digit_per_layer(self):
        """The item's own Done-when."""
        for unit in self.matvecs():
            self.assertLess(unit.cycles, 10, unit.name)

    def test_it_beats_the_folded_build_by_an_order_of_magnitude(self):
        folded = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_unrolled_ref"))
        self.assertLess(graph_latency(self.graph),
                        graph_latency(folded.hardware_graph) / 10)

    def test_it_costs_what_that_speed_costs(self):
        """Unrolling is not free and the report should not pretend it is."""
        folded = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("mlp_unrolled_ref2"))
        self.assertGreater(self.result.folding.total.dsp,
                           folded.folding.total.dsp * 10)


class UnrollRefusalTest(unittest.TestCase):
    def test_a_model_that_does_not_fit_is_refused_not_quietly_folded(self):
        """Asking for unrolled and being handed a folded design without being
        told is the kind of help that costs a day to notice."""
        with self.assertRaises(GraphError) as caught:
            Compiler(device="z7020", unroll=True).compile(
                Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
                Fixtures.build_dir("mlp_unrolled_toobig"))
        message = str(caught.exception)
        self.assertIn("z7020", message)
        self.assertIn("dsp", message)
        self.assertIn("%", message)


if __name__ == "__main__":
    unittest.main()
