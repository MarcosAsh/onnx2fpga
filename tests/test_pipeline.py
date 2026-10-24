"""Import, lowering, folding, stream insertion and emission."""

import json
import unittest

from support import Fixtures

from onnx2fpga.compile import Compiler
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.p3_ops.hardware_ops import (MatVecUnit, PoolUnit, SlidingWindowUnit,
                                           StreamFifo, WidthConverter)
from onnx2fpga.p3_ops.onnx_ops import ConvOp, GemmOp, ReluOp
from onnx2fpga.p4_quantize.lower import FuseRelu
from onnx2fpga.p5_schedule.streams import InsertWidthConverters, compatible_widths
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.targets.device import Device


def compile_model(name, target, out, **options):
    shape = Fixtures.CNN_SHAPE if name == "cnn" else Fixtures.MLP_SHAPE
    return Compiler(target_cycles=target, **options).compile(
        getattr(Fixtures.factory(), name)(), Fixtures.samples(shape), Fixtures.build_dir(out))


class ImportTest(unittest.TestCase):
    def test_conv_becomes_channel_last(self):
        graph = OnnxImporter(Fixtures.factory().cnn()).run()
        self.assertEqual(graph.tensor("x").shape, (1, 12, 12, 1))
        conv = next(n for n in graph.nodes if isinstance(n, ConvOp))
        self.assertEqual(graph.tensor(conv.inputs[1]).shape, (3, 3, 1, 4))
        self.assertEqual(graph.tensor("c0").shape, (1, 10, 10, 4))

    def test_relu_fuses_into_producer(self):
        graph = OnnxImporter(Fixtures.factory().mlp()).run()
        self.assertEqual(sum(isinstance(n, ReluOp) for n in graph.nodes), 2)
        FuseRelu().run(graph, CompilerContext(Device.get("sim")))
        self.assertEqual(sum(isinstance(n, ReluOp) for n in graph.nodes), 0)
        fused = [n for n in graph.nodes if isinstance(n, GemmOp) and n.attrs.get("relu")]
        self.assertEqual(len(fused), 2)
        self.assertEqual(graph.outputs, ["y"])


class LoweringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = compile_model("cnn", 400, "cnn").hardware_graph

    def test_conv_becomes_window_and_matvec(self):
        kinds = {type(n) for n in self.graph.nodes}
        self.assertTrue({SlidingWindowUnit, MatVecUnit, PoolUnit} <= kinds)
        for node in self.graph.nodes:
            self.assertIsNotNone(getattr(node, "module", None), node)

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
    def test_target_is_met(self):
        result = compile_model("mlp", 128, "mlp_target", device="vu9p")
        self.assertLessEqual(result.folding.bottleneck_cycles, 128)

    def test_tighter_target_costs_more_and_runs_faster(self):
        loose = compile_model("mlp", 512, "mlp_loose", device="vu9p")
        tight = compile_model("mlp", 32, "mlp_tight", device="vu9p")
        self.assertLess(tight.folding.bottleneck_cycles, loose.folding.bottleneck_cycles)
        self.assertGreater(tight.folding.total.dsp, loose.folding.total.dsp)

    def test_utilisation_limit_holds(self):
        result = compile_model("mlp", None, "mlp_small", device="z7020",
                               utilisation_limit=0.5)
        budget = Device.get("z7020").budget
        self.assertLessEqual(result.folding.total.worst_utilisation(budget), 0.5)


class BeatWidthTest(unittest.TestCase):
    def test_edges_agree_and_converters_take_one_step(self):
        """Neighbouring widths meet at an integer ratio, or the converter throttles the edge."""
        budget = Device.get("vu9p").budget
        for target in (400, 200, 100, 50, 25):
            result = compile_model("cnn", target, "align_%d" % target, device="vu9p")
            graph = result.hardware_graph
            self.assertLessEqual(result.folding.total.worst_utilisation(budget), 0.80)
            for node in graph.topological_order():
                if isinstance(node, WidthConverter):
                    with self.subTest(target=target, converter=node.name):
                        self.assertTrue(compatible_widths(node.in_elems, node.out_elems))
                for slot, name in enumerate(node.stream_inputs(graph)):
                    producer = graph.producer(name)
                    if producer is None:
                        continue
                    supplied = producer.output_beat_elems(graph)[producer.outputs.index(name)]
                    with self.subTest(target=target, edge=name):
                        self.assertEqual(supplied, node.input_beat_elems(graph)[slot])

    def test_coprime_widths_take_two_steps(self):
        self.assertEqual(InsertWidthConverters._stages(4, 2), [(4, 2)])
        self.assertEqual(InsertWidthConverters._stages(2, 8), [(2, 8)])
        self.assertEqual(InsertWidthConverters._stages(2, 5), [(2, 1), (1, 5)])
        self.assertEqual(InsertWidthConverters._stages(6, 4), [(6, 2), (2, 4)])


class EmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = compile_model("mlp", 64, "emit", device="vu9p")
        cls.build = cls.result.build.root
        cls.manifest = json.loads((cls.build / "manifest.json").read_text())

    def test_project_is_complete(self):
        for relative in ("rtl/top.sv", "Makefile", "manifest.json",
                         "golden/input.hex", "golden/expected.hex"):
            self.assertTrue((self.build / relative).exists(), relative)
        self.assertTrue(list((self.build / "mem").glob("*.hex")))
        self.assertTrue(list((self.build / "rtl").glob("otf_*.sv")))

    def test_manifest_describes_ports(self):
        self.assertEqual(self.manifest["input"]["dtype"], "INT8")
        self.assertEqual(self.manifest["input"]["shape"], [1, 32])
        self.assertEqual(self.manifest["output"]["shape"], [1, 10])
        self.assertGreater(self.manifest["input"]["scale"], 0)
        self.assertEqual(self.manifest["cycles_per_frame"], 64)

    def test_golden_beat_counts(self):
        stimulus = (self.build / "golden/input.hex").read_text().split()
        expected = (self.build / "golden/expected.hex").read_text().split()
        self.assertEqual(len(stimulus), self.manifest["input"]["beats"])
        self.assertEqual(len(expected), self.manifest["output"]["beats"])

    def test_weight_memory_has_a_word_per_lane_and_address(self):
        unit = next(n for n in self.result.hardware_graph.nodes if isinstance(n, MatVecUnit))
        words = (self.build / ("mem/%s_weights.hex" % unit.name)).read_text().split()
        self.assertEqual(len(words), unit.pe * unit.neuron_fold * unit.synapse_fold)
        self.assertEqual(len(words[0]) * 4, unit.simd * unit.weight_dtype.bits)


if __name__ == "__main__":
    unittest.main()
