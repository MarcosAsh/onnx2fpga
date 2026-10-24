"""The Hardcaml backend's Python half, which needs no OCaml."""

import json
import os
import stat
import unittest

from support import Fixtures

from onnx2fpga.compile import Compiler
from onnx2fpga.p3_ops.hardware_ops import DuplicateUnit, MatVecUnit
from onnx2fpga.p6_emit.hardcaml import DesignDescription, HardcamlError, HardcamlGenerator


class DesignDescriptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().residual(), Fixtures.samples((1, 16)),
            Fixtures.build_dir("hardcaml_description"))
        cls.graph = result.hardware_graph
        cls.design = DesignDescription(cls.graph).build()

    def test_every_node_appears_once_in_netlist_order(self):
        self.assertEqual([n["name"] for n in self.design["nodes"]],
                         [n.name for n in self.graph.topological_order()])

    def test_every_stream_a_node_touches_has_a_width(self):
        widths = {s["name"]: s["width"] for s in self.design["streams"]}
        for node in self.design["nodes"]:
            for name in node["inputs"] + node["outputs"]:
                self.assertIn(name, widths)
        self.assertEqual(widths[self.design["input"]],
                         self.graph.tensor(self.graph.inputs[0]).elems_per_beat * 8)

    def test_a_fork_lists_its_branches_in_port_order(self):
        fork = next(n for n in self.graph.nodes if isinstance(n, DuplicateUnit))
        entry = next(n for n in self.design["nodes"] if n["name"] == fork.name)
        self.assertEqual(entry["outputs"], fork.outputs)
        self.assertEqual(entry["params"]["FANOUT"], len(fork.outputs))

    def test_memories_are_the_bit_patterns_the_hex_images_hold(self):
        unit = next(n for n in self.graph.nodes if isinstance(n, MatVecUnit))
        entry = next(n for n in self.design["nodes"] if n["name"] == unit.name)
        self.assertNotIn("WEIGHT_FILE", entry["params"])
        weights = entry["memories"]["weights"]
        self.assertEqual(weights["lanes"], unit.simd)
        self.assertEqual(len(weights["values"]),
                         unit.pe * unit.neuron_fold * unit.synapse_fold * unit.simd)
        self.assertTrue(all(0 <= v < 256 for v in weights["values"]))
        packed = unit.memories(self.graph)["weights"].array.reshape(-1)
        self.assertEqual(weights["values"], [int(v) & 0xff for v in packed])

    def test_rendering_is_stable(self):
        first = DesignDescription(self.graph).render()
        self.assertEqual(first, DesignDescription(self.graph).render())
        self.assertEqual(json.loads(first), self.design)


class GeneratorAvailabilityTest(unittest.TestCase):
    """A stand in for dune decides what the build reports."""

    def fake_dune(self, name, stderr, code):
        folder = Fixtures.build_dir("hardcaml_fake_dune")
        script = folder / name
        script.write_text("#!/bin/sh\necho '%s' >&2\nexit %d\n" % (stderr, code))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return HardcamlGenerator(dune=str(script))

    def test_a_switch_without_hardcaml_is_unavailable(self):
        generator = self.fake_dune("dune_missing", 'Error: Library "hardcaml" not found.', 1)
        self.assertIn("hardcaml", generator.unavailable())
        with self.assertRaises(HardcamlError):
            generator.executable

    def test_a_project_that_fails_to_build_is_broken_not_unavailable(self):
        """Otherwise an OCaml compile error would skip every Hardcaml test rather than fail one."""
        generator = self.fake_dune("dune_broken", "Error: This expression has type int", 1)
        self.assertIsNone(generator.unavailable())
        with self.assertRaises(HardcamlError) as caught:
            generator.executable
        self.assertIn("This expression has type int", str(caught.exception))

    def test_no_dune_is_unavailable(self):
        generator = HardcamlGenerator(dune="no-such-dune-anywhere")
        self.assertIn("dune is not on PATH", generator.unavailable())

    def test_an_override_without_a_project_is_unavailable(self):
        previous = os.environ.get(HardcamlGenerator.ENV)
        os.environ[HardcamlGenerator.ENV] = str(Fixtures.build_dir("hardcaml_nowhere"))
        try:
            self.assertIn("no Hardcaml project", HardcamlGenerator().unavailable())
        finally:
            if previous is None:
                os.environ.pop(HardcamlGenerator.ENV, None)
            else:
                os.environ[HardcamlGenerator.ENV] = previous


if __name__ == "__main__":
    unittest.main()
