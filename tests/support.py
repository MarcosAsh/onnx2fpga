"""Shared fixtures. Imported by every test module."""

import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "examples"))

from make_models import ModelFactory  # noqa: E402


class Fixtures:
    MLP_SHAPE = (1, 32)
    CNN_SHAPE = (1, 1, 12, 12)

    @staticmethod
    def factory(seed=7):
        return ModelFactory(seed)

    @staticmethod
    def samples(shape, count=16, seed=3):
        rng = np.random.default_rng(seed)
        return [{"x": rng.normal(0, 1, shape).astype(np.float32)} for _ in range(count)]

    @staticmethod
    def build_dir(name):
        target = ROOT / "build" / "tests" / name
        target.mkdir(parents=True, exist_ok=True)
        return target


def run(command, cwd=None, timeout=900):
    return subprocess.run(command, cwd=cwd or ROOT, capture_output=True,
                          text=True, timeout=timeout)


class SingleUnitProject:
    """Emits a build directory containing one hardware unit.

    Uses the real netlist builder and project writer, so a unit test exercises
    the same emission path a full model does, and the golden vectors come from
    the unit's own reference implementation.
    """

    def __init__(self, node, input_shape, input_dtype, device="sim"):
        self.node = node
        self.input_shape = tuple(input_shape)
        self.input_dtype = input_dtype
        self.device = device

    def write(self, out_dir, input_array):
        from onnx2fpga.p2_graph.graph import Graph
        from onnx2fpga.p2_graph.tensor import Tensor
        from onnx2fpga.p6_emit.project import ProjectWriter
        from onnx2fpga.p6_emit.stream_io import StreamCodec
        from onnx2fpga.pipeline import CompilerContext
        from onnx2fpga.targets.device import Device

        graph = Graph("unit")
        graph.add_tensor(Tensor(self.node.inputs[0], self.input_shape, self.input_dtype))
        graph.inputs.append(self.node.inputs[0])
        graph.ensure_tensor(self.node.outputs[0])
        graph.add_node(self.node)
        self.node.infer(graph)
        graph.tensor(self.node.inputs[0]).elems_per_beat = \
            self.node.input_beat_elems(graph)[0]
        graph.outputs.append(self.node.outputs[0])

        context = CompilerContext(Device.get(self.device))
        build = ProjectWriter(graph, context, "otf_top").write(out_dir)

        expected = self.node.execute(graph, {self.node.inputs[0]: input_array})[0]
        StreamCodec(graph.tensor(self.node.inputs[0])).to_image(input_array).write(
            build.path("golden/input.hex"))
        StreamCodec(graph.tensor(self.node.outputs[0])).to_image(expected).write(
            build.path("golden/expected.hex"))
        return build, expected
