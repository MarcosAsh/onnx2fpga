"""The whole pipeline, end to end.

Compiler.compile is the only entry point most users need: an ONNX model and a
handful of calibration samples in, a build directory out.
"""

import copy

import numpy as np

from .p2_graph.onnx_importer import OnnxImporter
from .p4_quantize.calibrate import Calibrator
from .p4_quantize.plan import QuantizationPlan
from .p4_quantize.lower import FuseRelu, LowerToHardware
from .p5_schedule.folding import FoldingAllocator
from .p5_schedule.streams import (AlignBeatWidths, InsertDuplicates, InsertFifos,
                                  InsertWidthConverters, StreamsAreSingleConsumer)
from .p6_emit.project import ProjectWriter
from .p6_emit.stream_io import StreamCodec
from .pipeline import CompilerContext, PassManager
from .reference.runner import GraphRunner
from .targets.device import Device


class CompilationResult:
    def __init__(self, float_graph, hardware_graph, context, build):
        self.float_graph = float_graph
        self.hardware_graph = hardware_graph
        self.context = context
        self.build = build

    @property
    def folding(self):
        return self.context.artifacts.get("folding")

    @property
    def plan(self):
        return self.context.artifacts.get("plan")

    def summary(self):
        lines = list(self.context.log)
        if self.folding:
            lines.append("")
            lines.append(self.folding.render())
        return "\n".join(lines)


class Compiler:
    def __init__(self, device="vu9p", target_cycles=None, utilisation_limit=0.80,
                 mult_bits=18, top_name="otf_top", verbose=False, fifo_cap=None,
                 unroll=False):
        self.device = device if isinstance(device, Device) else Device.get(device)
        self.target_cycles = target_cycles
        self.unroll = unroll
        self.utilisation_limit = utilisation_limit
        self.mult_bits = mult_bits
        self.top_name = top_name
        self.verbose = verbose
        self.fifo_cap = fifo_cap

    def compile(self, model, samples=None, out_dir="build/out"):
        context = CompilerContext(self.device)
        float_graph = OnnxImporter(model).run()
        samples = [self._adapt(float_graph, feeds) for feeds in (samples or [])]

        prologue = PassManager([FuseRelu()], self.verbose)
        float_graph = prologue.run(float_graph, context)

        plan = self._plan(float_graph, samples, context)
        # Lowering rewrites in place, so take a copy first. Without it the two
        # graphs on the result would be one object under two names, and any
        # comparison between them would be vacuous.
        working = float_graph
        float_graph = copy.deepcopy(working)
        graph = PassManager([
            LowerToHardware(plan),
            InsertDuplicates(),
            FoldingAllocator(self.target_cycles, self.utilisation_limit,
                             unroll=self.unroll),
            AlignBeatWidths(),
            InsertWidthConverters(),
            InsertFifos(cap=self.fifo_cap),
            StreamsAreSingleConsumer(),
        ], self.verbose).run(working, context)

        writer = ProjectWriter(graph, context, self.top_name)
        build = writer.write(out_dir)
        self._write_golden(graph, plan, self._stimulus(graph, samples, plan), build)
        return CompilationResult(float_graph, graph, context, build)

    def _plan(self, graph, samples, context):
        """A model that was quantized before it got here already states every
        scale, so calibrating it would be guessing at an answer it gave."""
        if graph.annotations:
            context.note("plan: read %d grids from the model's quantize nodes"
                         % len(graph.annotations))
            return QuantizationPlan.from_annotations(
                graph.annotations, mult_bits=self.mult_bits)
        if not samples:
            raise ValueError("a float model needs calibration samples")
        context.note("plan: calibrated on %d samples" % len(samples))
        return Calibrator(graph).plan(samples, mult_bits=self.mult_bits)

    @staticmethod
    def _stimulus(graph, samples, plan):
        """Real calibration data if there is any, otherwise something with the
        right shape so the golden vectors still exercise the design."""
        name = graph.inputs[0]
        if samples:
            return np.asarray(list(samples[0].values())[0], dtype=np.float64)
        rng = np.random.default_rng(0)
        peak = plan.scale(name) * plan.act_dtype.max
        return rng.uniform(-peak, peak, graph.tensor(name).shape)

    @staticmethod
    def _adapt(graph, feeds):
        """Samples arrive in the model's declared layout; the graph works in
        channel-last."""
        return {name: graph.layouts[name].to_internal(array)
                if name in graph.layouts else array
                for name, array in feeds.items()}

    def _write_golden(self, graph, plan, sample, build):
        """Stimulus and expected output for the Verilator harness, taken from
        the integer reference model rather than from the float one."""
        in_name = graph.inputs[0]
        out_name = graph.outputs[0]
        in_tensor, out_tensor = graph.tensor(in_name), graph.tensor(out_name)

        raw = sample
        quantized = plan.act_dtype.clamp(
            np.rint(raw / plan.scale(in_name)) + plan.zero_point(in_name))
        produced = GraphRunner(graph).run({in_name: quantized})[out_name]

        StreamCodec(in_tensor).to_image(quantized).write(build.path("golden/input.hex"))
        StreamCodec(out_tensor).to_image(produced).write(
            build.path("golden/expected.hex"))
        return produced
