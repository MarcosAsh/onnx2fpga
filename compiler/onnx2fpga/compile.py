import copy

import numpy as np

from .errors import ConfigError, ModelError
from .hardware_layer import custom_passes
from .p2_graph.onnx_importer import OnnxImporter
from .p4_quantize.calibrate import Calibrator
from .p4_quantize.plan import QuantizationPlan
from .p4_quantize.lower import FuseRelu, LowerToHardware
from .p4_quantize.trees import plan_trees, tree_op
from .p5_schedule.fixed import UnrollForFixedLatency, check_fixed_ops
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

    @property
    def latency(self):
        """Cycles from input to output in fixed latency mode, None for a streaming design."""
        return self.context.artifacts.get("latency")


class Compiler:
    """`folding`, `weight_dtypes`, `arithmetic` and `custom_objects` are the per-layer settings built by onnx2fpga.api."""

    def __init__(self, device="vu9p", target_cycles=None, utilisation_limit=0.80,
                 mult_bits=18, top_name="otf_top", verbose=False, fifo_cap=None,
                 hdl="sv", folding=None, weight_dtypes=None, custom_objects=None,
                 mode="stream", arithmetic=None):
        self.device = device if isinstance(device, Device) else Device.get(device)
        self.target_cycles = target_cycles
        self.utilisation_limit = utilisation_limit
        self.mult_bits = mult_bits
        self.top_name = top_name
        self.verbose = verbose
        self.fifo_cap = fifo_cap
        self.hdl = hdl
        self.folding = dict(folding or {})
        self.weight_dtypes = dict(weight_dtypes or {})
        self.custom_handlers, self.custom_lowerings = custom_passes(custom_objects)
        self.mode = mode
        self.arithmetic = dict(arithmetic or {})

    def compile(self, model, samples=None, out_dir="build/out"):
        context = CompilerContext(self.device)
        float_graph = OnnxImporter(model, self.custom_handlers).run()
        samples = [self._adapt(float_graph, feeds) for feeds in (samples or [])]

        prologue = PassManager([FuseRelu()], self.verbose)
        float_graph = prologue.run(float_graph, context)
        if self.mode == "fixed":
            check_fixed_ops(float_graph)
        elif tree_op(float_graph):
            raise ConfigError("a tree ensemble compiles only with mode='fixed'")

        plan = self._plan(float_graph, samples, context)
        # Lowering rewrites in place, so copy first or the two graphs on the result would be one object.
        working = float_graph
        float_graph = copy.deepcopy(working)
        if self.mode == "fixed":
            passes = [LowerToHardware(plan, self.custom_lowerings),
                      UnrollForFixedLatency(self.utilisation_limit, self.arithmetic)]
        else:
            passes = [
                LowerToHardware(plan, self.custom_lowerings),
                InsertDuplicates(),
                FoldingAllocator(self.target_cycles, self.utilisation_limit, self.folding),
                AlignBeatWidths(),
                InsertWidthConverters(),
                InsertFifos(cap=self.fifo_cap),
                StreamsAreSingleConsumer(),
            ]
        graph = PassManager(passes, self.verbose).run(working, context)

        writer = ProjectWriter(graph, context, self.top_name, hdl=self.hdl)
        build = writer.write(out_dir)
        self._write_golden(graph, plan, self._stimulus(graph, samples, plan), build)
        return CompilationResult(float_graph, graph, context, build)

    def _plan(self, graph, samples, context):
        """A pre-quantized model already states every scale, so calibrating it would be guessing."""
        if tree_op(graph):
            context.note("plan: feature grids from the tree thresholds")
            plan = plan_trees(graph, self.mult_bits)
        elif graph.annotations:
            context.note("plan: read %d grids from the model's quantize nodes"
                         % len(graph.annotations))
            plan = QuantizationPlan.from_annotations(
                graph.annotations, mult_bits=self.mult_bits)
        elif not samples:
            raise ModelError("a float model needs calibration samples")
        else:
            context.note("plan: calibrated on %d samples" % len(samples))
            plan = Calibrator(graph).plan(samples, mult_bits=self.mult_bits)
        plan.layer_weight_dtypes = dict(self.weight_dtypes)
        return plan

    @staticmethod
    def _stimulus(graph, samples, plan):
        """Real calibration data if any, otherwise random data of the right shape."""
        name = graph.inputs[0]
        if samples:
            return np.asarray(list(samples[0].values())[0], dtype=np.float64)
        rng = np.random.default_rng(0)
        peak = plan.scale(name) * plan.act_dtype.max
        return rng.uniform(-peak, peak, graph.tensor(name).shape)

    @staticmethod
    def _adapt(graph, feeds):
        """Samples arrive in the model layout; the graph works in channel-last."""
        return {name: graph.layouts[name].to_internal(array)
                if name in graph.layouts else array
                for name, array in feeds.items()}

    def _write_golden(self, graph, plan, sample, build):
        """Stimulus and expected output for the Verilator harness, from the integer reference model."""
        in_name, out_name = graph.inputs[0], graph.outputs[0]
        quantized = plan.quantize(in_name, sample)
        produced = GraphRunner(graph).run({in_name: quantized})[out_name]
        StreamCodec(graph.tensor(in_name)).to_image(quantized).write(
            build.path("golden/input.hex"))
        StreamCodec(graph.tensor(out_name)).to_image(produced).write(
            build.path("golden/expected.hex"))
