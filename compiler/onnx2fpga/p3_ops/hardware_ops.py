"""Streaming hardware nodes. Each maps one-to-one onto a SystemVerilog module.

Every node reports three things the rest of the compiler needs: the cycles it
takes to process one frame at its current folding, the resources that folding
costs, and the exact integer function it computes.
"""

import numpy as np

from ..targets.resources import MemoryStyle, Resources
from ..p2_graph.datatype import INT32, IntType
from ..p2_graph.node import HwNode, MemorySpec
from .onnx_ops import Im2Col


def divisors(value, limit=None):
    limit = value if limit is None else min(limit, value)
    return [d for d in range(1, limit + 1) if value % d == 0]


class StreamingNode(HwNode):
    """Common plumbing: beat geometry, cycle cost, and beat level timing.

    The schedules are what the buffer sizing reads. Each says, in cycles from
    this unit's own start and assuming it never stalls, when it takes each
    input beat and when it hands over each output beat. That is enough to work
    out how far ahead one branch of a fork can run before the join drains it,
    which a cycle count alone cannot express.
    """

    #: Every module registers its output, so a beat presented to it is visible
    #: to the next unit one cycle later. The FIFO is the clearest case: it
    #: writes mem[wptr] on a clock edge and reads mem[rptr] combinationally, so
    #: even a pass-through costs a cycle. The schedules describe beat order,
    #: not this, so latency has to add it back.
    REGISTER_STAGES = 1

    @property
    def frames(self):
        return int(self.attrs.get("frames", 1))

    def input_beats(self, graph, index=0):
        name = self.stream_inputs(graph)[index]
        return graph.tensor(name).numel // self.input_beat_elems(graph)[index]

    def output_beats(self, graph, index=0):
        name = self.outputs[index]
        return graph.tensor(name).numel // self.output_beat_elems(graph)[index]

    def _uniform(self, beats):
        """Evenly spread, never faster than one beat per cycle."""
        if beats <= 0:
            return []
        span = max(self.cycles, beats)
        return [i * span // beats for i in range(beats)]

    def input_schedule(self, graph, index=0):
        return self._uniform(self.input_beats(graph, index))

    def output_schedule(self, graph, index=0):
        return self._uniform(self.output_beats(graph, index))

    def latency(self, graph, index=0, output_index=0):
        """Cycles from this unit's first input beat to its first output beat.

        This is the unit's contribution to end-to-end latency, and it is not
        `cycles`. `cycles` is the initiation interval, how long one frame
        occupies the unit; latency is how long the first answer takes to
        appear. A matrix vector unit folded over four synapse groups has to
        accumulate all four before it can emit anything, so it is late by
        three cycles no matter how many vectors follow.

        It is read straight off the schedules rather than stated separately,
        so a unit cannot report a latency that disagrees with the beat timing
        the buffer sizing already believes. The cost is that a unit on the
        uniform default inherits that approximation here too, and reports
        zero. For the elementwise units that is very nearly true. For the
        sliding window generator it is not: it buffers rows before it can
        emit a window, and until it reports a real schedule its latency is
        understated. That is the same soft spot tokens.py already has, and it
        is why the graph total is checked against the harness rather than
        trusted.
        """
        outputs = self.output_schedule(graph, output_index)
        if not outputs:
            return 0
        inputs = self.input_schedule(graph, index)
        first_input = inputs[0] if inputs else 0
        return max(0, outputs[0] - first_input) + self.REGISTER_STAGES

    def throughput_cycles(self):
        return self.cycles

    def resources(self, device):
        return Resources.zero()

    def sv_params(self, graph):
        return {}


class MatVecUnit(StreamingNode):
    """Folded matrix-vector engine with fused bias and rescale.

    Weights are held as a (MW, MH) integer matrix. SIMD sets how many input
    elements are consumed per cycle, PE how many output channels are produced
    in parallel. One vector therefore costs (MW/SIMD) * (MH/PE) cycles, which
    is the knob the balancing pass turns to equalise the pipeline.
    """

    module = "otf_mvau"

    def __init__(self, name, inputs, outputs, weights, bias, requant,
                 act_dtype, out_dtype, weight_dtype, vectors=1, **attrs):
        self.weights = np.asarray(weights, dtype=np.int64)
        self.bias = np.asarray(bias, dtype=np.int64)
        self.requant = requant
        self.act_dtype = act_dtype
        self.out_dtype = out_dtype
        self.weight_dtype = weight_dtype
        attrs.setdefault("vectors", vectors)
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)

    @property
    def mw(self):
        return int(self.weights.shape[0])

    @property
    def mh(self):
        return int(self.weights.shape[1])

    @property
    def vectors(self):
        return int(self.attrs["vectors"])

    @property
    def simd(self):
        return self.folding["simd"]

    @property
    def pe(self):
        return self.folding["pe"]

    @property
    def synapse_fold(self):
        return self.mw // self.simd

    @property
    def neuron_fold(self):
        return self.mh // self.pe

    def default_folding(self):
        return {"simd": 1, "pe": 1}

    def folding_options(self):
        options = []
        for simd in divisors(self.mw):
            for pe in divisors(self.mh):
                options.append({"simd": simd, "pe": pe})
        options.sort(key=lambda c: (c["simd"] * c["pe"], c["pe"]))
        return options

    @property
    def cycles(self):
        return self.vectors * self.synapse_fold * self.neuron_fold

    @property
    def accumulator_dtype(self):
        lo = int(np.minimum(self.weights, 0).sum(axis=0).min()) * self.act_dtype.max
        hi = int(np.maximum(self.weights, 0).sum(axis=0).max()) * self.act_dtype.max
        if self.act_dtype.signed:
            span = int(np.abs(self.weights).sum(axis=0).max()) * max(
                abs(self.act_dtype.min), self.act_dtype.max)
            lo, hi = -span, span
        lo += int(self.bias.min(initial=0))
        hi += int(self.bias.max(initial=0))
        return IntType.smallest_for(lo, hi)

    def resources(self, device):
        acc_bits = self.accumulator_dtype.bits
        macs = self.pe * self.simd
        dsp = device.dsp_cost(macs, self.act_dtype.bits, self.weight_dtype.bits)
        dsp += self.pe if acc_bits > 18 else 0
        adder_lut = self.pe * max(0, self.simd - 1) * acc_bits * 0.7
        control_ff = self.pe * acc_bits + self.simd * self.act_dtype.bits
        compute = Resources(lut=adder_lut, ff=control_ff, dsp=dsp)

        depth = self.synapse_fold * self.neuron_fold
        width = self.simd * self.weight_dtype.bits
        style = MemoryStyle.select(depth, width)
        weight_mem = style.cost(depth, width) * self.pe

        buffer_cost = Resources.zero()
        if self.neuron_fold > 1:
            buffer_cost = MemoryStyle.select(
                self.synapse_fold, self.simd * self.act_dtype.bits
            ).cost(self.synapse_fold, self.simd * self.act_dtype.bits)
        return compute + weight_mem + buffer_cost

    def input_beat_elems(self, graph):
        return [self.simd]

    def output_beat_elems(self, graph):
        return [self.pe]

    def infer(self, graph):
        """Spatial structure survives when the input's innermost axis is
        already the vector being multiplied, as it is downstream of a sliding
        window unit. After a flatten it does not, and the output collapses to
        one vector per row."""
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        if source.shape and int(source.shape[-1]) == self.mw:
            out.shape = tuple(source.shape[:-1]) + (self.mh,)
        else:
            out.shape = (source.numel // self.mw, self.mh)
        out.dtype = self.out_dtype
        out.elems_per_beat = self.pe

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.int64).reshape(-1, self.mw)
        acc = x @ self.weights + self.bias
        out = self.requant.apply(acc)
        shape = graph.tensor(self.outputs[0]).shape
        return [out.reshape(shape)]

    def input_schedule(self, graph, index=0):
        stride = self.synapse_fold * self.neuron_fold
        return [v * stride + sf
                for v in range(self.vectors)
                for sf in range(self.synapse_fold)]

    def output_schedule(self, graph, index=0):
        stride = self.synapse_fold * self.neuron_fold
        return [v * stride + nf * self.synapse_fold + self.synapse_fold - 1
                for v in range(self.vectors)
                for nf in range(self.neuron_fold)]

    def memories(self, graph):
        """Weight layout mirrors the RTL: one memory per PE, addressed by
        (neuron fold, synapse fold), each word holding SIMD weights."""
        pe, simd = self.pe, self.simd
        packed = np.zeros((pe, self.neuron_fold * self.synapse_fold, simd), dtype=np.int64)
        for p in range(pe):
            for nf in range(self.neuron_fold):
                out_channel = nf * pe + p
                for sf in range(self.synapse_fold):
                    window = self.weights[sf * simd:(sf + 1) * simd, out_channel]
                    packed[p, nf * self.synapse_fold + sf, :] = window
        return {
            "weights": MemorySpec(packed, self.weight_dtype.bits, packed=True),
            "bias": MemorySpec(self.bias.reshape(self.neuron_fold, pe).T,
                               self.accumulator_dtype.bits),
            "mult": MemorySpec(self.requant.broadcast_to(self.mh).multipliers.reshape(
                self.neuron_fold, pe).T, self.mult_bits),
        }

    @property
    def mult_bits(self):
        """Narrowed to the values it has to carry, unless the representation
        was pinned, in which case it is the width that was pinned: a fitted
        width follows the weights and takes the netlist with it."""
        if self.requant.width is not None:
            return int(self.requant.width)
        return int(max(2, int(np.abs(self.requant.multipliers).max()).bit_length() + 1))

    def sv_params(self, graph):
        paths = self.memory_paths()
        return {
            "MW": self.mw, "MH": self.mh, "SIMD": self.simd, "PE": self.pe,
            "ACT_BITS": self.act_dtype.bits, "WGT_BITS": self.weight_dtype.bits,
            "ACC_BITS": self.accumulator_dtype.bits,
            "OUT_BITS": self.out_dtype.bits,
            "MULT_BITS": self.mult_bits,
            "SHIFT": self.requant.shift,
            "OUT_MIN": self.requant.lower_clamp, "OUT_MAX": self.requant.upper_clamp,
            "OUT_ZP": self.requant.zero_point,
            "WEIGHT_FILE": paths.get("weights", ""),
            "BIAS_FILE": paths.get("bias", ""),
            "MULT_FILE": paths.get("mult", ""),
        }


class SlidingWindowUnit(StreamingNode):
    """Turns a channel-last image stream into a stream of flattened windows.

    Two strategies, same interface and same output. `line` keeps only the span
    a window can reach, so capture and replay overlap and the unit costs its
    replay term alone. `frame` stores the whole feature map and replays it
    afterwards, which is simpler to reason about and is kept as a reference for
    differential testing.
    """

    STRATEGIES = {"line": "otf_swu_line", "frame": "otf_swu"}

    def __init__(self, name, inputs, outputs, ifm_dim, ifm_ch, kernel, strides,
                 pads, dilations, dtype, strategy="line", pad_value=0, **attrs):
        self.ifm_dim = tuple(ifm_dim)
        self.ifm_ch = int(ifm_ch)
        self.kernel = tuple(kernel)
        self.strides = tuple(strides)
        self.pads = tuple(pads)
        self.dilations = tuple(dilations)
        self.dtype = dtype
        self.strategy = strategy
        self.pad_value = int(pad_value)
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)
        if strategy not in self.STRATEGIES:
            raise NotImplementedError("sliding window strategy %r" % strategy)

    @property
    def module(self):
        return self.STRATEGIES[self.strategy]

    def window(self):
        return Im2Col(self.kernel, self.strides, self.pads, self.dilations)

    @property
    def ofm_dim(self):
        return self.window().output_extent(*self.ifm_dim)

    @property
    def window_elems(self):
        return self.kernel[0] * self.kernel[1] * self.ifm_ch

    @property
    def span_pixels(self):
        """Pixels one window reaches across in raster order."""
        return ((self.kernel[0] - 1) * self.dilations[0] * self.ifm_dim[1]
                + (self.kernel[1] - 1) * self.dilations[1] + 1)

    @property
    def backward_reach(self):
        """How far the reader walks back when it wraps to the next output row.

        With top padding two output rows can start on the same input row, so
        the reader returns to a column it has already passed. The buffer has to
        still hold it."""
        out_w = self.ofm_dim[1]
        return max(0, (out_w - 1) * self.strides[1] - self.pads[1])

    @property
    def buffer_pixels(self):
        """Depth of the pixel buffer the chosen strategy needs, in pixels."""
        if self.strategy == "frame":
            return self.ifm_dim[0] * self.ifm_dim[1]
        required = self.span_pixels + self.backward_reach + 1
        depth = 1
        while depth < required:
            depth <<= 1
        return depth

    @property
    def capture_cycles(self):
        return self.ifm_dim[0] * self.ifm_dim[1] * self.ifm_ch // self.sub_elems

    @property
    def replay_cycles(self):
        out_h, out_w = self.ofm_dim
        return out_h * out_w * self.window_elems // self.simd

    @property
    def simd(self):
        """Elements per output beat, counted along the flattened window."""
        return self.folding["simd"]

    @property
    def sub_elems(self):
        """Elements per input beat. Capped at one pixel: beyond that the unit
        widens by reading more columns, not by taking wider input."""
        return min(self.simd, self.ifm_ch)

    @property
    def parallel_columns(self):
        """Adjacent kernel columns emitted per beat. One means the classic
        channel folded mode."""
        return self.simd // self.sub_elems

    def default_folding(self):
        return {"simd": 1}

    def folding_options(self):
        """Channel folding is available to both strategies. Column parallelism
        needs the replicated read ports only the line buffer has, and is what
        lets a layer with one input channel fold at all."""
        options = [{"simd": s} for s in divisors(self.ifm_ch)]
        if self.strategy == "line":
            options += [{"simd": self.ifm_ch * m}
                        for m in divisors(self.kernel[1]) if m > 1]
        return sorted(options, key=lambda c: c["simd"])

    @property
    def cycles(self):
        """The line strategy overlaps capture and replay, so a frame costs
        whichever side moves more beats. The frame strategy serialises them."""
        if self.strategy == "frame":
            return self.capture_cycles + self.replay_cycles
        return max(self.capture_cycles, self.replay_cycles)

    def resources(self, device):
        """Column parallelism buys read ports by replicating the buffer, so
        memory scales with it while the control logic does not."""
        depth = self.buffer_pixels * self.ifm_ch // self.sub_elems
        width = self.sub_elems * self.dtype.bits
        style = MemoryStyle.select(depth, width)
        memory = style.cost(depth, width) * self.parallel_columns
        counters = Resources(lut=260 + 40 * self.parallel_columns, ff=200)
        return memory + counters

    def input_beat_elems(self, graph):
        return [self.sub_elems]

    def output_beat_elems(self, graph):
        return [self.simd]

    def infer(self, graph):
        out_h, out_w = self.ofm_dim
        out = graph.ensure_tensor(self.outputs[0])
        out.shape = (graph.tensor(self.inputs[0]).shape[0], out_h, out_w,
                     self.window_elems)
        out.dtype = self.dtype
        out.elems_per_beat = self.simd

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.int64)
        return [self.window().apply(x, pad_value=self.pad_value)]

    def sv_params(self, graph):
        out_h, out_w = self.ofm_dim
        params = {
            "IFM_H": self.ifm_dim[0], "IFM_W": self.ifm_dim[1], "IFM_CH": self.ifm_ch,
            "OFM_H": out_h, "OFM_W": out_w,
            "KH": self.kernel[0], "KW": self.kernel[1],
            "STRIDE_H": self.strides[0], "STRIDE_W": self.strides[1],
            "PAD_T": self.pads[0], "PAD_L": self.pads[1],
            "DIL_H": self.dilations[0], "DIL_W": self.dilations[1],
            "SIMD": self.sub_elems, "DATA_BITS": self.dtype.bits,
            "PAD_VALUE": self.pad_value,
        }
        if self.strategy == "line":
            params["PAR"] = self.parallel_columns
        return params


class ActivationUnit(StreamingNode):
    """Standalone rescale and clamp, used where no matvec absorbed it."""

    module = "otf_act"

    def __init__(self, name, inputs, outputs, channels, elements, requant,
                 in_dtype, out_dtype, in_zero_point=0, **attrs):
        self.channels = int(channels)
        self.elements = int(elements)
        self.requant = requant
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.in_zero_point = int(in_zero_point)
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)

    @property
    def pe(self):
        return self.folding["pe"]

    def default_folding(self):
        return {"pe": 1}

    def folding_options(self):
        return [{"pe": p} for p in divisors(self.channels)]

    @property
    def cycles(self):
        return self.elements // self.pe

    def resources(self, device):
        dsp = self.pe if self.in_dtype.bits > 18 else 0
        return Resources(lut=self.pe * self.in_dtype.bits * 2.0,
                         ff=self.pe * self.out_dtype.bits, dsp=dsp)

    def input_beat_elems(self, graph):
        return [self.pe]

    def output_beat_elems(self, graph):
        return [self.pe]

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = source.shape, self.out_dtype
        out.elems_per_beat = self.pe

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.int64)
        flat = x.reshape(-1, self.channels) - self.in_zero_point
        return [self.requant.apply(flat).reshape(x.shape)]

    @property
    def mult_bits(self):
        """Narrowed to the values it has to carry, unless the representation
        was pinned, in which case it is the width that was pinned: a fitted
        width follows the weights and takes the netlist with it."""
        if self.requant.width is not None:
            return int(self.requant.width)
        return int(max(2, int(np.abs(self.requant.multipliers).max()).bit_length() + 1))

    def memories(self, graph):
        table = self.requant.broadcast_to(self.channels).multipliers.reshape(
            self.channels // self.pe, self.pe).T
        return {"mult": MemorySpec(table, self.mult_bits)}

    def sv_params(self, graph):
        return {
            "CHANNELS": self.channels, "PE": self.pe,
            "IN_BITS": self.in_dtype.bits, "OUT_BITS": self.out_dtype.bits,
            "MULT_BITS": self.mult_bits,
            "SHIFT": self.requant.shift,
            "OUT_MIN": self.requant.lower_clamp, "OUT_MAX": self.requant.upper_clamp,
            "IN_ZP": self.in_zero_point, "OUT_ZP": self.requant.zero_point,
            "MULT_FILE": self.memory_paths().get("mult", ""),
        }


class PoolUnit(StreamingNode):
    """Max reduction over the windows produced by a SlidingWindowUnit."""

    module = "otf_pool"

    def __init__(self, name, inputs, outputs, channels, window_size, windows,
                 dtype, **attrs):
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.windows = int(windows)
        self.dtype = dtype
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)

    @property
    def pe(self):
        return self.folding["pe"]

    def default_folding(self):
        return {"pe": 1}

    def folding_options(self):
        return [{"pe": p} for p in divisors(self.channels)]

    @property
    def cycles(self):
        return self.windows * self.window_size * self.channels // self.pe

    def resources(self, device):
        return Resources(lut=self.pe * self.dtype.bits * 3.0,
                         ff=self.pe * self.dtype.bits * 2)

    def input_beat_elems(self, graph):
        return [self.pe]

    def output_beat_elems(self, graph):
        return [self.pe]

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape = tuple(source.shape[:-1]) + (self.channels,)
        out.dtype = self.dtype
        out.elems_per_beat = self.pe

    def output_schedule(self, graph, index=0):
        folds = self.channels // self.pe
        return [w * self.window_size * folds + (self.window_size - 1) * folds + nf
                for w in range(self.windows)
                for nf in range(folds)]

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.int64)
        windows = x.reshape(-1, self.window_size, self.channels)
        reduced = windows.max(axis=1)
        return [reduced.reshape(graph.tensor(self.outputs[0]).shape)]

    def sv_params(self, graph):
        return {"CHANNELS": self.channels, "PE": self.pe,
                "WINDOW": self.window_size,
                "DATA_BITS": self.dtype.bits, "DATA_MIN": self.dtype.min}


class AddUnit(StreamingNode):
    """Two-input elementwise add. Each side is rescaled to the output grid."""

    module = "otf_add"

    def __init__(self, name, inputs, outputs, elements, channels, left_mult,
                 right_mult, shift, in_dtype, out_dtype, lower_clamp=None,
                 bias=0, zero_point=0, **attrs):
        self.elements = int(elements)
        self.channels = int(channels)
        self.left_mult = int(left_mult)
        self.right_mult = int(right_mult)
        self.shift = int(shift)
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.bias = int(bias)
        self.zero_point = int(zero_point)
        self.lower_clamp = out_dtype.min if lower_clamp is None else lower_clamp
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)

    @property
    def pe(self):
        return self.folding["pe"]

    def default_folding(self):
        return {"pe": 1}

    def folding_options(self):
        return [{"pe": p} for p in divisors(self.channels)]

    @property
    def cycles(self):
        return self.elements // self.pe

    def resources(self, device):
        return Resources(lut=self.pe * self.out_dtype.bits * 4.0,
                         ff=self.pe * self.out_dtype.bits)

    def sv_input_prefixes(self):
        return ["a", "b"]

    def input_beat_elems(self, graph):
        return [self.pe, self.pe]

    def output_beat_elems(self, graph):
        return [self.pe]

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = source.shape, self.out_dtype
        out.elems_per_beat = self.pe

    def execute(self, graph, values):
        left = np.asarray(values[self.inputs[0]], dtype=np.int64)
        right = np.asarray(values[self.inputs[1]], dtype=np.int64)
        acc = left * self.left_mult + right * self.right_mult + self.bias
        if self.shift > 0:
            acc = (acc + (1 << (self.shift - 1))) >> self.shift
        return [np.clip(acc + self.zero_point, self.lower_clamp, self.out_dtype.max)]

    def sv_params(self, graph):
        span = self.in_dtype.bits + max(
            abs(self.left_mult), abs(self.right_mult)).bit_length() + 2
        return {"PE": self.pe,
                "IN_BITS": self.in_dtype.bits, "OUT_BITS": self.out_dtype.bits,
                "ACC_BITS": span,
                "LEFT_MULT": self.left_mult, "RIGHT_MULT": self.right_mult,
                "SHIFT": self.shift, "OUT_MIN": self.lower_clamp,
                "OUT_MAX": self.out_dtype.max,
                "BIAS": self.bias, "OUT_ZP": self.zero_point}


class DuplicateUnit(StreamingNode):
    """Fork for skip connections.

    A stream cannot simply be wired to two consumers: both would drive its
    ready line and both would see the same beat. This unit holds each beat
    until every branch has taken it, which also means the branches stall each
    other, and that is what the buffer sizing has to account for.

    Its width follows the stream rather than a folding parameter, because a
    fork is not a place where parallelism is chosen.
    """

    module = "otf_dup"

    def __init__(self, name, inputs, outputs, elements, dtype, **attrs):
        self.elements = int(elements)
        self.dtype = dtype
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)

    def beat_elems(self, graph):
        return graph.tensor(self.inputs[0]).elems_per_beat

    @property
    def cycles(self):
        return self.elements

    def resources(self, device):
        width = self.elements and self.dtype.bits
        return Resources(lut=width * 0.5 * len(self.outputs),
                         ff=width + len(self.outputs))

    def input_beat_elems(self, graph):
        return [self.beat_elems(graph)]

    def output_beat_elems(self, graph):
        return [self.beat_elems(graph)] * len(self.outputs)

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        for name in self.outputs:
            out = graph.ensure_tensor(name)
            out.shape, out.dtype = source.shape, source.dtype
            out.elems_per_beat = source.elems_per_beat

    def execute(self, graph, values):
        x = values[self.inputs[0]]
        return [x for _ in self.outputs]

    def sv_connections(self, graph, nets):
        source = self.stream_inputs(graph)[0]
        branches = list(reversed(self.outputs))
        return {
            "clk": "clk", "rst_n": "rst_n",
            "s_tdata": nets.data(source),
            "s_tvalid": nets.valid(source),
            "s_tready": nets.ready(source),
            "m_tdata": "{%s}" % ", ".join(nets.data(b) for b in branches),
            "m_tvalid": "{%s}" % ", ".join(nets.valid(b) for b in branches),
            "m_tready": "{%s}" % ", ".join(nets.ready(b) for b in branches),
        }

    def sv_params(self, graph):
        return {"WIDTH": self.beat_elems(graph) * self.dtype.bits,
                "FANOUT": len(self.outputs)}


class StreamFifo(StreamingNode):
    module = "otf_fifo"

    def __init__(self, name, inputs, outputs, depth, width, **attrs):
        self.depth = int(depth)
        self.width = int(width)
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)

    @property
    def cycles(self):
        return 0

    def resources(self, device):
        style = MemoryStyle.select(self.depth, self.width)
        return style.cost(self.depth, self.width) + Resources(ff=32)

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = source.shape, source.dtype
        out.elems_per_beat = source.elems_per_beat

    def execute(self, graph, values):
        return [values[self.inputs[0]]]

    def input_beat_elems(self, graph):
        return [graph.tensor(self.inputs[0]).elems_per_beat]

    def output_beat_elems(self, graph):
        return [graph.tensor(self.inputs[0]).elems_per_beat]

    def sv_params(self, graph):
        return {"WIDTH": self.width, "DEPTH": self.depth}


class WidthConverter(StreamingNode):
    """Repacks beats when neighbouring folding factors disagree."""

    module = "otf_dwc"

    def __init__(self, name, inputs, outputs, in_elems, out_elems, elem_bits,
                 total_elems, **attrs):
        self.in_elems = int(in_elems)
        self.out_elems = int(out_elems)
        self.elem_bits = int(elem_bits)
        self.total_elems = int(total_elems)
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)
        wide, narrow = max(self.in_elems, self.out_elems), min(self.in_elems, self.out_elems)
        if wide % narrow:
            raise NotImplementedError(
                "width conversion %d -> %d is not an integer ratio"
                % (self.in_elems, self.out_elems))

    @property
    def cycles(self):
        return self.total_elems // min(self.in_elems, self.out_elems)

    @property
    def ratio(self):
        wide, narrow = max(self.in_elems, self.out_elems), min(self.in_elems, self.out_elems)
        return wide // narrow

    def input_schedule(self, graph, index=0):
        beats = self.input_beats(graph, index)
        step = self.ratio if self.in_elems > self.out_elems else 1
        return [i * step for i in range(beats)]

    def output_schedule(self, graph, index=0):
        beats = self.output_beats(graph, index)
        if self.out_elems > self.in_elems:
            return [k * self.ratio + self.ratio - 1 for k in range(beats)]
        return list(range(beats))

    def resources(self, device):
        return Resources(lut=max(self.in_elems, self.out_elems) * self.elem_bits * 0.4,
                         ff=max(self.in_elems, self.out_elems) * self.elem_bits)

    def input_beat_elems(self, graph):
        return [self.in_elems]

    def output_beat_elems(self, graph):
        return [self.out_elems]

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = source.shape, source.dtype
        out.elems_per_beat = self.out_elems

    def execute(self, graph, values):
        return [values[self.inputs[0]]]

    def sv_params(self, graph):
        return {"IN_ELEMS": self.in_elems, "OUT_ELEMS": self.out_elems,
                "ELEM_BITS": self.elem_bits, "TOTAL_ELEMS": self.total_elems}
