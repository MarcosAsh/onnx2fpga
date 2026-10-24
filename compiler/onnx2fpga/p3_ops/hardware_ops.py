"""Streaming hardware nodes, each one SystemVerilog module."""

from abc import abstractmethod

import numpy as np

from ..targets.resources import Resources, memory_cost
from ..p2_graph.datatype import IntType
from ..p2_graph.node import MemorySpec, Node
from .onnx_ops import Im2Col


def divisors(value, limit=None):
    limit = value if limit is None else min(limit, value)
    return [d for d in range(1, limit + 1) if value % d == 0]


def csd(value):
    """Canonical signed digits of value as (digit, position), most significant first."""
    digits, position = [], 0
    while value:
        if value & 1:
            digit = 2 - (value & 3)
            digits.append((digit, position))
            value -= digit
        value >>= 1
        position += 1
    return digits[::-1]


def odd_part(value):
    """(m, s) with abs(value) = m << s and m odd."""
    value, shift = abs(int(value)), 0
    while value % 2 == 0:
        value, shift = value // 2, shift + 1
    return value, shift


def multiplier_bits(requant):
    return int(max(2, int(np.abs(requant.multipliers).max()).bit_length() + 1))


class StreamingNode(Node):
    """A node that maps to one streaming module: folding, cycle cost, ports and beat schedules."""

    module = None
    input_ports = ("s",)

    def __init__(self, name, inputs, outputs, **attrs):
        Node.__init__(self, name, inputs, outputs, **attrs)
        self.folding = dict(self.default_folding())

    def default_folding(self):
        return {}

    def folding_options(self):
        """Legal folding assignments, cheapest first."""
        return [{}]

    def apply_folding(self, choice):
        self.folding.update(choice)

    @property
    def pinned_folding(self):
        """Factors a user fixed by hand; the scheduler chooses the rest."""
        return dict(self.attrs.get("pinned_folding", {}))

    def allowed_foldings(self):
        pinned = self.pinned_folding
        return [option for option in self.folding_options()
                if all(option.get(key) == value for key, value in pinned.items())]

    def starting_folding(self):
        if not self.pinned_folding:
            return self.default_folding()
        return dict(self.allowed_foldings()[0])

    @property
    @abstractmethod
    def cycles(self):
        """Cycles to process one frame once the pipeline is primed."""

    def input_beats(self, graph, index=0):
        name = self.stream_inputs(graph)[index]
        return graph.tensor(name).numel // self.input_beat_elems(graph)[index]

    def output_beats(self, graph, index=0):
        return graph.tensor(self.outputs[index]).numel // self.output_beat_elems(graph)[index]

    def _uniform(self, beats):
        """Evenly spread, never faster than one beat per cycle."""
        span = max(self.cycles, beats)
        return [i * span // beats for i in range(beats)]

    def input_schedule(self, graph, index=0):
        return self._uniform(self.input_beats(graph, index))

    def output_schedule(self, graph, index=0):
        return self._uniform(self.output_beats(graph, index))

    def input_beat_elems(self, graph):
        return [1] * len(self.stream_inputs(graph))

    def output_beat_elems(self, graph):
        return [1] * len(self.outputs)

    def resources(self, device):
        """Priced by the device's own cost model if it has one, else by the AMD model."""
        if device.cost_model is not None:
            return device.cost_model.estimate(self)
        return self.amd_resources(device)

    def amd_resources(self, device):
        return Resources()

    def memories(self, graph):
        """MemorySpec entries by name, written as $readmemh images."""
        return {}

    def memory_paths(self):
        return self.attrs.get("mem_paths", {})

    def rtl_files(self):
        """SystemVerilog sources this unit brings beyond the library, by file name."""
        return {}

    def sv_connections(self, graph, net):
        wiring = {"clk": "clk", "rst_n": "rst_n"}
        ports = list(zip(self.input_ports, self.stream_inputs(graph)))
        for prefix, name in ports + [("m", self.outputs[0])]:
            for signal in ("tdata", "tvalid", "tready"):
                wiring["%s_%s" % (prefix, signal)] = net(name, signal)
        return wiring


class MatVecUnit(StreamingNode):
    """Folded matrix-vector engine with fused bias and rescale; (MW/SIMD) * (MH/PE) cycles a vector."""

    module = "otf_mvau"

    def __init__(self, name, inputs, outputs, weights, bias, requant,
                 act_dtype, out_dtype, weight_dtype, vectors=1):
        self.weights = np.asarray(weights, dtype=np.int64)
        self.bias = np.asarray(bias, dtype=np.int64)
        self.requant = requant
        self.act_dtype = act_dtype
        self.out_dtype = out_dtype
        self.weight_dtype = weight_dtype
        self.vectors = int(vectors)
        StreamingNode.__init__(self, name, inputs, outputs)

    @property
    def mw(self):
        return int(self.weights.shape[0])

    @property
    def mh(self):
        return int(self.weights.shape[1])

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
        options = [{"simd": simd, "pe": pe}
                   for simd in divisors(self.mw) for pe in divisors(self.mh)]
        return sorted(options, key=lambda c: (c["simd"] * c["pe"], c["pe"]))

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

    @property
    def mult_bits(self):
        return multiplier_bits(self.requant)

    def amd_resources(self, device):
        acc_bits = self.accumulator_dtype.bits
        dsp = device.dsp_cost(self.pe * self.simd, self.act_dtype.bits, self.weight_dtype.bits)
        dsp += self.pe if acc_bits > 18 else 0
        compute = Resources(lut=self.pe * max(0, self.simd - 1) * acc_bits * 0.7,
                            ff=self.pe * acc_bits + self.simd * self.act_dtype.bits,
                            dsp=dsp)
        depth = self.synapse_fold * self.neuron_fold
        width = self.simd * self.weight_dtype.bits
        cost = compute + memory_cost(depth, width) * self.pe
        if self.neuron_fold > 1:
            width = self.simd * self.act_dtype.bits
            cost = cost + memory_cost(self.synapse_fold, width)
        return cost

    def input_beat_elems(self, graph):
        return [self.simd]

    def output_beat_elems(self, graph):
        return [self.pe]

    def infer(self, graph):
        """Keeps the spatial shape only when the innermost axis is already the multiplied vector."""
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
        out = self.requant.apply(x @ self.weights + self.bias)
        return [out.reshape(graph.tensor(self.outputs[0]).shape)]

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
        """One weight memory per PE, addressed by (neuron fold, synapse fold)."""
        pe, simd = self.pe, self.simd
        packed = np.zeros((pe, self.neuron_fold * self.synapse_fold, simd), dtype=np.int64)
        for p in range(pe):
            for nf in range(self.neuron_fold):
                column = self.weights[:, nf * pe + p].reshape(self.synapse_fold, simd)
                packed[p, nf * self.synapse_fold:(nf + 1) * self.synapse_fold] = column
        return {
            "weights": MemorySpec(packed, self.weight_dtype.bits, packed=True),
            "bias": MemorySpec(self.bias.reshape(self.neuron_fold, pe).T,
                               self.accumulator_dtype.bits),
            "mult": MemorySpec(self.requant.broadcast_to(self.mh).multipliers.reshape(
                self.neuron_fold, pe).T, self.mult_bits),
        }

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


class DenseUnit(MatVecUnit):
    """A matvec unrolled to one vector per cycle with no handshake; Hardcaml only."""

    module = "otf_dense"
    hardcaml_only = True
    latency = 3
    arithmetic = "multiplier"

    def default_folding(self):
        return {"simd": self.mw, "pe": self.mh}

    def folding_options(self):
        return [self.default_folding()]

    @property
    def multipliers(self):
        return self.mw * self.mh if self.arithmetic == "multiplier" else 0

    def constant_cost(self):
        """LUT and flip flops of the shift and add products and the adder trees, one LUT a bit."""
        act_bits, adders, ff = self.act_dtype.bits, 0, 0
        for row in self.weights:
            pairs = set()
            for m in {odd_part(w)[0] for w in row if w}:
                digits = csd(m)
                pairs.update((1 << (high - low)) + da * db for (da, high), (db, low)
                             in zip(digits[::2], digits[1::2]))
                adders += (len(digits) + 1) // 2 - 1
                ff += act_bits + (m - 1).bit_length()
            adders += len(pairs)
        tree = int(np.count_nonzero(self.weights)) * self.accumulator_dtype.bits
        return Resources(lut=tree + adders * (act_bits + self.weight_dtype.bits), ff=ff)

    def amd_resources(self, device):
        acc_bits, act_bits = self.accumulator_dtype.bits, self.act_dtype.bits
        dsp = 0
        if self.requant.shift < acc_bits + self.mult_bits:
            dsp += self.mh * (2 if acc_bits > 27 else 1)
        out = Resources(ff=self.mh * (acc_bits + self.out_dtype.bits), dsp=dsp)
        if self.arithmetic == "constant":
            return self.constant_cost() + out
        product_bits = act_bits + self.weight_dtype.bits
        return out + Resources(
            lut=self.multipliers * acc_bits, ff=self.multipliers * product_bits,
            dsp=device.dsp_cost(self.multipliers, act_bits, self.weight_dtype.bits))

    def memories(self, graph):
        return {
            "weights": MemorySpec(self.weights.T, self.weight_dtype.bits, packed=True),
            "bias": MemorySpec(self.bias, self.accumulator_dtype.bits),
            "mult": MemorySpec(self.requant.broadcast_to(self.mh).multipliers,
                               self.mult_bits),
        }

    def sv_params(self, graph):
        return {
            "MW": self.mw, "MH": self.mh,
            "ACT_BITS": self.act_dtype.bits, "WGT_BITS": self.weight_dtype.bits,
            "ACC_BITS": self.accumulator_dtype.bits, "OUT_BITS": self.out_dtype.bits,
            "MULT_BITS": self.mult_bits, "SHIFT": self.requant.shift,
            "OUT_MIN": self.requant.lower_clamp, "OUT_MAX": self.requant.upper_clamp,
            "OUT_ZP": self.requant.zero_point,
            "CONSTANT": int(self.arithmetic == "constant"),
        }


class SlidingWindowUnit(StreamingNode):
    """Flattened windows of a channel-last image; `line` buffers a span, `frame` the whole map."""

    MODULES = {"line": "otf_swu_line", "frame": "otf_swu"}

    def __init__(self, name, inputs, outputs, ifm_dim, ifm_ch, kernel, strides,
                 pads, dilations, dtype, strategy="line", pad_value=0):
        if strategy not in self.MODULES:
            raise NotImplementedError("sliding window strategy %r" % strategy)
        self.ifm_dim = tuple(ifm_dim)
        self.ifm_ch = int(ifm_ch)
        self.kernel = tuple(kernel)
        self.strides = tuple(strides)
        self.pads = tuple(pads)
        self.dilations = tuple(dilations)
        self.dtype = dtype
        self.strategy = strategy
        self.pad_value = int(pad_value)
        StreamingNode.__init__(self, name, inputs, outputs)

    @property
    def module(self):
        return self.MODULES[self.strategy]

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
        """How far the reader walks back when it wraps to the next output row."""
        return max(0, (self.ofm_dim[1] - 1) * self.strides[1] - self.pads[1])

    @property
    def buffer_pixels(self):
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
        """Elements per output beat, along the flattened window."""
        return self.folding["simd"]

    @property
    def sub_elems(self):
        """Elements per input beat, at most one pixel; wider beats read more columns."""
        return min(self.simd, self.ifm_ch)

    @property
    def parallel_columns(self):
        """Adjacent kernel columns emitted per beat."""
        return self.simd // self.sub_elems

    def default_folding(self):
        return {"simd": 1}

    def folding_options(self):
        """Column parallelism needs the replicated read ports only the line buffer has."""
        options = [{"simd": s} for s in divisors(self.ifm_ch)]
        if self.strategy == "line":
            options += [{"simd": self.ifm_ch * m}
                        for m in divisors(self.kernel[1]) if m > 1]
        return sorted(options, key=lambda c: c["simd"])

    @property
    def cycles(self):
        """The line buffer overlaps capture and replay; the frame buffer does not."""
        if self.strategy == "frame":
            return self.capture_cycles + self.replay_cycles
        return max(self.capture_cycles, self.replay_cycles)

    def amd_resources(self, device):
        depth = self.buffer_pixels * self.ifm_ch // self.sub_elems
        width = self.sub_elems * self.dtype.bits
        memory = memory_cost(depth, width) * self.parallel_columns
        return memory + Resources(lut=260 + 40 * self.parallel_columns, ff=200)

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


class ChannelUnit(StreamingNode):
    """Folded by `pe` lanes across the channels, `pe` elements per beat on every port."""

    def __init__(self, name, inputs, outputs, channels):
        self.channels = int(channels)
        StreamingNode.__init__(self, name, inputs, outputs)

    @property
    def pe(self):
        return self.folding["pe"]

    def default_folding(self):
        return {"pe": 1}

    def folding_options(self):
        return [{"pe": p} for p in divisors(self.channels)]

    def input_beat_elems(self, graph):
        return [self.pe] * len(self.input_ports)

    def output_beat_elems(self, graph):
        return [self.pe]


class ActivationUnit(ChannelUnit):
    """Standalone rescale and clamp, where no matvec absorbed it."""

    module = "otf_act"

    def __init__(self, name, inputs, outputs, channels, elements, requant,
                 in_dtype, out_dtype, in_zero_point=0):
        self.elements = int(elements)
        self.requant = requant
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.in_zero_point = int(in_zero_point)
        ChannelUnit.__init__(self, name, inputs, outputs, channels)

    @property
    def cycles(self):
        return self.elements // self.pe

    @property
    def mult_bits(self):
        return multiplier_bits(self.requant)

    def amd_resources(self, device):
        dsp = self.pe if self.in_dtype.bits > 18 else 0
        return Resources(lut=self.pe * self.in_dtype.bits * 2.0,
                         ff=self.pe * self.out_dtype.bits, dsp=dsp)

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = source.shape, self.out_dtype
        out.elems_per_beat = self.pe

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.int64)
        flat = x.reshape(-1, self.channels) - self.in_zero_point
        return [self.requant.apply(flat).reshape(x.shape)]

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


class PoolUnit(ChannelUnit):
    """Max over the windows a SlidingWindowUnit produces."""

    module = "otf_pool"

    def __init__(self, name, inputs, outputs, channels, window_size, windows, dtype):
        self.window_size = int(window_size)
        self.windows = int(windows)
        self.dtype = dtype
        ChannelUnit.__init__(self, name, inputs, outputs, channels)

    @property
    def cycles(self):
        return self.windows * self.window_size * self.channels // self.pe

    def amd_resources(self, device):
        return Resources(lut=self.pe * self.dtype.bits * 3.0,
                         ff=self.pe * self.dtype.bits * 2)

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
        reduced = x.reshape(-1, self.window_size, self.channels).max(axis=1)
        return [reduced.reshape(graph.tensor(self.outputs[0]).shape)]

    def sv_params(self, graph):
        return {"CHANNELS": self.channels, "PE": self.pe,
                "WINDOW": self.window_size,
                "DATA_BITS": self.dtype.bits, "DATA_MIN": self.dtype.min}


class AddUnit(ChannelUnit):
    """Elementwise add of two streams, each rescaled to the output grid."""

    module = "otf_add"
    input_ports = ("a", "b")

    def __init__(self, name, inputs, outputs, elements, channels, left_mult,
                 right_mult, shift, in_dtype, out_dtype, bias=0, zero_point=0):
        self.elements = int(elements)
        self.left_mult = int(left_mult)
        self.right_mult = int(right_mult)
        self.shift = int(shift)
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.bias = int(bias)
        self.zero_point = int(zero_point)
        ChannelUnit.__init__(self, name, inputs, outputs, channels)

    @property
    def cycles(self):
        return self.elements // self.pe

    def amd_resources(self, device):
        return Resources(lut=self.pe * self.out_dtype.bits * 4.0,
                         ff=self.pe * self.out_dtype.bits)

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
        return [np.clip(acc + self.zero_point, self.out_dtype.min, self.out_dtype.max)]

    def sv_params(self, graph):
        span = self.in_dtype.bits + max(
            abs(self.left_mult), abs(self.right_mult)).bit_length() + 2
        return {"PE": self.pe,
                "IN_BITS": self.in_dtype.bits, "OUT_BITS": self.out_dtype.bits,
                "ACC_BITS": span,
                "LEFT_MULT": self.left_mult, "RIGHT_MULT": self.right_mult,
                "SHIFT": self.shift, "OUT_MIN": self.out_dtype.min,
                "OUT_MAX": self.out_dtype.max,
                "BIAS": self.bias, "OUT_ZP": self.zero_point}


class DuplicateUnit(StreamingNode):
    """Fork for skip connections; a beat leaves only once every branch has taken it."""

    module = "otf_dup"

    def __init__(self, name, inputs, outputs, elements, dtype):
        self.elements = int(elements)
        self.dtype = dtype
        StreamingNode.__init__(self, name, inputs, outputs)

    def beat_elems(self, graph):
        return graph.tensor(self.inputs[0]).elems_per_beat

    @property
    def cycles(self):
        return self.elements

    def amd_resources(self, device):
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
        return [values[self.inputs[0]] for _ in self.outputs]

    def sv_connections(self, graph, net):
        source = self.stream_inputs(graph)[0]
        branches = list(reversed(self.outputs))
        wiring = {"clk": "clk", "rst_n": "rst_n"}
        for signal in ("tdata", "tvalid", "tready"):
            wiring["s_" + signal] = net(source, signal)
        for signal in ("tdata", "tvalid", "tready"):
            wiring["m_" + signal] = "{%s}" % ", ".join(net(b, signal) for b in branches)
        return wiring

    def sv_params(self, graph):
        return {"WIDTH": self.beat_elems(graph) * self.dtype.bits,
                "FANOUT": len(self.outputs)}


class StreamFifo(StreamingNode):
    module = "otf_fifo"

    def __init__(self, name, inputs, outputs, depth, width):
        self.depth = int(depth)
        self.width = int(width)
        StreamingNode.__init__(self, name, inputs, outputs)

    @property
    def cycles(self):
        return 0

    def amd_resources(self, device):
        return memory_cost(self.depth, self.width) + Resources(ff=32)

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
    """Repacks beats where neighbouring folding factors disagree, always at an integer ratio."""

    module = "otf_dwc"

    def __init__(self, name, inputs, outputs, in_elems, out_elems, elem_bits, total_elems):
        self.in_elems = int(in_elems)
        self.out_elems = int(out_elems)
        self.elem_bits = int(elem_bits)
        self.total_elems = int(total_elems)
        StreamingNode.__init__(self, name, inputs, outputs)

    @property
    def cycles(self):
        return self.total_elems // min(self.in_elems, self.out_elems)

    @property
    def ratio(self):
        return max(self.in_elems, self.out_elems) // min(self.in_elems, self.out_elems)

    def input_schedule(self, graph, index=0):
        step = self.ratio if self.in_elems > self.out_elems else 1
        return [i * step for i in range(self.input_beats(graph, index))]

    def output_schedule(self, graph, index=0):
        beats = self.output_beats(graph, index)
        if self.out_elems > self.in_elems:
            return [k * self.ratio + self.ratio - 1 for k in range(beats)]
        return list(range(beats))

    def amd_resources(self, device):
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
