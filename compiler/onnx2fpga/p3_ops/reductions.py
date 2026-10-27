"""Operators across the feature axis: ArgMax, the Reduce family, Softmax and the L1 Normalizer."""

import math

import numpy as np

from ..errors import UnsupportedOpError
from ..p2_graph.datatype import FLOAT32, IntType
from ..p2_graph.node import MemorySpec
from ..targets.resources import Resources
from .hardware_ops import StreamingNode
from .onnx_ops import FrontendNode

ARG = ("ArgMax", "ArgMin")
REDUCE = ARG + ("ReduceMax", "ReduceMin", "ReduceSum", "ReduceMean")
SOFTMAX = ("Softmax", "LogSoftmax")


def signed_bits(low, high):
    return IntType.smallest_for(min(int(low), -1), int(high)).bits


def ceil_log2(value):
    return (int(value) - 1).bit_length()


class ReduceOp(FrontendNode):
    """One of REDUCE over the last axis of a (1, N) vector; `keepdims` keeps the axis, `last` picks the last of a tie."""

    def __init__(self, name, inputs, outputs, op_type, **attrs):
        FrontendNode.__init__(self, name, inputs, outputs, **attrs)
        self.op_type = op_type

    def infer(self, graph):
        out = self._out(graph)
        out.shape = (1, 1) if self.attrs.get("keepdims", True) else (1,)
        out.dtype = FLOAT32

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.float64).reshape(-1)
        op = self.op_type
        if op in ARG:
            pick = np.argmax if op == "ArgMax" else np.argmin
            value = len(x) - 1 - pick(x[::-1]) if self.attrs.get("last") else pick(x)
        else:
            value = {"ReduceMax": np.max, "ReduceMin": np.min, "ReduceSum": np.sum,
                     "ReduceMean": np.mean}[op](x)
        return [np.full(self._out(graph).shape, value, dtype=np.float64)]


class SoftmaxOp(FrontendNode):
    """Softmax, LogSoftmax or a Normalizer on the last axis; MAX divides by the largest value, as ONNX Runtime does."""

    def __init__(self, name, inputs, outputs, op_type, **attrs):
        FrontendNode.__init__(self, name, inputs, outputs, **attrs)
        self.op_type = op_type

    def infer(self, graph):
        out = self._out(graph)
        out.shape = graph.tensor(self.inputs[0]).shape
        out.dtype = FLOAT32

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.float64)
        if self.op_type == "Normalizer":
            norm = self.attrs.get("norm", "L1")
            if norm == "L1":
                total = np.sum(np.abs(x), axis=-1, keepdims=True)
            elif norm == "L2":
                total = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
            else:
                total = np.max(x, axis=-1, keepdims=True)
            return [x / np.where(total == 0, 1, total)]
        shifted = x - np.max(x, axis=-1, keepdims=True)
        total = np.sum(np.exp(shifted), axis=-1, keepdims=True)
        if self.op_type == "LogSoftmax":
            return [shifted - np.log(total)]
        return [np.exp(shifted) / total]


def feature_vector(node, importer, name):
    """The live input's name, refusing anything but a (1, N) vector."""
    shape = importer.onnx_shape(node.input[0])
    if len(shape) != 2 or shape[0] != 1:
        raise UnsupportedOpError(node.op_type, name, message="%s %r takes a %s tensor, and "
                                 "only a (1, N) vector is supported"
                                 % (node.op_type, name, shape))
    return importer.rename(node.input[0])


def refuse_axis(node, name, axes):
    raise UnsupportedOpError(node.op_type, name, message="%s %r works on axis %s, and only "
                             "the feature axis of a (1, N) vector is supported"
                             % (node.op_type, name, ", ".join(str(a) for a in axes)))


def import_reduce(node, importer):
    """Axes come from the attribute, or from a constant second input from opset 13 or 18 on."""
    op, attrs = node.op_type, node.attributes()
    name = node.name or importer.graph.fresh_name(op.lower())
    source = feature_vector(node, importer, name)
    if op in ARG:
        axes = [int(attrs.get("axis", 0))]
    elif len(node.input) > 1 and node.input[1]:
        axes = importer.constant_array(node.input[1]).astype(int).reshape(-1).tolist()
    else:
        axes = [int(a) for a in attrs.get("axes", [])]
    if not axes and attrs.get("noop_with_empty_axes"):
        raise UnsupportedOpError(op, name, message="%s %r has noop_with_empty_axes, which is "
                                 "not supported" % (op, name))
    if axes and 1 not in {a % 2 for a in axes}:
        refuse_axis(node, name, axes)
    producer = importer.graph.producer(source)
    # The ArgMax of a softmax is the ArgMax of its input, so the softmax goes.
    if (op in ARG and isinstance(producer, SoftmaxOp) and producer.op_type in SOFTMAX
            and importer.only_use(source)):
        importer.graph.remove_node(producer)
        source = producer.inputs[0]
    importer.emit(ReduceOp(name, [source], [importer.rename(node.output[0])], op,
                           keepdims=bool(attrs.get("keepdims", 1)),
                           last=bool(attrs.get("select_last_index", 0)), label=op in ARG))


def import_softmax(node, importer):
    """Any axis that names the last one of a (1, N) vector, whatever the opset's default."""
    name = node.name or importer.graph.fresh_name(node.op_type.lower())
    source = feature_vector(node, importer, name)
    axis = int(node.attributes().get("axis", -1))
    if axis % 2 != 1:
        refuse_axis(node, name, [axis])
    importer.emit(SoftmaxOp(name, [source], [importer.rename(node.output[0])], node.op_type))


def import_normalizer(node, importer):
    """L1 of a softmax is the softmax itself; any other Normalizer is a softmax unit."""
    name = node.name or importer.graph.fresh_name("normalizer")
    norm = node.attributes().get("norm", "MAX")
    norm = norm.decode() if isinstance(norm, bytes) else norm
    if norm not in ("L1", "L2", "MAX"):
        raise UnsupportedOpError(node.op_type, name, message="Normalizer %r has norm %s, and "
                                 "only MAX, L1 and L2 are supported" % (name, norm))
    source = feature_vector(node, importer, name)
    producer = importer.graph.producer(source)
    if norm == "L1" and isinstance(producer, SoftmaxOp) and producer.op_type == "Softmax":
        importer.alias(node.output[0], node.input[0])
        return
    importer.emit(SoftmaxOp(name, [source], [importer.rename(node.output[0])], "Normalizer",
                            norm=norm))


def compare_paths(delays, levels, bits):
    """A compare and select per level, with 0, 1 ... cuts spread over the levels."""
    step = delays.adder(bits) + delays.logic()
    return [delays.path(math.ceil(levels / (cuts + 1)) * step) for cuts in range(max(1, levels))]


def sum_paths(delays, levels, bits):
    return [delays.path(delays.adder_tree(math.ceil(levels / (cuts + 1)), bits))
            for cuts in range(max(1, levels))]


def rescale_paths(delays, acc, mult_bits, shift, fanout=1, multiply=True):
    """The cuts of Otf_requant after its multiply, rounding and zero point; Log multiplies by 1."""
    path, adder, logic = delays.path, delays.adder, delays.logic
    wide = acc + mult_bits + 1
    rounding = [adder(wide)] if 0 < shift < wide - 1 else []
    clamp = [logic()]
    if multiply:
        first = [delays.product(acc, mult_bits)]
        registered = delays.registered_product(acc, mult_bits, fanout)
    else:
        first, registered = [adder(acc)], path(adder(acc))
    return [path(*first, *rounding, adder(wide), *clamp, fanout=fanout),
            max(registered, path(*rounding, adder(wide), *clamp)),
            max(registered, path(*rounding), path(adder(wide), *clamp)),
            max(registered, path(*rounding), path(adder(wide)), path(*clamp))]


class ReduceUnit(StreamingNode):
    """A tree across the lanes: a compare and select for the arg and extreme ops, an adder tree and a rescale for the sum."""

    module = "otf_reduce"
    hardcaml_only = True
    OPS = ("argmax", "argmin", "max", "min", "sum")

    def __init__(self, name, inputs, outputs, lanes, op, dtype, shape, last=False,
                 zero_point=0, requant=None, mult_bits=18):
        self.lanes = int(lanes)
        self.op = op
        self.dtype = dtype
        self.shape = tuple(shape)
        self.last = bool(last)
        self.zero_point = int(zero_point)
        self.requant = requant
        self.mult_bits = int(mult_bits)
        self.stages = {"tree": 0} if op != "sum" else {"tree": 0, "rescale": 0}
        StreamingNode.__init__(self, name, inputs, outputs)

    @property
    def levels(self):
        return ceil_log2(self.lanes + (self.op == "sum"))

    @property
    def latency(self):
        return 1 + sum(self.stages.values()) + (self.op == "sum")

    @property
    def cycles(self):
        return 1

    @property
    def multipliers(self):
        return int(self.op == "sum")

    @property
    def out_dtype(self):
        if self.op in ("argmax", "argmin"):
            return IntType(max(1, ceil_log2(self.lanes)), False)
        return self.dtype

    @property
    def acc_dtype(self):
        span = self.lanes * (max(-self.dtype.min, self.dtype.max) + abs(self.zero_point))
        return IntType(max(self.dtype.bits + 1, signed_bits(-span, span)), True)

    def infer(self, graph):
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = self.shape, self.out_dtype
        out.elems_per_beat = 1

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.int64).reshape(-1)
        if self.op == "sum":
            value = self.requant.apply(np.sum(x) - self.lanes * self.zero_point)
        elif self.op in ("max", "min"):
            value = x.max() if self.op == "max" else x.min()
        else:
            pick = np.argmax if self.op == "argmax" else np.argmin
            value = len(x) - 1 - pick(x[::-1]) if self.last else pick(x)
        return [np.full(self.shape, value, dtype=np.int64)]

    def paths(self, delays):
        if self.op != "sum":
            return {"tree": compare_paths(delays, self.levels, self.dtype.bits)}
        acc = self.acc_dtype.bits
        return {"tree": sum_paths(delays, self.levels, acc),
                "rescale": rescale_paths(delays, acc, self.mult_bits, self.requant.shift)}

    def amd_resources(self, device):
        """A comparator and a mux of the code and index per node, or an adder per node and a DSP."""
        nodes = max(0, self.lanes - 1)
        if self.op != "sum":
            bits = self.dtype.bits + self.out_dtype.bits
            return Resources(lut=nodes * (self.dtype.bits + bits),
                             ff=self.stages["tree"] * self.lanes * bits + self.out_dtype.bits
                             + self.latency)
        acc = self.acc_dtype.bits
        return Resources(lut=(nodes + 1) * acc + 2 * acc,
                         ff=(self.stages["tree"] * self.lanes + 1) * acc
                         + self.stages["rescale"] * (acc + self.mult_bits) + self.dtype.bits
                         + self.latency, dsp=1)

    def sv_params(self, graph):
        requant = self.requant
        return {"LANES": self.lanes, "BITS": self.dtype.bits, "OP": self.OPS.index(self.op),
                "LAST": int(self.last), "ZP": self.zero_point,
                "MULT": int(requant.multipliers.flat[0]) if requant else 1,
                "MULT_BITS": self.mult_bits, "ACC_BITS": self.acc_dtype.bits,
                "SHIFT": requant.shift if requant else 0,
                "OUT_ZP": requant.zero_point if requant else 0,
                "OUT_MIN": requant.lower_clamp if requant else self.dtype.min,
                "OUT_MAX": requant.upper_clamp if requant else self.dtype.max,
                "TREE_STAGES": self.stages["tree"],
                "RESCALE_STAGES": self.stages.get("rescale", 0)}


class SoftmaxUnit(StreamingNode):
    """Lane tables, their sum normalised to a mantissa, then a reciprocal, its square root or a log; max reads one table at the largest code."""

    MODES = ("divide", "log", "root", "max")

    module = "otf_softmax"
    hardcaml_only = True

    def __init__(self, name, inputs, outputs, lanes, op, in_dtype, out_dtype, sum_table,
                 numer_table, min_exp, mantissa_bits, scale_table, exp_table, scale_bits,
                 shift, out_zero_point, subtract_max, mode=None):
        self.mode = mode or ("divide" if exp_table is None else "log")
        self.lanes = int(lanes)
        self.op = op
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.sum_table = np.asarray(sum_table, dtype=np.int64)
        self.numer_table = None if numer_table is None else np.asarray(numer_table,
                                                                          dtype=np.int64)
        self.min_exp = int(min_exp)
        self.mantissa_bits = int(mantissa_bits)
        self.scale_table = np.asarray(scale_table, dtype=np.int64)
        self.exp_table = None if exp_table is None else np.asarray(exp_table, dtype=np.int64)
        self.scale_bits = int(scale_bits)
        self.shift = int(shift)
        self.out_zero_point = int(out_zero_point)
        self.subtract_max = bool(subtract_max)
        if self.mode == "max":
            self.stages = {"max": 0, "table": 0, "rescale": 0}
        else:
            self.stages = {"max": 0, "table": 0, "sum": 0, "normalize": 0, "rescale": 0}
            if not self.subtract_max:
                del self.stages["max"]
        StreamingNode.__init__(self, name, inputs, outputs)

    @property
    def log(self):
        return self.mode == "log"

    @property
    def latency(self):
        if self.mode == "max":
            return 3 + sum(self.stages.values())
        return 4 + self.subtract_max + sum(self.stages.values())

    @property
    def cycles(self):
        return 1

    @property
    def multipliers(self):
        return 0 if self.log else self.lanes

    @property
    def levels(self):
        return ceil_log2(self.lanes)

    @property
    def sum_bits(self):
        return max(1, int(self.sum_table.max(initial=0)).bit_length())

    @property
    def numer_bits(self):
        if self.numer_table is None:
            return self.sum_bits + 1
        return signed_bits(self.numer_table.min(), self.numer_table.max())

    @property
    def total_bits(self):
        return max(1, int(self.lanes * self.sum_table.max(initial=0)).bit_length())

    @property
    def exps(self):
        return self.total_bits - self.min_exp

    @property
    def root_shifts(self):
        """Root shifts by half the leading one's position; its parity picks the table half."""
        return (self.exps - 1) // 2

    @property
    def acc_bits(self):
        if self.log:
            return max(self.numer_bits, self.scale_bits) + 1
        if self.mode == "max":
            return self.numer_bits
        if self.mode == "root":
            return self.numer_bits + self.root_shifts
        return self.numer_bits + self.exps - 1

    def normalise(self, total):
        """The sum's leading one at or above min_exp, and the mantissa_bits bits below it."""
        k = max(self.min_exp, int(total).bit_length() - 1)
        r = self.mantissa_bits
        mantissa = (total >> (k - r) if k >= r else total << (r - k)) & ((1 << r) - 1)
        return k, int(mantissa)

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.int64).reshape(-1)
        index = x.max() - x if self.subtract_max else x - self.in_dtype.min
        if self.mode == "max":
            acc, mult = self.numer_table[index], int(self.scale_table[x.max() - self.in_dtype.min])
        else:
            a = self.sum_table[index]
            b = a if self.numer_table is None else self.numer_table[index]
            total = int(a.sum())
            k, mantissa = self.normalise(total)
            if self.log:
                acc, mult = b - self.scale_table[mantissa] - self.exp_table[k - self.min_exp], 1
            elif self.mode == "root":
                half, parity = divmod(k - self.min_exp, 2)
                acc = b << (self.root_shifts - half)
                mult = (int(self.scale_table[(parity << self.mantissa_bits) | mantissa])
                        if total else 0)
            else:
                acc = b << (self.total_bits - 1 - k)
                mult = int(self.scale_table[mantissa]) if total else 0
        scaled = acc * mult
        if self.shift > 0:
            scaled = (scaled + (1 << (self.shift - 1))) >> self.shift
        out = np.clip(scaled + self.out_zero_point, self.out_dtype.min, self.out_dtype.max)
        return [out.reshape(np.shape(values[self.inputs[0]]))]

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = source.shape, self.out_dtype
        out.elems_per_beat = source.numel

    def paths(self, delays):
        """Each step's longest path with 0, 1 ... registers added."""
        path, logic, select = delays.path, delays.logic, delays.select
        lanes, bits = self.lanes, self.in_dtype.bits
        luts = max(1, (1 << bits) // 64)
        fanout = luts * (self.sum_bits + (0 if self.numer_table is None else self.numer_bits))
        table = delays.table(bits)
        found = {}
        if self.mode == "max":
            fanout = luts * self.scale_bits
            return {"max": compare_paths(delays, self.levels, bits),
                    "table": [path(table, fanout=fanout)] * 2,
                    "rescale": rescale_paths(delays, self.acc_bits, self.scale_bits, self.shift,
                                             fanout=lanes)}
        if self.subtract_max:
            found["max"] = compare_paths(delays, self.levels, bits)
            sub = delays.adder(bits + 1)
            found["table"] = [path(sub, table, fanout=fanout),
                              max(path(sub, fanout=lanes), path(table, fanout=fanout))]
        else:
            found["table"] = [path(table, fanout=fanout)] * 2
        found["sum"] = sum_paths(delays, self.levels, self.total_bits)
        detect = [logic(), select(ceil_log2(self.exps))]
        root = self.mode == "root"
        scale = [delays.table(self.mantissa_bits + root),
                 delays.adder(self.scale_bits) if self.log else logic()]
        shifts = self.root_shifts + 1 if root else self.exps
        shift = [] if self.log else [select(ceil_log2(shifts))]
        found["normalize"] = [
            max(path(*detect, *scale, fanout=self.exps), path(*detect, *shift)),
            max(path(*detect, fanout=self.exps), path(*scale), path(*shift, fanout=lanes))]
        if self.exps >= 3:
            # The second cut searches each half of the sum for its leading one apart.
            half = [logic(), select(ceil_log2(-(-self.exps // 2)))]
            found["normalize"].append(max(path(*half, fanout=self.exps), path(*scale),
                                          path(*shift, fanout=lanes)))
        found["rescale"] = rescale_paths(delays, self.acc_bits, 2 if self.log else
                                         self.scale_bits, self.shift, fanout=lanes,
                                         multiply=not self.log)
        return found

    def amd_resources(self, device):
        """The lane tables, the trees, the shift mux and the rescale per lane, and the shared scale table."""
        bits, lanes = self.in_dtype.bits, self.lanes
        luts = max(1, (1 << bits) // 64)
        table_bits = self.sum_bits + (0 if self.numer_table is None else self.numer_bits)
        scale = (len(self.scale_table) // 64 + 1) * self.scale_bits
        has_max = self.subtract_max or self.mode == "max"
        trees = (lanes - 1) * (2 * bits * has_max + (self.mode != "max") * self.total_bits)
        shifts = {"log": 1, "max": 1, "root": self.root_shifts + 1}.get(self.mode, self.exps)
        shift = 0 if shifts == 1 else lanes * self.acc_bits * max(1, ceil_log2(shifts))
        rescale = lanes * 3 * (self.acc_bits + self.scale_bits)
        lut = lanes * table_bits * luts + scale + trees + shift + rescale
        wide = table_bits + self.acc_bits
        ff = lanes * (bits + wide * (2 + sum(self.stages.values())) + self.out_dtype.bits)
        return Resources(lut=lut, ff=ff + self.total_bits + self.scale_bits + self.latency,
                         dsp=self.multipliers)

    def memories(self, graph):
        found = {"scale_table": MemorySpec(self.scale_table, self.scale_bits)}
        if self.sum_table.size:
            found["sum_table"] = MemorySpec(self.sum_table, self.sum_bits)
        if self.numer_table is not None:
            found["numer_table"] = MemorySpec(self.numer_table, self.numer_bits)
        if self.log:
            found["exp_table"] = MemorySpec(self.exp_table, self.scale_bits)
        return found

    def sv_params(self, graph):
        return {"LANES": self.lanes, "IN_BITS": self.in_dtype.bits,
                "MAX": int(self.subtract_max), "LOG": int(self.log),
                "MODE": self.MODES.index(self.mode),
                "SUM_BITS": self.sum_bits, "NUMER_BITS": self.numer_bits,
                "TOTAL_BITS": self.total_bits, "MIN_EXP": self.min_exp,
                "MANTISSA_BITS": self.mantissa_bits, "SCALE_BITS": self.scale_bits,
                "SHIFT": self.shift, "OUT_BITS": self.out_dtype.bits,
                "OUT_ZP": self.out_zero_point, "OUT_MIN": self.out_dtype.min,
                "OUT_MAX": self.out_dtype.max,
                "MAX_STAGES": self.stages.get("max", 0),
                "TABLE_STAGES": self.stages["table"], "SUM_STAGES": self.stages.get("sum", 0),
                "NORMALIZE_STAGES": self.stages.get("normalize", 0),
                "RESCALE_STAGES": self.stages["rescale"]}
