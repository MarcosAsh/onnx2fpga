"""Elementwise operators: the float ops the importer makes, and the two fixed latency units they become."""

import math

import numpy as np

from ..errors import ModelError, UnsupportedOpError
from ..p2_graph.datatype import FLOAT32, IntType
from ..p2_graph.node import MemorySpec
from ..targets.resources import Resources
from .hardware_ops import StreamingNode
from .onnx_ops import FrontendNode

erf = np.vectorize(math.erf, otypes=[np.float64])


def softplus(x):
    return np.logaddexp(0.0, x)


def gelu(x, approximate="none"):
    if approximate == "tanh":
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))


def truth(x):
    return np.asarray(x) != 0


# ONNX's definitions on real values; attributes arrive as keywords with ONNX's defaults.
FUNCTIONS = {
    "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "Tanh": np.tanh,
    "HardSigmoid": lambda x, alpha=0.2, beta=0.5: np.clip(alpha * x + beta, 0, 1),
    "HardSwish": lambda x: x * np.clip(x / 6 + 0.5, 0, 1),
    "Elu": lambda x, alpha=1.0: np.where(x >= 0, x, alpha * (np.exp(x) - 1)),
    "Selu": lambda x, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875:
        gamma * np.where(x > 0, x, alpha * (np.exp(x) - 1)),
    "Gelu": gelu,
    "Softplus": softplus,
    "Softsign": lambda x: x / (1 + np.abs(x)),
    "Exp": np.exp,
    "Log": np.log,
    "Sqrt": np.sqrt,
    "Reciprocal": lambda x: 1 / x,
    "Erf": erf,
    "Mish": lambda x: x * np.tanh(softplus(x)),
    "LeakyRelu": lambda x, alpha=0.01: np.where(x >= 0, x, alpha * x),
    "PRelu": lambda x, slope: np.where(x >= 0, x, slope * x),
    "Relu": lambda x: np.maximum(x, 0),
    "Neg": np.negative,
    "Abs": np.abs,
    "Not": lambda x: ~truth(x),
    "Clip": lambda x, min=-np.inf, max=np.inf: np.clip(x, min, max),
    "Add": np.add,
    "Sub": np.subtract,
    "Mul": np.multiply,
    "Div": np.divide,
    "Max": np.maximum,
    "Min": np.minimum,
    "Equal": np.equal,
    "Less": np.less,
    "Greater": np.greater,
    "LessOrEqual": np.less_equal,
    "GreaterOrEqual": np.greater_equal,
    "And": lambda a, b: truth(a) & truth(b),
    "Or": lambda a, b: truth(a) | truth(b),
    "Where": lambda c, a, b: np.where(truth(c), a, b),
}

# Their outputs are 0 or 1, carried on a grid of scale 1 and zero point 0.
BOOLEAN = {"Not", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or"}

ATTRIBUTES = {"HardSigmoid": ("alpha", "beta"), "Elu": ("alpha",), "Selu": ("alpha", "gamma"),
              "Gelu": ("approximate",), "LeakyRelu": ("alpha",), "Clip": ("min", "max")}

# Operands past the first that must be constants.
CONSTANT_OPERANDS = {"PRelu": "slope", "Div": "divisor"}


def evaluate(op_type, operands, attrs):
    """FUNCTIONS[op_type] as float64, quiet about the infinities Log and Reciprocal make."""
    with np.errstate(all="ignore"):
        return np.asarray(FUNCTIONS[op_type](*operands, **attrs), dtype=np.float64)


class ElementwiseOp(FrontendNode):
    """One of FUNCTIONS, broadcasting its operands like numpy; `function_attrs` are its ONNX attributes."""

    def __init__(self, name, inputs, outputs, op_type, **attrs):
        FrontendNode.__init__(self, name, inputs, outputs, **attrs)
        self.op_type = op_type

    def infer(self, graph):
        shapes = [graph.tensor(name).shape for name in self.inputs]
        try:
            shape = np.broadcast_shapes(*shapes)
        except ValueError:
            raise ModelError("%s (%s) takes shapes %s, which do not broadcast"
                             % (self.name, self.op_type, " and ".join(map(str, shapes))))
        out = self._out(graph)
        out.shape = tuple(shape)
        out.dtype = FLOAT32

    def execute(self, graph, values):
        operands = [values[name] if name in values else graph.tensor(name).data
                    for name in self.inputs]
        return [evaluate(self.op_type, operands, self.attrs.get("function_attrs", {}))]


def import_elementwise(node, importer):
    """Constants are moved to the channel-last layout of a 4D live operand, and Clip's bounds become attributes."""
    op, name = node.op_type, node.name or importer.graph.fresh_name(node.op_type.lower())
    raw = node.attributes()
    attrs = {key: raw[key] for key in ATTRIBUTES.get(op, ()) if key in raw}
    if isinstance(attrs.get("approximate"), bytes):
        attrs["approximate"] = attrs["approximate"].decode()
    names = list(node.input)
    if op == "Clip":
        for key, bound in zip(("min", "max"), names[1:3]):
            if not bound:
                continue
            if not is_constant(importer, bound):
                raise UnsupportedOpError(op, name, message="Clip %r has a %s bound that is "
                                         "not a constant" % (name, key))
            attrs[key] = float(importer.constant_array(bound).reshape(-1)[0])
        names = names[:1]
    if op in ("Max", "Min") and len(names) > 2:
        raise UnsupportedOpError(op, name, message="%s %r has %d inputs, and only two are "
                                 "supported" % (op, name, len(names)))
    live = [n for n in names if not is_constant(importer, n)]
    rank = len(importer.graph.tensor(importer.rename(live[0])).shape) if live else 0
    inputs = []
    for operand in names:
        if operand in live:
            inputs.append(importer.rename(operand))
            continue
        array = importer.constant_array(operand)
        if rank == 4 and array.ndim > 1:
            array = array.reshape((1,) * (4 - array.ndim) + array.shape).transpose(0, 2, 3, 1)
        inputs.append(importer.derive_constant(operand, array))
    if op in CONSTANT_OPERANDS and len(names) > 1 and names[1] in live:
        raise UnsupportedOpError(op, name, message="%s %r needs a constant %s"
                                 % (op, name, CONSTANT_OPERANDS[op]))
    importer.emit(ElementwiseOp(name, inputs, [importer.rename(node.output[0])], op,
                                function_attrs=attrs))


def is_constant(importer, name):
    resolved = importer.rename(name)
    if importer.graph.has_tensor(resolved):
        return importer.graph.tensor(resolved).is_constant
    return importer.model.has_initializer(resolved)


class TableUnit(StreamingNode):
    """A constant table per element on the input code, one cycle; `tables` hold the most negative code first."""

    module = "otf_table"
    hardcaml_only = True

    def __init__(self, name, inputs, outputs, tables, lane_table, in_dtype, out_dtype):
        self.tables = np.asarray(tables, dtype=np.int64).reshape(-1, 1 << in_dtype.bits)
        self.lane_table = np.asarray(lane_table, dtype=np.int64)
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.stages = {}
        StreamingNode.__init__(self, name, inputs, outputs)

    @classmethod
    def per_lane(cls, name, inputs, outputs, rows, in_dtype, out_dtype):
        """From one table row per lane, keeping each distinct row once."""
        tables, lane_table = np.unique(np.asarray(rows, dtype=np.int64), axis=0,
                                       return_inverse=True)
        return cls(name, inputs, outputs, tables, lane_table.reshape(-1), in_dtype, out_dtype)

    @property
    def lanes(self):
        return int(self.lane_table.size)

    @property
    def latency(self):
        return 1

    @property
    def cycles(self):
        return 1

    multipliers = 0

    def rows(self):
        return self.tables[self.lane_table]

    def paths(self, delays):
        luts = max(1, self.tables.shape[1] // 64)
        return {"table": [delays.path(delays.table(self.in_dtype.bits),
                                      fanout=self.out_dtype.bits * luts)]}

    def amd_resources(self, device):
        """A LUT6 per 64 entries of each output bit, the most a table takes; MUXF7 and MUXF8 are free."""
        luts = max(1, self.tables.shape[1] // 64)
        return Resources(lut=self.lanes * self.out_dtype.bits * luts,
                         ff=self.lanes * self.out_dtype.bits + 1)

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = source.shape, self.out_dtype
        out.elems_per_beat = source.numel

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.int64)
        flat = x.reshape(-1, self.lanes) - self.in_dtype.min
        return [self.rows()[np.arange(self.lanes), flat].reshape(x.shape)]

    def memories(self, graph):
        return {"tables": MemorySpec(self.tables.reshape(-1), self.out_dtype.bits),
                "lane_table": MemorySpec(self.lane_table,
                                         max(1, (self.tables.shape[0] - 1).bit_length()))}

    def sv_params(self, graph):
        return {"LANES": self.lanes, "IN_BITS": self.in_dtype.bits,
                "OUT_BITS": self.out_dtype.bits}


class EltwiseUnit(StreamingNode):
    """Lane by lane on two or three operands; Add, Mul and comparisons rescale by the contract, Max, Min and Where by table."""

    module = "otf_eltwise"
    hardcaml_only = True
    OPS = ("add", "mul", "less", "less_equal", "equal", "max", "min", "and", "or", "where")
    ARITHMETIC = ("add", "mul", "less", "less_equal", "equal")

    def __init__(self, name, inputs, outputs, shape, op, sources, zero_points, mults, mult,
                 shift, dtype, out_zero_point=0, mult_bits=18, tables=None):
        """`sources` holds a port index or an array of codes for each operand, `tables` an array or None."""
        self.shape = tuple(shape)
        self.tables = list(tables or [None] * len(sources))
        self.op = op
        self.sources = list(sources)
        self.zero_points = [int(z) for z in zero_points]
        self.mults = [int(m) for m in mults]
        self.mult = int(mult)
        self.shift = int(shift)
        self.dtype = dtype
        self.out_zero_point = int(out_zero_point)
        self.mult_bits = int(mult_bits)
        self.stages = {"product": 0, "rescale": 0}
        self.skew = []
        StreamingNode.__init__(self, name, inputs, outputs)

    @property
    def latency(self):
        if self.op in self.ARITHMETIC:
            return 2 + self.stages["product"] + self.stages["rescale"]
        return 1 + self.stages["rescale"]

    @property
    def cycles(self):
        return 1

    @property
    def lanes(self):
        return int(np.prod(self.shape))

    @property
    def multipliers(self):
        return self.lanes if self.op == "mul" else 0

    @property
    def acc_dtype(self):
        """Wide enough for the weighted sum of Add and the comparisons, or a product of two terms."""
        span = (1 << self.dtype.bits) - 1 + max(abs(z) for z in self.zero_points)
        if self.op == "mul":
            return IntType.smallest_for(-span * span, span * span)
        if self.op in self.ARITHMETIC:
            total = span * sum(abs(m) for m in self.mults)
            return IntType.smallest_for(-total, total)
        return IntType.smallest_for(-span, span)

    def rescale(self, acc, mult):
        scaled = np.asarray(acc, dtype=np.int64) * mult
        if self.shift > 0:
            scaled = (scaled + (1 << (self.shift - 1))) >> self.shift
        return np.clip(scaled + self.out_zero_point, self.dtype.min, self.dtype.max)

    def looked_up(self, k, codes):
        """A constant operand is already a code on the output grid."""
        if self.tables[k] is None:
            return codes
        return self.tables[k][codes - self.dtype.min]

    def execute(self, graph, values):
        ports = [np.asarray(values[n], dtype=np.int64).reshape(-1) for n in self.inputs]
        codes = [ports[s] if isinstance(s, int) else np.asarray(s, dtype=np.int64).reshape(-1)
                 for s in self.sources]
        terms = [c - z for c, z in zip(codes, self.zero_points)]
        weighted = sum(t * m for t, m in zip(terms, self.mults))
        op = self.op
        if op == "add":
            out = self.rescale(weighted, self.mult)
        elif op == "mul":
            out = self.rescale(terms[0] * terms[1], self.mult)
        elif op in ("less", "less_equal", "equal"):
            out = {"less": weighted < 0, "less_equal": weighted <= 0, "equal": weighted == 0}[op]
        elif op in ("max", "min"):
            pick = np.maximum if op == "max" else np.minimum
            out = pick(self.looked_up(0, codes[0]), self.looked_up(1, codes[1]))
        elif op in ("and", "or"):
            truth = [c != z for c, z in zip(codes, self.zero_points)]
            out = truth[0] & truth[1] if op == "and" else truth[0] | truth[1]
        else:
            out = np.where(codes[0] != self.zero_points[0], self.looked_up(1, codes[1]),
                           self.looked_up(2, codes[2]))
        return [np.asarray(out, dtype=np.int64).reshape(self.shape)]

    def infer(self, graph):
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = self.shape, self.dtype
        out.elems_per_beat = int(np.prod(self.shape))

    def paths(self, delays):
        """The longest path each stage governs, in ns, with 0, 1, 2 ... registers added."""
        path, adder, logic = delays.path, delays.adder, delays.logic
        wide, acc = self.dtype.bits + 1, self.acc_dtype.bits
        out = acc + self.mult_bits + 1
        rounding = [adder(out)] if 0 < self.shift < out - 1 else []
        clamp = [logic()]
        if self.op in ("and", "or"):
            return {"logic": [path(logic())]}
        if self.op in self.ARITHMETIC:
            if self.op == "mul":
                multiply = [delays.product(wide, wide), delays.product(wide, wide, mreg=True)]
                tail = []
            else:
                multiply = [delays.product(wide, self.mult_bits),
                            delays.product(wide, self.mult_bits, mreg=True)]
                tail = [adder(acc)]
            product = [path(adder(wide), multiply[0], *tail),
                       max(path(adder(wide), multiply[1]), path(*tail))]
            if self.op not in ("add", "mul"):
                return {"product": product, "test": [path(adder(acc))]}
            scale = [] if self.op == "add" else [delays.product(acc, self.mult_bits)]
            registered = (delays.registered_product(acc, self.mult_bits)
                          if self.op == "mul" else path())
            rescale = [path(*scale, *rounding, adder(out), *clamp),
                       max(registered, path(*rounding, adder(out), *clamp)),
                       max(registered, path(*rounding), path(adder(out), *clamp)),
                       max(registered, path(*rounding), path(adder(out)), path(*clamp))]
            if self.op == "add":
                rescale = rescale[1:]
            return {"product": product, "rescale": rescale}
        combine = [adder(self.dtype.bits), logic()] if self.op in ("max", "min") else [logic()]
        table = delays.table(self.dtype.bits)
        fanout = self.dtype.bits * max(1, (1 << self.dtype.bits) // 64)
        return {"rescale": [path(table, *combine, fanout=fanout),
                            max(path(table, fanout=fanout), path(*combine))]}

    def amd_resources(self, device):
        """Per lane: the operand adders, the rescale's multiply and adders, and the skew registers."""
        lanes, bits, acc = self.lanes, self.dtype.bits, self.acc_dtype.bits
        live = [s for s in self.sources if isinstance(s, int)]
        if self.op in ("and", "or"):
            cost = Resources(lut=lanes * 2, ff=lanes * bits)
        elif self.op in ("max", "min", "where"):
            luts = max(1, (1 << bits) // 64)
            cost = Resources(lut=lanes * len(live) * bits * luts + lanes * bits * 2,
                             ff=lanes * bits * (1 + self.stages["rescale"] * len(live)))
        else:
            dsp = lanes * (2 if self.op != "mul" else 1) + (lanes if self.op == "mul" else 0)
            cost = Resources(lut=lanes * (2 * (bits + 1) + 3 * acc),
                             ff=lanes * (acc * (1 + self.stages["product"]) + bits
                                         + acc * self.stages["rescale"]),
                             dsp=dsp)
        waiting = sum(self.skew) * lanes * bits
        return cost + Resources(ff=waiting + self.latency)

    def memories(self, graph):
        found = {"const%d" % k: MemorySpec(np.asarray(s, dtype=np.int64).reshape(-1),
                                           self.dtype.bits)
                 for k, s in enumerate(self.sources) if not isinstance(s, int)}
        found.update({"table%d" % k: MemorySpec(np.asarray(table), self.dtype.bits)
                      for k, table in enumerate(self.tables) if table is not None})
        return found

    def sv_params(self, graph):
        params = {"LANES": self.lanes, "BITS": self.dtype.bits, "OP": self.OPS.index(self.op),
                  "OPERANDS": len(self.sources), "MULT": self.mult,
                  "MULT_BITS": self.mult_bits, "ACC_BITS": self.acc_dtype.bits,
                  "SHIFT": self.shift, "OUT_ZP": self.out_zero_point,
                  "OUT_MIN": self.dtype.min, "OUT_MAX": self.dtype.max,
                  "PRODUCT_STAGES": self.stages["product"],
                  "RESCALE_STAGES": self.stages["rescale"]}
        for k, source in enumerate(self.sources):
            params["PORT%d" % k] = source if isinstance(source, int) else -1
            params["ZP%d" % k] = self.zero_points[k]
            params["MULT%d" % k] = self.mults[k]
        return params
