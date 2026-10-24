"""Custom hardware operators: one stream in and out on the same grid, folded by pe, with the library port contract."""

import numpy as np

from .errors import ConfigError, ModelError
from .p2_graph.datatype import FLOAT32
from .p2_graph.node import MemorySpec
from .p2_graph.onnx_importer import OpHandler
from .p3_ops.hardware_ops import StreamingNode, divisors
from .p3_ops.onnx_ops import FrontendNode
from .p4_quantize.lower import Lowering
from .targets.resources import Resources


class Grid:
    """The integer grid a stream lives on: real = scale * (q - zero_point)."""

    def __init__(self, scale, zero_point, dtype):
        self.scale = float(scale)
        self.zero_point = int(zero_point)
        self.dtype = dtype

    @property
    def bits(self):
        return self.dtype.bits

    @property
    def min(self):
        return self.dtype.min

    @property
    def max(self):
        return self.dtype.max

    def quantize(self, real):
        codes = self.dtype.clamp(np.rint(np.asarray(real, dtype=np.float64) / self.scale)
                                 + self.zero_point)
        return int(codes) if codes.ndim == 0 else codes

    def dequantize(self, codes):
        return self.scale * (np.asarray(codes, dtype=np.float64) - self.zero_point)

    def __repr__(self):
        return "Grid(scale=%.6g, zero_point=%d, %s)" % (self.scale, self.zero_point,
                                                        self.dtype)


class HardwareLayer:
    """Subclass this to give the compiler an operator it does not have."""

    module = None
    rtl = None

    @classmethod
    def from_onnx(cls, attributes, constants):
        """The layer for one ONNX node: attributes by name and constant inputs after the first in order."""
        if constants:
            raise ModelError(
                "%s receives %d constant inputs besides its stream; override "
                "%s.from_onnx(attributes, constants) to read them"
                % (cls.__name__, len(constants), cls.__name__))
        try:
            return cls(**attributes)
        except TypeError as error:
            raise ModelError(
                "%s(**%r) failed: %s. Accept these ONNX attributes in __init__, or "
                "override from_onnx" % (cls.__name__, attributes, error))

    def build(self, input_shape, grid):
        """Called with the channel-last input shape and the grid, which the output shares."""
        self.input_shape = tuple(input_shape)
        self.grid = grid

    @property
    def channels(self):
        return int(self.input_shape[-1])

    def call(self, x):
        """The float function, on a numpy array."""
        raise NotImplementedError("%s.call(x) is the float reference and must be "
                                  "defined" % type(self).__name__)

    def call_integer(self, q):
        """The integer function on codes of self.grid, which the hardware must reproduce exactly."""
        raise NotImplementedError("%s.call_integer(q) is the integer reference and "
                                  "must be defined" % type(self).__name__)

    def parameters(self, pe):
        """Module parameters for a unit `pe` lanes wide; each memory gets a <NAME>_FILE parameter."""
        return {}

    def memories(self, pe):
        """Initialised memories by name, each (array, element bits), laid out as the module reads it."""
        return {}

    def resources(self, device, pe):
        """A rough figure for the folding allocator."""
        bits = self.grid.bits
        return Resources(lut=pe * bits * 2 + 20, ff=pe * bits)

    def __repr__(self):
        return "<%s module=%s>" % (type(self).__name__, self.module)


class CustomOp(FrontendNode):
    """A HardwareLayer in the float graph."""

    def __init__(self, name, inputs, outputs, layer, op_type):
        FrontendNode.__init__(self, name, inputs, outputs)
        self.layer = layer
        self.op_type = op_type

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = self._out(graph)
        out.shape, out.dtype = source.shape, FLOAT32

    def execute(self, graph, values):
        x = np.asarray(values[self.inputs[0]], dtype=np.float64)
        return [np.asarray(self.layer.call(x), dtype=np.float64).reshape(x.shape)]


class CustomHandler(OpHandler):
    def __init__(self, op_type, layer_class):
        self.op_types = (op_type,)
        self.layer_class = layer_class

    def handle(self, node, importer):
        constants = [importer.constant_array(name) for name in node.input[1:] if name]
        layer = self.layer_class.from_onnx(node.attributes(), constants)
        importer.emit(CustomOp(node.name or node.op_type.lower(),
                               [importer.rename(node.input[0])],
                               [importer.rename(node.output[0])],
                               layer, node.op_type))


class CustomUnit(StreamingNode):
    """A HardwareLayer in the hardware graph."""

    def __init__(self, name, inputs, outputs, layer, **attrs):
        self.layer = layer
        StreamingNode.__init__(self, name, inputs, outputs, **attrs)

    @property
    def module(self):
        return self.layer.module

    @property
    def channels(self):
        return self.layer.channels

    @property
    def elements(self):
        return int(np.prod(self.layer.input_shape))

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
        return self.layer.resources(device, self.pe)

    def input_beat_elems(self, graph):
        return [self.pe]

    def output_beat_elems(self, graph):
        return [self.pe]

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = graph.ensure_tensor(self.outputs[0])
        out.shape, out.dtype = source.shape, self.layer.grid.dtype
        out.elems_per_beat = self.pe

    def execute(self, graph, values):
        q = np.asarray(values[self.inputs[0]], dtype=np.int64)
        produced = np.asarray(self.layer.call_integer(q), dtype=np.int64)
        return [self.layer.grid.dtype.clamp(produced).reshape(q.shape)]

    def memories(self, graph):
        return {key: MemorySpec(np.asarray(array, dtype=np.int64), bits)
                for key, (array, bits) in self.layer.memories(self.pe).items()}

    def sv_params(self, graph):
        params = dict(self.layer.parameters(self.pe))
        paths = self.memory_paths()
        for key in self.layer.memories(self.pe):
            params["%s_FILE" % key.upper()] = paths.get(key, "")
        return params

    def rtl_files(self):
        if not self.layer.rtl:
            return {}
        return {"%s.sv" % self.module: self.layer.rtl}


class CustomLowering(Lowering):
    handles = (CustomOp,)

    def apply(self, node, ctx):
        """The output shares the input grid, so its calibrated scale is replaced by the input one."""
        source = node.inputs[0]
        ctx.require_same_grid(source, node.outputs[0], node.op_type)
        ctx.plan.propagate(source, node.outputs[0])
        grid = Grid(ctx.scale(source), ctx.zero_point(source), ctx.plan.act_dtype)
        node.layer.build(ctx.graph.tensor(source).shape, grid)
        return [CustomUnit(node.name, [source], node.outputs, node.layer)]


def custom_passes(custom_objects):
    """The importer handlers and lowering for a mapping of ONNX op type to HardwareLayer."""
    handlers = []
    for op_type, layer_class in sorted((custom_objects or {}).items()):
        if not (isinstance(layer_class, type) and issubclass(layer_class, HardwareLayer)):
            raise ConfigError("custom_objects[%r] must be a HardwareLayer subclass, "
                              "not %r" % (op_type, layer_class))
        handlers.append(CustomHandler(op_type, layer_class))
    return handlers, ([CustomLowering()] if handlers else [])
