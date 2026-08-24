"""ONNX to internal IR.

ONNX carries images as NCHW; the streaming architecture wants channels
innermost so they can be split across parallel lanes. The importer therefore
transposes every weight and shape into channel-last form once, here, and the
rest of the compiler never sees NCHW again.
"""

import numpy as np

from ..p1_ingest import TensorProto
from ..p2_graph.datatype import FLOAT32, INT8, IntType, QuantSpec, UINT8
from ..p2_graph.graph import Graph
from ..p2_graph.layout import LayoutAdapter
from ..p2_graph.tensor import Tensor
from ..p3_ops.onnx_ops import AddOp, ConvOp, FlattenOp, GemmOp, MaxPoolOp, ReluOp


class ImportError_(Exception):
    pass


class OpHandler:
    """Translates one ONNX op type. Subclasses register themselves by name."""

    op_types = ()

    def handle(self, node, importer):
        raise NotImplementedError

    @staticmethod
    def spatial(attrs, key, default):
        value = attrs.get(key, default)
        return tuple(int(v) for v in value)

    @staticmethod
    def resolve_pads(attrs, kernel, strides, dilations, in_dim):
        """Returns (top, left, bottom, right)."""
        auto = attrs.get("auto_pad", "NOTSET") or "NOTSET"
        if auto in ("NOTSET", ""):
            pads = attrs.get("pads", [0, 0, 0, 0])
            return (int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3]))
        if auto == "VALID":
            return (0, 0, 0, 0)
        totals = []
        for axis in range(2):
            effective = (kernel[axis] - 1) * dilations[axis] + 1
            out = -(-in_dim[axis] // strides[axis])
            need = max(0, (out - 1) * strides[axis] + effective - in_dim[axis])
            totals.append(need)
        if auto == "SAME_UPPER":
            return (totals[0] // 2, totals[1] // 2,
                    totals[0] - totals[0] // 2, totals[1] - totals[1] // 2)
        return (totals[0] - totals[0] // 2, totals[1] - totals[1] // 2,
                totals[0] // 2, totals[1] // 2)


class IdentityHandler(OpHandler):
    """Identity carries no computation, so it becomes an alias."""

    op_types = ("Identity",)

    def handle(self, node, importer):
        importer.alias(node.output[0], importer.rename(node.input[0]))


class QdqHandler(OpHandler):
    """QuantizeLinear and DequantizeLinear.

    Neither is computation. They record which integer grid a tensor lives on,
    which is exactly what the compiler needs and would otherwise have to
    measure by calibration. Both collapse to an alias carrying that record.

    A DequantizeLinear whose input is an integer initializer is the usual way a
    quantized model stores its weights, so that case materialises the float
    constant the rest of the frontend expects, with the grid remembered so the
    lowering re-derives the very same integers instead of picking its own.
    """

    op_types = ("QuantizeLinear", "DequantizeLinear")

    QUANTIZED = {TensorProto.INT8: IntType(8, True),
                 TensorProto.UINT8: IntType(8, False)}

    def handle(self, node, importer):
        spec = self._spec(node, importer)
        source = importer.rename(node.input[0])
        target = node.output[0]

        if importer.model.has_initializer(source):
            stored = importer.model.initializer_dtype(source)
            if stored in self.QUANTIZED:
                raw = importer.model.initializer(source).astype(np.float64)
                importer.add_constant(target, spec.dequantize(raw))
                importer.annotate(target, spec)
                return
        importer.alias(target, source)
        importer.annotate(source, spec)
        importer.annotate(target, spec)

    def _spec(self, node, importer):
        scale = importer.constant_array(node.input[1])
        zero_point = np.zeros_like(scale, dtype=np.int64)
        dtype = UINT8
        if len(node.input) > 2 and node.input[2]:
            zero_point = importer.constant_array(node.input[2]).astype(np.int64)
            stored = importer.model.initializer_dtype(node.input[2])
            dtype = self.QUANTIZED.get(stored, UINT8)
        axis = node.attributes().get("axis")
        return QuantSpec(scale, zero_point, axis, dtype)


class ConvHandler(OpHandler):
    op_types = ("Conv",)

    def handle(self, node, importer):
        attrs = node.attributes()
        if int(attrs.get("group", 1)) != 1:
            raise ImportError_("grouped convolution is not supported yet: %s" % node.name)
        weights = importer.constant_array(node.input[1])
        kernel = self.spatial(attrs, "kernel_shape", weights.shape[2:])
        strides = self.spatial(attrs, "strides", (1, 1))
        dilations = self.spatial(attrs, "dilations", (1, 1))
        source = importer.graph.tensor(importer.rename(node.input[0]))
        pads = self.resolve_pads(attrs, kernel, strides, dilations,
                                 (source.shape[1], source.shape[2]))

        weight_name = importer.derive_constant(node.input[1],
                                               np.transpose(weights, (2, 3, 1, 0)))
        inputs = [importer.rename(node.input[0]), weight_name]
        if len(node.input) > 2:
            inputs.append(importer.derive_constant(node.input[2],
                                                   importer.constant_array(node.input[2])))
        importer.emit(ConvOp(node.name or "conv", inputs,
                             [importer.rename(node.output[0])],
                             kernel_shape=kernel, strides=strides, pads=pads,
                             dilations=dilations))


class GemmHandler(OpHandler):
    op_types = ("Gemm", "MatMul")

    def handle(self, node, importer):
        attrs = node.attributes()
        weights = importer.constant_array(node.input[1])
        if node.op_type == "Gemm":
            if float(attrs.get("alpha", 1.0)) != 1.0 or float(attrs.get("beta", 1.0)) != 1.0:
                raise ImportError_("Gemm alpha/beta scaling is not supported: %s" % node.name)
            if int(attrs.get("transA", 0)):
                raise ImportError_("Gemm transA is not supported: %s" % node.name)
            if int(attrs.get("transB", 0)):
                weights = weights.T
        weight_name = importer.derive_constant(node.input[1], weights)
        inputs = [importer.rename(node.input[0]), weight_name]
        if len(node.input) > 2:
            inputs.append(importer.derive_constant(node.input[2],
                                                   importer.constant_array(node.input[2])))
        importer.emit(GemmOp(node.name or "gemm", inputs,
                             [importer.rename(node.output[0])]))


class _FloatNode:
    """A QOperator node dressed as its float equivalent.

    QLinearConv is a Conv whose operands happen to arrive already quantized,
    so once the grids are recorded and the weights dequantized there is
    nothing left for a separate importer to do. Handing the existing handler a
    node of this shape is what stops the two paths drifting: QDQ and
    QOperator models reach the same ConvOp by the same code."""

    def __init__(self, op_type, name, inputs, outputs, attributes):
        self.op_type = op_type
        self.name = name
        self.input = list(inputs)
        self.output = list(outputs)
        self._attributes = dict(attributes)

    def attributes(self):
        return self._attributes


class QOperatorHandler(OpHandler):
    """QLinearConv and QLinearMatMul.

    The QDQ form states a tensor's grid beside the tensor; the QOperator form
    folds it into the operator's own signature. Same information, so it lands
    the same way: annotate the activation grids, dequantize the stored weights
    so the float frontend has something to look at, and emit the operator the
    rest of the compiler already knows. The lowering then re-derives exactly
    the integers the model shipped, rather than picking its own.
    """

    op_types = ("QLinearConv", "QLinearMatMul")

    STORED = {TensorProto.INT8: IntType(8, True),
              TensorProto.UINT8: IntType(8, False)}

    def handle(self, node, importer):
        if node.op_type == "QLinearConv":
            self._convolution(node, importer)
        else:
            self._matmul(node, importer)

    def _convolution(self, node, importer):
        # x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, and maybe B.
        x, w = node.input[0], node.input[3]
        importer.annotate(importer.rename(x), self._spec(node, importer, 1, 2))
        weights = self._dequantized(node, importer, 3, 4, 5, channel_axis=0)
        inputs = [importer.rename(x), weights]
        if len(node.input) > 8 and node.input[8]:
            inputs.append(self._bias(node, importer, 8, 1, 4))
        importer.annotate(importer.rename(node.output[0]),
                          self._spec(node, importer, 6, 7))
        ConvHandler().handle(
            _FloatNode("Conv", node.name or "qconv", inputs, node.output,
                       node.attributes()), importer)

    def _matmul(self, node, importer):
        # a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp.
        a, b = node.input[0], node.input[3]
        importer.annotate(importer.rename(a), self._spec(node, importer, 1, 2))
        weights = self._dequantized(node, importer, 3, 4, 5, channel_axis=1)
        importer.annotate(importer.rename(node.output[0]),
                          self._spec(node, importer, 6, 7))
        GemmHandler().handle(
            _FloatNode("MatMul", node.name or "qmatmul",
                       [importer.rename(a), weights], node.output, {}), importer)

    def _spec(self, node, importer, scale_index, zero_index):
        scale = importer.constant_array(node.input[scale_index])
        zero_point = np.zeros_like(scale, dtype=np.int64)
        dtype = UINT8
        name = node.input[zero_index] if len(node.input) > zero_index else ""
        if name:
            zero_point = importer.constant_array(name).astype(np.int64)
            dtype = self.STORED.get(importer.model.initializer_dtype(name), UINT8)
        return QuantSpec(scale, zero_point, None, dtype)

    def _dequantized(self, node, importer, weight_index, scale_index,
                     zero_index, channel_axis):
        """Weights come back as floats carrying the grid they were stored on,
        which is what lets the quantizer land on the model's own integers
        instead of recalibrating to something close but different."""
        name = node.input[weight_index]
        if not importer.model.has_initializer(name):
            raise ImportError_(
                "%s needs its weights as an initializer, not a stream: %s"
                % (node.op_type, node.name))
        stored = importer.model.initializer(name).astype(np.float64)
        scale = np.asarray(importer.constant_array(node.input[scale_index]),
                           dtype=np.float64)
        zero = np.zeros_like(scale, dtype=np.float64)
        if len(node.input) > zero_index and node.input[zero_index]:
            zero = np.asarray(importer.constant_array(node.input[zero_index]),
                              dtype=np.float64)
        shape = [1] * stored.ndim
        if scale.ndim:
            shape[channel_axis] = scale.size
        real = (stored - zero.reshape(shape)) * scale.reshape(shape)

        spec_dtype = UINT8
        zp_name = node.input[zero_index] if len(node.input) > zero_index else ""
        if zp_name:
            spec_dtype = self.STORED.get(importer.model.initializer_dtype(zp_name),
                                         UINT8)
        target = importer.graph.fresh_name(name + "_deq")
        importer.add_constant(target, real)
        importer.annotate(target, QuantSpec(scale, np.asarray(zero, dtype=np.int64),
                                            channel_axis if scale.ndim else None,
                                            spec_dtype))
        return target

    def _bias(self, node, importer, bias_index, x_scale_index, w_scale_index):
        """int32, on the product of the input and weight grids, which is what
        an integer accumulator lands on before it is rescaled."""
        raw = importer.model.initializer(node.input[bias_index]).astype(np.float64)
        x_scale = np.asarray(importer.constant_array(node.input[x_scale_index]),
                             dtype=np.float64)
        w_scale = np.asarray(importer.constant_array(node.input[w_scale_index]),
                             dtype=np.float64)
        target = importer.graph.fresh_name(node.input[bias_index] + "_deq")
        importer.add_constant(target, raw * x_scale * w_scale)
        return target

class ReluHandler(OpHandler):
    op_types = ("Relu",)

    def handle(self, node, importer):
        importer.emit(ReluOp(node.name or "relu", [importer.rename(node.input[0])],
                             [importer.rename(node.output[0])]))


class MaxPoolHandler(OpHandler):
    op_types = ("MaxPool",)

    def handle(self, node, importer):
        attrs = node.attributes()
        kernel = self.spatial(attrs, "kernel_shape", (2, 2))
        strides = self.spatial(attrs, "strides", kernel)
        dilations = self.spatial(attrs, "dilations", (1, 1))
        source = importer.graph.tensor(importer.rename(node.input[0]))
        pads = self.resolve_pads(attrs, kernel, strides, dilations,
                                 (source.shape[1], source.shape[2]))
        importer.emit(MaxPoolOp(node.name or "pool", [importer.rename(node.input[0])],
                                [importer.rename(node.output[0])],
                                kernel_shape=kernel, strides=strides, pads=pads,
                                dilations=dilations))


class AddHandler(OpHandler):
    op_types = ("Add",)

    def handle(self, node, importer):
        inputs = []
        for name in node.input:
            if importer.model.has_initializer(importer.rename(name)):
                array = importer.constant_array(name)
                inputs.append(importer.derive_constant(name, np.squeeze(array)))
            else:
                inputs.append(importer.rename(name))
        importer.emit(AddOp(node.name or "add", inputs,
                            [importer.rename(node.output[0])]))


class FlattenHandler(OpHandler):
    op_types = ("Flatten", "Reshape", "Squeeze")

    def handle(self, node, importer):
        importer.emit(FlattenOp(node.name or "flatten",
                                [importer.rename(node.input[0])],
                                [importer.rename(node.output[0])]))


class OnnxImporter:
    HANDLERS = (ConvHandler(), GemmHandler(), ReluHandler(), MaxPoolHandler(),
                AddHandler(), FlattenHandler(), QdqHandler(), QOperatorHandler(),
                IdentityHandler())

    def __init__(self, model):
        self.model = model
        self.graph = Graph(model.graph.name or "imported")
        self._dispatch = {}
        for handler in self.HANDLERS:
            for op_type in handler.op_types:
                self._dispatch[op_type] = handler
        self._renames = {}
        self.annotations = {}

    def run(self):
        for name in self.model.graph_inputs():
            declared = self.model.value_shape(name)
            adapter = LayoutAdapter.for_rank(len(declared or ()))
            self.graph.layouts[name] = adapter
            self.graph.add_tensor(Tensor(name, self._to_channel_last(declared), FLOAT32))
            self.graph.inputs.append(name)
        for node in self.model.nodes():
            if self._is_constant_only(node):
                continue
            handler = self._dispatch.get(node.op_type)
            if handler is None:
                raise ImportError_("unsupported ONNX op %r in node %r"
                                   % (node.op_type, node.name))
            handler.handle(node, self)
        for name in self.model.graph_outputs():
            self.graph.outputs.append(self.rename(name))
        self.graph.annotations = {self.rename(k): v
                                  for k, v in self.annotations.items()}
        self.graph.infer()
        return self.graph

    def _is_constant_only(self, node):
        """Quantize and dequantize nodes are annotations, so they are handled
        even when every input is constant."""
        if node.op_type in QdqHandler.op_types:
            return False
        return bool(node.output) and all(self.model.has_initializer(i) for i in node.input)

    def emit(self, node):
        """Shapes are inferred as each node lands, because later handlers need
        the extent of their input to resolve padding and window geometry."""
        for name in node.outputs:
            self.graph.ensure_tensor(name)
        self.graph.add_node(node)
        node.infer(self.graph)

    def rename(self, name):
        return self._renames.get(name, name)

    def constant_array(self, name):
        """Constants can come from the model, or from a DequantizeLinear the
        frontend has already folded into the graph."""
        resolved = self.rename(name)
        for candidate in (resolved, name):
            if self.graph.has_tensor(candidate) and self.graph.tensor(candidate).is_constant:
                return np.asarray(self.graph.tensor(candidate).data, dtype=np.float64)
        if self.model.has_initializer(resolved):
            return self.model.initializer(resolved).astype(np.float64)
        raise ImportError_("expected %r to be a constant initializer" % name)

    def add_constant(self, name, array):
        if self.graph.has_tensor(name):
            return name
        self.graph.add_tensor(Tensor.constant(name, np.asarray(array, dtype=np.float64)))
        return name

    def derive_constant(self, base, array):
        """A reshaped or transposed view of a constant, keeping whatever
        quantization grid the original carried."""
        name = self.graph.fresh_name(base + "_cl")
        self.graph.add_tensor(Tensor.constant(name, np.asarray(array, dtype=np.float64)))
        source = self.rename(base)
        for candidate in (base, source):
            if candidate in self.annotations:
                self.annotations[name] = self.annotations[candidate]
                break
        return name

    def alias(self, name, target):
        self._renames[name] = target

    def annotate(self, name, spec):
        self.annotations[name] = spec

    @staticmethod
    def _to_channel_last(shape):
        if shape is None:
            raise ImportError_("graph input has no static shape")
        dims = [1 if isinstance(d, str) else int(d) for d in shape]
        if len(dims) == 4:
            return (dims[0], dims[2], dims[3], dims[1])
        return tuple(dims)
