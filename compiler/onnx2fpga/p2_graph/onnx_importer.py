"""ONNX to the internal IR, transposing every weight and shape to channel-last once, here."""

import collections

import numpy as np

from ..errors import ModelError, UnsupportedOpError
from ..p1_ingest import TensorProto
from ..p1_ingest.onnx_model import tensor_to_numpy
from ..p1_ingest.onnx_schema import NodeProto
from ..p2_graph.datatype import FLOAT32, IntType, QuantSpec, UINT8
from ..p2_graph.graph import Graph
from ..p2_graph.layout import LayoutAdapter
from ..p2_graph.tensor import Tensor
from ..p3_ops.elementwise import FUNCTIONS, ElementwiseOp, import_elementwise
from ..p3_ops.onnx_ops import (AddOp, BinarizerOp, ConvOp, FlattenOp, GemmOp, MaxPoolOp,
                               ReluOp)
from ..p3_ops.reductions import (REDUCE, SOFTMAX, ReduceOp, import_normalizer, import_reduce,
                                 import_softmax)
from ..p3_ops.trees import LEAF, MODES, Ensemble, TreeEnsembleOp
from .constants import FOLD, LAYOUT, evaluate

QUANTIZED = {np.dtype(np.int8): IntType(8, True), np.dtype(np.uint8): IntType(8, False)}
QDQ = ("QuantizeLinear", "DequantizeLinear")

# y[j] = mul[j] * x[index[j]] + add[j] over the flat elements of live tensor x, not built yet.
View = collections.namedtuple("View", "source index mul add origin")


class OpHandler:
    """Translates the ONNX op types in `op_types` with handle(node, importer)."""

    op_types = ()

    def handle(self, node, importer):
        raise NotImplementedError


def refuse(node, text):
    raise UnsupportedOpError(node.op_type, node.name, message="%s %r %s"
                             % (node.op_type, node.name, text))


def rewritten(node, op_type, inputs, outputs=None):
    return NodeProto(op_type=op_type, name=node.name, input=list(inputs),
                     output=list(outputs or node.output), attribute=list(node.attribute))


def spatial(attrs, key, default):
    return tuple(int(v) for v in attrs.get(key, default))


def resolve_pads(attrs, kernel, strides, dilations, in_dim):
    """Returns (top, left, bottom, right)."""
    auto = attrs.get("auto_pad", "NOTSET") or "NOTSET"
    if auto == "NOTSET":
        pads = attrs.get("pads", [0, 0, 0, 0])
        return tuple(int(p) for p in pads[:4])
    if auto == "VALID":
        return (0, 0, 0, 0)
    totals = []
    for axis in range(2):
        effective = (kernel[axis] - 1) * dilations[axis] + 1
        out = -(-in_dim[axis] // strides[axis])
        totals.append(max(0, (out - 1) * strides[axis] + effective - in_dim[axis]))
    small = (totals[0] // 2, totals[1] // 2)
    large = (totals[0] - small[0], totals[1] - small[1])
    return small + large if auto == "SAME_UPPER" else large + small


def window_attrs(node, importer, kernel_default):
    attrs = node.attributes()
    kernel = spatial(attrs, "kernel_shape", kernel_default)
    strides = spatial(attrs, "strides", (1, 1) if node.op_type == "Conv" else kernel)
    dilations = spatial(attrs, "dilations", (1, 1))
    source = importer.graph.tensor(importer.rename(node.input[0]))
    pads = resolve_pads(attrs, kernel, strides, dilations, source.shape[1:3])
    return dict(kernel_shape=kernel, strides=strides, pads=pads, dilations=dilations)


def import_identity(node, importer):
    importer.alias(node.output[0], node.input[0])


def grid(importer, scale, zero=None):
    """The grid of a scale and optional zero point input, typed by the zero point."""
    if not zero:
        return QuantSpec(importer.constant_array(scale), 0, UINT8)
    raw = importer.constants[importer.rename(zero)]
    return QuantSpec(importer.constant_array(scale), raw.astype(np.int64),
                     QUANTIZED.get(raw.dtype, UINT8))


def dequantize(spec, raw, axis):
    """A per axis scale and zero point apply along `axis`."""
    scale, zero = spec.scale, spec.zero_point
    if scale.size > 1:
        shape = [1] * raw.ndim
        shape[axis] = -1
        scale, zero = scale.reshape(shape), (zero.reshape(shape) if zero.size > 1 else zero)
    return scale * (raw.astype(np.float64) - zero)


def import_qdq(node, importer):
    """Records the integer grid of a tensor and collapses to an alias carrying it."""
    spec = grid(importer, node.input[1], node.input[2] if len(node.input) > 2 else None)
    source = importer.rename(node.input[0])
    target = node.output[0]
    raw = importer.constants.get(source)
    if (node.op_type == "DequantizeLinear" and raw is not None
            and np.issubdtype(raw.dtype, np.integer)):
        importer.constants[target] = dequantize(spec, raw, node.attributes().get("axis", 1))
        importer.annotate(target, spec)
        return
    importer.alias(target, node.input[0], reads=False)
    importer.annotate(source, spec)
    importer.annotate(target, spec)


def real(importer, name, spec, axis=-1):
    """The real tensor behind integer tensor `name` on grid `spec`, as a QDQ pair would give."""
    raw = importer.constants.get(importer.rename(name))
    if raw is None:
        importer.annotate(importer.rename(name), spec)
        return name
    fresh = importer.graph.fresh_name(name + "_real")
    importer.constants[fresh] = dequantize(spec, raw, axis)
    importer.annotate(fresh, spec)
    return fresh


def import_qlinear(node, importer):
    """QLinearMatMul, QLinearConv and QLinearAdd become their float op between QDQ grids."""
    names = node.input
    a = real(importer, names[0], grid(importer, names[1], names[2]))
    b = real(importer, names[3], grid(importer, names[4], names[5]),
             axis=0 if node.op_type == "QLinearConv" else -1)
    inputs = [a, b]
    if node.op_type == "QLinearConv" and len(names) > 8 and names[8]:
        scale = importer.constant_array(names[1]) * importer.constant_array(names[4])
        bias = importer.graph.fresh_name(names[8] + "_real")
        importer.constants[bias] = importer.constant_array(names[8]) * scale
        inputs.append(bias)
    op_type = {"QLinearMatMul": "MatMul", "QLinearConv": "Conv", "QLinearAdd": "Add"}
    plain = rewritten(node, op_type[node.op_type], inputs)
    if node.op_type != "QLinearAdd":
        BUILTIN[plain.op_type](plain, importer)
    elif not importer.arithmetic(plain):
        import_add(plain, importer)
    spec = grid(importer, names[6], names[7])
    importer.annotate(importer.rename(node.output[0]), spec)
    importer.annotate(node.output[0], spec)


def import_integer(node, importer):
    """MatMulInteger or ConvInteger on a grid of scale s is a layer with weights (b - b_zero) / s."""
    names = list(node.input) + [""] * (4 - len(node.input))
    spec = importer.annotations.get(importer.rename(names[0]))
    if spec is None:
        refuse(node, "reads %r, which has no fixed grid, so it needs dynamic quantization"
               % names[0])
    conv = node.op_type == "ConvInteger"
    weights = importer.constants[importer.rename(names[1])].astype(np.float64)
    if names[3]:
        zero = importer.constant_array(names[3])
        weights = weights - (zero.reshape(-1, *[1] * (weights.ndim - 1)) if conv else zero)
    offset = spec.zero_point.reshape(-1) - (importer.constant_array(names[2]).reshape(-1)
                                            if names[2] else 0)
    inputs = [names[0], importer.graph.fresh_name(names[1] + "_real")]
    importer.constants[inputs[1]] = weights / (spec.scale if conv else
                                               spec.scale.reshape(-1, 1))
    if np.any(offset):
        if conv:
            refuse(node, "has an input zero point unlike its grid's, which padding breaks")
        inputs.append(importer.graph.fresh_name(names[1] + "_offset"))
        importer.constants[inputs[2]] = np.broadcast_to(offset, weights.shape[:1]) @ weights
    BUILTIN["Conv" if conv else "MatMul"](
        rewritten(node, "Conv" if conv else "MatMul", inputs), importer)


def refuse_dynamic(node, importer):
    refuse(node, "computes its scale from the data at run time, which a fixed point "
           "design cannot do")


def import_conv(node, importer):
    if int(node.attributes().get("group", 1)) != 1:
        raise ModelError("grouped convolution is not supported yet: %s" % node.name)
    weights = importer.constant_array(node.input[1])
    attrs = window_attrs(node, importer, weights.shape[2:])
    inputs = [importer.rename(node.input[0]),
              importer.derive_constant(node.input[1], np.transpose(weights, (2, 3, 1, 0)))]
    if len(node.input) > 2 and node.input[2]:
        inputs.append(importer.derive_constant(node.input[2],
                                               importer.constant_array(node.input[2])))
    importer.emit(ConvOp(node.name or "conv", inputs, [importer.rename(node.output[0])],
                         **attrs))


def import_gemm(node, importer):
    attrs = node.attributes()
    weights = importer.constant_array(node.input[1])
    bias = None
    if len(node.input) > 2 and node.input[2]:
        bias = importer.constant_array(node.input[2]) * float(attrs.get("beta", 1.0))
    if node.op_type == "Gemm":
        if int(attrs.get("transA", 0)):
            raise ModelError("Gemm transA is not supported: %s" % node.name)
        if int(attrs.get("transB", 0)):
            weights = weights.T
        weights = weights * float(attrs.get("alpha", 1.0))
    importer.dense(node.name or "gemm", node.input[0], weights, bias, node.output[0],
                   grid_from=node.input[1])


def import_linear(node, importer):
    """LinearRegressor, and LinearClassifier decided on the margin for two classes or by ArgMax for more."""
    attrs = node.attributes()
    transform = attrs.get("post_transform", "NONE")
    coefficients = np.asarray(attrs["coefficients"], dtype=np.float64)
    features = importer.onnx_shape(node.input[0])[-1]
    rows = coefficients.reshape(-1, features)
    intercepts = np.asarray(attrs.get("intercepts") or np.zeros(len(rows)), dtype=np.float64)
    name = node.name or node.op_type
    if node.op_type == "LinearRegressor":
        if transform != "NONE":
            refuse(node, "has post_transform %s, and only NONE is supported" % transform)
        importer.dense(name, node.input[0], rows.T, intercepts, node.output[0])
        return
    classes = len(attrs.get("classlabels_ints") or attrs.get("classlabels_strings") or [])
    if classes == 2:
        # One row is the margin itself, as ONNX Runtime reads it.
        margin, offset = rows[-1], intercepts[-1]
        if len(rows) > 1:
            margin, offset = margin - rows[0], offset - intercepts[0]
        importer.dense(name, node.input[0], margin[:, None], [offset], node.output[0],
                       label=True)
        importer.drop(node.output[1])
    else:
        # Every post transform is monotone, so the label is the ArgMax of the scores.
        scores = importer.graph.fresh_name(name + "_scores")
        importer.dense(name, node.input[0], rows.T, intercepts, scores)
        importer.emit(ReduceOp(name + "_label", [scores], [importer.rename(node.output[0])],
                               "ArgMax", keepdims=False, label=True))
        importer.drop(node.output[1])


def import_scaler(node, importer):
    attrs = node.attributes()
    scale = np.asarray(attrs.get("scale", [1.0]), dtype=np.float64)
    offset = np.asarray(attrs.get("offset", [0.0]), dtype=np.float64)
    importer.scale_shift(node.output[0], node.input[0], scale, -offset * scale, node.name)


def import_batchnorm(node, importer):
    attrs = node.attributes()
    if attrs.get("training_mode"):
        refuse(node, "is in training mode")
    scale, bias, mean, var = (importer.constant_array(name) for name in node.input[1:5])
    mul = scale / np.sqrt(var + float(attrs.get("epsilon", 1e-5)))
    importer.scale_shift(node.output[0], node.input[0], mul, bias - mean * mul,
                         node.name or "batchnorm")
    for name in node.output[1:]:
        importer.drop(name)


def import_layout(node, importer):
    """Reshape, Transpose, Slice, Gather and the like on live data, run on element indices."""
    shape = importer.onnx_shape(node.input[0])
    others = [importer.constants.get(importer.rename(name)) if name else None
              for name in node.input[1:]]
    moved = evaluate(node, [np.arange(int(np.prod(shape))).reshape(shape)] + others)[0]
    importer.rearrange(node, moved)


def import_cast(node, importer):
    """Only a float cast, or an integer cast of a label, leaves live values unchanged."""
    source = importer.rename(node.input[0])
    producer = importer.graph.producer(source)
    dtype = np.dtype(TensorProto.NUMPY_DTYPE[node.attributes()["to"]])
    if not (np.issubdtype(dtype, np.floating) or (producer and producer.attrs.get("label"))):
        refuse(node, "turns live data into %s, which would truncate it" % dtype)
    importer.alias(node.output[0], node.input[0])


def import_concat(node, importer):
    """Pieces that are views of one feature vector or Gemms on it become one view or one Gemm."""
    parts = []
    for raw in node.input:
        name = importer.rename(raw)
        producer = importer.graph.producer(name)
        shape = importer.constants[name].shape if name in importer.constants else (
            importer.onnx_shape(raw))
        if len(shape) != 2 or shape[0] != 1 or node.attributes()["axis"] % 2 != 1:
            refuse(node, "joins tensors that are not feature vectors")
        if name in importer.constants:
            width = shape[1]
            parts.append(View(None, np.zeros(width, dtype=int), np.zeros(width),
                              importer.constant_array(name).reshape(-1), None))
        elif name in importer.views:
            parts.append(importer.views[name])
        elif (isinstance(producer, GemmOp) and importer.only_use(name)
              and not producer.attrs.get("label") and name not in importer.annotations):
            parts.append(producer)
        else:
            width = importer.graph.tensor(name).numel
            parts.append(View(name, np.arange(width), np.ones(width), np.zeros(width), None))
    sources = {p.source if isinstance(p, View) else p.inputs[0] for p in parts} - {None}
    if len(sources) != 1:
        refuse(node, "joins features of more than one live tensor")
    source = sources.pop()
    width = importer.graph.tensor(source).numel
    if not any(isinstance(p, GemmOp) for p in parts):
        joined = [np.concatenate([getattr(p, key) for p in parts])
                  for key in ("index", "mul", "add")]
        importer.views[node.output[0]] = View(source, *joined, node.name)
        importer.shapes[node.output[0]] = (1, len(joined[0]))
        return
    weights, biases = [], []
    for part in parts:
        if isinstance(part, View):
            weights.append(importer.selection(part, width))
            biases.append(part.add)
        else:
            weights.append(importer.graph.tensor(part.inputs[1]).data)
            biases.append(importer.graph.tensor(part.inputs[2]).data.reshape(-1)
                          if len(part.inputs) > 2 else np.zeros(weights[-1].shape[1]))
            importer.graph.remove_node(part)
    importer.dense(node.name or "concat", source, np.hstack(weights), np.concatenate(biases),
                   node.output[0])


def import_relu(node, importer):
    importer.emit(ReluOp(node.name or "relu", [importer.rename(node.input[0])],
                         [importer.rename(node.output[0])]))


def import_binarizer(node, importer):
    importer.emit(BinarizerOp(node.name or "binarizer", [importer.rename(node.input[0])],
                              [importer.rename(node.output[0])],
                              threshold=float(node.attributes().get("threshold", 0.0))))


def import_maxpool(node, importer):
    importer.emit(MaxPoolOp(node.name or "pool", [importer.rename(node.input[0])],
                            [importer.rename(node.output[0])],
                            **window_attrs(node, importer, (2, 2))))


def import_add(node, importer):
    inputs = []
    for name in node.input:
        if importer.is_constant(name):
            array = importer.constant_array(name)
            inputs.append(importer.derive_constant(name, np.squeeze(array)))
        else:
            inputs.append(importer.rename(name))
    importer.emit(AddOp(node.name or "add", inputs, [importer.rename(node.output[0])]))


def import_trees(node, importer):
    """TreeEnsembleRegressor with one target, or a binary TreeEnsembleClassifier, from ai.onnx.ml 1 or 3."""
    attrs = node.attributes()
    op, name = node.op_type, node.name or node.op_type
    classifier = op == "TreeEnsembleClassifier"

    def floats(key):
        if key + "_as_tensor" in attrs:
            return tensor_to_numpy(attrs[key + "_as_tensor"]).astype(np.float64).reshape(-1)
        return np.asarray(attrs.get(key, []), dtype=np.float64)

    def refuse(text):
        raise UnsupportedOpError(op, name, message="%s %r %s" % (op, name, text))

    transforms = ("NONE", "LOGISTIC") if classifier else ("NONE",)
    transform = attrs.get("post_transform", "NONE")
    if transform not in transforms:
        refuse("has post_transform %s, and only %s is supported"
               % (transform, " or ".join(transforms)))
    aggregate = attrs.get("aggregate_function", "SUM")
    if aggregate not in ("SUM", "AVERAGE"):
        refuse("has aggregate_function %s, and only SUM and AVERAGE are supported" % aggregate)
    unknown = sorted(set(attrs["nodes_modes"]) - set(MODES))
    if unknown:
        refuse("has a node mode %s that is not supported" % unknown[0])
    prefix = "class" if classifier else "target"
    ids = attrs.get(prefix + "_ids", [])
    if classifier:
        labels = attrs.get("classlabels_int64s") or attrs.get("classlabels_strings") or []
        if len(labels) != 2 or len(set(ids)) > 1:
            refuse("has %d classes scored %d ways, and only binary classifiers with one "
                   "score are supported" % (len(labels), len(set(ids))))
    elif int(attrs.get("n_targets", 1)) != 1:
        refuse("has %d targets, and only one is supported" % int(attrs["n_targets"]))
    base = floats("base_values")
    if len(base) > 1:
        refuse("has %d base values, and only one is supported" % len(base))

    trees = attrs["nodes_treeids"]
    index = {key: k for k, key in enumerate(zip(trees, attrs["nodes_nodeids"]))}
    mode = np.array([MODES.index(m) for m in attrs["nodes_modes"]])
    count = len(mode)
    if_true, if_false = np.arange(count), np.arange(count)
    for k in np.flatnonzero(mode != LEAF):
        if_true[k] = index[(trees[k], attrs["nodes_truenodeids"][k])]
        if_false[k] = index[(trees[k], attrs["nodes_falsenodeids"][k])]
    children = set(if_true[mode != LEAF]) | set(if_false[mode != LEAF])
    roots = {}
    for k, tree in enumerate(trees):
        if k not in children:
            roots.setdefault(tree, k)
    value = np.zeros(count)
    for tree, nodeid, weight in zip(attrs[prefix + "_treeids"], attrs[prefix + "_nodeids"],
                                    floats(prefix + "_weights")):
        value[index[(tree, nodeid)]] += weight
    if aggregate == "AVERAGE":
        value /= len(roots)
    ensemble = Ensemble([roots[tree] for tree in sorted(roots)], attrs["nodes_featureids"],
                        floats("nodes_values"), mode, if_true, if_false, value)
    # ONNX Runtime labels a binary one score classifier on the raw score, at 0.5 when no weight is negative.
    decision = None
    if classifier:
        decision = 0.5 if np.all(floats("class_weights") >= 0) else 0.0
        importer.drop(node.output[1])
    importer.emit(TreeEnsembleOp(name, [importer.rename(node.input[0])],
                                 [importer.rename(node.output[0])], ensemble,
                                 base[0] if len(base) else 0.0, decision, op))


def refuse_tree_ensemble(node, importer):
    raise UnsupportedOpError(node.op_type, node.name, message=(
        "TreeEnsemble %r is ai.onnx.ml opset 5, which is not supported yet; export "
        "TreeEnsembleRegressor or TreeEnsembleClassifier instead" % node.name))


LAYOUT_OPS = ("Flatten", "Reshape", "Squeeze", "Unsqueeze", "Transpose", "Slice", "Gather",
              "Expand", "Dropout", "ArrayFeatureExtractor")

BUILTIN = {
    "Conv": import_conv, "Gemm": import_gemm, "MatMul": import_gemm,
    "Relu": import_relu, "MaxPool": import_maxpool, "Add": import_add,
    "QuantizeLinear": import_qdq, "DequantizeLinear": import_qdq,
    "QLinearMatMul": import_qlinear, "QLinearConv": import_qlinear,
    "QLinearAdd": import_qlinear, "MatMulInteger": import_integer,
    "ConvInteger": import_integer, "DynamicQuantizeLinear": refuse_dynamic,
    "Identity": import_identity, "ZipMap": import_identity, "Cast": import_cast,
    "Concat": import_concat, "BatchNormalization": import_batchnorm,
    "LinearRegressor": import_linear, "LinearClassifier": import_linear,
    "Scaler": import_scaler, "Binarizer": import_binarizer, "Normalizer": import_normalizer,
    "TreeEnsembleRegressor": import_trees, "TreeEnsembleClassifier": import_trees,
    "TreeEnsemble": refuse_tree_ensemble,
    **{op: import_elementwise for op in FUNCTIONS if op not in ("Relu", "Add")},
    **dict.fromkeys(REDUCE, import_reduce), **dict.fromkeys(SOFTMAX, import_softmax),
}
BUILTIN.update(dict.fromkeys(LAYOUT_OPS, import_layout))

# Rules that read a view themselves; any other rule gets it built first.
ABSORBING = {import_gemm, import_linear, import_scaler, import_batchnorm, import_layout,
             import_cast, import_concat, import_identity}
ARITHMETIC = ("Add", "Sub", "Mul", "Div")


class OnnxImporter:
    def __init__(self, model, handlers=()):
        """`handlers` are OpHandlers, and take over any built in op type they name."""
        self.model = model
        self.graph = Graph(model.graph.name or "imported")
        self.rules = dict(BUILTIN)
        for handler in handlers:
            for op_type in handler.op_types:
                self.rules[op_type] = handler.handle
        self._renames = {}
        self.annotations = {}
        self.dropped = set()
        self.constants = {}
        self.views = {}
        self.shapes = {}
        self.uses = collections.Counter()
        self.reads = collections.Counter()

    def run(self):
        for name in self.model.graph_inputs():
            declared = self.model.value_shape(name)
            if declared is None:
                raise ModelError("graph input %r has no static shape" % name)
            self.graph.layouts[name] = LayoutAdapter(len(declared))
            dims = [1 if isinstance(d, str) else int(d) for d in declared]
            if len(dims) == 4:
                dims = [dims[0], dims[2], dims[3], dims[1]]
            self.graph.add_tensor(Tensor(name, dims, FLOAT32))
            self.graph.inputs.append(name)
        for tensor in self.model.graph.initializer:
            self.constants[tensor.name] = tensor_to_numpy(tensor)
        nodes = self.model.nodes()
        # A QDQ pair on live data is an alias, not another reader.
        self.uses.update(name for node in nodes if node.op_type not in QDQ
                         for name in node.input if name)
        self.uses.update(self.model.graph_outputs())
        for node in nodes:
            self.visit(node)
        for name in self.model.graph_outputs():
            name = self.rename(name)
            if name in self.views:
                self.materialize(name)
            if name not in self.dropped:
                self.graph.outputs.append(name)
        self.graph.annotations = {self.rename(k): v for k, v in self.annotations.items()}
        self.graph.infer()
        return self.graph

    def visit(self, node):
        inputs = [name for name in node.input if name]
        if any(self.rename(name) in self.dropped for name in inputs):
            for name in node.output:
                self.drop(name)
            return
        live = [name for name in inputs if not self.is_constant(name)]
        # QDQ nodes carry annotations, so they are handled even on constants.
        if node.op_type in FOLD and node.op_type not in QDQ and (
                not live or node.op_type in ("Shape", "Size")):
            self.fold(node)
            return
        if node.op_type in ARITHMETIC and self.arithmetic(node):
            return
        rule = self.rules.get(node.op_type)
        if rule is None:
            raise UnsupportedOpError(node.op_type, node.name, sorted(self.rules))
        if rule not in ABSORBING:
            for name in live:
                if self.rename(name) in self.views:
                    self.materialize(self.rename(name))
        rule(node, self)

    def fold(self, node):
        values = []
        for name in node.input:
            if not name:
                values.append(None)
            elif self.is_constant(name):
                values.append(self.constants[self.rename(name)])
            else:
                values.append(np.zeros(self.onnx_shape(name)))
        for name, value in zip(node.output, evaluate(node, values)):
            self.constants[name] = value
            source = self.rename(node.input[0]) if node.input else None
            if node.op_type in LAYOUT and source in self.annotations:
                self.annotations[name] = self.annotations[source]

    def arithmetic(self, node):
        """A constant Add, Sub, Mul or Div along the features folds into a neighbouring layer."""
        if len(node.input) != 2 or self.is_constant(node.input[0]) == self.is_constant(
                node.input[1]):
            return False
        first = not self.is_constant(node.input[0])
        live, other = node.input if first else node.input[::-1]
        shape = self.onnx_shape(live)
        value = self.constant_array(other)
        if np.broadcast_shapes(value.shape, shape) != tuple(shape):
            return False
        axis = 1 if len(shape) == 4 else len(shape) - 1
        rows = np.moveaxis(np.broadcast_to(value, shape), axis, -1).reshape(-1, shape[axis])
        if np.any(rows != rows[0]) or (node.op_type == "Div" and not first):
            return False
        vector, ones, zeros = rows[0], np.ones(shape[axis]), np.zeros(shape[axis])
        mul, add = {"Add": (ones, vector), "Mul": (vector, zeros), "Div": (1 / vector, zeros),
                    "Sub": (ones, -vector) if first else (-ones, vector)}[node.op_type]
        self.scale_shift(node.output[0], live, mul, add, node.name)
        return True

    def scale_shift(self, out, name, mul, add, origin):
        """out = mul * name + add per channel: into the Gemm or Conv before, a view, or a 1x1 Conv."""
        source = self.rename(name)
        shape = self.onnx_shape(name)
        channels = shape[1] if len(shape) == 4 else shape[-1]
        mul, add = np.broadcast_to(mul, (channels,)), np.broadcast_to(add, (channels,))
        if len(shape) not in (2, 4) or shape[0] != 1:
            raise UnsupportedOpError(origin, origin, message="%r scales a %dD tensor, and only "
                                     "feature vectors and maps are supported"
                                     % (origin, len(shape)))
        view = self.views.get(source)
        producer = self.graph.producer(source)
        if view is not None:
            axis = [1, -1] + [1] * (len(shape) - 2)
            mul, add = (np.broadcast_to(v.reshape(axis), shape).reshape(-1) for v in (mul, add))
            self.views[out] = View(view.source, view.index, view.mul * mul,
                                   view.add * mul + add, view.origin)
        elif isinstance(producer, (GemmOp, ConvOp)) and self.only_use(source) and not (
                producer.attrs.get("label")):
            if np.any(mul != 1):
                weights = self.graph.tensor(producer.inputs[1])
                weights.data = weights.data * mul
                self.annotations.pop(producer.inputs[1], None)
            if len(producer.inputs) > 2:
                bias = self.graph.tensor(producer.inputs[2])
                bias.data = bias.data.reshape(-1) * mul + add
                bias.shape = bias.data.shape
            else:
                producer.inputs.append(self.derive_constant(source + "_bias", add, grid=False))
            for key in [k for k in self.annotations if self.rename(k) == source]:
                del self.annotations[key]
            self.alias(out, name)
        elif len(shape) == 4:
            kernel = np.diag(mul)[None, None]
            self.emit(ConvOp(origin, [source, self.derive_constant(out + "_kernel", kernel,
                                                                   grid=False),
                                      self.derive_constant(out + "_bias", add, grid=False)],
                             [out], kernel_shape=(1, 1), strides=(1, 1), pads=(0, 0, 0, 0),
                             dilations=(1, 1)))
        else:
            self.views[out] = View(source, np.arange(channels), mul, add, origin)
            self.shapes[out] = shape

    def rearrange(self, node, moved):
        """`moved` holds, in the output's ONNX shape, the ONNX index of each input element."""
        source, out = self.rename(node.input[0]), node.output[0]
        shape = self.onnx_shape(node.input[0])
        view = self.views.get(source)
        # A view keeps ONNX order; a real map is stored channel-last.
        order = np.arange(int(np.prod(shape)))
        if view is None and len(shape) == 4:
            n, c, h, w = shape
            order = order.reshape(n, h, w, c).transpose(0, 3, 1, 2).reshape(-1)
        index = order[moved.reshape(-1)]
        stored = order[(moved.transpose(0, 2, 3, 1) if moved.ndim == 4 else moved).reshape(-1)]
        self.shapes[out] = moved.shape
        if view is not None:
            self.views[out] = View(view.source, view.index[index], view.mul[index],
                                   view.add[index], view.origin)
        elif not np.array_equal(stored, np.arange(order.size)):
            if len(shape) == 4:
                flat = self.graph.fresh_name(out + "_nhwc")
                self.emit(FlattenOp(node.name or "flatten", [source], [flat]))
                source = flat
            self.views[out] = View(source, index, np.ones(len(index)), np.zeros(len(index)),
                                   node.name)
        elif moved.ndim != 4 and len(shape) != 4:
            self.alias(out, node.input[0])
            self.shapes[out] = moved.shape
        else:
            dims = moved.shape if moved.ndim != 4 else moved.transpose(0, 2, 3, 1).shape
            self.emit(FlattenOp(node.name or "reshape", [source], [out],
                                shape=dims if moved.ndim > 1 else (1,) + dims))

    def dense(self, name, x, weights, bias, out, grid_from=None, **attrs):
        """A Gemm on live tensor x, taking in x's view if it has one."""
        source = self.rename(x)
        view = self.views.get(source)
        if view is not None:
            width = self.graph.tensor(view.source).numel
            bias = (0 if bias is None else bias) + view.add @ weights
            weights = self.selection(view, width) @ weights
            source, grid_from = view.source, None
        inputs = [source, self.derive_constant(grid_from or name + "_weights", weights,
                                               grid=grid_from is not None)]
        if bias is not None:
            inputs.append(self.derive_constant(name + "_bias", np.broadcast_to(
                np.asarray(bias, dtype=np.float64).reshape(-1), weights.shape[1:]), grid=False))
        self.emit(GemmOp(name, inputs, [self.rename(out)], **attrs))

    def selection(self, view, width):
        """The view as a matrix; scaling the model's own inputs asks for a grid per feature."""
        if view.source in self.graph.inputs and (np.any(view.mul != 1) or np.any(view.add)):
            self.graph.feature_grids.add(view.source)
        matrix = np.zeros((width, len(view.index)))
        matrix[view.index, np.arange(len(view.index))] = view.mul
        return matrix

    def materialize(self, name):
        """A view nothing absorbed becomes a Gemm of its own, if it is a feature vector."""
        view = self.views.pop(name)
        if len(self.shapes[name]) > 2:
            raise UnsupportedOpError(view.origin, view.origin, message=(
                "%r reorders live data into a %dD tensor, which only a Gemm reading it as a "
                "feature vector can absorb" % (view.origin, len(self.shapes[name]))))
        width = self.graph.tensor(view.source).numel
        if len(self.shapes[name]) == 2 and np.array_equal(view.index, np.arange(width)) and (
                np.any(view.mul != 1) or np.any(view.add)):
            return self.affine(name, view)
        self.emit(GemmOp(view.origin or "view", [
            view.source, self.derive_constant(name + "_weights", self.selection(view, width),
                                              grid=False),
            self.derive_constant(name + "_bias", view.add, grid=False)], [name]))
        return name

    def affine(self, name, view):
        """A view that only scales and shifts each feature is a table per lane, not a Gemm."""
        origin = view.origin or "view"

        def const(tag, value):
            return self.derive_constant(name + "_" + tag, value.reshape(1, -1), grid=False)

        if np.all(view.mul == -1):
            steps = [("Sub", [const("add", view.add), None])]
        else:
            steps = [(op, [None, const(op.lower(), value)])
                     for op, value in (("Mul", view.mul), ("Add", view.add))
                     if np.any(value != (op == "Mul"))]
        source = view.source
        for k, (op, inputs) in enumerate(steps):
            last = k == len(steps) - 1
            out = name if last else self.graph.fresh_name(name + "_scaled")
            inputs = [source if i is None else i for i in inputs]
            self.emit(ElementwiseOp(origin if last else origin + "_scale", inputs, [out], op,
                                    function_attrs={}))
            source = out
        return name

    def only_use(self, name):
        """One reader, not counting the aliases whose own readers are counted instead."""
        uses = sum(count for raw, count in self.uses.items() if self.rename(raw) == name)
        return uses - self.reads[name] == 1

    def onnx_shape(self, name):
        """The shape ONNX gives tensor `name`, which is NCHW where the graph keeps NHWC."""
        if name in self.shapes:
            return tuple(self.shapes[name])
        shape = tuple(self.graph.tensor(self.rename(name)).shape)
        return (shape[0], shape[3], shape[1], shape[2]) if len(shape) == 4 else shape

    def emit(self, node):
        """Shapes are inferred as each node lands, since later ops need them for padding."""
        names = {n.name for n in self.graph.nodes}
        if node.name in names:
            node.name = next("%s_%d" % (node.name, k) for k in range(1, len(names) + 2)
                             if "%s_%d" % (node.name, k) not in names)
        for name in node.outputs:
            self.graph.ensure_tensor(name)
        self.graph.add_node(node)
        node.infer(self.graph)

    def rename(self, name):
        return self._renames.get(name, name)

    def is_constant(self, name):
        return self.rename(name) in self.constants

    def constant_array(self, name):
        """From the model, or folded from it at import."""
        resolved = self.rename(name)
        if resolved in self.constants:
            return np.asarray(self.constants[resolved], dtype=np.float64)
        if self.graph.has_tensor(resolved) and self.graph.tensor(resolved).is_constant:
            return np.asarray(self.graph.tensor(resolved).data, dtype=np.float64)
        raise ModelError("expected %r to be a constant initializer" % name)

    def derive_constant(self, base, array, grid=True):
        """A reshaped or transposed copy of a constant, keeping its quantization grid."""
        name = self.graph.fresh_name(base + "_cl")
        self.graph.add_tensor(Tensor.constant(name, np.asarray(array, dtype=np.float64)))
        for candidate in (base, self.rename(base)) if grid else ():
            if candidate in self.annotations:
                self.annotations[name] = self.annotations[candidate]
                break
        return name

    def drop(self, name):
        """An output the hardware does not compute, such as a classifier's probabilities."""
        self.dropped.add(name)

    def alias(self, name, source, reads=True):
        """`name` is `source`, both as ONNX names, with source's shape unless told otherwise."""
        target = self.rename(source)
        self._renames[name] = target
        self.reads[target] += reads
        if source in self.shapes:
            self.shapes[name] = self.shapes[source]

    def annotate(self, name, spec):
        self.annotations[name] = spec
