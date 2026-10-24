"""ONNX to the internal IR, transposing every weight and shape to channel-last once, here."""

import numpy as np

from ..errors import ModelError, UnsupportedOpError
from ..p1_ingest import TensorProto
from ..p1_ingest.onnx_model import tensor_to_numpy
from ..p2_graph.datatype import FLOAT32, IntType, QuantSpec, UINT8
from ..p2_graph.graph import Graph
from ..p2_graph.layout import LayoutAdapter
from ..p2_graph.tensor import Tensor
from ..p3_ops.onnx_ops import AddOp, ConvOp, FlattenOp, GemmOp, MaxPoolOp, ReluOp
from ..p3_ops.trees import LEAF, MODES, Ensemble, TreeEnsembleOp

QUANTIZED = {TensorProto.INT8: IntType(8, True), TensorProto.UINT8: IntType(8, False)}
QDQ = ("QuantizeLinear", "DequantizeLinear")


class OpHandler:
    """Translates the ONNX op types in `op_types` with handle(node, importer)."""

    op_types = ()

    def handle(self, node, importer):
        raise NotImplementedError


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
    importer.alias(node.output[0], importer.rename(node.input[0]))


def import_qdq(node, importer):
    """Records the integer grid of a tensor and collapses to an alias carrying it."""
    scale = importer.constant_array(node.input[1])
    zero_point = np.zeros_like(scale, dtype=np.int64)
    dtype = UINT8
    if len(node.input) > 2 and node.input[2]:
        zero_point = importer.constant_array(node.input[2]).astype(np.int64)
        dtype = QUANTIZED.get(importer.model.initializer_dtype(node.input[2]), UINT8)
    spec = QuantSpec(scale, zero_point, dtype)
    source = importer.rename(node.input[0])
    target = node.output[0]
    if (importer.model.has_initializer(source)
            and importer.model.initializer_dtype(source) in QUANTIZED):
        raw = importer.model.initializer(source).astype(np.float64)
        importer.add_constant(target, spec.dequantize(raw))
        importer.annotate(target, spec)
        return
    importer.alias(target, source)
    importer.annotate(source, spec)
    importer.annotate(target, spec)


def matmul_inputs(node, importer, weights):
    inputs = [importer.rename(node.input[0]), importer.derive_constant(node.input[1], weights)]
    if len(node.input) > 2:
        inputs.append(importer.derive_constant(node.input[2],
                                               importer.constant_array(node.input[2])))
    return inputs


def import_conv(node, importer):
    if int(node.attributes().get("group", 1)) != 1:
        raise ModelError("grouped convolution is not supported yet: %s" % node.name)
    weights = importer.constant_array(node.input[1])
    attrs = window_attrs(node, importer, weights.shape[2:])
    inputs = matmul_inputs(node, importer, np.transpose(weights, (2, 3, 1, 0)))
    importer.emit(ConvOp(node.name or "conv", inputs, [importer.rename(node.output[0])],
                         **attrs))


def import_gemm(node, importer):
    attrs = node.attributes()
    weights = importer.constant_array(node.input[1])
    if node.op_type == "Gemm":
        if float(attrs.get("alpha", 1.0)) != 1.0 or float(attrs.get("beta", 1.0)) != 1.0:
            raise ModelError("Gemm alpha/beta scaling is not supported: %s" % node.name)
        if int(attrs.get("transA", 0)):
            raise ModelError("Gemm transA is not supported: %s" % node.name)
        if int(attrs.get("transB", 0)):
            weights = weights.T
    importer.emit(GemmOp(node.name or "gemm", matmul_inputs(node, importer, weights),
                         [importer.rename(node.output[0])]))


def import_relu(node, importer):
    importer.emit(ReluOp(node.name or "relu", [importer.rename(node.input[0])],
                         [importer.rename(node.output[0])]))


def import_maxpool(node, importer):
    importer.emit(MaxPoolOp(node.name or "pool", [importer.rename(node.input[0])],
                            [importer.rename(node.output[0])],
                            **window_attrs(node, importer, (2, 2))))


def import_add(node, importer):
    inputs = []
    for name in node.input:
        if importer.model.has_initializer(importer.rename(name)):
            array = importer.constant_array(name)
            inputs.append(importer.derive_constant(name, np.squeeze(array)))
        else:
            inputs.append(importer.rename(name))
    importer.emit(AddOp(node.name or "add", inputs, [importer.rename(node.output[0])]))


def import_flatten(node, importer):
    importer.emit(FlattenOp(node.name or "flatten", [importer.rename(node.input[0])],
                            [importer.rename(node.output[0])]))


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


def drop_zipmap(node, importer):
    importer.drop(node.output[0])


BUILTIN = {
    "Conv": import_conv, "Gemm": import_gemm, "MatMul": import_gemm,
    "Relu": import_relu, "MaxPool": import_maxpool, "Add": import_add,
    "Flatten": import_flatten, "Reshape": import_flatten, "Squeeze": import_flatten,
    "QuantizeLinear": import_qdq, "DequantizeLinear": import_qdq,
    "Identity": import_identity,
    "TreeEnsembleRegressor": import_trees, "TreeEnsembleClassifier": import_trees,
    "TreeEnsemble": refuse_tree_ensemble, "ZipMap": drop_zipmap,
}


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
        for node in self.model.nodes():
            # QDQ nodes carry annotations, so they are handled even on constants.
            if (node.op_type not in QDQ and node.output
                    and all(self.model.has_initializer(i) for i in node.input)):
                continue
            rule = self.rules.get(node.op_type)
            if rule is None:
                raise UnsupportedOpError(node.op_type, node.name, sorted(self.rules))
            rule(node, self)
        for name in self.model.graph_outputs():
            if self.rename(name) not in self.dropped:
                self.graph.outputs.append(self.rename(name))
        self.graph.annotations = {self.rename(k): v for k, v in self.annotations.items()}
        self.graph.infer()
        return self.graph

    def emit(self, node):
        """Shapes are inferred as each node lands, since later ops need them for padding."""
        for name in node.outputs:
            self.graph.ensure_tensor(name)
        self.graph.add_node(node)
        node.infer(self.graph)

    def rename(self, name):
        return self._renames.get(name, name)

    def constant_array(self, name):
        """From the model, or from a DequantizeLinear already folded into the graph."""
        resolved = self.rename(name)
        for candidate in (resolved, name):
            if self.graph.has_tensor(candidate) and self.graph.tensor(candidate).is_constant:
                return np.asarray(self.graph.tensor(candidate).data, dtype=np.float64)
        if self.model.has_initializer(resolved):
            return self.model.initializer(resolved).astype(np.float64)
        raise ModelError("expected %r to be a constant initializer" % name)

    def add_constant(self, name, array):
        if not self.graph.has_tensor(name):
            self.graph.add_tensor(Tensor.constant(name, np.asarray(array, dtype=np.float64)))

    def derive_constant(self, base, array):
        """A reshaped or transposed copy of a constant, keeping its quantization grid."""
        name = self.graph.fresh_name(base + "_cl")
        self.graph.add_tensor(Tensor.constant(name, np.asarray(array, dtype=np.float64)))
        for candidate in (base, self.rename(base)):
            if candidate in self.annotations:
                self.annotations[name] = self.annotations[candidate]
                break
        return name

    def drop(self, name):
        """An output the hardware does not compute, such as a classifier's probabilities."""
        self.dropped.add(name)

    def alias(self, name, target):
        self._renames[name] = target

    def annotate(self, name, spec):
        self.annotations[name] = spec
