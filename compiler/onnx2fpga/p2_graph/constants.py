"""Numpy evaluation of the ONNX ops that appear on constants, for folding at import."""

import numpy as np

from ..p1_ingest import TensorProto
from ..p1_ingest.onnx_model import tensor_to_numpy


def ints(array):
    return [int(v) for v in np.asarray(array).reshape(-1)]


def axes(attrs, inputs, index):
    """From the attribute before opset 13, else from an optional input."""
    if "axes" in attrs:
        return list(attrs["axes"])
    if len(inputs) > index and inputs[index] is not None:
        return ints(inputs[index])
    return None


def constant(attrs, inputs):
    if "value" in attrs:
        return tensor_to_numpy(attrs["value"])
    for key, dtype in (("value_float", np.float32), ("value_floats", np.float32),
                       ("value_int", np.int64), ("value_ints", np.int64)):
        if key in attrs:
            return np.asarray(attrs[key], dtype=dtype)
    raise NotImplementedError("Constant with %s" % sorted(attrs))


def shape(attrs, inputs):
    dims = np.asarray(inputs[0].shape, dtype=np.int64)
    return dims[attrs.get("start", 0):attrs.get("end", len(dims))]


def squeeze(attrs, inputs):
    x, chosen = inputs[0], axes(attrs, inputs, 1)
    return np.squeeze(x) if chosen is None else np.squeeze(
        x, axis=tuple(a % x.ndim for a in chosen))


def unsqueeze(attrs, inputs):
    x, chosen = inputs[0], axes(attrs, inputs, 1)
    for axis in sorted(a % (x.ndim + len(chosen)) for a in chosen):
        x = np.expand_dims(x, axis)
    return x


def reshape(attrs, inputs):
    x, dims = inputs[0], ints(inputs[1])
    if not attrs.get("allowzero", 0):
        dims = [x.shape[i] if d == 0 else d for i, d in enumerate(dims)]
    return x.reshape(dims)


def flatten(attrs, inputs):
    x = inputs[0]
    axis = attrs.get("axis", 1) % (x.ndim + 1)
    return x.reshape(int(np.prod(x.shape[:axis])), -1)


def transpose(attrs, inputs):
    return np.transpose(inputs[0], attrs.get("perm"))


def slice_(attrs, inputs):
    x = inputs[0]
    if "starts" in attrs:
        starts, ends, chosen, steps = attrs["starts"], attrs["ends"], attrs.get("axes"), None
    else:
        starts, ends = ints(inputs[1]), ints(inputs[2])
        chosen = axes({}, inputs, 3)
        steps = ints(inputs[4]) if len(inputs) > 4 and inputs[4] is not None else None
    chosen = range(len(starts)) if chosen is None else chosen
    steps = steps or [1] * len(starts)
    index = [slice(None)] * x.ndim
    for start, end, axis, step in zip(starts, ends, chosen, steps):
        index[axis] = slice(int(start), int(end), int(step))
    return x[tuple(index)]


def expand(attrs, inputs):
    x = inputs[0]
    return np.broadcast_to(x, np.broadcast_shapes(x.shape, tuple(ints(inputs[1])))).copy()


def constant_of_shape(attrs, inputs):
    value = tensor_to_numpy(attrs["value"]) if "value" in attrs else np.zeros(1, np.float32)
    return np.full(ints(inputs[0]), value.reshape(-1)[0], dtype=value.dtype)


def cast(attrs, inputs):
    return inputs[0].astype(TensorProto.NUMPY_DTYPE[attrs["to"]])


def divide(a, b):
    """ONNX truncates an integer division toward zero."""
    if np.issubdtype(a.dtype, np.integer):
        return np.trunc(a / b).astype(a.dtype)
    return a / b


def one(function):
    return lambda attrs, inputs: function(*inputs)


FOLD = {
    "Constant": constant, "Shape": shape, "Size": lambda attrs, inputs: np.int64(inputs[0].size),
    "Cast": cast, "Squeeze": squeeze, "Unsqueeze": unsqueeze, "Reshape": reshape,
    "Flatten": flatten, "Transpose": transpose, "Slice": slice_, "Expand": expand,
    "Gather": lambda attrs, inputs: np.take(inputs[0], inputs[1], axis=attrs.get("axis", 0)),
    "ArrayFeatureExtractor": lambda attrs, inputs: np.take(
        inputs[0], inputs[1].reshape(-1), axis=-1),
    "Concat": lambda attrs, inputs: np.concatenate(inputs, axis=attrs["axis"]),
    "ConstantOfShape": constant_of_shape,
    "Range": one(lambda start, limit, delta: np.arange(start, limit, delta, dtype=start.dtype)),
    "Identity": one(lambda x: x), "Dropout": lambda attrs, inputs: [inputs[0]],
    "Add": one(np.add), "Sub": one(np.subtract), "Mul": one(np.multiply), "Div": one(divide),
    "Pow": one(np.power), "Neg": one(np.negative), "Abs": one(np.abs), "Sqrt": one(np.sqrt),
    "Reciprocal": one(np.reciprocal), "Floor": one(np.floor), "Ceil": one(np.ceil),
    "MatMul": one(np.matmul), "Equal": one(np.equal), "Less": one(np.less),
    "Greater": one(np.greater), "Not": one(np.logical_not), "Where": one(np.where),
    "Max": one(lambda *xs: np.maximum.reduce(xs)), "Min": one(lambda *xs: np.minimum.reduce(xs)),
}

# Ops that only move elements, so a quantized constant keeps its grid through them.
LAYOUT = {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose"}


def evaluate(node, inputs):
    """The outputs of `node` on numpy `inputs`, None standing for an omitted input."""
    result = FOLD[node.op_type](node.attributes(), inputs)
    return list(result) if isinstance(result, list) else [np.asarray(result)]
