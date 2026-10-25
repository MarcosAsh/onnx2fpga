"""A numpy view of the ONNX protos, and a builder for small models."""

import numpy as np

from .onnx_schema import (AttributeProto, GraphProto, ModelProto, NodeProto,
                          OperatorSetIdProto, TensorProto, TensorShapeDimension,
                          TensorShapeProto, TypeProto, TypeTensor, ValueInfoProto)

_NUMPY_TO_ONNX = {
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float64"): TensorProto.DOUBLE,
    np.dtype("int8"): TensorProto.INT8,
    np.dtype("uint8"): TensorProto.UINT8,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("int64"): TensorProto.INT64,
    np.dtype("bool"): TensorProto.BOOL,
}


def tensor_to_numpy(proto):
    dtype = TensorProto.NUMPY_DTYPE.get(proto.data_type)
    if dtype is None:
        raise NotImplementedError("ONNX data_type %d" % proto.data_type)
    if proto.raw_data:
        array = np.frombuffer(proto.raw_data, dtype=dtype)
    else:
        typed = (proto.float_data or proto.double_data or proto.int64_data
                 or proto.int32_data or proto.uint64_data)
        array = np.asarray(typed, dtype=dtype)
    return array.reshape(tuple(int(d) for d in proto.dims))


def tensor_from_numpy(array, name):
    array = np.ascontiguousarray(array)
    data_type = _NUMPY_TO_ONNX.get(array.dtype)
    if data_type is None:
        raise NotImplementedError("numpy dtype %s" % array.dtype)
    return TensorProto(name=name, data_type=data_type,
                       dims=list(array.shape), raw_data=array.tobytes())


class OnnxModel:
    def __init__(self, proto):
        self.proto = proto
        self._initializers = {t.name: t for t in proto.graph.initializer}

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            return cls(ModelProto.decode(handle.read()))

    def save(self, path):
        with open(path, "wb") as handle:
            handle.write(self.proto.encode())

    @property
    def graph(self):
        return self.proto.graph

    @property
    def opset(self):
        for entry in self.proto.opset_import:
            if entry.domain in ("", "ai.onnx"):
                return int(entry.version)
        return 0

    def nodes(self):
        return list(self.graph.node)

    def has_initializer(self, name):
        return name in self._initializers

    def initializer(self, name):
        return tensor_to_numpy(self._initializers[name])

    def initializer_dtype(self, name):
        return int(self._initializers[name].data_type)

    def value_shape(self, name):
        for collection in (self.graph.input, self.graph.output, self.graph.value_info):
            for info in collection:
                if info.name != name:
                    continue
                tensor = info.type and info.type.tensor_type
                if tensor is None or tensor.shape is None:
                    return None
                return tuple(int(d.dim_value) if d.dim_value else d.dim_param or 1
                             for d in tensor.shape.dim)
        return None

    def graph_inputs(self):
        return [i.name for i in self.graph.input if i.name not in self._initializers]

    def graph_outputs(self):
        return [o.name for o in self.graph.output]


class ModelBuilder:
    """Builds ONNX models without the onnx package."""

    def __init__(self, name="model", opset=13, ir_version=8):
        self._graph = GraphProto(name=name)
        self._opset = opset
        self._ir_version = ir_version

    def add_input(self, name, shape, elem_type=TensorProto.FLOAT):
        self._graph.input.append(self._value_info(name, shape, elem_type))
        return self

    def add_output(self, name, shape, elem_type=TensorProto.FLOAT):
        self._graph.output.append(self._value_info(name, shape, elem_type))
        return self

    def add_initializer(self, name, array):
        self._graph.initializer.append(tensor_from_numpy(array, name))
        return self

    def add_node(self, op_type, inputs, outputs, name=None, **attributes):
        node = NodeProto(op_type=op_type, input=list(inputs), output=list(outputs),
                         name=name or "%s_%d" % (op_type.lower(), len(self._graph.node)))
        for key, value in attributes.items():
            node.attribute.append(self._attribute(key, value))
        self._graph.node.append(node)
        return self

    def build(self):
        model = ModelProto(ir_version=self._ir_version, producer_name="onnx2fpga",
                           graph=self._graph)
        model.opset_import.append(OperatorSetIdProto(domain="", version=self._opset))
        return OnnxModel(model)

    @staticmethod
    def _value_info(name, shape, elem_type):
        dims = TensorShapeProto(dim=[
            TensorShapeDimension(dim_param=extent) if isinstance(extent, str)
            else TensorShapeDimension(dim_value=int(extent)) for extent in shape])
        return ValueInfoProto(name=name,
                              type=TypeProto(tensor_type=TypeTensor(elem_type=elem_type,
                                                                    shape=dims)))

    @staticmethod
    def _attribute(name, value):
        if isinstance(value, bool):
            return AttributeProto(name=name, type=AttributeProto.INT, i=int(value))
        if isinstance(value, int):
            return AttributeProto(name=name, type=AttributeProto.INT, i=value)
        if isinstance(value, float):
            return AttributeProto(name=name, type=AttributeProto.FLOAT, f=value)
        if isinstance(value, str):
            return AttributeProto(name=name, type=AttributeProto.STRING,
                                  s=value.encode("utf-8"))
        if isinstance(value, np.ndarray):
            return AttributeProto(name=name, type=AttributeProto.TENSOR,
                                  t=tensor_from_numpy(value, name))
        if isinstance(value, (list, tuple)):
            return AttributeProto(name=name, type=AttributeProto.INTS,
                                  ints=[int(v) for v in value])
        raise NotImplementedError("attribute %r of type %s" % (name, type(value)))
