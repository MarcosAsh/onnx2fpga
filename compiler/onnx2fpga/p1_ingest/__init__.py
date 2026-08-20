"""Stage 1: read a .onnx file.

ONNX is protobuf. Rather than depend on the 100 MB reference onnx package this
stage carries its own decoder, so the compiler runs with nothing but Python and
numpy installed.

protobuf_codec  the wire format
onnx_schema     the subset of onnx.proto3 this compiler reads and writes
onnx_model      numpy-facing view over the protos, plus a builder for fixtures
"""

from .onnx_model import ModelBuilder, OnnxModel, TensorCodec
from .onnx_schema import (AttributeProto, GraphProto, ModelProto, NodeProto,
                          TensorProto, ValueInfoProto)

__all__ = ["ModelBuilder", "OnnxModel", "TensorCodec", "AttributeProto",
           "GraphProto", "ModelProto", "NodeProto", "TensorProto", "ValueInfoProto"]
