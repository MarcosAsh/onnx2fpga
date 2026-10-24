"""The subset of onnx.proto3 the compiler reads and writes; unknown fields are skipped."""

import struct

from .protobuf_codec import Decoder, Encoder, LEN

PACKABLE = ("varint", "float", "double")


def signed64(raw):
    return raw - (1 << 64) if raw >= (1 << 63) else raw


class Field:
    def __init__(self, number, name, kind, repeated=False, message=None):
        self.number = number
        self.name = name
        self.kind = kind
        self.repeated = repeated
        self.message = message

    def default(self):
        if self.repeated:
            return []
        return {"varint": 0, "float": 0.0, "double": 0.0,
                "string": "", "bytes": b"", "message": None}[self.kind]


class Message:
    FIELDS = ()

    def __init_subclass__(cls):
        cls.BY_NUMBER = {f.number: f for f in cls.FIELDS}

    def __init__(self, **kwargs):
        for field in self.FIELDS:
            setattr(self, field.name, field.default())
        for name, value in kwargs.items():
            if not any(f.name == name for f in self.FIELDS):
                raise AttributeError("%s has no field %r" % (type(self).__name__, name))
            setattr(self, name, value)

    @classmethod
    def decode(cls, buf, start=0, end=None):
        message = cls()
        decoder = Decoder(buf, start, end)
        for number, wire_type in decoder.fields():
            field = cls.BY_NUMBER.get(number)
            if field is None:
                decoder.skip(wire_type)
            elif field.repeated and field.kind in PACKABLE and wire_type == LEN:
                message._read_packed(decoder.blob(), field)
            elif field.repeated:
                getattr(message, field.name).append(message._read_value(decoder, field))
            else:
                setattr(message, field.name, message._read_value(decoder, field))
        return message

    @staticmethod
    def _read_value(decoder, field):
        if field.kind == "varint":
            return signed64(decoder.varint())
        if field.kind == "float":
            return struct.unpack("<f", decoder.fixed32())[0]
        if field.kind == "double":
            return struct.unpack("<d", decoder.fixed64())[0]
        if field.kind == "string":
            return decoder.blob().decode("utf-8")
        if field.kind == "bytes":
            return bytes(decoder.blob())
        return field.message.decode(decoder.blob())

    def _read_packed(self, blob, field):
        target = getattr(self, field.name)
        if field.kind == "float":
            target.extend(struct.unpack("<%df" % (len(blob) // 4), blob))
        elif field.kind == "double":
            target.extend(struct.unpack("<%dd" % (len(blob) // 8), blob))
        else:
            inner = Decoder(blob)
            while not inner.exhausted:
                target.append(signed64(inner.varint()))

    def encode(self):
        encoder = Encoder()
        for field in self.FIELDS:
            value = getattr(self, field.name)
            if field.repeated and field.kind == "varint":
                if value:
                    encoder.packed_varints(field.number, value)
            elif field.repeated and field.kind in ("float", "double"):
                if value:
                    encoder.packed_field(field.number, value, field.kind[0])
            elif field.repeated:
                for item in value:
                    self._write(encoder, field, item)
            elif value not in (None, "", b"", 0):
                self._write(encoder, field, value)
        return encoder.to_bytes()

    @staticmethod
    def _write(encoder, field, value):
        if field.kind == "varint":
            encoder.varint_field(field.number, value)
        elif field.kind == "float":
            encoder.float_field(field.number, value)
        elif field.kind == "string":
            encoder.string_field(field.number, value)
        elif field.kind == "bytes":
            encoder.blob_field(field.number, value)
        elif field.kind == "message":
            encoder.blob_field(field.number, value.encode())
        else:
            raise ValueError("cannot encode a single %s field" % field.kind)


class TensorProto(Message):
    FIELDS = (
        Field(1, "dims", "varint", repeated=True),
        Field(2, "data_type", "varint"),
        Field(4, "float_data", "float", repeated=True),
        Field(5, "int32_data", "varint", repeated=True),
        Field(7, "int64_data", "varint", repeated=True),
        Field(8, "name", "string"),
        Field(9, "raw_data", "bytes"),
        Field(10, "double_data", "double", repeated=True),
        Field(11, "uint64_data", "varint", repeated=True),
    )

    UNDEFINED, FLOAT, UINT8, INT8, UINT16, INT16, INT32, INT64 = range(8)
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32, UINT64 = 12, 13

    NUMPY_DTYPE = {
        FLOAT: "<f4", UINT8: "|u1", INT8: "|i1", UINT16: "<u2", INT16: "<i2",
        INT32: "<i4", INT64: "<i8", BOOL: "|b1", FLOAT16: "<f2", DOUBLE: "<f8",
        UINT32: "<u4", UINT64: "<u8",
    }


class TensorShapeDimension(Message):
    FIELDS = (
        Field(1, "dim_value", "varint"),
        Field(2, "dim_param", "string"),
    )


class TensorShapeProto(Message):
    FIELDS = (Field(1, "dim", "message", repeated=True, message=TensorShapeDimension),)


class TypeTensor(Message):
    FIELDS = (
        Field(1, "elem_type", "varint"),
        Field(2, "shape", "message", message=TensorShapeProto),
    )


class TypeProto(Message):
    FIELDS = (Field(1, "tensor_type", "message", message=TypeTensor),)


class ValueInfoProto(Message):
    FIELDS = (
        Field(1, "name", "string"),
        Field(2, "type", "message", message=TypeProto),
    )


class AttributeProto(Message):
    UNDEFINED, FLOAT, INT, STRING, TENSOR, GRAPH = range(6)
    FLOATS, INTS, STRINGS, TENSORS, GRAPHS = range(6, 11)

    FIELDS = (
        Field(1, "name", "string"),
        Field(2, "f", "float"),
        Field(3, "i", "varint"),
        Field(4, "s", "bytes"),
        Field(5, "t", "message", message=TensorProto),
        Field(7, "floats", "float", repeated=True),
        Field(8, "ints", "varint", repeated=True),
        Field(9, "strings", "bytes", repeated=True),
        Field(20, "type", "varint"),
    )

    def value(self):
        if self.type == self.FLOAT:
            return self.f
        if self.type == self.INT:
            return self.i
        if self.type == self.STRING:
            return self.s.decode("utf-8")
        if self.type == self.TENSOR:
            return self.t
        if self.type == self.FLOATS:
            return list(self.floats)
        if self.type == self.INTS:
            return list(self.ints)
        if self.type == self.STRINGS:
            return [s.decode("utf-8") for s in self.strings]
        return None


class NodeProto(Message):
    FIELDS = (
        Field(1, "input", "string", repeated=True),
        Field(2, "output", "string", repeated=True),
        Field(3, "name", "string"),
        Field(4, "op_type", "string"),
        Field(5, "attribute", "message", repeated=True, message=AttributeProto),
        Field(7, "domain", "string"),
    )

    def attributes(self):
        return {a.name: a.value() for a in self.attribute}


class GraphProto(Message):
    FIELDS = (
        Field(1, "node", "message", repeated=True, message=NodeProto),
        Field(2, "name", "string"),
        Field(5, "initializer", "message", repeated=True, message=TensorProto),
        Field(11, "input", "message", repeated=True, message=ValueInfoProto),
        Field(12, "output", "message", repeated=True, message=ValueInfoProto),
        Field(13, "value_info", "message", repeated=True, message=ValueInfoProto),
    )


class OperatorSetIdProto(Message):
    FIELDS = (
        Field(1, "domain", "string"),
        Field(2, "version", "varint"),
    )


class ModelProto(Message):
    FIELDS = (
        Field(1, "ir_version", "varint"),
        Field(2, "producer_name", "string"),
        Field(3, "producer_version", "string"),
        Field(7, "graph", "message", message=GraphProto),
        Field(8, "opset_import", "message", repeated=True, message=OperatorSetIdProto),
    )
