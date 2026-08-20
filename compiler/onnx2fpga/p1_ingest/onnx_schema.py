"""ONNX message schema, expressed against the wire codec in wire.py.

Field numbers come from onnx/onnx.proto3. Only the subset the compiler reads
or writes is declared; unknown fields are skipped on decode and dropped on
re-encode.
"""

import struct

from .protobuf_codec import Decoder, Encoder, FIXED32, FIXED64, LEN, VARINT


class Field:
    def __init__(self, number, name, kind, repeated=False, message=None):
        self.number = number
        self.name = name
        self.kind = kind
        self.repeated = repeated
        self.message = message

    @property
    def is_scalar_list(self):
        return self.repeated and self.kind in ("varint", "float", "double")

    def default(self):
        if self.repeated:
            return []
        return {"varint": 0, "float": 0.0, "double": 0.0,
                "string": "", "bytes": b"", "message": None}[self.kind]


class Message:
    FIELDS = ()

    def __init__(self, **kwargs):
        for field in self.FIELDS:
            setattr(self, field.name, field.default())
        for name, value in kwargs.items():
            if not any(f.name == name for f in self.FIELDS):
                raise AttributeError("%s has no field %r" % (type(self).__name__, name))
            setattr(self, name, value)

    @classmethod
    def _by_number(cls):
        cached = cls.__dict__.get("_FIELD_MAP")
        if cached is None:
            cached = {f.number: f for f in cls.FIELDS}
            setattr(cls, "_FIELD_MAP", cached)
        return cached

    @classmethod
    def decode(cls, buf, start=0, end=None):
        instance = cls()
        instance._read(Decoder(buf, start, end))
        return instance

    def _read(self, decoder):
        table = type(self)._by_number()
        for number, wire_type in decoder.fields():
            field = table.get(number)
            if field is None:
                decoder.skip(wire_type)
                continue
            self._read_field(decoder, field, wire_type)

    def _read_field(self, decoder, field, wire_type):
        if field.is_scalar_list and wire_type == LEN:
            self._read_packed(decoder.blob(), field)
            return
        value = self._read_scalar(decoder, field, wire_type)
        if field.repeated:
            getattr(self, field.name).append(value)
        else:
            setattr(self, field.name, value)

    def _read_scalar(self, decoder, field, wire_type):
        if field.kind == "varint":
            raw = decoder.varint()
            return raw - (1 << 64) if raw >= (1 << 63) else raw
        if field.kind == "float":
            return struct.unpack("<f", decoder.fixed32())[0]
        if field.kind == "double":
            return struct.unpack("<d", decoder.fixed64())[0]
        if field.kind == "string":
            return decoder.blob().decode("utf-8")
        if field.kind == "bytes":
            return bytes(decoder.blob())
        if field.kind == "message":
            blob = decoder.blob()
            return field.message.decode(blob)
        raise ValueError("unhandled kind %r" % field.kind)

    def _read_packed(self, blob, field):
        target = getattr(self, field.name)
        if field.kind == "float":
            target.extend(struct.unpack("<%df" % (len(blob) // 4), blob))
        elif field.kind == "double":
            target.extend(struct.unpack("<%dd" % (len(blob) // 8), blob))
        else:
            inner = Decoder(blob)
            while not inner.exhausted:
                raw = inner.varint()
                target.append(raw - (1 << 64) if raw >= (1 << 63) else raw)

    def encode(self):
        encoder = Encoder()
        for field in self.FIELDS:
            value = getattr(self, field.name)
            if field.repeated:
                if not value:
                    continue
                self._write_repeated(encoder, field, value)
            elif value not in (None, "", b"", 0, 0.0):
                self._write_scalar(encoder, field, value)
        return encoder.to_bytes()

    def _write_repeated(self, encoder, field, values):
        if field.kind == "varint":
            encoder.packed_varints(field.number, values)
        elif field.kind == "float":
            encoder.packed_field(field.number, values, "f")
        elif field.kind == "double":
            encoder.packed_field(field.number, values, "d")
        else:
            for item in values:
                self._write_scalar(encoder, field, item)

    def _write_scalar(self, encoder, field, value):
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
            raise ValueError("unhandled kind %r" % field.kind)

    def __repr__(self):
        shown = []
        for field in self.FIELDS:
            value = getattr(self, field.name)
            if value not in (None, "", b"", 0, 0.0, []):
                shown.append("%s=%r" % (field.name, value))
        return "%s(%s)" % (type(self).__name__, ", ".join(shown))


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
        return {
            self.FLOAT: lambda: self.f,
            self.INT: lambda: self.i,
            self.STRING: lambda: self.s.decode("utf-8"),
            self.TENSOR: lambda: self.t,
            self.FLOATS: lambda: list(self.floats),
            self.INTS: lambda: list(self.ints),
            self.STRINGS: lambda: [s.decode("utf-8") for s in self.strings],
        }.get(self.type, lambda: None)()


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
