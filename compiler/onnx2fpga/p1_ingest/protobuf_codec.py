"""Protobuf wire-format codec. Enough of proto3 to read and write ONNX."""

import struct

VARINT = 0
FIXED64 = 1
LEN = 2
FIXED32 = 5


class WireError(Exception):
    pass


class Decoder:
    def __init__(self, buf, start=0, end=None):
        self._buf = buf
        self._pos = start
        self._end = len(buf) if end is None else end

    @property
    def exhausted(self):
        return self._pos >= self._end

    def varint(self):
        result = 0
        shift = 0
        while True:
            if self._pos >= self._end:
                raise WireError("truncated varint")
            byte = self._buf[self._pos]
            self._pos += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
            if shift > 63:
                raise WireError("varint too long")

    def zigzag(self):
        raw = self.varint()
        return (raw >> 1) ^ -(raw & 1)

    def fixed32(self):
        value = self._buf[self._pos:self._pos + 4]
        self._pos += 4
        return value

    def fixed64(self):
        value = self._buf[self._pos:self._pos + 8]
        self._pos += 8
        return value

    def blob(self):
        length = self.varint()
        if self._pos + length > self._end:
            raise WireError("truncated length-delimited field")
        value = self._buf[self._pos:self._pos + length]
        self._pos += length
        return value

    def skip(self, wire_type):
        if wire_type == VARINT:
            self.varint()
        elif wire_type == FIXED64:
            self._pos += 8
        elif wire_type == LEN:
            length = self.varint()
            self._pos += length
        elif wire_type == FIXED32:
            self._pos += 4
        else:
            raise WireError("unsupported wire type %d" % wire_type)

    def fields(self):
        while not self.exhausted:
            key = self.varint()
            yield key >> 3, key & 0x07


class Encoder:
    def __init__(self):
        self._parts = []

    def to_bytes(self):
        return b"".join(self._parts)

    def _raw(self, data):
        self._parts.append(data)

    def _varint(self, value):
        if value < 0:
            value += 1 << 64
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
        self._raw(bytes(out))

    def _key(self, number, wire_type):
        self._varint((number << 3) | wire_type)

    def varint_field(self, number, value):
        self._key(number, VARINT)
        self._varint(value)

    def blob_field(self, number, data):
        self._key(number, LEN)
        self._varint(len(data))
        self._raw(data)

    def string_field(self, number, text):
        self.blob_field(number, text.encode("utf-8"))

    def float_field(self, number, value):
        self._key(number, FIXED32)
        self._raw(struct.pack("<f", value))

    def packed_field(self, number, values, fmt):
        self.blob_field(number, struct.pack("<%d%s" % (len(values), fmt), *values))

    def packed_varints(self, number, values):
        inner = Encoder()
        for value in values:
            inner._varint(value)
        self.blob_field(number, inner.to_bytes())
