"""Datatypes carried on graph edges. Everything past quantization is integer."""

import numpy as np


class DataType:
    _registry = {}
    is_integer = False

    def __init__(self, name, bits):
        self.name = name
        self.bits = bits

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, DataType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def register(cls, instance):
        cls._registry[instance.name] = instance
        return instance


class IntType(DataType):
    is_integer = True

    def __init__(self, bits, signed):
        DataType.__init__(self, ("INT%d" if signed else "UINT%d") % bits, bits)
        self.signed = signed

    @property
    def min(self):
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max(self):
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    def clamp(self, values):
        return np.clip(values, self.min, self.max).astype(np.int64)

    @staticmethod
    def smallest_for(low, high):
        if low >= 0:
            return IntType(max(1, int(high).bit_length()), False)
        bits = 1
        while not (-(1 << (bits - 1)) <= low and high <= (1 << (bits - 1)) - 1):
            bits += 1
        return IntType(bits, True)


class FloatType(DataType):
    signed = True

    def __init__(self, bits=32):
        DataType.__init__(self, "FLOAT%d" % bits, bits)

    def clamp(self, values):
        return values


INT4 = DataType.register(IntType(4, True))
INT8 = DataType.register(IntType(8, True))
INT16 = DataType.register(IntType(16, True))
INT32 = DataType.register(IntType(32, True))
UINT8 = DataType.register(IntType(8, False))
FLOAT32 = DataType.register(FloatType(32))


class QuantSpec:
    """Affine quantization, real = scale * (integer - zero_point)."""

    REBASE = 128

    def __init__(self, scale, zero_point=0, dtype=None, rebased=False):
        self.scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
        self.zero_point = np.atleast_1d(np.asarray(zero_point, dtype=np.int64))
        self.dtype = dtype
        self.rebased = rebased

    def to_signed(self):
        """The same grid on a signed type, shifting the zero point if it was unsigned."""
        if self.dtype is None or self.dtype.signed:
            return self
        return QuantSpec(self.scale, self.zero_point - self.REBASE,
                         IntType(self.dtype.bits, True), rebased=True)

    def dequantize(self, integers):
        return self.scale * (np.asarray(integers, dtype=np.float64) - self.zero_point)

    @property
    def is_symmetric(self):
        return bool(np.all(self.zero_point == 0))

    def __repr__(self):
        return "QuantSpec(scale=%s, zp=%s%s)" % (
            self.scale, self.zero_point, ", rebased" if self.rebased else "")
