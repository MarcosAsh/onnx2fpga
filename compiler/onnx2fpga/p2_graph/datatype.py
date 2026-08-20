"""Datatypes carried on graph edges. Everything past quantization is integer."""

import numpy as np


class DataType:
    _registry = {}

    def __init__(self, name, bits):
        self.name = name
        self.bits = bits

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, DataType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    @property
    def is_integer(self):
        return False

    @property
    def signed(self):
        raise NotImplementedError

    def clamp(self, values):
        raise NotImplementedError

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    @classmethod
    def register(cls, instance):
        cls._registry[instance.name] = instance
        return instance


class IntType(DataType):
    def __init__(self, bits, signed):
        DataType.__init__(self, ("INT%d" if signed else "UINT%d") % bits, bits)
        self._signed = signed

    @property
    def is_integer(self):
        return True

    @property
    def signed(self):
        return self._signed

    @property
    def min(self):
        return -(1 << (self.bits - 1)) if self._signed else 0

    @property
    def max(self):
        return (1 << (self.bits - 1)) - 1 if self._signed else (1 << self.bits) - 1

    def numpy_dtype(self):
        for width in (8, 16, 32, 64):
            if self.bits <= width:
                return np.dtype(("i%d" if self._signed else "u%d") % (width // 8))
        return np.dtype("int64")

    def clamp(self, values):
        return np.clip(values, self.min, self.max).astype(np.int64)

    def to_unsigned_bits(self, values):
        """Two's complement bit pattern, for hex weight files and stream packing."""
        mask = (1 << self.bits) - 1
        return np.asarray(values, dtype=np.int64) & mask

    @staticmethod
    def smallest_for(low, high):
        if low >= 0:
            bits = max(1, int(high).bit_length())
            return IntType(bits, False)
        bits = 1
        while not (-(1 << (bits - 1)) <= low and high <= (1 << (bits - 1)) - 1):
            bits += 1
        return IntType(bits, True)


class FloatType(DataType):
    def __init__(self, bits=32):
        DataType.__init__(self, "FLOAT%d" % bits, bits)

    @property
    def signed(self):
        return True

    def numpy_dtype(self):
        return np.dtype("float%d" % self.bits)

    def clamp(self, values):
        return values


INT4 = DataType.register(IntType(4, True))
INT8 = DataType.register(IntType(8, True))
INT16 = DataType.register(IntType(16, True))
INT32 = DataType.register(IntType(32, True))
UINT8 = DataType.register(IntType(8, False))
FLOAT32 = DataType.register(FloatType(32))


class QuantSpec:
    """Affine quantization: real = scale * (integer - zero_point).

    ONNX writes unsigned activations as often as signed ones. Rather than carry
    signedness through the whole datapath, an unsigned spec is rebased onto the
    signed grid: value u with zero point z is exactly value u-128 with zero
    point z-128, same scale. The shift is a boundary concern, recorded here so
    the host can apply it, and nothing inside the accelerator sees it.
    """

    REBASE = 128

    def __init__(self, scale, zero_point=0, axis=None, dtype=None, rebased=False):
        self.scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
        self.zero_point = np.atleast_1d(np.asarray(zero_point, dtype=np.int64))
        self.axis = axis
        self.dtype = dtype
        self.rebased = rebased

    def to_signed(self):
        """Signed equivalent of this spec, shifting the zero point if the
        source grid was unsigned."""
        if self.dtype is None or self.dtype.signed:
            return self
        return QuantSpec(self.scale, self.zero_point - self.REBASE, self.axis,
                         IntType(self.dtype.bits, True), rebased=True)

    @property
    def offset(self):
        """What the host must subtract from an incoming value, and add to an
        outgoing one, to reach the grid the hardware works on."""
        return self.REBASE if self.rebased else 0

    @property
    def per_channel(self):
        return self.scale.size > 1

    def dequantize(self, integers):
        return self.scale * (np.asarray(integers, dtype=np.float64) - self.zero_point)

    def quantize(self, reals, dtype):
        raw = np.rint(np.asarray(reals, dtype=np.float64) / self.scale) + self.zero_point
        return dtype.clamp(raw)

    @property
    def is_symmetric(self):
        return bool(np.all(self.zero_point == 0))

    def __repr__(self):
        tail = ", rebased" if self.rebased else ""
        if self.per_channel:
            return "QuantSpec(per-channel[%d], zp=%d%s)" % (
                self.scale.size, int(self.zero_point.flat[0]), tail)
        return "QuantSpec(scale=%.6g, zp=%d%s)" % (
            float(self.scale.flat[0]), int(self.zero_point.flat[0]), tail)
