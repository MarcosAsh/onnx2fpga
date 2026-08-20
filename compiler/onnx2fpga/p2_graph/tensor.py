"""Graph edges. In the generated hardware every non-constant tensor is a stream."""

import numpy as np

from .datatype import FLOAT32, DataType


class Tensor:
    def __init__(self, name, shape=None, dtype=FLOAT32, data=None, quant=None):
        self.name = name
        self.shape = tuple(shape) if shape is not None else None
        self.dtype = dtype
        self.data = data
        self.quant = quant
        self.elems_per_beat = 1

    @classmethod
    def constant(cls, name, array, dtype=None, quant=None):
        array = np.asarray(array)
        return cls(name, array.shape, dtype or FLOAT32, data=array, quant=quant)

    @property
    def is_constant(self):
        return self.data is not None

    @property
    def numel(self):
        if self.shape is None:
            return None
        count = 1
        for extent in self.shape:
            count *= int(extent)
        return count

    @property
    def channels(self):
        """Innermost dimension, which is the axis folded across PE/SIMD lanes."""
        return int(self.shape[-1]) if self.shape else 1

    @property
    def stream_width(self):
        return self.elems_per_beat * self.dtype.bits

    @property
    def beats(self):
        return self.numel // self.elems_per_beat

    def clone(self, name):
        copy = Tensor(name, self.shape, self.dtype, self.data, self.quant)
        copy.elems_per_beat = self.elems_per_beat
        return copy

    def __repr__(self):
        kind = "const" if self.is_constant else "stream"
        return "Tensor(%s, %s, %s, %s)" % (self.name, self.shape, self.dtype, kind)
