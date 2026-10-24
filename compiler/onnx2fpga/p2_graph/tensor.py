"""Graph edges. In the generated hardware every non-constant tensor is a stream."""

import numpy as np

from .datatype import FLOAT32


class Tensor:
    def __init__(self, name, shape=None, dtype=FLOAT32, data=None):
        self.name = name
        self.shape = tuple(shape) if shape is not None else None
        self.dtype = dtype
        self.data = data
        self.elems_per_beat = 1

    @classmethod
    def constant(cls, name, array):
        array = np.asarray(array)
        return cls(name, array.shape, FLOAT32, data=array)

    @property
    def is_constant(self):
        return self.data is not None

    @property
    def numel(self):
        if self.shape is None:
            return None
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def stream_width(self):
        return self.elems_per_beat * self.dtype.bits

    @property
    def beats(self):
        return self.numel // self.elems_per_beat

    def __repr__(self):
        kind = "const" if self.is_constant else "stream"
        return "Tensor(%s, %s, %s, %s)" % (self.name, self.shape, self.dtype, kind)
