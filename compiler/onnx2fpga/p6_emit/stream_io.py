"""Conversion between a logical tensor and the beats that carry it.

The stream order is the tensor flattened channel-last. Each beat holds
`elems_per_beat` consecutive elements with element 0 in the least significant
bits, which is the layout every RTL module slices with.
"""

import numpy as np

from .memories import HexImage


class StreamCodec:
    def __init__(self, tensor):
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.elems_per_beat = tensor.elems_per_beat

    @property
    def beats(self):
        count = 1
        for extent in self.shape:
            count *= int(extent)
        return count // self.elems_per_beat

    def to_image(self, array):
        flat = np.asarray(array, dtype=np.int64).reshape(-1)
        if flat.size % self.elems_per_beat:
            raise ValueError("tensor of %d elements does not divide into beats of %d"
                             % (flat.size, self.elems_per_beat))
        return HexImage.packed(flat.reshape(-1, self.elems_per_beat), self.dtype.bits)

    def from_words(self, words):
        mask = (1 << self.dtype.bits) - 1
        sign = 1 << (self.dtype.bits - 1)
        values = []
        for word in words:
            for lane in range(self.elems_per_beat):
                raw = (word >> (lane * self.dtype.bits)) & mask
                values.append((raw ^ sign) - sign if self.dtype.signed else raw)
        return np.asarray(values, dtype=np.int64).reshape(self.shape)
