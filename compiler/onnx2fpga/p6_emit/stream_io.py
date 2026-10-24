"""Conversion between a tensor and its beats: channel-last, element 0 in the low bits."""

import numpy as np

from .memories import HexImage


class StreamCodec:
    def __init__(self, tensor):
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.elems_per_beat = tensor.elems_per_beat

    def to_image(self, array):
        flat = np.asarray(array, dtype=np.int64).reshape(-1, self.elems_per_beat)
        return HexImage.packed(flat, self.dtype.bits)

    def from_words(self, words):
        bits = self.dtype.bits
        mask = (1 << bits) - 1
        sign = 1 << (bits - 1)
        values = []
        for word in words:
            for lane in range(self.elems_per_beat):
                raw = (word >> (lane * bits)) & mask
                values.append((raw ^ sign) - sign if self.dtype.signed else raw)
        return np.asarray(values, dtype=np.int64).reshape(self.shape)
