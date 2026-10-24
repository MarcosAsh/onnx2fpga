"""The layout of a graph input; everything inside the compiler is channel-last."""

import numpy as np


class LayoutAdapter:
    def __init__(self, rank):
        self.permutation = (0, 2, 3, 1) if rank == 4 else None
        self.source = "NCHW" if rank == 4 else "channel-last"

    def to_internal(self, array):
        return self._transpose(array, self.permutation)

    def to_external(self, array):
        inverse = tuple(np.argsort(self.permutation)) if self.permutation else None
        return self._transpose(array, inverse)

    def _transpose(self, array, order):
        array = np.asarray(array)
        if order is None or array.ndim != len(order):
            return array
        return np.transpose(array, order)

    def describe(self):
        return {"source_layout": self.source, "internal_layout": "channel-last",
                "permutation": list(self.permutation) if self.permutation else None}
