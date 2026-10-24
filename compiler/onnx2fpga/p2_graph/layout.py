"""Tracks which layout a given input is in; everything inside the compiler is channel-last."""

import numpy as np


class LayoutAdapter:
    def __init__(self, permutation=None, source="unchanged"):
        self.permutation = tuple(permutation) if permutation else None
        self.source = source

    @classmethod
    def for_rank(cls, rank):
        if rank == 4:
            return cls((0, 2, 3, 1), source="NCHW")
        return cls(None, source="channel-last")

    @property
    def inverse(self):
        if self.permutation is None:
            return None
        order = [0] * len(self.permutation)
        for position, axis in enumerate(self.permutation):
            order[axis] = position
        return tuple(order)

    def to_internal(self, array):
        array = np.asarray(array)
        if self.permutation is None or array.ndim != len(self.permutation):
            return array
        return np.transpose(array, self.permutation)

    def to_external(self, array):
        array = np.asarray(array)
        if self.permutation is None or array.ndim != len(self.permutation):
            return array
        return np.transpose(array, self.inverse)

    def describe(self):
        return {"source_layout": self.source, "internal_layout": "channel-last",
                "permutation": list(self.permutation) if self.permutation else None}

    def __repr__(self):
        return "LayoutAdapter(%s -> channel-last)" % self.source
