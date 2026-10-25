"""Measures activation ranges on the float graph so scales can be fixed."""

import numpy as np

from ..p2_graph.datatype import QuantSpec
from ..reference.runner import GraphRunner
from .plan import QuantizationPlan


def feature_grid(values, dtype):
    """Asymmetric per feature, each spanning that feature's range over the samples."""
    low, high = values.min(axis=0), values.max(axis=0)
    span = np.where(high > low, high - low, np.maximum(np.abs(high), 1.0))
    scale = span / (dtype.max - dtype.min)
    return QuantSpec(scale, np.rint(dtype.min - low / scale))


class Calibrator:
    """Symmetric quantization from the peak magnitude of every activation over the samples."""

    def __init__(self, graph):
        self.graph = graph
        self.peaks = {}

    def observe(self, name, array):
        """Infinities and NaN, as from Log at 0, say nothing about the range and are skipped."""
        array = np.asarray(array, dtype=np.float64)
        peak = float(np.max(np.abs(array[np.isfinite(array)]), initial=0.0))
        self.peaks[name] = max(self.peaks.get(name, 0.0), peak)

    def plan(self, samples, mult_bits=18):
        """An input the model scales itself, read only by Gemms, gets a grid per feature."""
        runner = GraphRunner(self.graph)
        for feeds in samples:
            for name, array in feeds.items():
                self.observe(name, array)
            runner.run(feeds, capture=self.observe)
        plan = QuantizationPlan.from_peaks(self.peaks, mult_bits=mult_bits)
        for name in self.graph.feature_grids:
            if all(node.op_type == "Gemm" for node in self.graph.consumers(name)):
                values = np.concatenate([np.reshape(feeds[name], (-1, feeds[name].shape[-1]))
                                         for feeds in samples])
                plan.annotate(name, feature_grid(values, plan.act_dtype))
        return plan
