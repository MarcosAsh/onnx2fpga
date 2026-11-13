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
    """Use the full code range for nonnegative tensors; symmetric grids otherwise."""

    def __init__(self, graph):
        self.graph = graph
        self.peaks = {}
        self.minima = {}

    def observe(self, name, array):
        """Infinities and NaN, as from Log at 0, say nothing about the range and are skipped."""
        array = np.asarray(array, dtype=np.float64)
        finite = array[np.isfinite(array)]
        peak = float(np.max(np.abs(finite), initial=0.0))
        self.peaks[name] = max(self.peaks.get(name, 0.0), peak)
        self.minima[name] = min(self.minima.get(name, np.inf),
                                float(np.min(finite, initial=np.inf)))

    def plan(self, samples, mult_bits=18):
        """An input the model scales itself, read only by Gemms, gets a grid per feature."""
        runner = GraphRunner(self.graph)
        for feeds in samples:
            for name, array in feeds.items():
                self.observe(name, array)
            runner.run(feeds, capture=self.observe)
        plan = QuantizationPlan.from_peaks(self.peaks, mult_bits=mult_bits)
        for name, peak in self.peaks.items():
            if self.minima[name] >= 0 and peak > 0:
                dtype = plan.act_dtype
                plan.annotate(name, QuantSpec(peak / (dtype.max - dtype.min), dtype.min))
        for name in self.graph.feature_grids:
            if all(node.op_type == "Gemm" for node in self.graph.consumers(name)):
                values = np.concatenate([np.reshape(feeds[name], (-1, feeds[name].shape[-1]))
                                         for feeds in samples])
                plan.annotate(name, feature_grid(values, plan.act_dtype))
        return plan
