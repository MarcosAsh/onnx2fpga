"""Measures activation ranges on the float graph so scales can be fixed."""

import numpy as np

from ..reference.runner import GraphRunner
from .plan import QuantizationPlan


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
        runner = GraphRunner(self.graph)
        for feeds in samples:
            for name, array in feeds.items():
                self.observe(name, array)
            runner.run(feeds, capture=self.observe)
        return QuantizationPlan.from_peaks(self.peaks, mult_bits=mult_bits)
