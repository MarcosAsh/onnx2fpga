"""Measures activation ranges on the float graph so scales can be fixed."""

import numpy as np

from ..reference.runner import GraphRunner
from .plan import QuantizationPlan


class RangeObserver:
    """Symmetric dynamic range of every activation seen so far."""

    def __init__(self):
        self.peaks = {}

    def __call__(self, name, array):
        peak = float(np.max(np.abs(np.asarray(array, dtype=np.float64))))
        self.peaks[name] = max(self.peaks.get(name, 0.0), peak)

    def observe_inputs(self, feeds):
        for name, array in feeds.items():
            self(name, array)


class Calibrator:
    def __init__(self, graph, observer=None):
        self.graph = graph
        self.observer = observer or RangeObserver()

    def run(self, samples):
        runner = GraphRunner(self.graph)
        for feeds in samples:
            self.observer.observe_inputs(feeds)
            runner.run(feeds, capture=self.observer)
        return self.observer.peaks

    def plan(self, samples, **kwargs):
        return QuantizationPlan.from_peaks(self.run(samples), **kwargs)
