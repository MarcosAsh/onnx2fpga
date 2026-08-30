"""Measures activation ranges on the float graph so scales can be fixed."""

import numpy as np

from ..reference.runner import GraphRunner
from .plan import QuantizationPlan


class RangeObserver:
    """Symmetric dynamic range of every activation seen so far.

    Tensors named in `per_feature` get one range per column instead of one for
    the whole tensor. That is worth doing for an engineered feature vector,
    whose columns can differ by orders of magnitude — a single scale fitted to
    the largest of them leaves the small ones with almost no integers to sit
    on. It is not worth doing for a convolution's activations, where the
    channels are comparable by construction, so it is opt in rather than the
    default."""

    def __init__(self, per_feature=()):
        self.peaks = {}
        self.per_feature = set(per_feature)

    def __call__(self, name, array):
        values = np.abs(np.asarray(array, dtype=np.float64))
        if name in self.per_feature and values.ndim:
            peak = np.max(values, axis=tuple(range(values.ndim - 1)))
        else:
            peak = float(np.max(values))
        seen = self.peaks.get(name)
        self.peaks[name] = peak if seen is None else np.maximum(seen, peak)

    def observe_inputs(self, feeds):
        for name, array in feeds.items():
            self(name, array)


class Calibrator:
    def __init__(self, graph, observer=None, per_feature_inputs=False):
        self.graph = graph
        self.observer = observer or RangeObserver(
            per_feature=graph.inputs if per_feature_inputs else ())

    def run(self, samples):
        runner = GraphRunner(self.graph)
        for feeds in samples:
            self.observer.observe_inputs(feeds)
            runner.run(feeds, capture=self.observer)
        return self.observer.peaks

    def plan(self, samples, **kwargs):
        return QuantizationPlan.from_peaks(self.run(samples), **kwargs)
