"""Runs a graph in numpy: calibration on float graphs, the golden model on hardware graphs."""

import numpy as np


class GraphRunner:
    def __init__(self, graph):
        self.graph = graph

    def run(self, feeds, capture=None, bounds=None):
        """Every tensor's value by name, clipped to `bounds` if given; `capture` sees each one."""
        bounds = bounds or {}
        values = {name: np.clip(array, *bounds[name]) if name in bounds else array
                  for name, array in feeds.items()}
        for node in self.graph.nodes:
            for name in node.inputs:
                tensor = self.graph.tensor(name)
                if tensor.is_constant:
                    values.setdefault(name, tensor.data)
        for node in self.graph.topological_order():
            for name, array in zip(node.outputs, node.execute(self.graph, values)):
                if name in bounds:
                    array = np.clip(array, *bounds[name])
                values[name] = array
                if capture is not None:
                    capture(name, array)
        return values
