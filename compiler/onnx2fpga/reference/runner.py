"""Executes a graph in numpy: calibration on float graphs, the bit exact golden model on hardware graphs."""

import numpy as np


class GraphRunner:
    def __init__(self, graph):
        self.graph = graph

    def run(self, feeds, capture=None):
        values = dict(feeds)
        for name, tensor in list(self._constants()):
            values.setdefault(name, tensor.data)
        for node in self.graph.topological_order():
            produced = node.execute(self.graph, values)
            for name, array in zip(node.outputs, produced):
                values[name] = array
                if capture is not None:
                    capture(name, array)
        return values

    def outputs(self, feeds):
        values = self.run(feeds)
        return [values[name] for name in self.graph.outputs]

    def _constants(self):
        for node in self.graph.nodes:
            for name in node.inputs:
                tensor = self.graph.tensor(name)
                if tensor.is_constant:
                    yield name, tensor
