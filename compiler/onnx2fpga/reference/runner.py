"""Executes a graph in numpy.

On the float graph this drives calibration; on the hardware graph it is the
golden model, where every operation is exact integer arithmetic and the
Verilator harness must reproduce it bit for bit."""

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
