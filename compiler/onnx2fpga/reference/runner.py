"""Runs a graph in numpy: calibration on float graphs, the golden model on hardware graphs."""


class GraphRunner:
    def __init__(self, graph):
        self.graph = graph

    def run(self, feeds, capture=None):
        """Every tensor's value by name; `capture(name, array)` sees each node output."""
        values = dict(feeds)
        for node in self.graph.nodes:
            for name in node.inputs:
                tensor = self.graph.tensor(name)
                if tensor.is_constant:
                    values.setdefault(name, tensor.data)
        for node in self.graph.topological_order():
            for name, array in zip(node.outputs, node.execute(self.graph, values)):
                values[name] = array
                if capture is not None:
                    capture(name, array)
        return values
