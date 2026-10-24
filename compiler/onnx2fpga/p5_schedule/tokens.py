"""Buffer sizing from an as-soon-as-possible schedule: peak edge occupancy, since undersizing deadlocks forks."""


class EdgeOccupancy:
    """Peak number of beats in flight on one edge."""

    def __init__(self, produced, consumed):
        self.produced = produced
        self.consumed = consumed

    @property
    def peak(self):
        """Occupancy only rises at a production, and both sequences are sorted, so one walk does it."""
        peak, taken = 0, 0
        for index, moment in enumerate(self.produced):
            while taken < len(self.consumed) and self.consumed[taken] <= moment:
                taken += 1
            peak = max(peak, index + 1 - taken)
        return peak


class TokenAnalysis:
    def __init__(self, graph):
        self.graph = graph
        self.arrivals = {}
        self.edges = {}

    def run(self):
        graph = self.graph
        for name in graph.inputs:
            tensor = graph.tensor(name)
            beats = tensor.numel // max(tensor.elems_per_beat, 1)
            self.arrivals[name] = list(range(beats))

        for node in graph.topological_order():
            start = self._start_of(node)
            for index, name in enumerate(node.outputs):
                offsets = node.output_schedule(graph, index)
                self.arrivals[name] = [start + offset for offset in offsets]

        for node in graph.topological_order():
            for index, name in enumerate(node.stream_inputs(graph)):
                if name not in self.arrivals:
                    continue
                start = self._start_of(node)
                consumed = [start + offset
                            for offset in node.input_schedule(graph, index)]
                self.edges[name] = EdgeOccupancy(self.arrivals[name], consumed)
        return self.edges

    def _start_of(self, node):
        """Earliest the unit can begin without waiting mid stream: every beat has arrived when wanted."""
        start = 0
        for index, name in enumerate(node.stream_inputs(self.graph)):
            arrivals = self.arrivals.get(name)
            if not arrivals:
                continue
            offsets = node.input_schedule(self.graph, index)
            for arrival, offset in zip(arrivals, offsets):
                start = max(start, arrival - offset)
        return start

    def depth_for(self, name):
        edge = self.edges.get(name)
        return 2 if edge is None else max(2, edge.peak)
