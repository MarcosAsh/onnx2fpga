"""Buffer sizing from an as-soon-as-possible schedule.

The question a FIFO answers is how far one producer can run ahead of its
consumer. On a straight chain that is small. Across a fork that rejoins, it is
the whole difference in depth between the two branches, because the join
cannot take a beat from the short side until the long side catches up, and
everything the short side produced in the meantime has to sit somewhere.

Undersizing there is not a slowdown, it is a deadlock: the short branch fills,
which stalls the fork, which stalls the long branch, which never reaches the
join. So the analysis below computes the peak occupancy of every edge under
unbounded buffers, and each FIFO is sized to it. Real backpressure can only
lower occupancy relative to that schedule, so the answer is safe.
"""


class EdgeOccupancy:
    """Peak number of beats in flight on one edge."""

    def __init__(self, name, produced, consumed):
        self.name = name
        self.produced = produced
        self.consumed = consumed

    @property
    def peak(self):
        """Occupancy only ever rises at a production, so checking those is
        enough. Both sequences are sorted, so one walk does it."""
        peak, taken = 0, 0
        for index, moment in enumerate(self.produced):
            while taken < len(self.consumed) and self.consumed[taken] <= moment:
                taken += 1
            peak = max(peak, index + 1 - taken)
        return peak

    def __repr__(self):
        return "EdgeOccupancy(%s, peak=%d)" % (self.name, self.peak)


class TokenAnalysis:
    def __init__(self, graph, source_rate=1):
        self.graph = graph
        self.source_rate = source_rate
        self.arrivals = {}
        self.edges = {}

    def run(self):
        graph = self.graph
        for name in graph.inputs:
            tensor = graph.tensor(name)
            beats = tensor.numel // max(tensor.elems_per_beat, 1)
            self.arrivals[name] = [i * self.source_rate for i in range(beats)]

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
                self.edges[name] = EdgeOccupancy(name, self.arrivals[name], consumed)
        return self.edges

    def _start_of(self, node):
        """Earliest the unit can begin without ever waiting mid stream: every
        beat must have arrived by the moment the schedule wants it."""
        start = 0
        for index, name in enumerate(node.stream_inputs(self.graph)):
            arrivals = self.arrivals.get(name)
            if not arrivals:
                continue
            offsets = node.input_schedule(self.graph, index)
            for arrival, offset in zip(arrivals, offsets):
                start = max(start, arrival - offset)
        return start

    def depth_for(self, tensor_name, minimum=2):
        edge = self.edges.get(tensor_name)
        return minimum if edge is None else max(minimum, edge.peak)

    def report(self):
        rows = ["%-28s %8s" % ("edge", "peak")]
        for name, edge in sorted(self.edges.items()):
            rows.append("%-28s %8d" % (name, edge.peak))
        return "\n".join(rows)
