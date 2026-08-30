"""End-to-end latency: when the first output beat of a frame appears.

This is not the sum of the units' latencies, and the difference is large. A
unit does not start when its producer emits its first beat; it starts when
enough beats have arrived, at the rate the producer emits them. A matrix
vector unit folded four ways needs four input beats before its first output
exists, and if its producer hands those over one every eight cycles then it
waits twenty four cycles that no per-unit number mentions. On the example MLP
the sum is 17 cycles against a measured 132.

So the schedules are composed rather than added. Each unit publishes when it
takes each input beat and when it hands over each output beat, both counted
from its own start. Placing a unit means finding the earliest start at which
none of its input beats is needed before the producer has produced it:

    start[n] = max over inputs i, beats j of
                   start[producer] + producer_out[j] - n_in[j]

and the frame's latency is the last unit's start plus its own latency.
"""

from ..p3_ops.hardware_ops import StreamingNode


class LatencyModel:
    """Placement of every unit in time, and the graph total that falls out.

    Exact where the units report exact schedules. Where a unit is on the
    uniform default it is modelled as spreading its output evenly across the
    frame, which is slower than the truth for a unit that bursts, so the total
    is an upper bound rather than an estimate that could go either way. That is
    the safe direction for a latency claim and the wrong direction for a tight
    one: the example CNN comes out at 612 against a measured 474, and all of
    that gap is the sliding window generator, which has no real schedule yet.
    """

    def __init__(self, graph):
        self.graph = graph
        self.starts = {}
        self.gates = {}
        self._place()

    def _place(self):
        for node in self.graph.topological_order():
            if not isinstance(node, StreamingNode):
                continue
            self.starts[node.name] = self._earliest_start(node)

    def _earliest_start(self, node):
        start, gate = 0, None
        for index, name in enumerate(node.stream_inputs(self.graph)):
            producer = self.graph.producer(name)
            if not isinstance(producer, StreamingNode):
                continue
            emitted = producer.output_schedule(self.graph,
                                               producer.outputs.index(name))
            taken = node.input_schedule(self.graph, index)
            if not emitted or not taken:
                continue
            # A beat leaves its producer through that producer's output
            # register, so it is visible one cycle after the schedule says it
            # was formed. The schedules describe beat order and say nothing
            # about the flip flop, so the delay is added at every hop.
            base = self.starts[producer.name] + producer.REGISTER_STAGES
            for beat in range(min(len(emitted), len(taken))):
                candidate = base + emitted[beat] - taken[beat]
                if candidate > start:
                    start, gate = candidate, producer.name
        self.gates[node.name] = gate
        return start

    @property
    def units(self):
        return [n for n in self.graph.topological_order()
                if isinstance(n, StreamingNode)]

    def start_of(self, node):
        return self.starts[node.name]

    def first_output_of(self, node):
        return self.starts[node.name] + node.latency(self.graph)

    @property
    def cycles(self):
        """Cycles from the first input beat to the first output beat."""
        units = self.units
        if not units:
            return 0
        return max(self.first_output_of(n) for n in units)

    @property
    def critical_path(self):
        """The units in the order they gate one another, each with the cycle
        its first output appears. What to look at when the total is too big."""
        return [(n.name, self.first_output_of(n)) for n in self.units]

    @property
    def gating_path(self):
        """The chain that sets the total, latest last."""
        units = self.units
        if not units:
            return []
        by_name = {n.name: n for n in units}
        node, chain, seen = max(units, key=self.first_output_of), [], set()
        while node is not None and node.name not in seen:
            seen.add(node.name)
            chain.append(node)
            node = by_name.get(self.gates.get(node.name))
        return list(reversed(chain))


def graph_latency(graph):
    return LatencyModel(graph).cycles
