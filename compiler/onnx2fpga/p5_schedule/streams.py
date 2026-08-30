"""Stream plumbing inserted after folding is decided.

Two things can go wrong between neighbouring units once their folding factors
differ. Their beat widths stop matching, which a width converter fixes, and
their instantaneous rates stop matching, which a FIFO absorbs. Undersized FIFOs
stall a dataflow pipeline or deadlock it outright when a fork rejoins, so the
sizer is deliberately conservative on reconvergent paths.
"""

import math

from ..p2_graph.graph import GraphError
from ..p2_graph.tensor import Tensor
from ..p3_ops.hardware_ops import DuplicateUnit, StreamFifo, WidthConverter
from ..pipeline import Pass
from .tokens import TokenAnalysis


def compatible_widths(supplied, wanted):
    """Two beat widths convert in one step only at an integer ratio."""
    return supplied % wanted == 0 or wanted % supplied == 0


class InsertDuplicates(Pass):
    """Gives every stream exactly one consumer.

    Wiring one stream to two units leaves both driving its ready line and both
    seeing the same beat. Verilator accepts that netlist without a word, so it
    would have shipped: the check that follows this pass is what makes the
    guarantee, not the simulator.
    """

    name = "insert-dup"

    def run(self, graph, context):
        for name in [t for t in graph.tensor_names()
                     if not graph.tensor(t).is_constant]:
            consumers = graph.consumers(name)
            if len(consumers) < 2:
                continue
            self._fork(graph, name, consumers, context)
        graph.infer()
        return graph

    @staticmethod
    def _fork(graph, name, consumers, context):
        source = graph.tensor(name)
        producer = graph.producer(name)
        branches = []
        for index, consumer in enumerate(consumers):
            branch = graph.fresh_name("%s_br%d" % (name, index))
            graph.add_tensor(Tensor(branch, source.shape, source.dtype))
            consumer.inputs = [branch if i == name else i for i in consumer.inputs]
            branches.append(branch)
        unit = DuplicateUnit(graph.fresh_name(name + "_dup"), [name], branches,
                             elements=source.numel or 0, dtype=source.dtype)
        if producer is None:
            graph.add_node(unit)
        else:
            graph.insert_after(producer, unit)
        context.note("insert-dup: %s feeds %d consumers" % (name, len(branches)))


class StreamsAreSingleConsumer(Pass):
    """Structural check, run last. A stream with two consumers is a netlist
    that simulates without complaint and computes the wrong thing."""

    name = "check-streams"

    def run(self, graph, context):
        for name in graph.tensor_names():
            if graph.tensor(name).is_constant:
                continue
            consumers = graph.consumers(name)
            if len(consumers) > 1:
                raise GraphError(
                    "stream %r feeds %d consumers (%s); it needs a duplicate unit"
                    % (name, len(consumers), ", ".join(c.name for c in consumers)))
        return graph


class AlignBeatWidths(Pass):
    """Nudges folding so neighbouring units meet at an integer beat ratio.

    Folding is chosen per unit against a resource budget, which can leave two
    neighbours at widths like 2 and 5. That still works, but only through a
    converter that has to step down to their common divisor and back up, which
    throttles the edge to one element per cycle and quietly undoes the balance
    the allocator just computed. Retuning one side is nearly always cheaper.
    """

    name = "align-widths"

    def run(self, graph, context):
        adjusted = []
        for node in graph.topological_order():
            for slot, name in enumerate(node.stream_inputs(graph)):
                producer = graph.producer(name)
                if producer is None:
                    continue
                supplied = producer.output_beat_elems(graph)[
                    producer.outputs.index(name)]
                if compatible_widths(supplied, node.input_beat_elems(graph)[slot]):
                    continue
                choice = self._retune(node, graph, slot, supplied, context.device)
                if choice is None:
                    continue
                before = node.cycles
                node.apply_folding(choice)
                adjusted.append("%s %s (%d cycles, was %d)"
                                % (node.name, choice, node.cycles, before))
        if adjusted:
            context.note("align-widths: retuned " + "; ".join(adjusted))
        graph.infer()
        return graph

    @staticmethod
    def _retune(node, graph, slot, supplied, device):
        """Cheapest compatible folding that is no more expensive than the one
        the allocator picked. Reducing parallelism is always affordable, so a
        candidate always exists unless nothing is compatible at all."""
        current = dict(node.folding)
        ceiling = node.resources(device).pressure(device.budget)
        best, best_cycles = None, None
        for option in node.folding_options():
            node.apply_folding(option)
            width = node.input_beat_elems(graph)[slot]
            cycles = node.cycles
            pressure = node.resources(device).pressure(device.budget)
            node.folding = dict(current)
            if not compatible_widths(supplied, width) or pressure > ceiling:
                continue
            if best_cycles is None or cycles < best_cycles:
                best, best_cycles = dict(option), cycles
        return best


class InsertWidthConverters(Pass):
    name = "insert-dwc"

    def run(self, graph, context):
        graph.infer()
        for node in list(graph.topological_order()):
            for slot, name in enumerate(node.stream_inputs(graph)):
                producer = graph.producer(name)
                if producer is None:
                    continue
                supplied = self._supplied_elems(producer, graph, name)
                wanted = node.input_beat_elems(graph)[slot]
                if supplied == wanted:
                    continue
                self._splice(graph, producer, node, name, supplied, wanted)
        self._resolve_io_widths(graph)
        graph.infer()
        return graph

    @staticmethod
    def _resolve_io_widths(graph):
        """Graph inputs have no producer, so their beat width is whatever the
        first unit consumes."""
        for name in graph.inputs:
            for consumer in graph.consumers(name):
                slot = consumer.stream_inputs(graph).index(name)
                graph.tensor(name).elems_per_beat = consumer.input_beat_elems(graph)[slot]
                break

    @staticmethod
    def _supplied_elems(producer, graph, name):
        return producer.output_beat_elems(graph)[producer.outputs.index(name)]

    @staticmethod
    def _stages(supplied, wanted):
        """One converter at an integer ratio, otherwise down to the common
        divisor and back up. The two step path is correct but slow, which is
        why AlignBeatWidths runs first to avoid needing it."""
        if compatible_widths(supplied, wanted):
            return [(supplied, wanted)]
        common = math.gcd(supplied, wanted)
        return [(supplied, common), (common, wanted)]

    @staticmethod
    def _splice(graph, producer, consumer, name, supplied, wanted):
        source = graph.tensor(name)
        anchor, upstream = producer, name
        for step_in, step_out in InsertWidthConverters._stages(supplied, wanted):
            staged = graph.fresh_name(name + "_dwc")
            graph.add_tensor(Tensor(staged, source.shape, source.dtype))
            converter = WidthConverter(graph.fresh_name(name + "_dwc_unit"),
                                       [upstream], [staged], in_elems=step_in,
                                       out_elems=step_out,
                                       elem_bits=source.dtype.bits,
                                       total_elems=source.numel)
            graph.insert_after(anchor, converter)
            anchor, upstream = converter, staged
        consumer.inputs = [upstream if i == name else i for i in consumer.inputs]


class InsertFifos(Pass):
    """Places a buffer on every internal edge and sizes it from the schedule.

    Depth comes from TokenAnalysis, which computes how many beats can be in
    flight on each edge under an as-soon-as-possible schedule. On a chain that
    is a handful; across a fork that rejoins it is the whole skew between the
    branches, and getting it wrong there deadlocks rather than slows.

    `cap` exists so a test can deliberately starve the buffers and show that
    the sizing is load bearing.
    """

    name = "insert-fifo"

    def __init__(self, minimum_depth=2, cap=None):
        self.minimum_depth = minimum_depth
        self.cap = cap

    def run(self, graph, context):
        analysis = TokenAnalysis(graph)
        analysis.run()
        context.artifacts["tokens"] = analysis

        planned = {}
        for node in list(graph.topological_order()):
            for name in list(node.stream_inputs(graph)):
                producer = graph.producer(name)
                if producer is None or isinstance(producer, StreamFifo):
                    continue
                planned[name] = (producer, node, self.depth_for(analysis, name))

        for name, (producer, consumer, depth) in planned.items():
            self._splice(graph, producer, consumer, name, depth)

        deepest = max(planned.values(), key=lambda entry: entry[2], default=None)
        if deepest is not None:
            context.note("insert-fifo: %d buffers, deepest %d beats"
                         % (len(planned), deepest[2]))
        graph.infer()
        return graph

    def depth_for(self, analysis, name):
        depth = analysis.depth_for(name, self.minimum_depth)
        if self.cap is not None:
            depth = min(depth, self.cap)
        return max(1, depth)

    @staticmethod
    def _splice(graph, producer, consumer, name, depth):
        source = graph.tensor(name)
        staged = graph.fresh_name(name + "_fifo")
        graph.add_tensor(Tensor(staged, source.shape, source.dtype))
        buffer = StreamFifo(graph.fresh_name(name + "_fifo_unit"), [name], [staged],
                            depth=depth, width=source.stream_width)
        consumer.inputs = [staged if i == name else i for i in consumer.inputs]
        graph.insert_after(producer, buffer)


class StripStreamPlumbing(Pass):
    """Removes every width converter and FIFO, leaving the units and forks."""

    name = "strip-streams"

    def run(self, graph, context):
        removed = 0
        for node in graph.topological_order():
            if not isinstance(node, (WidthConverter, StreamFifo)):
                continue
            staged = node.outputs[0]
            graph.rewire(staged, node.inputs[0])
            graph.remove_node(node)
            graph.remove_tensor(staged)
            removed += 1
        if removed:
            context.note("strip-streams: removed %d converters and buffers"
                         % removed)
        graph.infer()
        return graph
