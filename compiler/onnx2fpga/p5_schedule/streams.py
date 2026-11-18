"""Stream plumbing after folding: width converters and FIFOs, sized conservatively on reconvergent paths."""

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
    """Gives every stream exactly one consumer; the check after this pass is the guarantee, not the simulator."""

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
    """Structural check, run last; a stream with two consumers simulates fine and computes the wrong thing."""

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
    """Nudges folding so neighbours meet at an integer beat ratio, avoiding a throttling two-step converter."""

    name = "align-widths"

    def run(self, graph, context):
        report = context.artifacts["folding"]
        limit = report.bottleneck_cycles
        adjusted = []
        for node in graph.topological_order():
            for slot, name in enumerate(node.stream_inputs(graph)):
                producer = graph.producer(name)
                if producer is None or self._agree(graph, node, slot):
                    continue
                choice = self._retune(producer, node, graph, slot, context.device, limit)
                if choice is None:
                    continue
                for unit, option in zip((producer, node), choice):
                    before = unit.cycles
                    if option != unit.folding:
                        unit.apply_folding(option)
                        adjusted.append("%s %s (%d cycles, was %d)"
                                        % (unit.name, option, unit.cycles, before))
        if adjusted:
            context.note("align-widths: retuned " + "; ".join(adjusted))
            report.record(graph)
            if report.bottleneck_cycles > limit:
                context.note("align-widths: bottleneck now %d cycles"
                             % report.bottleneck_cycles)
        graph.infer()
        return graph

    @staticmethod
    def _agree(graph, node, slot):
        name = node.stream_inputs(graph)[slot]
        producer = graph.producer(name)
        if producer is None:
            return True
        supplied = producer.output_beat_elems(graph)[producer.outputs.index(name)]
        return compatible_widths(supplied, node.input_beat_elems(graph)[slot])

    @classmethod
    def _retune(cls, producer, node, graph, slot, device, limit):
        """Cheapest foldings for both ends of an edge that keep the bottleneck and cost no more together."""
        def pressure(unit):
            return unit.resources(device).pressure(device.budget)

        current = dict(producer.folding), dict(node.folding)
        ceiling = pressure(producer) + pressure(node)
        best, best_key = None, None
        for upstream in producer.allowed_foldings():
            producer.apply_folding(upstream)
            if not all(cls._agree(graph, producer, s)
                       for s in range(len(producer.stream_inputs(graph)))):
                continue
            for option in node.allowed_foldings():
                node.apply_folding(option)
                cost = pressure(producer) + pressure(node)
                if cost > ceiling or not cls._agree(graph, node, slot):
                    continue
                key = (max(producer.cycles, node.cycles, limit), cost)
                if best_key is None or key < best_key:
                    best, best_key = (dict(upstream), dict(option)), key
        producer.folding, node.folding = current
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
                supplied = producer.output_beat_elems(graph)[producer.outputs.index(name)]
                wanted = node.input_beat_elems(graph)[slot]
                if supplied == wanted:
                    continue
                self._splice(graph, producer, node, name, supplied, wanted)
        self._resolve_io_widths(graph)
        graph.infer()
        return graph

    @staticmethod
    def _resolve_io_widths(graph):
        """Graph inputs have no producer, so their beat width is whatever the first unit consumes."""
        for name in graph.inputs:
            for consumer in graph.consumers(name):
                slot = consumer.stream_inputs(graph).index(name)
                graph.tensor(name).elems_per_beat = consumer.input_beat_elems(graph)[slot]
                break

    @staticmethod
    def _stages(supplied, wanted):
        """One converter at an integer ratio, else down to the common divisor and back up (slow)."""
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
    """Places a sized buffer on every internal edge; `cap` lets a test starve them to show sizing matters."""

    name = "insert-fifo"

    def __init__(self, cap=None):
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
                depth = analysis.depth_for(name)
                if self.cap is not None:
                    depth = min(depth, self.cap)
                planned[name] = (producer, node, max(1, depth))

        for name, (producer, consumer, depth) in planned.items():
            self._splice(graph, producer, consumer, name, depth)

        deepest = max(planned.values(), key=lambda entry: entry[2], default=None)
        if deepest is not None:
            context.note("insert-fifo: %d buffers, deepest %d beats"
                         % (len(planned), deepest[2]))
        graph.infer()
        return graph

    @staticmethod
    def _splice(graph, producer, consumer, name, depth):
        source = graph.tensor(name)
        staged = graph.fresh_name(name + "_fifo")
        graph.add_tensor(Tensor(staged, source.shape, source.dtype))
        buffer = StreamFifo(graph.fresh_name(name + "_fifo_unit"), [name], [staged],
                            depth=depth, width=source.stream_width)
        consumer.inputs = [staged if i == name else i for i in consumer.inputs]
        graph.insert_after(producer, buffer)
