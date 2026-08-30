"""Folding chosen against end-to-end latency, on the graph that gets built.

Runs after stream insertion. Scoring a candidate before the converters and
FIFOs exist is 7.1x optimistic: the first attempt stopped at a modelled 90
cycles against a built 640. A candidate is therefore scored by building it.
Cheap, because the plumbing is a deterministic function of the foldings, so a
dict of foldings is a complete snapshot and reverting is re-applying it.
"""

from ..pipeline import CompilerContext, Pass
from .folding import FoldingReport
from .latency import LatencyModel
from .streams import (AlignBeatWidths, InsertFifos, InsertWidthConverters,
                      StripStreamPlumbing)


class LatencyDirectedFolding(Pass):
    name = "refold-latency"

    def __init__(self, target_cycles=None, utilisation_limit=0.80, fifo_cap=None):
        self.target_cycles = target_cycles
        self.utilisation_limit = utilisation_limit
        self.replumb_passes = [
            StripStreamPlumbing(),
            AlignBeatWidths(),
            InsertWidthConverters(),
            InsertFifos(cap=fifo_cap),
        ]

    def run(self, graph, context):
        units = [n for n in graph.topological_order()
                 if n.folding_options() != [{}]]
        if not units:
            return graph

        scratch = CompilerContext(context.device, context.config)
        started = best = LatencyModel(graph).cycles
        frame_ceiling = FoldingReport(context.device).record(graph).bottleneck_cycles

        while self.target_cycles is None or best > self.target_cycles:
            step = self._best_step(graph, scratch, units, best, frame_ceiling)
            if step is None:
                break
            node, choice, best = step
            node.apply_folding(choice)
            self._replumb(graph, scratch)

        self._replumb(graph, context)

        report = FoldingReport(context.device).record(graph)
        report.latency_cycles = LatencyModel(graph).cycles
        context.artifacts["folding"] = report
        if report.latency_cycles < started:
            context.note("refold-latency: %d -> %d cycles to first output, dsp %.0f"
                         % (started, report.latency_cycles, report.total.dsp))
        else:
            context.note("refold-latency: %d cycles to first output, nothing "
                         "bought it less within the budget" % started)
        return graph

    def _best_step(self, graph, scratch, units, best, frame_ceiling):
        budget = scratch.device.budget
        snapshot = {node.name: dict(node.folding) for node in units}
        gating = [n for n in LatencyModel(graph).gating_path
                  if n.folding_options() != [{}]]

        chosen = None
        for node in gating:
            for option in node.folding_options():
                node.apply_folding(option)
                self._replumb(graph, scratch)
                report = FoldingReport(scratch.device).record(graph)
                latency = LatencyModel(graph).cycles
                pressure = report.total.pressure(budget)
                fits = (report.total.worst_utilisation(budget)
                        <= self.utilisation_limit)
                frame = report.bottleneck_cycles
                self._restore(graph, scratch, snapshot)

                if not fits or latency >= best or frame > frame_ceiling:
                    continue
                key = (latency, frame, pressure)
                if chosen is None or key < chosen[3]:
                    chosen = (node, dict(option), latency, key)
        return chosen[:3] if chosen else None

    def _restore(self, graph, scratch, snapshot):
        for node in graph.nodes:
            if node.name in snapshot:
                node.folding = dict(snapshot[node.name])
        self._replumb(graph, scratch)

    def _replumb(self, graph, context):
        for stage in self.replumb_passes:
            graph = stage(graph, context) or graph
        return graph
