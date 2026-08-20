"""Folding allocation: how parallel each unit gets.

Throughput of a streaming pipeline is set by its slowest stage, so spending
resources anywhere else is waste. The allocator repeatedly finds the current
bottleneck, buys it the cheapest folding step that actually makes it faster,
and stops when the budget is gone or the target is met. That is what keeps the
stages balanced instead of merely fast in one place.
"""

from ..pipeline import Pass
from ..targets.resources import Resources


class FoldingReport:
    def __init__(self, device):
        self.device = device
        self.rows = []
        self.bottleneck_cycles = 0
        self.total = Resources.zero()

    def record(self, graph):
        self.rows = []
        self.total = Resources.zero()
        for node in graph.topological_order():
            cost = node.resources(self.device)
            self.total = self.total + cost
            self.rows.append((node.name, type(node).__name__, dict(node.folding),
                              node.cycles, cost))
        self.bottleneck_cycles = max((r[3] for r in self.rows), default=0)
        return self

    @property
    def frames_per_second(self):
        if not self.bottleneck_cycles:
            return 0.0
        return self.device.fmax_mhz * 1e6 / self.bottleneck_cycles

    def render(self):
        lines = ["%-24s %-18s %-18s %10s %8s %8s %8s" %
                 ("node", "kind", "folding", "cycles", "lut", "dsp", "bram")]
        for name, kind, folding, cycles, cost in self.rows:
            marker = " <-- bottleneck" if cycles == self.bottleneck_cycles else ""
            lines.append("%-24s %-18s %-18s %10d %8d %8.1f %8.1f%s" % (
                name, kind, ",".join("%s=%d" % kv for kv in sorted(folding.items())),
                cycles, int(cost.lut), cost.dsp, cost.bram36, marker))
        used = self.total.utilisation(self.device.budget)
        lines.append("total %s" % self.total)
        lines.append("utilisation " + "  ".join("%s %.1f%%" % (k, 100 * v)
                                                for k, v in sorted(used.items())))
        lines.append("bottleneck %d cycles -> %.1f frames/s at %.0f MHz"
                     % (self.bottleneck_cycles, self.frames_per_second,
                        self.device.fmax_mhz))
        return "\n".join(lines)


class FoldingAllocator(Pass):
    name = "fold"

    def __init__(self, target_cycles=None, utilisation_limit=0.80):
        self.target_cycles = target_cycles
        self.utilisation_limit = utilisation_limit

    def run(self, graph, context):
        device = context.device
        nodes = [n for n in graph.topological_order() if n.folding_options() != [{}]]
        for node in nodes:
            node.apply_folding(node.default_folding())

        frozen = []
        while True:
            movable = [n for n in nodes if n not in frozen]
            if not movable:
                break
            bottleneck = max(movable, key=lambda n: n.cycles)
            target = self._effective_target(frozen)
            if target is not None and bottleneck.cycles <= target:
                break
            upgrade = self._cheapest_upgrade(bottleneck, graph, device, target)
            if upgrade is None:
                frozen.append(bottleneck)
                continue
            bottleneck.apply_folding(upgrade)

        report = FoldingReport(device).record(graph)
        context.artifacts["folding"] = report
        if frozen:
            context.note("fold: %s cannot fold further and set the floor at %d cycles"
                         % (", ".join(n.name for n in frozen),
                            max(n.cycles for n in frozen)))
        context.note("fold: bottleneck %d cycles, %.1f frames/s, dsp %.0f"
                     % (report.bottleneck_cycles, report.frames_per_second,
                        report.total.dsp))
        return graph

    def _effective_target(self, frozen):
        """A stage that cannot fold any further sets the floor for the whole
        pipeline. Nothing upstream or downstream of it gains anything from
        running faster than that, so the target rises to meet it."""
        floor = max((n.cycles for n in frozen), default=None)
        if floor is None:
            return self.target_cycles
        if self.target_cycles is None:
            return floor
        return max(self.target_cycles, floor)

    @staticmethod
    def _bottleneck(graph):
        candidates = [n for n in graph.nodes if n.folding_options() != [{}]]
        return max(candidates, key=lambda n: n.cycles, default=None)

    def _total_without(self, graph, node, device):
        return sum((n.resources(device) for n in graph.nodes if n is not node),
                   Resources.zero())

    def _cheapest_upgrade(self, node, graph, device, target):
        """Pick the next folding step for the bottleneck.

        With a target set, take the cheapest option that reaches it, so the
        allocator does not spend a thousand DSPs to overshoot a goal a hundred
        would have met. Without one, take the option that buys the most cycles
        per unit of resource, which converges on a balanced pipeline.
        """
        baseline = node.cycles
        others = self._total_without(graph, node, device)
        current = dict(node.folding)
        base_pressure = node.resources(device).pressure(device.budget)

        feasible = []
        for option in node.folding_options():
            node.apply_folding(option)
            cycles, cost = node.cycles, node.resources(device)
            node.folding = dict(current)
            if cycles >= baseline:
                continue
            if (others + cost).worst_utilisation(device.budget) > self.utilisation_limit:
                continue
            feasible.append((option, cycles, cost.pressure(device.budget)))
        if not feasible:
            return None

        if target:
            meeting = [entry for entry in feasible if entry[1] <= target]
            if meeting:
                return min(meeting, key=lambda entry: entry[2])[0]

        def efficiency(entry):
            _, cycles, pressure = entry
            spend = max(pressure - base_pressure, 1e-9)
            return (baseline - cycles) / spend

        return max(feasible, key=efficiency)[0]
