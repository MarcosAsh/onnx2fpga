"""Folding allocation: repeatedly speeds up the bottleneck until the budget is gone or the target met."""

from ..errors import BudgetError, ConfigError
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
        block_ram = self.device.block_ram
        lines = ["%-24s %-18s %-18s %10s %8s %8s %8s" %
                 ("node", "kind", "folding", "cycles", "lut", "dsp",
                  "bram" if block_ram == "bram36" else block_ram)]
        for name, kind, folding, cycles, cost in self.rows:
            marker = " <-- bottleneck" if cycles == self.bottleneck_cycles else ""
            lines.append("%-24s %-18s %-18s %10d %8d %8.1f %8.1f%s" % (
                name, kind, ",".join("%s=%d" % kv for kv in sorted(folding.items())),
                cycles, int(cost.lut), cost.dsp, getattr(cost, block_ram), marker))
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

    def __init__(self, target_cycles=None, utilisation_limit=0.80, pins=None):
        """`pins` maps a unit name to folding factors fixed by hand; see HwNode.allowed_foldings."""
        self.target_cycles = target_cycles
        self.utilisation_limit = utilisation_limit
        self.pins = dict(pins or {})

    def run(self, graph, context):
        device = context.device
        nodes = [n for n in graph.topological_order() if n.folding_options() != [{}]]
        self._apply_pins(nodes)
        target = self.target_cycles
        frozen, held = self._allocate(nodes, graph, device, target)
        missed = dict(held)
        if held and target is not None:
            target = self._settle(nodes, graph, device, target)
            frozen, held = self._allocate(nodes, graph, device, target)

        report = FoldingReport(device).record(graph)
        context.artifacts["folding"] = report
        pinned = [n for n in nodes if n.pinned_folding]
        self._check_pins(pinned, report, device)
        for node in pinned:
            context.note("fold: %s pinned at %s, %d cycles"
                         % (node.name, _render(node.folding), node.cycles))
        floor = [n for n in frozen if n not in held and not n.pinned_folding]
        if floor:
            context.note("fold: %s cannot fold further and set the floor at %d cycles"
                         % (", ".join(n.name for n in floor),
                            max(n.cycles for n in floor)))
        if target != self.target_cycles:
            for node, (field, used, allowed) in missed.items():
                context.note("fold: %d cycles does not fit the %s budget of %s: the next "
                             "step for %s would take the design to %.0f %s, past the "
                             "%.0f allowed. Balanced at %d cycles instead"
                             % (self.target_cycles, field, device.name, node.name,
                                used, field, allowed, target))
        for node, (field, used, allowed) in held.items():
            context.note("fold: %s held at %d cycles by the %s budget: its next step "
                         "would take the design to %.0f %s, past the %.0f allowed on %s"
                         % (node.name, node.cycles, field, used, field, allowed,
                            device.name))
        context.note("fold: bottleneck %d cycles, %.1f frames/s, dsp %.0f"
                     % (report.bottleneck_cycles, report.frames_per_second,
                        report.total.dsp))
        return graph

    def _allocate(self, nodes, graph, device, target_cycles):
        """One greedy pass from the default folding; returns the stages stopped early and which budget stopped them."""
        for node in nodes:
            node.apply_folding(node.starting_folding())
        frozen, held = [], {}
        while True:
            movable = [n for n in nodes if n not in frozen]
            if not movable:
                break
            bottleneck = max(movable, key=lambda n: n.cycles)
            target = self._effective_target(frozen, target_cycles)
            if target is not None and bottleneck.cycles <= target:
                break
            upgrade, blocked = self._cheapest_upgrade(bottleneck, graph, device, target)
            if upgrade is None:
                frozen.append(bottleneck)
                if blocked:
                    held[bottleneck] = blocked
                continue
            bottleneck.apply_folding(upgrade)
        return frozen, held

    def _settle(self, nodes, graph, device, target_cycles):
        """The lowest bottleneck the budget can hold every stage to, found by bisection over achievable cycle counts."""
        candidates = set()
        for node in nodes:
            current = dict(node.folding)
            for option in node.allowed_foldings():
                node.apply_folding(option)
                candidates.add(node.cycles)
            node.folding = current
        candidates = sorted(c for c in candidates if c > target_cycles)
        low, high = 0, len(candidates) - 1
        while low < high:
            middle = (low + high) // 2
            if self._allocate(nodes, graph, device, candidates[middle])[1]:
                low = middle + 1
            else:
                high = middle
        return candidates[low] if candidates else target_cycles

    def _apply_pins(self, nodes):
        """Checks every pin against what the unit can do first, so a typo fails here naming the choices."""
        units = {node.name: node for node in nodes}
        for name, factors in self.pins.items():
            node = units.get(name)
            if node is None:
                raise ConfigError(
                    "cannot fold %r: the compiled design has no unit by that name that "
                    "folds. Units that fold: %s" % (name, ", ".join(sorted(units))))
            options = node.folding_options()
            known = sorted(node.default_folding())
            unknown = sorted(set(factors) - set(known))
            if unknown:
                raise ConfigError(
                    "%s (%s) folds by %s, not %s. Pass only those, for example "
                    "fold(%s=%d)" % (name, type(node).__name__, " and ".join(known),
                                     ", ".join(unknown), known[0], options[-1][known[0]]))
            for key, value in sorted(factors.items()):
                valid = sorted({option[key] for option in options})
                if value not in valid:
                    raise ConfigError(
                        "%s cannot fold to %s=%r. Valid %s values for it: %s%s"
                        % (name, key, value, key, ", ".join(str(v) for v in valid),
                           _fold_hint(node, key)))
            node.attrs["pinned_folding"] = dict(factors)
            if not node.allowed_foldings():
                raise ConfigError(
                    "%s cannot fold to %s: each value is valid alone but not together. "
                    "Valid foldings: %s" % (name, _render(factors),
                                            "; ".join(_render(o) for o in options)))

    def _check_pins(self, pinned, report, device):
        """Only a pin can take a design past its budget; the allocator checks its own choices first."""
        over = report.total.overruns(device.budget, device.limits(self.utilisation_limit))
        if not pinned or not over:
            return
        field, used, allowed = over[0]
        names = ", ".join("%s to %s" % (n.name, _render(n.folding)) for n in pinned)
        raise BudgetError(
            "the design does not fit %s: folding %s by hand takes it to %.0f %s, past "
            "the %.0f allowed at a utilisation limit of %.2f. Fold less, drop the pin "
            "and let the allocator choose, or compile for a larger device"
            % (device.name, names, used, field, allowed, self.utilisation_limit))

    def _effective_target(self, frozen, target_cycles):
        """A stage that cannot fold further sets the floor for the pipeline, so the target rises to meet it."""
        floor = max((n.cycles for n in frozen), default=None)
        if floor is None:
            return target_cycles
        if target_cycles is None:
            return floor
        return max(target_cycles, floor)

    @staticmethod
    def _bottleneck(graph):
        candidates = [n for n in graph.nodes if n.folding_options() != [{}]]
        return max(candidates, key=lambda n: n.cycles, default=None)

    def _total_without(self, graph, node, device):
        return sum((n.resources(device) for n in graph.nodes if n is not node),
                   Resources.zero())

    def _cheapest_upgrade(self, node, graph, device, target):
        """Picks the next folding step: cheapest that reaches the target, else most cycles per resource."""
        baseline = node.cycles
        others = self._total_without(graph, node, device)
        current = dict(node.folding)
        base_pressure = node.resources(device).pressure(device.budget)
        limits = device.limits(self.utilisation_limit)

        feasible, rejected = [], []
        for option in node.allowed_foldings():
            node.apply_folding(option)
            cycles, cost = node.cycles, node.resources(device)
            node.folding = dict(current)
            if cycles >= baseline:
                continue
            over = (others + cost).overruns(device.budget, limits)
            if over:
                rejected.append((cost.pressure(device.budget), over[0]))
                continue
            feasible.append((option, cycles, cost.pressure(device.budget)))
        if not feasible:
            return None, min(rejected)[1] if rejected else None

        if target:
            meeting = [entry for entry in feasible if entry[1] <= target]
            if meeting:
                return min(meeting, key=lambda entry: entry[2])[0], None

        def efficiency(entry):
            _, cycles, pressure = entry
            spend = max(pressure - base_pressure, 1e-9)
            return (baseline - cycles) / spend

        return max(feasible, key=efficiency)[0], None


def _render(folding):
    return ",".join("%s=%d" % item for item in sorted(folding.items()))


def _fold_hint(node, key):
    """What each factor has to divide, for units where that is a fact about the layer."""
    if hasattr(node, "mw") and hasattr(node, "mh"):
        return {"simd": " (simd divides the %d inputs each output sums over)" % node.mw,
                "pe": " (pe divides the %d outputs)" % node.mh}.get(key, "")
    if hasattr(node, "channels") and key == "pe":
        return " (pe divides the %d channels)" % node.channels
    return ""
