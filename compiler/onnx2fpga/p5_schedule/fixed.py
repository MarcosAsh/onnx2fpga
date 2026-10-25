"""Fixed latency mode: every dense layer unrolled to one vector per cycle, with the latency summed along the chain."""

from ..errors import BudgetError, ConfigError
from ..p3_ops.hardware_ops import DenseUnit, MatVecUnit, TreeEnsembleUnit
from ..p3_ops.onnx_ops import FlattenOp, GemmOp
from ..p3_ops.trees import TreeEnsembleOp
from ..pipeline import Pass
from .folding import FoldingReport

# What each stage knob of a unit governs, for the summary and the errors.
PATHS = {"product": "multiplies", "sum": "adder tree", "rescale": "rescale",
         "compare": "comparators", "select": "leaf select"}


def slowest(unit, delays):
    """(ns, path name) of the unit's longest path at the stages it has."""
    return max((options[unit.stages.get(knob, 0)], PATHS[knob])
               for knob, options in unit.paths(delays).items())


def predicted_fmax(units, delays):
    """The clock the delay model expects the whole design to reach, or None without a model."""
    if delays is None:
        return None
    return 1000.0 / max(slowest(unit, delays)[0] for unit in units)


def check_fixed_ops(graph):
    for node in graph.topological_order():
        if not isinstance(node, (GemmOp, FlattenOp, TreeEnsembleOp)):
            kind = node.op_type + ("+Relu" if node.attrs.get("relu") else "")
            raise ConfigError("mode='fixed' cannot build %s (%s) yet" % (node.name, kind))


class UnrollForFixedLatency(Pass):
    name = "unroll"

    # The words the 0x8000 register window holds, one per value.
    BANK_WORDS = 8192

    def __init__(self, utilisation_limit=0.80, arithmetic=None, reloadable=False,
                 clock_mhz=None):
        self.utilisation_limit = utilisation_limit
        self.arithmetic = dict(arithmetic or {})
        self.reloadable = reloadable
        self.clock_mhz = clock_mhz

    def run(self, graph, context):
        for node in list(graph.topological_order()):
            if isinstance(node, TreeEnsembleUnit):
                continue
            if node.vectors != 1:
                raise ConfigError("mode='fixed' cannot build %s, which takes %d vectors "
                                  "per input" % (node.name, node.vectors))
            graph.replace_node(node, [DenseUnit(
                node.name, node.inputs, node.outputs, node.weights, node.bias,
                node.requant, node.act_dtype, node.out_dtype, node.weight_dtype)])
        graph.infer()
        units = graph.topological_order()
        graph.tensor(graph.inputs[0]).elems_per_beat = graph.tensor(graph.inputs[0]).numel
        for unit in units:
            if isinstance(unit, DenseUnit) and self.reloadable:
                self._make_reloadable(unit, context.artifacts["plan"].mult_bits)
            elif isinstance(unit, DenseUnit):
                unit.arithmetic = self._arithmetic(unit, context.device)
        if self.clock_mhz is not None:
            self._pipeline(units, context.device)
        self._check_budget(units, context.device)
        words = sum(getattr(unit, "words", 0) for unit in units)
        if words > self.BANK_WORDS:
            raise BudgetError("the reloadable weights need %d words a bank, past the %d of "
                              "the register window" % (words, self.BANK_WORDS))

        latency = sum(unit.latency for unit in units)
        context.artifacts["latency"] = latency
        context.artifacts["clock_mhz"] = self.clock_mhz or context.device.fmax_mhz
        context.artifacts["fmax_mhz"] = predicted_fmax(units, context.device.delays)
        context.artifacts["folding"] = FoldingReport(context.device).record(graph)
        context.note("unroll: %d layers, latency %d cycles" % (len(units), latency))
        return graph

    def _pipeline(self, units, device):
        """Each stage knob takes the fewest registers whose path fits the clock period."""
        if device.delays is None:
            raise ConfigError("%s has no delay model, so clock_mhz needs one of %s"
                              % (device.name, ", ".join(
                                  name for name in device.names()
                                  if device.get(name).delays is not None)))
        period = 1000.0 / self.clock_mhz / (1 + device.delays.margin)
        floors = []
        for unit in units:
            for knob, options in unit.paths(device.delays).items():
                floors.append((min(options), unit.name, PATHS[knob]))
                fits = [stages for stages, delay in enumerate(options) if delay <= period]
                if fits and knob in unit.stages:
                    unit.stages[knob] = fits[0]
        delay, name, path = max(floors)
        if delay > period:
            raise BudgetError("%g MHz is out of reach on %s: the slowest path, through the %s "
                              "of %s, takes %.2f ns however it is pipelined, so the fastest "
                              "clock is %.1f MHz"
                              % (self.clock_mhz, device.name, path, name, delay,
                                 1000.0 / delay / (1 + device.delays.margin)))

    @staticmethod
    def _make_reloadable(unit, mult_bits):
        unit.make_reloadable(mult_bits)
        needed = MatVecUnit.accumulator_dtype.fget(unit).bits
        if needed > unit.accumulator_dtype.bits:
            raise ConfigError("%s has a bias too large for reloadable weights, which need a "
                              "%d bit accumulator against the %d the grid gives"
                              % (unit.name, needed, unit.accumulator_dtype.bits))

    def _arithmetic(self, unit, device):
        """Auto takes constant arithmetic unless it leaves the layer's busiest resource busier."""
        chosen = self.arithmetic.get(unit.name, "auto")
        if chosen != "auto":
            return chosen

        def worst(arithmetic):
            unit.arithmetic = arithmetic
            return unit.resources(device).worst_utilisation(device.budget)

        return "constant" if worst("constant") <= worst("multiplier") else "multiplier"

    def _check_budget(self, units, device):
        total = sum((unit.resources(device) for unit in units[1:]),
                    units[0].resources(device))
        over = total.overruns(device.budget, device.limits(self.utilisation_limit))
        if not over:
            return
        field, used, allowed = over[0]
        if isinstance(units[0], TreeEnsembleUnit):
            ensemble = units[0].ensemble
            raise BudgetError("the tree ensemble needs %d %s, past the %d allowed on %s "
                              "(%d trees of depth %d)" % (used, field.upper(), allowed,
                                                          device.name, ensemble.trees,
                                                          ensemble.depth))
        largest = max(units, key=lambda unit: unit.weights.size)
        raise BudgetError("the unrolled model needs %d %s, past the %d allowed on %s "
                          "(largest layer %s has %d weights)"
                          % (used, field.upper(), allowed, device.name, largest.name,
                             largest.weights.size))

