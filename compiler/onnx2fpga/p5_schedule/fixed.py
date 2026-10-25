"""Fixed latency mode: every dense layer unrolled to one vector per cycle, with the latency summed along the chain."""

from ..errors import BudgetError, ConfigError
from ..p3_ops.hardware_ops import DenseUnit, MatVecUnit, TreeEnsembleUnit
from ..p3_ops.onnx_ops import FlattenOp, GemmOp
from ..p3_ops.trees import TreeEnsembleOp
from ..pipeline import Pass
from .folding import FoldingReport


def check_fixed_ops(graph):
    for node in graph.topological_order():
        if not isinstance(node, (GemmOp, FlattenOp, TreeEnsembleOp)):
            kind = node.op_type + ("+Relu" if node.attrs.get("relu") else "")
            raise ConfigError("mode='fixed' cannot build %s (%s) yet" % (node.name, kind))


class UnrollForFixedLatency(Pass):
    name = "unroll"

    # The words the 0x8000 register window holds, one per value.
    BANK_WORDS = 8192

    def __init__(self, utilisation_limit=0.80, arithmetic=None, reloadable=False):
        self.utilisation_limit = utilisation_limit
        self.arithmetic = dict(arithmetic or {})
        self.reloadable = reloadable

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
        self._check_budget(units, context.device)
        words = sum(getattr(unit, "words", 0) for unit in units)
        if words > self.BANK_WORDS:
            raise BudgetError("the reloadable weights need %d words a bank, past the %d of "
                              "the register window" % (words, self.BANK_WORDS))

        latency = sum(unit.latency for unit in units)
        context.artifacts["latency"] = latency
        context.artifacts["folding"] = FoldingReport(context.device).record(graph)
        context.note("unroll: %d layers, latency %d cycles" % (len(units), latency))
        return graph

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

