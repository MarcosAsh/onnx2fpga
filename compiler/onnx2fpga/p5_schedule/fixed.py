"""Fixed latency mode: every dense layer unrolled to one vector per cycle, with the latency summed along the chain."""

from ..errors import BudgetError, ConfigError
from ..p3_ops.hardware_ops import DenseUnit
from ..p3_ops.onnx_ops import FlattenOp, GemmOp
from ..pipeline import Pass
from .folding import FoldingReport


def check_fixed_ops(graph):
    for node in graph.topological_order():
        if not isinstance(node, (GemmOp, FlattenOp)):
            kind = node.op_type + ("+Relu" if node.attrs.get("relu") else "")
            raise ConfigError("mode='fixed' cannot build %s (%s) yet, only Gemm and MatMul "
                              "with an optional fused Relu, Flatten and Reshape"
                              % (node.name, kind))


class UnrollForFixedLatency(Pass):
    name = "unroll"

    def __init__(self, utilisation_limit=0.80):
        self.utilisation_limit = utilisation_limit

    def run(self, graph, context):
        for node in list(graph.topological_order()):
            if node.vectors != 1:
                raise ConfigError("mode='fixed' takes one vector per cycle, but %s multiplies "
                                  "%d vectors per input" % (node.name, node.vectors))
            graph.replace_node(node, [DenseUnit(
                node.name, node.inputs, node.outputs, node.weights, node.bias,
                node.requant, node.act_dtype, node.out_dtype, node.weight_dtype)])
        graph.infer()
        units = graph.topological_order()
        graph.tensor(graph.inputs[0]).elems_per_beat = units[0].mw
        self._check_budget(units, context.device)

        latency = sum(unit.latency for unit in units)
        context.artifacts["latency"] = latency
        context.artifacts["folding"] = FoldingReport(context.device).record(graph)
        context.note("unroll: %d layers, latency %d cycles" % (len(units), latency))
        return graph

    def _check_budget(self, units, device):
        total = sum((unit.resources(device) for unit in units[1:]),
                    units[0].resources(device))
        over = total.overruns(device.budget, device.limits(self.utilisation_limit))
        if not over:
            return
        field, used, allowed = over[0]
        largest = max(units, key=lambda unit: unit.multipliers)
        raise BudgetError(
            "the model does not fit %s unrolled: it needs %d %s against the %d allowed, and "
            "its largest layer %s alone has %d multipliers; use a larger device or the "
            "streaming mode" % (device.name, used, field.upper(), allowed, largest.name,
                                largest.multipliers))

