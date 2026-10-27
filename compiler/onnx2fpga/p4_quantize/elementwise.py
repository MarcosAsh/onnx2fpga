"""Elementwise ops in fixed latency mode: one live operand becomes a table, two or three an otf_eltwise unit."""

import numpy as np

from ..errors import ConfigError
from ..p2_graph.datatype import QuantSpec
from ..p3_ops.elementwise import (BOOLEAN, ConcatOp, ConcatUnit, ElementwiseOp, EltwiseUnit,
                                  TableUnit, evaluate)
from ..p3_ops.reductions import ReduceOp, SoftmaxOp
from .requantize import Requantizer

COMPARISONS = {"Less": ("less", False), "Greater": ("less", True),
               "LessOrEqual": ("less_equal", False), "GreaterOrEqual": ("less_equal", True),
               "Equal": ("equal", False)}


def lookup_table(function, in_scale, in_zero_point, out_scale, out_zero_point, dtype):
    """function(x) rounded onto the output grid for each input code along the last axis, most negative first; NaN gives the zero point."""
    codes = np.arange(dtype.min, dtype.max + 1)
    with np.errstate(all="ignore"):
        ratio = np.asarray(function(in_scale * (codes - in_zero_point)), dtype=np.float64)
        ratio = ratio / out_scale
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=2.0 ** 40, neginf=-2.0 ** 40)
    return dtype.clamp(np.rint(ratio) + out_zero_point)


def check_stream_ops(graph):
    for node in graph.topological_order():
        if isinstance(node, (ElementwiseOp, ReduceOp, SoftmaxOp, ConcatOp)):
            raise ConfigError("mode='stream' cannot build %s (%s); it needs mode='fixed'"
                              % (node.name, node.op_type))


def lower_elementwise(node, ctx):
    graph, plan = ctx.graph, ctx.plan
    op, out = node.op_type, node.outputs[0]
    live = list(dict.fromkeys(n for n in node.inputs if not graph.tensor(n).is_constant))
    if not live:
        raise ConfigError("%s (%s) has no live input, so it is a constant" % (node.name, op))
    shape = graph.tensor(live[0]).shape
    if any(graph.tensor(n).shape != shape for n in live):
        raise ConfigError("%s (%s) takes tensors of shapes %s, and live operands must have one "
                          "shape" % (node.name, op, " and ".join(
                              str(graph.tensor(n).shape) for n in live)))
    if graph.tensor(out).numel != graph.tensor(live[0]).numel:
        raise ConfigError("%s (%s) broadcasts a constant past the shape %s of its live operand"
                          % (node.name, op, shape))
    if op in BOOLEAN:
        plan.annotate(out, QuantSpec(1.0, 0, plan.act_dtype))
    if len(live) == 1:
        return [table(node, ctx, live[0], shape)]
    return [eltwise(node, ctx, live, shape)]


def table(node, ctx, source, shape):
    """Each lane's table, its constants broadcast to the live operand's shape."""
    graph, plan = ctx.graph, ctx.plan
    lanes = int(np.prod(shape))
    constants = {n: np.broadcast_to(graph.tensor(n).data, shape).reshape(lanes, 1)
                 for n in node.inputs if n != source}

    def function(x):
        operands = [x if n == source else constants[n] for n in node.inputs]
        return evaluate(node.op_type, operands, node.attrs.get("function_attrs", {}))

    rows = lookup_table(function, plan.scale(source), plan.zero_point(source),
                        plan.scale(node.outputs[0]), plan.zero_point(node.outputs[0]),
                        plan.act_dtype)
    rows = np.broadcast_to(rows, (lanes, rows.shape[-1]))
    return TableUnit.per_lane(node.name, [source], node.outputs, rows, plan.act_dtype,
                              plan.act_dtype)


def lower_concat(node, ctx):
    """Each piece's codes moved onto the output grid by a table per lane, a feature at a time."""
    graph, plan, dtype = ctx.graph, ctx.plan, ctx.plan.act_dtype
    out, rows, widths = node.outputs[0], [], []
    for name in node.inputs:
        width = graph.tensor(name).numel
        scale = np.broadcast_to(plan.scale(name), width).reshape(-1, 1)
        zero = np.broadcast_to(plan.zero_point(name), width).reshape(-1, 1)
        rows.append(lookup_table(lambda x: x, scale, zero, plan.scale(out), plan.zero_point(out),
                                 dtype))
        widths.append(width)
    return [ConcatUnit.per_lane(node.name, node.inputs, node.outputs, np.vstack(rows), dtype,
                                dtype, widths)]


def eltwise(node, ctx, live, shape):
    plan = ctx.plan
    op, out = node.op_type, node.outputs[0]
    dtype, lanes = plan.act_dtype, int(np.prod(shape))
    inputs = list(node.inputs)
    out_scale, out_zero = plan.scale(out), plan.zero_point(out)

    def multipliers(reals):
        requant = Requantizer.from_real_multipliers(reals, dtype, plan.mult_bits)
        return [int(m) for m in requant.multipliers], requant.shift

    def unit(kind, operands, mults, mult=1, shift=0, sources=None, zero_points=None,
             tables=None):
        sources = sources or [live.index(n) for n in operands]
        zero_points = zero_points or [plan.zero_point(n) for n in operands]
        return EltwiseUnit(node.name, live, node.outputs, shape, kind, sources, zero_points,
                           mults, mult, shift, dtype, out_zero, plan.mult_bits, tables)

    if op in ("Add", "Sub"):
        sign = -1 if op == "Sub" else 1
        mults, shift = multipliers([plan.scale(inputs[0]) / out_scale,
                                    sign * plan.scale(inputs[1]) / out_scale])
        return unit("add", inputs, mults, shift=shift)
    if op == "Mul":
        mults, shift = multipliers([plan.scale(inputs[0]) * plan.scale(inputs[1]) / out_scale])
        return unit("mul", inputs, [1, 1], mult=mults[0], shift=shift)
    if op in COMPARISONS:
        kind, swap = COMPARISONS[op]
        operands = inputs[::-1] if swap else inputs
        scales = [plan.scale(n) for n in operands]
        mults, _ = multipliers([s / max(scales) for s in scales])
        return unit(kind, operands, [mults[0], -mults[1]])
    if op in ("Max", "Min"):
        return unit(op.lower(), inputs, [0, 0], tables=[onto(plan, n, out) for n in inputs])
    if op in ("And", "Or"):
        return unit(op.lower(), inputs, [1, 1])
    if op == "Where":
        return where(node, ctx, live, lanes, unit)
    raise ConfigError("%s (%s) needs every operand but one to be a constant" % (node.name, op))


def onto(plan, source, target):
    """The table that moves a code of `source` onto the grid of `target`."""
    return lookup_table(lambda x: x, plan.scale(source), plan.zero_point(source),
                        plan.scale(target), plan.zero_point(target), plan.act_dtype)


def where(node, ctx, live, lanes, unit):
    """A constant condition is 0 or 1 per lane; a constant branch is its code on the output grid."""
    graph, plan = ctx.graph, ctx.plan
    out = node.outputs[0]
    sources, zero_points, tables = [], [], []
    for position, name in enumerate(node.inputs):
        if name in live:
            sources.append(live.index(name))
            zero_points.append(plan.zero_point(name))
            tables.append(onto(plan, name, out) if position else None)
            continue
        data = np.broadcast_to(graph.tensor(name).data, graph.tensor(live[0]).shape)
        if position == 0:
            codes = (data != 0).astype(np.int64)
        else:
            codes = plan.act_dtype.clamp(np.rint(data / plan.scale(out)) + plan.zero_point(out))
        sources.append(codes.reshape(lanes))
        zero_points.append(0)
        tables.append(None)
    return unit("where", node.inputs, [0, 0, 0], sources=sources, zero_points=zero_points,
                tables=tables)
