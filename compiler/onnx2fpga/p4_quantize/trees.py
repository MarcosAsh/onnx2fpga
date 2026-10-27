"""Feature grids taken from the tree thresholds, and the integer ensemble that agrees with the float one on them."""

import numpy as np

from ..p2_graph.datatype import IntType, QuantSpec
from ..p3_ops.hardware_ops import TreeEnsembleUnit
from ..p3_ops.trees import EQ, GT, GTE, LEAF, LEQ, LT, NEQ, Ensemble, TreeEnsembleOp
from .plan import QuantizationPlan

FEATURE_BITS = 16
LEAF_BITS = 16


def tree_op(graph):
    return next((node for node in graph.nodes if isinstance(node, TreeEnsembleOp)), None)


def leaf_scale(ensemble, bits):
    peak = float(np.max(np.abs(ensemble.value[ensemble.mode == LEAF])))
    return peak / ((1 << (bits - 1)) - 1) if peak > 0 else 1.0


def plan_trees(graph, mult_bits=18, features=None, leaves=None):
    """Per feature grids with a power of two step, so round thresholds land on codes, spanning the thresholds with a code to spare each end."""
    op = tree_op(graph)
    ensemble = op.ensemble
    dtype = features or IntType(FEATURE_BITS, True)
    leaves = leaves or IntType(LEAF_BITS, True)
    features = graph.tensor(op.inputs[0]).numel
    scales, zeros = np.ones(features), np.zeros(features, dtype=np.int64)
    split = ensemble.mode != LEAF
    for feature in range(features):
        thresholds = ensemble.threshold[split & (ensemble.feature == feature)]
        if thresholds.size:
            low, high = float(thresholds.min()), float(thresholds.max())
            step = ((high - low) or max(abs(low), 1.0)) / (dtype.max - dtype.min - 2)
            scales[feature] = 2.0 ** np.ceil(np.log2(step))
            zeros[feature] = int(round(dtype.min + 1 - low / scales[feature]))
    plan = QuantizationPlan(mult_bits)
    plan.act_dtype = dtype
    plan.weight_dtype = leaves
    plan.source = "thresholds"
    plan.annotate(op.inputs[0], QuantSpec(scales, zeros, dtype))
    plan.annotate(op.outputs[0], QuantSpec(
        1.0 if op.decision is not None or op.label else leaf_scale(ensemble, leaves.bits), 0))
    return plan


def largest_code(threshold, scale, zero, dtype, strict):
    """The largest code whose value scale * (code - zero) is <= threshold, or < it when strict; dtype.min - 1 if none."""
    def below(code):
        real = scale * (code - zero)
        return real < threshold if strict else real <= threshold

    code = np.clip(np.floor(threshold / scale + zero), dtype.min - 1, dtype.max)
    while True:
        up = (code < dtype.max) & below(code + 1)
        down = (code >= dtype.min) & ~below(code)
        if not (up.any() or down.any()):
            return code.astype(np.int64)
        code = code + up - down


def lower_trees(node, ctx):
    """Every split becomes x <= T or x == T on the codes; a split that cannot go both ways is cut out."""
    spec, dtype = ctx.plan.spec(node.inputs[0]), ctx.plan.act_dtype
    source = node.ensemble
    scale, zero = spec.scale[source.feature], spec.zero_point[source.feature]
    leq = largest_code(source.threshold, scale, zero, dtype, strict=False)
    lt = largest_code(source.threshold, scale, zero, dtype, strict=True)
    equal = (leq >= dtype.min) & (scale * (leq - zero) == source.threshold)
    threshold = np.select([source.mode == m for m in (LEQ, LT, GT, GTE)], [leq, lt, leq, lt], leq)
    swapped = np.isin(source.mode, (GT, GTE, NEQ))
    is_eq = np.isin(source.mode, (EQ, NEQ))
    never = np.where(is_eq, ~equal, threshold < dtype.min)
    ever = ~is_eq & (threshold >= dtype.max)

    scale_out = leaf_scale(source, ctx.plan.weight_dtype.bits)
    nodes = []

    def build(k, value):
        if source.mode[k] == LEAF:
            nodes.append([LEAF, 0, 0, 0, 0, int(np.rint(value[k] / scale_out))])
            return len(nodes) - 1
        taken, other = ((source.if_false[k], source.if_true[k]) if swapped[k]
                        else (source.if_true[k], source.if_false[k]))
        if never[k] or ever[k]:
            return build(taken if ever[k] else other, value)
        index = len(nodes)
        nodes.append([EQ if is_eq[k] else LEQ, source.feature[k], threshold[k], 0, 0, 0])
        nodes[index][3] = build(taken, value)
        nodes[index][4] = build(other, value)
        return index

    # With a score per class, each class gets its own copy of every tree that scores it.
    columns = source.value.T if node.label else [source.value]
    roots, scores = [], []
    for score, value in enumerate(columns):
        for root, leaves in zip(source.roots, source.tree_leaves()):
            if not node.label or np.any(value[leaves]):
                roots.append(build(root, value))
                scores.append(score)
    mode, feature, thresholds, if_true, if_false, value = (np.array(c) for c in zip(*nodes))
    leaf = mode == LEAF
    if_true[leaf] = if_false[leaf] = np.flatnonzero(leaf)
    ensemble = Ensemble(roots, feature, thresholds, mode, if_true, if_false, value)
    bias = np.rint(np.broadcast_to(node.base, len(columns)) / scale_out).astype(np.int64)
    if node.decision is not None:
        bias[0] -= int(np.floor(node.decision / scale_out)) + 1
    return [TreeEnsembleUnit(node.name, node.inputs, node.outputs, ensemble, bias,
                             node.decision is not None, len(spec.scale), dtype,
                             ctx.plan.weight_dtype, scores)]
