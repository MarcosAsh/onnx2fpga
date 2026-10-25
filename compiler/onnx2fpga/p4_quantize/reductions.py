"""Reductions and softmax in fixed latency mode: otf_reduce and otf_softmax units."""

import math

import numpy as np

from ..errors import ConfigError
from ..p2_graph.datatype import IntType, QuantSpec
from ..p3_ops.hardware_ops import MatVecUnit
from ..p3_ops.reductions import ARG, ReduceUnit, SoftmaxUnit
from .requantize import Requantizer

# Fraction bits of the exponential table, mantissa bits of the sum, fraction bits of a log.
EXP_BITS = 12
MANTISSA_BITS = 10
LOG_BITS = 6


def lower_reduce(node, ctx):
    graph, plan = ctx.graph, ctx.plan
    source, out = node.inputs[0], node.outputs[0]
    lanes, dtype = graph.tensor(source).numel, plan.act_dtype
    shape = graph.tensor(out).shape
    op = {"ArgMax": "argmax", "ArgMin": "argmin", "ReduceMax": "max", "ReduceMin": "min",
          "ReduceSum": "sum", "ReduceMean": "sum"}[node.op_type]
    if node.op_type in ARG:
        if graph.consumers(out):
            raise ConfigError("%s (%s) gives an index, which only a graph output can take"
                              % (node.name, node.op_type))
        plan.annotate(out, QuantSpec(1.0))
        return [ReduceUnit(node.name, [source], [out], lanes, op, widen(graph, source, dtype),
                           shape, last=node.attrs.get("last"))]
    if op != "sum":
        ctx.require_same_grid(source, out, node.op_type)
        plan.propagate(source, out)
        return [ReduceUnit(node.name, [source], [out], lanes, op, dtype, shape)]
    real = plan.scale(source) / plan.scale(out) / (lanes if node.op_type == "ReduceMean" else 1)
    requant = Requantizer.from_real_multipliers([real], dtype, plan.mult_bits,
                                                zero_point=plan.zero_point(out))
    return [ReduceUnit(node.name, [source], [out], lanes, op, dtype, shape,
                       zero_point=plan.zero_point(source), requant=requant,
                       mult_bits=plan.mult_bits)]


def widen(graph, source, dtype, extra=3):
    """A Gemm read only by an ArgMax gives its scores on a grid 2^extra times finer, so close classes rarely tie."""
    producer = graph.producer(source)
    if (not isinstance(producer, MatVecUnit) or len(graph.consumers(source)) != 1
            or source in graph.outputs or producer.requant.lower_clamp != dtype.min):
        return dtype
    requant, wide = producer.requant, IntType(dtype.bits + extra, True)
    shift, multipliers = requant.shift - extra, requant.multipliers
    if shift < 0:
        shift, multipliers = 0, multipliers << -shift
    producer.requant = Requantizer(multipliers, shift, wide,
                                   zero_point=requant.zero_point << extra)
    producer.out_dtype = wide
    return wide


def lower_softmax(node, ctx):
    """Softmax and the Normalizer give 0 to 1 or -1 to 1, so a calibrated output takes the grid 1/127."""
    graph, plan = ctx.graph, ctx.plan
    source, out = node.inputs[0], node.outputs[0]
    lanes, dtype = graph.tensor(source).numel, plan.act_dtype
    if node.op_type != "LogSoftmax" and not (plan.from_model and plan.has(out)):
        plan.annotate(out, QuantSpec(1.0 / dtype.max))
    scale, zero = plan.scale(source), plan.zero_point(source)
    out_scale = plan.scale(out)
    index = np.arange(1 << dtype.bits)
    mantissas = 1 + (np.arange(1 << MANTISSA_BITS) + 0.5) / (1 << MANTISSA_BITS)
    common = dict(lanes=lanes, op=node.op_type, in_dtype=dtype, out_dtype=dtype,
                  mantissa_bits=MANTISSA_BITS, out_zero_point=plan.zero_point(out))
    if node.op_type == "Normalizer":
        codes = index + dtype.min - zero
        return [SoftmaxUnit(node.name, [source], [out], sum_table=np.abs(codes),
                            numer_table=codes, min_exp=0, subtract_max=False,
                            **divide(lanes * int(np.abs(codes).max()), out_scale, plan),
                            **common)]
    exponential = np.rint(2.0 ** EXP_BITS * np.exp(-scale * index))
    total_bits = int(lanes * 2 ** EXP_BITS).bit_length()
    if node.op_type == "Softmax":
        return [SoftmaxUnit(node.name, [source], [out], sum_table=exponential,
                            numer_table=None, min_exp=EXP_BITS, subtract_max=True,
                            **divide(lanes * 2 ** EXP_BITS, out_scale, plan), **common)]
    # LogSoftmax: s * (x - m) - ln(S / 2^12), with ln S split into the exponent and the mantissa.
    step = 2.0 ** LOG_BITS / out_scale
    mantissa_log = np.rint(np.log(mantissas) * step)
    exponent_log = np.rint(np.arange(total_bits - EXP_BITS) * math.log(2) * step)
    largest = int(mantissa_log.max() + exponent_log.max())
    return [SoftmaxUnit(node.name, [source], [out], sum_table=exponential,
                        numer_table=np.rint(-index * scale * step), min_exp=EXP_BITS,
                        subtract_max=True, scale_table=mantissa_log, exp_table=exponent_log,
                        scale_bits=max(2, (largest.bit_length() + 1)), shift=LOG_BITS,
                        **common)]


def divide(largest, out_scale, plan):
    """The reciprocal table of the mantissa, K * 2^P / m with K one over the output scale."""
    limit = (1 << (plan.mult_bits - 1)) - 1
    k = 1.0 / out_scale
    shift = max(0, int(math.floor(math.log2(limit / k))))
    mantissas = 1 + (np.arange(1 << MANTISSA_BITS) + 0.5) / (1 << MANTISSA_BITS)
    table = np.minimum(np.rint(k * 2.0 ** shift / mantissas), limit)
    return dict(scale_table=table, exp_table=None, scale_bits=plan.mult_bits,
                shift=shift + int(largest).bit_length() - 1)
