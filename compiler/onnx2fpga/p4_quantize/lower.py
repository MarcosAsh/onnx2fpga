"""Rewrites float operators into integer streaming hardware nodes.

This is the stage where the graph stops being a mathematical description and
starts being a machine: after it runs, every node corresponds to exactly one
SystemVerilog module and every edge to one stream.
"""

import numpy as np

from ..p2_graph.datatype import INT8, INT32
from ..p3_ops.onnx_ops import AddOp, ConvOp, FlattenOp, GemmOp, MaxPoolOp, ReluOp
from ..p3_ops.hardware_ops import (ActivationUnit, AddUnit, MatVecUnit, PoolUnit,
                                   SlidingWindowUnit)
from ..pipeline import Pass
from .requantize import Requantizer


class FuseRelu(Pass):
    """Folds a following Relu into the producer's output clamp."""

    name = "fuse-relu"

    def run(self, graph, context):
        for node in list(graph.topological_order()):
            if not isinstance(node, ReluOp):
                continue
            producer = graph.producer(node.inputs[0])
            if not isinstance(producer, (GemmOp, ConvOp)):
                continue
            if len(graph.consumers(node.inputs[0])) != 1:
                continue
            if node.inputs[0] in graph.outputs:
                continue
            producer.attrs["relu"] = True
            producer.outputs = list(node.outputs)
            graph.remove_node(node)
        return graph


class Lowering:
    """Rewrites one frontend node into one or more hardware nodes."""

    handles = ()

    def apply(self, node, ctx):
        raise NotImplementedError


class LoweringContext:
    def __init__(self, graph, plan):
        self.graph = graph
        self.plan = plan

    def scale(self, name):
        return self.plan.scale(name)

    def zero_point(self, name):
        return self.plan.zero_point(name)

    def weight_spec(self, name):
        return self.plan.specs.get(name)

    def require_same_grid(self, source, target, op_name):
        """Max is scale preserving, so the unit does no rescale. If the model
        asked for one anyway, say so rather than quietly ignoring it."""
        if not self.plan.from_model or not self.plan.has(target):
            return
        same_scale = abs(self.scale(source) - self.scale(target)) <= 1e-9 * max(
            self.scale(source), self.scale(target), 1e-12)
        if not same_scale or self.zero_point(source) != self.zero_point(target):
            raise NotImplementedError(
                "%s changes the quantization grid between %s and %s; this unit "
                "cannot rescale" % (op_name, source, target))

    def stage_tensor(self, stem, shape, dtype):
        from ..p2_graph.tensor import Tensor
        name = self.graph.fresh_name(stem)
        self.graph.add_tensor(Tensor(name, shape, dtype))
        return name

    def matvec_requant(self, out_name, in_scale, weight_scales, relu):
        reals = in_scale * weight_scales / self.scale(out_name)
        return Requantizer.from_real_multipliers(
            reals, self.plan.act_dtype, self.plan.mult_bits, relu=relu,
            zero_point=self.zero_point(out_name))

    def quantize_bias(self, node, index, unit_scale, channels, weights, in_zero):
        """The model's own bias plus the correction an asymmetric input needs.

        Expanding sum_k (x_k - zx) * w_kc leaves -zx * sum_k w_kc, which is a
        per channel constant, so it costs nothing to fold in here."""
        folded = np.zeros(channels, dtype=np.int64)
        if len(node.inputs) > index:
            raw = np.asarray(self.graph.tensor(node.inputs[index]).data,
                             dtype=np.float64).reshape(-1)
            folded = np.rint(raw / unit_scale).astype(np.int64)
        if in_zero:
            folded = folded - in_zero * np.sum(weights, axis=0).astype(np.int64)
        return INT32.clamp(folded)


class GemmLowering(Lowering):
    handles = (GemmOp,)

    def apply(self, node, ctx):
        source = ctx.graph.tensor(node.inputs[0])
        weights = np.asarray(ctx.graph.tensor(node.inputs[1]).data, dtype=np.float64)
        scales = ctx.plan.weight_scales(weights, ctx.weight_spec(node.inputs[1]))
        quantized = ctx.plan.quantize_weights(weights, scales)
        in_scale = ctx.scale(node.inputs[0])
        in_zero = ctx.zero_point(node.inputs[0])
        bias = ctx.quantize_bias(node, 2, in_scale * scales, weights.shape[1],
                                 quantized, in_zero)
        requant = ctx.matvec_requant(node.outputs[0], in_scale, scales,
                                     bool(node.attrs.get("relu")))
        vectors = source.numel // weights.shape[0]
        return [MatVecUnit(node.name, [node.inputs[0]], node.outputs, quantized,
                           bias, requant, ctx.plan.act_dtype, ctx.plan.act_dtype,
                           ctx.plan.weight_dtype, vectors=vectors)]


class ConvLowering(Lowering):
    handles = (ConvOp,)

    def apply(self, node, ctx):
        source = ctx.graph.tensor(node.inputs[0])
        weights = np.asarray(ctx.graph.tensor(node.inputs[1]).data, dtype=np.float64)
        kh, kw, cin, cout = weights.shape
        matrix = weights.reshape(kh * kw * cin, cout)
        scales = ctx.plan.weight_scales(matrix, ctx.weight_spec(node.inputs[1]))
        quantized = ctx.plan.quantize_weights(matrix, scales)
        in_scale = ctx.scale(node.inputs[0])
        in_zero = ctx.zero_point(node.inputs[0])
        bias = ctx.quantize_bias(node, 2, in_scale * scales, cout, quantized, in_zero)
        requant = ctx.matvec_requant(node.outputs[0], in_scale, scales,
                                     bool(node.attrs.get("relu")))

        window = SlidingWindowUnit(
            node.name + "_swu", [node.inputs[0]],
            [ctx.stage_tensor(node.name + "_win", None, ctx.plan.act_dtype)],
            ifm_dim=(source.shape[1], source.shape[2]), ifm_ch=cin,
            kernel=(kh, kw), strides=node.attrs["strides"], pads=node.attrs["pads"],
            dilations=node.attrs["dilations"], dtype=ctx.plan.act_dtype,
            pad_value=in_zero)
        out_h, out_w = window.ofm_dim
        matvec = MatVecUnit(node.name, window.outputs, node.outputs, quantized,
                            bias, requant, ctx.plan.act_dtype, ctx.plan.act_dtype,
                            ctx.plan.weight_dtype, vectors=out_h * out_w)
        return [window, matvec]


class MaxPoolLowering(Lowering):
    """Max is scale preserving, so the pooled stream keeps the input's grid."""

    handles = (MaxPoolOp,)

    def apply(self, node, ctx):
        source = ctx.graph.tensor(node.inputs[0])
        channels = source.shape[3]
        kernel = node.attrs["kernel_shape"]
        window = SlidingWindowUnit(
            node.name + "_swu", [node.inputs[0]],
            [ctx.stage_tensor(node.name + "_win", None, ctx.plan.act_dtype)],
            ifm_dim=(source.shape[1], source.shape[2]), ifm_ch=channels,
            kernel=kernel, strides=node.attrs["strides"], pads=node.attrs["pads"],
            dilations=node.attrs.get("dilations", (1, 1)), dtype=ctx.plan.act_dtype,
            pad_value=ctx.zero_point(node.inputs[0]))
        out_h, out_w = window.ofm_dim
        ctx.require_same_grid(node.inputs[0], node.outputs[0], "MaxPool")
        ctx.plan.propagate(node.inputs[0], node.outputs[0])
        pool = PoolUnit(node.name, window.outputs, node.outputs, channels=channels,
                        window_size=kernel[0] * kernel[1], windows=out_h * out_w,
                        dtype=ctx.plan.act_dtype)
        return [window, pool]


class ReluLowering(Lowering):
    """Only reached when the Relu could not be fused into its producer."""

    handles = (ReluOp,)

    def apply(self, node, ctx):
        source = ctx.graph.tensor(node.inputs[0])
        channels = source.shape[-1]
        ratio = ctx.scale(node.inputs[0]) / ctx.scale(node.outputs[0])
        requant = Requantizer.from_real_multipliers(
            np.full(channels, ratio), ctx.plan.act_dtype, ctx.plan.mult_bits,
            relu=True, zero_point=ctx.zero_point(node.outputs[0]))
        return [ActivationUnit(node.name, node.inputs, node.outputs, channels=channels,
                               elements=source.numel, requant=requant,
                               in_dtype=ctx.plan.act_dtype, out_dtype=ctx.plan.act_dtype,
                               in_zero_point=ctx.zero_point(node.inputs[0]))]


class AddLowering(Lowering):
    handles = (AddOp,)

    def apply(self, node, ctx):
        streams = [n for n in node.inputs if not ctx.graph.tensor(n).is_constant]
        if len(streams) != 2:
            raise NotImplementedError(
                "Add with a constant operand should have been folded into a bias: %s"
                % node.name)
        source = ctx.graph.tensor(streams[0])
        out_scale = ctx.scale(node.outputs[0])
        left = ctx.scale(streams[0]) / out_scale
        right = ctx.scale(streams[1]) / out_scale
        shift = Requantizer.from_real_multipliers([left, right], ctx.plan.act_dtype,
                                                  ctx.plan.mult_bits).shift
        left_mult = int(round(left * 2 ** shift))
        right_mult = int(round(right * 2 ** shift))
        correction = (-ctx.zero_point(streams[0]) * left_mult
                      - ctx.zero_point(streams[1]) * right_mult)
        return [AddUnit(node.name, streams, node.outputs, elements=source.numel,
                        channels=source.shape[-1],
                        left_mult=left_mult, right_mult=right_mult, shift=shift,
                        in_dtype=ctx.plan.act_dtype, out_dtype=ctx.plan.act_dtype,
                        bias=correction,
                        zero_point=ctx.zero_point(node.outputs[0]))]


class FlattenLowering(Lowering):
    """A channel-last flatten is a relabelling, not a hardware operation."""

    handles = (FlattenOp,)

    def apply(self, node, ctx):
        graph = ctx.graph
        old, new = node.outputs[0], node.inputs[0]
        graph.rewire(old, new)
        ctx.plan.propagate(old, new)
        return []


class LowerToHardware(Pass):
    name = "lower-to-hardware"

    LOWERINGS = (GemmLowering(), ConvLowering(), MaxPoolLowering(), ReluLowering(),
                 AddLowering(), FlattenLowering())

    def __init__(self, plan):
        self.plan = plan
        self._dispatch = {}
        for lowering in self.LOWERINGS:
            for op_class in lowering.handles:
                self._dispatch[op_class] = lowering

    def run(self, graph, context):
        ctx = LoweringContext(graph, self.plan)
        for name in graph.inputs:
            graph.tensor(name).dtype = self.plan.act_dtype
        for node in list(graph.topological_order()):
            lowering = self._dispatch.get(type(node))
            if lowering is None:
                raise NotImplementedError("no lowering for %s" % type(node).__name__)
            graph.replace_node(node, lowering.apply(node, ctx))
        graph.infer()
        context.artifacts["plan"] = self.plan
        return graph
