"""Rewrites float operators into integer streaming units."""

import numpy as np

from ..p2_graph.datatype import INT32, IntType, QuantSpec
from ..p2_graph.tensor import Tensor
from ..p3_ops.elementwise import ElementwiseOp
from ..p3_ops.onnx_ops import AddOp, BinarizerOp, ConvOp, FlattenOp, GemmOp, MaxPoolOp, ReluOp
from ..p3_ops.hardware_ops import (ActivationUnit, AddUnit, MatVecUnit, PoolUnit,
                                   SlidingWindowUnit)
from ..pipeline import Pass
from ..p3_ops.trees import TreeEnsembleOp
from .elementwise import lower_elementwise
from .requantize import Requantizer
from .trees import lower_trees


class FuseRelu(Pass):
    """Folds a Relu into the output clamp of the Gemm or Conv before it."""

    name = "fuse-relu"

    def run(self, graph, context):
        for node in list(graph.topological_order()):
            if not isinstance(node, ReluOp):
                continue
            producer = graph.producer(node.inputs[0])
            if (isinstance(producer, (GemmOp, ConvOp))
                    and len(graph.consumers(node.inputs[0])) == 1
                    and node.inputs[0] not in graph.outputs):
                producer.attrs["relu"] = True
                producer.outputs = list(node.outputs)
                graph.remove_node(node)
        return graph


class Lowering:
    """Rewrites the frontend node types in `handles` into hardware nodes."""

    handles = ()

    def apply(self, node, ctx):
        raise NotImplementedError


class LoweringContext:
    """`fixed` lowers for fixed latency mode, where Relu and Add become elementwise units."""

    def __init__(self, graph, plan, fixed=False):
        self.graph = graph
        self.plan = plan
        self.fixed = fixed

    def scale(self, name):
        return self.plan.scale(name)

    def zero_point(self, name):
        return self.plan.zero_point(name)

    def require_same_grid(self, source, target, op_name):
        """For units that cannot rescale, when the model gave the output its own grid."""
        if not self.plan.from_model or not self.plan.has(target):
            return
        same_scale = abs(self.scale(source) - self.scale(target)) <= 1e-9 * max(
            self.scale(source), self.scale(target), 1e-12)
        if not same_scale or self.zero_point(source) != self.zero_point(target):
            raise NotImplementedError("%s cannot change the quantization grid from %s to %s"
                                      % (op_name, source, target))

    def stage_tensor(self, stem):
        name = self.graph.fresh_name(stem)
        self.graph.add_tensor(Tensor(name, None, self.plan.act_dtype))
        return name

    def window(self, node, kernel, dilations):
        source = self.graph.tensor(node.inputs[0])
        return SlidingWindowUnit(
            node.name + "_swu", [node.inputs[0]], [self.stage_tensor(node.name + "_win")],
            ifm_dim=(source.shape[1], source.shape[2]), ifm_ch=source.shape[3],
            kernel=kernel, strides=node.attrs["strides"], pads=node.attrs["pads"],
            dilations=dilations, dtype=self.plan.act_dtype,
            pad_value=self.zero_point(node.inputs[0]))

    def matvec(self, node, matrix, inputs, vectors):
        """Per feature input scales fold into the weights, zero points into the bias."""
        plan = self.plan
        spec = plan.spec(node.inputs[0])
        weight_spec = plan.specs.get(node.inputs[1])
        in_scale, in_zero = float(spec.scale[0]), np.broadcast_to(spec.zero_point,
                                                                  matrix.shape[:1])
        if spec.scale.size > 1:
            matrix, in_scale, weight_spec = matrix * spec.scale[:, None], 1.0, None
        scales = plan.weight_scales(matrix, weight_spec, node.name)
        weights = plan.quantize_weights(matrix, scales, node.name)
        raw = np.zeros(matrix.shape[1])
        if len(node.inputs) > 2:
            raw = np.asarray(self.graph.tensor(node.inputs[2]).data,
                             dtype=np.float64).reshape(-1)
        # A large zero point times the rounded weights would scale their rounding error by it.
        bias = np.rint((raw - in_scale * (in_zero @ matrix)) / (in_scale * scales))
        requant = Requantizer.from_real_multipliers(
            in_scale * scales / self.scale(node.outputs[0]), plan.act_dtype,
            plan.mult_bits, relu=bool(node.attrs.get("relu")),
            zero_point=self.zero_point(node.outputs[0]))
        return MatVecUnit(node.name, inputs, node.outputs, weights, INT32.clamp(bias),
                          requant, plan.act_dtype, plan.act_dtype,
                          plan.weight_dtype_for(node.name), vectors=vectors)


def lower_gemm(node, ctx):
    """A label is one bit, the sign of the margin: a multiplier of 1 and a clamp to 0 and 1."""
    source = ctx.graph.tensor(node.inputs[0])
    matrix = np.asarray(ctx.graph.tensor(node.inputs[1]).data, dtype=np.float64)
    if node.attrs.get("label"):
        ctx.plan.annotate(node.outputs[0], QuantSpec(1.0))
    unit = ctx.matvec(node, matrix, [node.inputs[0]], source.numel // matrix.shape[0])
    if node.attrs.get("label"):
        unit.out_dtype = IntType(1, False)
        unit.requant = Requantizer(np.ones(1), 0, unit.out_dtype)
    return [unit]


def lower_binarizer(node, ctx):
    """Identity weights and a bias of minus the floor of the threshold's code, clamped to 0 and 1."""
    source = ctx.graph.tensor(node.inputs[0])
    spec, width = ctx.plan.spec(node.inputs[0]), source.shape[-1]
    code = np.broadcast_to(node.attrs["threshold"] / spec.scale + spec.zero_point, (width,))
    ctx.plan.annotate(node.outputs[0], QuantSpec(1.0))
    dtype = ctx.plan.act_dtype
    requant = Requantizer(np.ones(width), 0, dtype, lower_clamp=0, upper_clamp=1)
    return [MatVecUnit(node.name, node.inputs, node.outputs, np.eye(width),
                       -np.clip(np.floor(code), dtype.min - 1, dtype.max), requant, dtype,
                       dtype, ctx.plan.weight_dtype_for(node.name),
                       vectors=source.numel // width)]


def lower_conv(node, ctx):
    weights = np.asarray(ctx.graph.tensor(node.inputs[1]).data, dtype=np.float64)
    kh, kw, cin, cout = weights.shape
    window = ctx.window(node, (kh, kw), node.attrs["dilations"])
    out_h, out_w = window.ofm_dim
    matrix = weights.reshape(kh * kw * cin, cout)
    return [window, ctx.matvec(node, matrix, window.outputs, out_h * out_w)]


def lower_maxpool(node, ctx):
    """Max preserves scale, so the pooled stream keeps the input's grid."""
    kernel = node.attrs["kernel_shape"]
    window = ctx.window(node, kernel, node.attrs.get("dilations", (1, 1)))
    out_h, out_w = window.ofm_dim
    ctx.require_same_grid(node.inputs[0], node.outputs[0], "MaxPool")
    ctx.plan.propagate(node.inputs[0], node.outputs[0])
    pool = PoolUnit(node.name, window.outputs, node.outputs, channels=window.ifm_ch,
                    window_size=kernel[0] * kernel[1], windows=out_h * out_w,
                    dtype=ctx.plan.act_dtype)
    return [window, pool]


def lower_relu(node, ctx):
    """Only reached when the Relu could not be fused into its producer."""
    if ctx.fixed:
        return lower_elementwise(node, ctx)
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


def lower_add(node, ctx):
    if ctx.fixed:
        return lower_elementwise(node, ctx)
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
                    channels=source.shape[-1], left_mult=left_mult,
                    right_mult=right_mult, shift=shift,
                    in_dtype=ctx.plan.act_dtype, out_dtype=ctx.plan.act_dtype,
                    bias=correction, zero_point=ctx.zero_point(node.outputs[0]))]


def lower_flatten(node, ctx):
    """A channel-last flatten is a relabelling, not hardware."""
    ctx.graph.rewire(node.outputs[0], node.inputs[0])
    ctx.plan.propagate(node.outputs[0], node.inputs[0])
    return []


BUILTIN = {GemmOp: lower_gemm, ConvOp: lower_conv, MaxPoolOp: lower_maxpool,
           ReluOp: lower_relu, AddOp: lower_add, FlattenOp: lower_flatten,
           BinarizerOp: lower_binarizer, TreeEnsembleOp: lower_trees,
           ElementwiseOp: lower_elementwise}


class LowerToHardware(Pass):
    name = "lower-to-hardware"

    def __init__(self, plan, lowerings=(), fixed=False):
        self.plan = plan
        self.fixed = fixed
        self.rules = dict(BUILTIN)
        for lowering in lowerings:
            for op_class in lowering.handles:
                self.rules[op_class] = lowering.apply

    def run(self, graph, context):
        ctx = LoweringContext(graph, self.plan, self.fixed)
        for name in graph.inputs:
            graph.tensor(name).dtype = self.plan.act_dtype
        for node in list(graph.topological_order()):
            rule = self.rules.get(type(node))
            if rule is None:
                raise NotImplementedError("no lowering for %s" % type(node).__name__)
            graph.replace_node(node, rule(node, ctx))
        graph.infer()
        context.artifacts["plan"] = self.plan
        return graph
