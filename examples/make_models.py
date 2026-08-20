"""Builds the example ONNX models used by the tests and the README walkthrough.

Written with the compiler's own ModelBuilder so the repository has no build
time dependency on the reference onnx package.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "compiler"))

from onnx2fpga.p1_ingest import ModelBuilder, TensorProto


class ModelFactory:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def dense(self, fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

    def kernel(self, out_ch, in_ch, kh, kw):
        limit = np.sqrt(6.0 / (in_ch * kh * kw + out_ch))
        return self.rng.uniform(-limit, limit, (out_ch, in_ch, kh, kw)).astype(np.float32)

    def bias(self, size):
        return self.rng.uniform(-0.1, 0.1, size).astype(np.float32)

    def mlp(self, widths=(32, 64, 16, 10)):
        builder = ModelBuilder("mlp").add_input("x", (1, widths[0]))
        cursor = "x"
        for layer, (fan_in, fan_out) in enumerate(zip(widths, widths[1:])):
            last = layer == len(widths) - 2
            builder.add_initializer("w%d" % layer, self.dense(fan_in, fan_out))
            builder.add_initializer("b%d" % layer, self.bias(fan_out))
            target = "y" if last else "h%d" % layer
            builder.add_node("Gemm", [cursor, "w%d" % layer, "b%d" % layer],
                             [target if last else target + "_pre"],
                             name="fc%d" % layer)
            if last:
                cursor = target
            else:
                builder.add_node("Relu", [target + "_pre"], [target],
                                 name="relu%d" % layer)
                cursor = target
        return builder.add_output("y", (1, widths[-1])).build()

    def residual(self, width=16, classes=10):
        """A skip connection, which is the shape that makes buffer sizing
        matter: the join cannot take a beat from the short branch until the
        long one catches up, and everything in between has to be held."""
        builder = ModelBuilder("residual").add_input("x", (1, width))
        for index, (fan_in, fan_out) in enumerate(
                [(width, width), (width, width), (width, width)]):
            builder.add_initializer("w%d" % index, self.dense(fan_in, fan_out))
            builder.add_initializer("b%d" % index, self.bias(fan_out))
        builder.add_initializer("wo", self.dense(width, classes))
        builder.add_initializer("bo", self.bias(classes))

        builder.add_node("Gemm", ["x", "w0", "b0"], ["h_pre"], name="stem")
        builder.add_node("Relu", ["h_pre"], ["h"], name="stem_relu")
        builder.add_node("Gemm", ["h", "w1", "b1"], ["a_pre"], name="branch0")
        builder.add_node("Relu", ["a_pre"], ["a"], name="branch0_relu")
        builder.add_node("Gemm", ["a", "w2", "b2"], ["b"], name="branch1")
        builder.add_node("Add", ["h", "b"], ["sum"], name="residual_add")
        builder.add_node("Gemm", ["sum", "wo", "bo"], ["y"], name="head")
        return builder.add_output("y", (1, classes)).build()

    def cnn(self, image=12, in_ch=1, conv_ch=4, classes=10):
        conv_out = image - 2
        pooled = conv_out // 2
        flat = pooled * pooled * conv_ch
        builder = (ModelBuilder("cnn")
                   .add_input("x", (1, in_ch, image, image))
                   .add_initializer("k0", self.kernel(conv_ch, in_ch, 3, 3))
                   .add_initializer("cb0", self.bias(conv_ch))
                   .add_initializer("w1", self.dense(flat, classes))
                   .add_initializer("b1", self.bias(classes))
                   .add_node("Conv", ["x", "k0", "cb0"], ["c0"], name="conv0",
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0],
                             dilations=[1, 1], group=1)
                   .add_node("Relu", ["c0"], ["r0"], name="relu0")
                   .add_node("MaxPool", ["r0"], ["p0"], name="pool0",
                             kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
                   .add_node("Flatten", ["p0"], ["f0"], name="flat0", axis=1)
                   .add_node("Gemm", ["f0", "w1", "b1"], ["y"], name="fc1"))
        return builder.add_output("y", (1, classes)).build()


class Grid:
    """One tensor's integer grid, in the form ONNX records it."""

    def __init__(self, scale, zero_point, dtype):
        self.scale = np.asarray(scale, dtype=np.float32)
        self.zero_point = np.asarray(zero_point)
        self.dtype = dtype

    @classmethod
    def for_activation(cls, values, dtype=TensorProto.UINT8):
        """Asymmetric, the way a static quantizer fits a range it measured."""
        low, high = float(np.min(values)), float(np.max(values))
        low, high = min(low, 0.0), max(high, 0.0)
        if dtype == TensorProto.UINT8:
            scale = max((high - low) / 255.0, 1e-9)
            zero = int(np.clip(round(-low / scale), 0, 255))
            return cls(np.float32(scale), np.uint8(zero), dtype)
        scale = max(max(abs(low), abs(high)) / 127.0, 1e-9)
        return cls(np.float32(scale), np.int8(0), dtype)

    @classmethod
    def for_weights(cls, weights, channel_axis):
        """Symmetric per output channel, which is what accelerators need."""
        moved = np.moveaxis(weights, channel_axis, 0).reshape(weights.shape[channel_axis], -1)
        scale = np.maximum(np.max(np.abs(moved), axis=1), 1e-9) / 127.0
        return cls(scale.astype(np.float32),
                   np.zeros(scale.shape, dtype=np.int8), TensorProto.INT8)

    def quantize(self, values, channel_axis=None):
        scale, zero = self._broadcast(values, channel_axis)
        limits = (0, 255) if self.dtype == TensorProto.UINT8 else (-128, 127)
        raw = np.clip(np.rint(values / scale) + zero, *limits)
        return raw.astype(np.uint8 if self.dtype == TensorProto.UINT8 else np.int8)

    def dequantize(self, values, channel_axis=None):
        scale, zero = self._broadcast(values, channel_axis)
        return (values.astype(np.float64) - zero) * scale

    def fake(self, values, channel_axis=None):
        return self.dequantize(self.quantize(values, channel_axis), channel_axis)

    def _broadcast(self, values, channel_axis):
        if channel_axis is None or self.scale.size == 1:
            return np.float64(self.scale.reshape(-1)[0]), np.int64(self.zero_point.reshape(-1)[0])
        shape = [1] * values.ndim
        shape[channel_axis] = -1
        return (self.scale.astype(np.float64).reshape(shape),
                self.zero_point.astype(np.int64).reshape(shape))


class QdqModelFactory(ModelFactory):
    """Builds models already quantized, in the QuantizeLinear and
    DequantizeLinear form a static quantizer emits."""

    def __init__(self, seed=0, activation_dtype=TensorProto.UINT8):
        ModelFactory.__init__(self, seed)
        self.activation_dtype = activation_dtype
        self._counter = 0

    def _grid_initializers(self, builder, stem, grid):
        builder.add_initializer(stem + "_scale", grid.scale)
        builder.add_initializer(stem + "_zp", grid.zero_point)
        return stem + "_scale", stem + "_zp"

    def qdq(self, builder, tensor, grid):
        """Quantize then dequantize, which is how a QDQ model states that a
        tensor lives on a particular integer grid."""
        stem = "%s_q%d" % (tensor, self._counter)
        self._counter += 1
        scale, zero = self._grid_initializers(builder, stem, grid)
        builder.add_node("QuantizeLinear", [tensor, scale, zero], [stem + "_i"],
                         name=stem + "_quant")
        builder.add_node("DequantizeLinear", [stem + "_i", scale, zero], [stem + "_d"],
                         name=stem + "_dequant")
        return stem + "_d"

    def quantized_weight(self, builder, stem, weights, channel_axis):
        grid = Grid.for_weights(weights, channel_axis)
        builder.add_initializer(stem + "_i", grid.quantize(weights, channel_axis))
        scale, zero = self._grid_initializers(builder, stem, grid)
        builder.add_node("DequantizeLinear", [stem + "_i", scale, zero], [stem + "_d"],
                         name=stem + "_dequant", axis=channel_axis)
        return stem + "_d", grid

    def mlp(self, widths=(32, 64, 16, 10), samples=32):
        layers = [(self.dense(a, b), self.bias(b)) for a, b in zip(widths, widths[1:])]
        probe = self.rng.normal(0, 1, (samples, widths[0])).astype(np.float32)
        ranges = self._mlp_ranges(layers, probe)
        self._reference = (layers, ranges)

        builder = ModelBuilder("qdq_mlp").add_input("x", (1, widths[0]))
        cursor = self.qdq(builder, "x", ranges[0])
        for index, (weights, bias) in enumerate(layers):
            name, _ = self.quantized_weight(builder, "w%d" % index, weights, 1)
            builder.add_initializer("b%d" % index, bias)
            last = index == len(layers) - 1
            raw = "y_raw" if last else "h%d_raw" % index
            builder.add_node("Gemm", [cursor, name, "b%d" % index], [raw],
                             name="fc%d" % index, transB=0)
            if last:
                cursor = self.qdq(builder, raw, ranges[index + 1])
            else:
                builder.add_node("Relu", [raw], ["h%d" % index], name="relu%d" % index)
                cursor = self.qdq(builder, "h%d" % index, ranges[index + 1])
        builder.add_node("Identity", [cursor], ["y"], name="out")
        return builder.add_output("y", (1, widths[-1])).build()

    def reference(self, values):
        """What the QDQ model means, evaluated the way a runtime evaluates it:
        float arithmetic between each quantize and dequantize pair. This is the
        specification the compiler's integer pipeline has to match."""
        layers, grids = self._reference
        cursor = grids[0].fake(np.asarray(values, dtype=np.float64))
        for index, (weights, bias) in enumerate(layers):
            wgrid = Grid.for_weights(weights, 1)
            cursor = cursor @ wgrid.fake(weights, 1) + bias
            if index < len(layers) - 1:
                cursor = np.maximum(cursor, 0)
            cursor = grids[index + 1].fake(cursor)
        return cursor

    def _mlp_ranges(self, layers, probe):
        """Fake quantize forward, so every recorded grid is one the model
        actually produces."""
        grids = [Grid.for_activation(probe, self.activation_dtype)]
        cursor = grids[0].fake(probe)
        for index, (weights, bias) in enumerate(layers):
            wgrid = Grid.for_weights(weights, 1)
            cursor = cursor @ wgrid.fake(weights, 1) + bias
            if index < len(layers) - 1:
                cursor = np.maximum(cursor, 0)
            grids.append(Grid.for_activation(cursor, self.activation_dtype))
            cursor = grids[-1].fake(cursor)
        return grids


def main():
    out = pathlib.Path(__file__).resolve().parent / "models"
    out.mkdir(exist_ok=True)
    factory = ModelFactory(seed=7)
    factory.mlp().save(out / "mlp.onnx")
    factory.cnn().save(out / "cnn.onnx")
    factory.residual().save(out / "residual.onnx")
    QdqModelFactory(seed=7).mlp().save(out / "qdq_mlp_uint8.onnx")
    QdqModelFactory(seed=7, activation_dtype=TensorProto.INT8).mlp().save(
        out / "qdq_mlp_int8.onnx")
    for path in sorted(out.glob("*.onnx")):
        print("wrote %s (%d bytes)" % (path, path.stat().st_size))


if __name__ == "__main__":
    main()
