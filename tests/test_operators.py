"""Import of the ops spec 009 adds: each float graph against numpy's reading of the ONNX model."""

import unittest

import numpy as np

import support  # noqa: F401

from onnx2fpga.errors import UnsupportedOpError
from onnx2fpga.p1_ingest import ModelBuilder, TensorProto
from onnx2fpga.p2_graph.datatype import QuantSpec
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.p4_quantize.lower import LowerToHardware
from onnx2fpga.p4_quantize.plan import QuantizationPlan
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.reference.runner import GraphRunner
from onnx2fpga.targets.device import Device

RNG = np.random.default_rng(5)


def weights(*shape):
    return RNG.normal(0, 1, shape).astype(np.float32)


def run(model, x):
    """The imported graph on x in the model's layout, and its node types."""
    graph = OnnxImporter(model).run()
    name = graph.inputs[0]
    out = GraphRunner(graph).run({name: graph.layouts[name].to_internal(x)})[graph.outputs[0]]
    return out, [node.op_type for node in graph.nodes]


class Folding(unittest.TestCase):
    def test_shape_plumbing(self):
        """A dynamic batch reshape, as PyTorch writes it, folds away."""
        w = weights(8, 4)
        model = (ModelBuilder("plumbing").add_input("x", ("batch", 8))
                 .add_initializer("w", w).add_initializer("minus", np.array([-1], np.int64))
                 .add_initializer("zero", np.array([0], np.int64))
                 .add_node("Shape", ["x"], ["shape"])
                 .add_node("Gather", ["shape", "zero"], ["batch"], axis=0)
                 .add_node("Concat", ["batch", "minus"], ["target"], axis=0)
                 .add_node("Constant", [], ["half"], value=np.array([0.5], np.float32))
                 .add_node("MatMul", ["x", "w"], ["y0"])
                 .add_node("Unsqueeze", ["y0"], ["u"], axes=[1])
                 .add_node("Squeeze", ["u"], ["s"], axes=[1])
                 .add_node("Dropout", ["s"], ["y1"])
                 .add_node("Mul", ["y1", "half"], ["y2"])
                 .add_node("Reshape", ["y2", "target"], ["y"])
                 .add_output("y", ("batch", 4)).build())
        x = weights(1, 8)
        out, kinds = run(model, x)
        np.testing.assert_allclose(out, x @ w * 0.5, rtol=1e-6)
        self.assertEqual(kinds, ["Gemm"])

    def test_flatten_order(self):
        """A Flatten of a map means NCHW order, which the next Gemm's rows take in."""
        w = weights(18, 3)
        x = weights(1, 2, 3, 3)
        for perm in (None, [0, 2, 3, 1]):
            builder = ModelBuilder("flat").add_input("x", (1, 2, 3, 3)).add_initializer("w", w)
            source = "x"
            if perm:
                builder.add_node("Transpose", ["x"], ["t"], perm=perm)
                source = "t"
            model = (builder.add_node("Flatten", [source], ["f"], axis=1)
                     .add_node("Gemm", ["f", "w"], ["y"]).add_output("y", (1, 3)).build())
            flat = np.transpose(x, perm).reshape(1, -1) if perm else x.reshape(1, -1)
            out, kinds = run(model, x)
            np.testing.assert_allclose(out, flat @ w, rtol=1e-5)
            self.assertEqual(kinds, ["Flatten", "Gemm"])

    def test_batchnorm(self):
        """On the inputs it joins the first Gemm, after a Gemm it folds into it."""
        w0, w1 = weights(4, 6), weights(6, 2)
        stats = [np.abs(weights(n)) + 0.5 for n in (4, 4, 4, 4, 6, 6, 6, 6)]
        builder = ModelBuilder("bn").add_input("x", (1, 4)).add_initializer("w0", w0)
        builder.add_initializer("w1", w1)
        for k, array in enumerate(stats):
            builder.add_initializer("s%d" % k, array)
        model = (builder
                 .add_node("BatchNormalization", ["x", "s0", "s1", "s2", "s3"], ["a"])
                 .add_node("MatMul", ["a", "w0"], ["b"])
                 .add_node("BatchNormalization", ["b", "s4", "s5", "s6", "s7"], ["c"],
                           epsilon=0.01)
                 .add_node("Relu", ["c"], ["d"]).add_node("MatMul", ["d", "w1"], ["y"])
                 .add_output("y", (1, 2)).build())

        def norm(v, s, b, m, var, eps):
            return s * (v - m) / np.sqrt(var + eps) + b

        x = weights(1, 4)
        hidden = norm(norm(x, *stats[:4], 1e-5) @ w0, *stats[4:], 0.01)
        out, kinds = run(model, x)
        np.testing.assert_allclose(out, np.maximum(hidden, 0) @ w1, rtol=1e-5)
        self.assertEqual(kinds, ["Gemm", "Relu", "Gemm"])

    def test_batchnorm_conv(self):
        """After a Conv it scales the kernels; on a map with no Conv it is a 1x1 Conv."""
        kernel = weights(2, 1, 3, 3)
        stats = [np.abs(weights(2)) + 0.5 for _ in range(8)]
        builder = ModelBuilder("bnc").add_input("x", (1, 1, 5, 5)).add_initializer("k", kernel)
        for k, array in enumerate(stats):
            builder.add_initializer("s%d" % k, array)
        model = (builder.add_node("Conv", ["x", "k"], ["c"], kernel_shape=[3, 3])
                 .add_node("BatchNormalization", ["c", "s0", "s1", "s2", "s3"], ["b"])
                 .add_node("Relu", ["b"], ["r"])
                 .add_node("BatchNormalization", ["r", "s4", "s5", "s6", "s7"], ["y"])
                 .add_output("y", (1, 2, 3, 3)).build())
        x = weights(1, 1, 5, 5)
        windows = np.lib.stride_tricks.sliding_window_view(x[0, 0], (3, 3))
        conv = np.einsum("hwij,cij->hwc", windows, kernel[:, 0])

        def norm(v, s, b, m, var):
            return s * (v - m) / np.sqrt(var + 1e-5) + b

        out, kinds = run(model, x)
        np.testing.assert_allclose(out[0], norm(np.maximum(norm(conv, *stats[:4]), 0),
                                                *stats[4:]), rtol=1e-5)
        self.assertEqual(kinds, ["Conv", "Relu", "Conv"])

    def test_concat(self):
        """x beside a Gemm on x is one Gemm; selections of x stay a view."""
        w = weights(4, 3)
        model = (ModelBuilder("cat").add_input("x", (1, 4)).add_initializer("w", w)
                 .add_initializer("pick", np.array([3, 0], np.int64))
                 .add_initializer("w2", weights(9, 2))
                 .add_node("MatMul", ["x", "w"], ["h"])
                 .add_node("Gather", ["x", "pick"], ["g"], axis=1)
                 .add_node("Concat", ["x", "h", "g"], ["c"], axis=1)
                 .add_node("MatMul", ["c", "w2"], ["y"]).add_output("y", (1, 2)).build())
        x = weights(1, 4)
        joined = np.concatenate([x, x @ w, x[:, [3, 0]]], axis=1)
        out, kinds = run(model, x)
        np.testing.assert_allclose(out, joined @ model.initializer("w2"), rtol=1e-5)
        self.assertEqual(kinds, ["Gemm", "Gemm"])

    def test_concat_live(self):
        """x beside relu(x @ w) is two tensors, so they join as they are."""
        w = weights(4, 3)
        model = (ModelBuilder("wide").add_input("x", (1, 4)).add_initializer("w", w)
                 .add_initializer("w2", weights(7, 2))
                 .add_node("MatMul", ["x", "w"], ["h"]).add_node("Relu", ["h"], ["r"])
                 .add_node("Concat", ["x", "r"], ["c"], axis=1)
                 .add_node("MatMul", ["c", "w2"], ["y"]).add_output("y", (1, 2)).build())
        x = weights(1, 4)
        joined = np.concatenate([x, np.maximum(x @ w, 0)], axis=1)
        out, kinds = run(model, x)
        np.testing.assert_allclose(out, joined @ model.initializer("w2"), rtol=1e-5)
        self.assertEqual(kinds, ["Gemm", "Relu", "Concat", "Gemm"])


class Quantized(unittest.TestCase):
    def test_qlinear(self):
        """QLinearMatMul then QLinearAdd is one Gemm on the model's grids."""
        b = RNG.integers(-127, 128, (4, 3)).astype(np.int8)
        bias = RNG.integers(-50, 50, 3).astype(np.int8)
        builder = ModelBuilder("ql").add_input("x", (1, 4))
        for name, value in (("xs", 0.05), ("bs", 0.01), ("ys", 0.02), ("cs", 0.03),
                            ("ds", 0.04)):
            builder.add_initializer(name, np.array(value, np.float32))
        for name in ("xz", "yz", "cz", "dz"):
            builder.add_initializer(name, np.array(128, np.uint8))
        model = (builder.add_initializer("b", b).add_initializer("bz", np.array(0, np.int8))
                 .add_initializer("c", bias)
                 .add_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"])
                 .add_node("QLinearMatMul", ["xq", "xs", "xz", "b", "bs", "bz", "ys", "yz"],
                           ["yq"])
                 .add_node("QLinearAdd", ["yq", "ys", "yz", "c", "cs", "bz", "ds", "dz"],
                           ["dq"])
                 .add_node("DequantizeLinear", ["dq", "ds", "dz"], ["y"])
                 .add_output("y", (1, 3)).build())
        graph = OnnxImporter(model).run()
        x = weights(1, 4)
        out = GraphRunner(graph).run({"x": x})[graph.outputs[0]]
        np.testing.assert_allclose(out, x @ (b * 0.01) + bias * 0.03, rtol=1e-5)
        self.assertEqual([n.op_type for n in graph.nodes], ["Gemm"])
        self.assertAlmostEqual(float(graph.annotations[graph.outputs[0]].scale[0]), 0.04)

    def test_integer(self):
        """MatMulInteger on a grid, with ONNX Runtime's Cast and scale Mul after it."""
        b = RNG.integers(-127, 128, (4, 3)).astype(np.int8)
        model = (ModelBuilder("mmi").add_input("x", (1, 4))
                 .add_initializer("xs", np.array(0.05, np.float32))
                 .add_initializer("xz", np.array(3, np.uint8))
                 .add_initializer("b", b).add_initializer("bz", np.array(2, np.int8))
                 .add_initializer("scale", np.array(0.05 * 0.01, np.float32))
                 .add_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"])
                 .add_node("MatMulInteger", ["xq", "b", "xz", "bz"], ["acc"])
                 .add_node("Cast", ["acc"], ["accf"], to=TensorProto.FLOAT)
                 .add_node("Mul", ["accf", "scale"], ["y"])
                 .add_output("y", (1, 3)).build())
        x = weights(1, 4)
        out, kinds = run(model, x)
        np.testing.assert_allclose(out, x @ ((b - 2) * 0.01), rtol=1e-5)
        self.assertEqual(kinds, ["Gemm"])

    def test_conv(self):
        """QLinearConv and ConvInteger become a Conv with dequantized weights."""
        w = RNG.integers(-127, 128, (2, 1, 3, 3)).astype(np.int8)
        x = weights(1, 1, 5, 5)
        for op in ("QLinearConv", "ConvInteger"):
            builder = (ModelBuilder("qc").add_input("x", (1, 1, 5, 5))
                       .add_initializer("xs", np.array(0.05, np.float32))
                       .add_initializer("xz", np.array(0, np.uint8))
                       .add_initializer("w", w).add_initializer("ws", np.array(0.01, np.float32))
                       .add_initializer("wz", np.array(0, np.int8))
                       .add_initializer("ys", np.array(0.1, np.float32))
                       .add_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"]))
            if op == "QLinearConv":
                builder.add_node(op, ["xq", "xs", "xz", "w", "ws", "wz", "ys", "xz"], ["y"],
                                 kernel_shape=[3, 3])
                scale = 0.01
            else:
                builder.add_node(op, ["xq", "w", "xz", "wz"], ["y"], kernel_shape=[3, 3])
                scale = 1 / 0.05
            graph = OnnxImporter(builder.add_output("y", (1, 2, 3, 3)).build()).run()
            out = GraphRunner(graph).run({"x": graph.layouts["x"].to_internal(x)})["y"]
            windows = np.lib.stride_tricks.sliding_window_view(x[0, 0], (3, 3))
            expected = np.einsum("hwij,cij->hwc", windows, w[:, 0] * scale)
            np.testing.assert_allclose(out[0], expected, rtol=1e-5)
            self.assertEqual([n.op_type for n in graph.nodes], ["Conv"])


class Contrib(unittest.TestCase):
    """ONNX Runtime's QOperator ops, each between a QuantizeLinear and a DequantizeLinear."""

    b = RNG.integers(-127, 128, (3, 4)).astype(np.int8)
    c = RNG.integers(-500, 500, 3).astype(np.int32)
    k = RNG.integers(-127, 128, 4).astype(np.int8)

    def run_op(self, op, inputs, width, **attrs):
        builder = ModelBuilder("contrib").add_input("x", (1, 4))
        for name, value in (("xs", 0.05), ("bs", 0.01), ("ys", 0.02)):
            builder.add_initializer(name, np.array(value, np.float32))
        for name in ("xz", "yz"):
            builder.add_initializer(name, np.array(128, np.uint8))
        for name, value in (("bz", np.array(0, np.int8)), ("b", self.b), ("c", self.c),
                            ("k", self.k)):
            builder.add_initializer(name, value)
        model = (builder.add_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"])
                 .add_node(op, inputs, ["yq"], **attrs)
                 .add_node("DequantizeLinear", ["yq", "ys", "yz"], ["y"])
                 .add_output("y", (1, width)).build())
        graph = OnnxImporter(model).run()
        x = weights(1, 4)
        out = GraphRunner(graph).run({"x": x})[graph.outputs[0]]
        self.assertAlmostEqual(float(graph.annotations[graph.outputs[0]].scale[0]), 0.02)
        return x, out.reshape(1, -1), [node.op_type for node in graph.nodes]

    def test_qgemm(self):
        """The int32 bias is on the grid alpha * a_scale * b_scale."""
        x, out, kinds = self.run_op("QGemm", ["xq", "xs", "xz", "b", "bs", "bz", "c", "ys",
                                              "yz"], 3, transB=1, alpha=0.5)
        expected = 0.5 * (x @ (self.b.T * 0.01) + self.c * 0.05 * 0.01)
        np.testing.assert_allclose(out, expected, rtol=1e-5)
        self.assertEqual(kinds, ["Gemm"])

    def test_lanewise(self):
        cases = [("QLinearMul", ["xq", "xs", "xz", "k", "bs", "bz", "ys", "yz"], {},
                  lambda x: x * self.k * 0.01),
                 ("QLinearSigmoid", ["xq", "xs", "xz", "ys", "yz"], {},
                  lambda x: 1 / (1 + np.exp(-x))),
                 ("QLinearLeakyRelu", ["xq", "xs", "xz", "ys", "yz"], {"alpha": 0.1},
                  lambda x: np.where(x > 0, x, 0.1 * x))]
        for op, inputs, attrs, function in cases:
            with self.subTest(op):
                x, out, _ = self.run_op(op, inputs, 4, **attrs)
                np.testing.assert_allclose(out, function(x), rtol=1e-5, atol=1e-6)

    def test_across_lanes(self):
        x, out, kinds = self.run_op("QLinearSoftmax", ["xq", "xs", "xz", "ys", "yz"], 4,
                                    axis=-1, opset=13)
        np.testing.assert_allclose(out, np.exp(x) / np.exp(x).sum(), rtol=1e-5)
        self.assertEqual(kinds, ["Softmax"])
        x, out, kinds = self.run_op("QLinearReduceMean", ["xq", "xs", "xz", "ys", "yz"], 1,
                                    axes=[1], keepdims=1)
        np.testing.assert_allclose(out, x.mean(keepdims=True), rtol=1e-5)
        self.assertEqual(kinds, ["ReduceMean"])

    def test_concat(self):
        x, out, _ = self.run_op("QLinearConcat", ["ys", "yz", "xq", "xs", "xz", "xq", "xs",
                                                  "xz"], 8, axis=1)
        np.testing.assert_allclose(out, np.hstack([x, x]), rtol=1e-6)


class Binarizer(unittest.TestCase):
    def test_every_code(self):
        """The integer unit agrees with x > threshold on every input code."""
        model = (ModelBuilder("bin").add_input("x", (1, 3))
                 .add_node("Binarizer", ["x"], ["y"], threshold=0.3)
                 .add_output("y", (1, 3)).build())
        graph = OnnxImporter(model).run()
        plan = QuantizationPlan()
        plan.annotate("x", QuantSpec(np.array([0.01, 0.002, 0.05]), np.array([4, -9, 0])))
        LowerToHardware(plan).run(graph, CompilerContext(Device.get("sim")))
        codes = np.arange(-128, 128)[:, None].repeat(3, axis=1)
        runner = GraphRunner(graph)
        out = np.concatenate([runner.run({"x": row[None]})["y"] for row in codes])
        real = plan.spec("x").dequantize(codes)
        np.testing.assert_array_equal(out, real > 0.3)


class Refusals(unittest.TestCase):
    def refused(self, op, inputs, text, **attrs):
        builder = ModelBuilder("no").add_input("x", (1, 1, 2, 2) if op == "Transpose" else (1, 4))
        builder.add_node(op, ["x"] + inputs, ["y"], **attrs).add_output("y", (1, 4))
        with self.assertRaisesRegex(UnsupportedOpError, text):
            OnnxImporter(builder.build()).run()

    def test_messages(self):
        self.refused("Normalizer", [], "only MAX, L1 and L2 are supported", norm="L3")
        self.refused("DynamicQuantizeLinear", [], "at run time")
        self.refused("Cast", [], "would truncate", to=TensorProto.INT32)
        self.refused("Transpose", [], "reorders live data into a 4D", perm=[0, 1, 3, 2])
        builder = (ModelBuilder("no").add_input("x", (1, 4))
                   .add_initializer("w", weights(4, 4)).add_initializer("k", weights(1, 2))
                   .add_node("MatMul", ["x", "w"], ["h"]).add_node("Relu", ["h"], ["r"])
                   .add_node("Concat", ["x", "r", "k"], ["y"], axis=1))
        with self.assertRaisesRegex(UnsupportedOpError, "joins a constant to features of more"):
            OnnxImporter(builder.add_output("y", (1, 10)).build()).run()


if __name__ == "__main__":
    unittest.main()
