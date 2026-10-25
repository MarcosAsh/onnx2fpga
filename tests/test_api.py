"""The Python API, from otf.load to predict on a board; tests skip where an optional tool is missing."""

import contextlib
import io
import json
import os
import shutil
import threading
import unittest
from unittest import mock

import numpy as np

from support import Fixtures, Hdl, ROOT

import onnx2fpga as otf
from onnx2fpga import api
from onnx2fpga.board import IcepiZero
from onnx2fpga.errors import (BudgetError, ConfigError, ModelError, NotCompiledError,
                              NotDeployedError, ToolchainError, UnknownLayerError,
                              UnsupportedOpError)
from onnx2fpga.p1_ingest import ModelBuilder
from onnx2fpga.p6_emit.hardcaml import HardcamlGenerator
from onnx2fpga.p6_emit.stream_io import StreamCodec
from onnx2fpga.reference.runner import GraphRunner
from onnx2fpga.serial_link import Link, read_hex
from onnx2fpga.targets.device import Device
from onnx2fpga.targets.resources import Resources

MODELS = {"mlp": (Fixtures.MLP_SHAPE, 64), "cnn": (Fixtures.CNN_SHAPE, 400)}


def batch(shape, count=8, seed=5):
    return np.random.default_rng(seed).normal(0, 1, (count,) + tuple(shape[1:])) \
        .astype(np.float32)


def model(name):
    return getattr(Fixtures.factory(), name)()


def compiled(name, hdl="sv", device="vu9p", **options):
    shape, target = MODELS[name]
    acc = otf.load(model(name), name=name)
    options.setdefault("target_cycles", target)
    acc.compile(device=device, hdl=hdl, calibration=batch(shape),
                out_dir=Fixtures.build_dir("api_%s_%s_%s" % (name, device, hdl)),
                **options)
    return acc


def summary_text(acc):
    lines = []
    acc.summary(print_fn=lines.append)
    return "\n".join(lines)


@contextlib.contextmanager
def environment(**values):
    with mock.patch.dict(os.environ):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield


def without_hardcaml():
    nowhere = Fixtures.build_dir("api_no_hardcaml")
    return environment(**{HardcamlGenerator.ENV: str(nowhere)})


class Clip(otf.HardwareLayer):
    """The worked example: ONNX Clip on the library otf_act unit with unit multiplier and no shift, so it only clamps."""

    module = "otf_act"

    def __init__(self, min=0.0, max=6.0):
        self.min, self.max = min, max

    def build(self, input_shape, grid):
        super().build(input_shape, grid)
        self.low, self.high = grid.quantize(self.min), grid.quantize(self.max)

    def call(self, x):
        return np.clip(x, self.min, self.max)

    def call_integer(self, q):
        return np.clip(q, self.low, self.high)

    def parameters(self, pe):
        grid = self.grid
        return {"CHANNELS": self.channels, "PE": pe, "IN_BITS": grid.bits,
                "OUT_BITS": grid.bits, "MULT_BITS": 2, "SHIFT": 0,
                "OUT_MIN": self.low, "OUT_MAX": self.high,
                "IN_ZP": grid.zero_point, "OUT_ZP": grid.zero_point}

    def memories(self, pe):
        return {"mult": (np.ones((pe, self.channels // pe), dtype=np.int64), 2)}


NEG_RTL = """`default_nettype none

module otf_neg #(
    parameter int PE        = 1,
    parameter int DATA_BITS = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire                      clk,
    input  wire                      rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire  [PE*DATA_BITS-1:0]  s_tdata,
    input  wire                      s_tvalid,
    output wire                      s_tready,
    output wire  [PE*DATA_BITS-1:0]  m_tdata,
    output wire                      m_tvalid,
    input  wire                      m_tready
);
    localparam logic signed [DATA_BITS-1:0] LOWEST = {1'b1, {(DATA_BITS-1){1'b0}}};
    localparam logic signed [DATA_BITS-1:0] HIGHEST = {1'b0, {(DATA_BITS-1){1'b1}}};

    for (genvar lane = 0; lane < PE; lane++) begin : g_lane
        wire signed [DATA_BITS-1:0] x = s_tdata[lane*DATA_BITS +: DATA_BITS];
        assign m_tdata[lane*DATA_BITS +: DATA_BITS] = (x == LOWEST) ? HIGHEST : -x;
    end
    assign m_tvalid = s_tvalid;
    assign s_tready = m_tready;
endmodule

`default_nettype wire
"""


class Neg(otf.HardwareLayer):
    """A layer with its own SystemVerilog, so sv backend only; negation saturates where -128 has no int8 opposite."""

    module = "otf_neg"
    rtl = NEG_RTL

    def call(self, x):
        return -x

    def call_integer(self, q):
        return np.clip(-q, self.grid.min, self.grid.max)

    def parameters(self, pe):
        return {"PE": pe, "DATA_BITS": self.grid.bits}


def custom_model(op_type, **attributes):
    factory = Fixtures.factory()
    builder = ModelBuilder("custom").add_input("x", (1, 32))
    builder.add_initializer("w0", factory.dense(32, 16))
    builder.add_initializer("b0", factory.bias(16))
    builder.add_initializer("w1", factory.dense(16, 10))
    builder.add_initializer("b1", factory.bias(10))
    builder.add_node("Gemm", ["x", "w0", "b0"], ["h"], name="fc0")
    builder.add_node(op_type, ["h"], ["c"], name="custom", **attributes)
    builder.add_node("Gemm", ["c", "w1", "b1"], ["y"], name="fc1")
    return builder.add_output("y", (1, 10)).build()


class Basics(unittest.TestCase):
    """load, compile, summary, evaluate, predict, on the integer model."""

    def check(self, name):
        shape, target = MODELS[name]
        x = batch(shape)
        acc = compiled(name)
        self.assertEqual(acc.hdl, "sv")
        self.assertIn("%d cycles a frame" % acc.result.folding.bottleneck_cycles,
                      repr(acc))

        text = summary_text(acc)
        for layer in acc.layers:
            if layer.startswith("flat"):
                continue
            self.assertIn(layer + " (", text)
        for part in ("<- bottleneck", "Budget on vu9p", "Throughput:", "Used"):
            self.assertIn(part, text)

        metrics = acc.evaluate(x, np.arange(len(x)) % 10)
        self.assertEqual(sorted(metrics), ["accuracy", "agreement", "float_accuracy"])
        self.assertGreaterEqual(metrics["agreement"], 0.75)

        # The golden frame is the first calibration sample, so predict on it must give the golden output.
        graph = acc.result.hardware_graph
        sink = StreamCodec(graph.tensor(graph.outputs[0]))
        golden = sink.from_words(read_hex(acc.out_dir / "golden" / "expected.hex"))
        np.testing.assert_array_equal(acc.predict(x[:1], quantized=True), golden)

        # And on every other input, the same integer model run by hand.
        plan = acc.result.plan
        name_in, name_out = graph.inputs[0], graph.outputs[0]
        layout = graph.layouts.get(name_in)
        codes = []
        for sample in x:
            raw = sample[None].astype(np.float64)
            raw = layout.to_internal(raw) if layout else raw
            quantized = plan.act_dtype.clamp(np.rint(raw / plan.scale(name_in))
                                             + plan.zero_point(name_in))
            codes.append(GraphRunner(graph).run({name_in: quantized})[name_out])
        codes = np.concatenate(codes)
        np.testing.assert_array_equal(acc.predict(x, quantized=True), codes)
        np.testing.assert_allclose(
            acc.predict(x), plan.scale(name_out) * (codes - plan.zero_point(name_out)))

    def test_mlp(self):
        self.check("mlp")

    def test_cnn(self):
        self.check("cnn")

    def test_summary_before_compile(self):
        acc = otf.load(model("cnn"))
        acc.layers["conv0"].fold(pe=2)
        acc.layers["fc1"].quantize(weights="int4")
        text = summary_text(acc)
        for part in ("conv0 (Conv+Relu)", "pool0 (MaxPool)", "fold pe=2",
                     "weights int4", "Total params: 1050", "not compiled"):
            self.assertIn(part, text)
        self.assertEqual(repr(acc), '<Accelerator "cnn": 4 layers, not compiled>')

    def test_qdq_needs_no_calibration(self):
        acc = otf.load(example_model("qdq_mlp_uint8"))
        with warnings_caught() as caught:
            acc.compile(hdl="sv", out_dir=Fixtures.build_dir("api_qdq"))
        self.assertEqual(caught, [])
        self.assertIn("from the model's QDQ nodes", summary_text(acc))

    def test_calibration_warning(self):
        acc = otf.load(model("mlp"))
        with self.assertWarnsRegex(UserWarning, "calibration=x_train"):
            acc.compile(hdl="sv", out_dir=Fixtures.build_dir("api_nocal"))

    def test_target_fps_and_fit(self):
        acc = otf.load(model("mlp"))
        x = batch(Fixtures.MLP_SHAPE)
        acc.compile(device="icepi25", hdl="sv", target_fps=50_000, calibration=x,
                    out_dir=Fixtures.build_dir("api_fps"))
        self.assertLessEqual(acc.result.folding.bottleneck_cycles, 400)
        acc.compile(target="fit")
        fit = acc.result.folding
        self.assertLess(fit.bottleneck_cycles, 400)
        self.assertLessEqual(fit.total.dsp, 28)


def example_model(name):
    path = ROOT / "examples" / "models" / ("%s.onnx" % name)
    if not path.exists():
        raise unittest.SkipTest("%s is missing; run make models" % path)
    return path


@contextlib.contextmanager
def warnings_caught():
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


class Fold(unittest.TestCase):
    def test_full_pin(self):
        acc = otf.load(model("mlp"))
        acc.layers["fc1"].fold(simd=4, pe=2)
        acc.compile(device="vu9p", hdl="sv", target_cycles=128,
                    calibration=batch(Fixtures.MLP_SHAPE),
                    out_dir=Fixtures.build_dir("api_pin_full"))
        self.assertEqual(acc.layers["fc1"].folding, {"simd": 4, "pe": 2})
        self.assertEqual(acc.layers["fc1"].cycles, (64 // 4) * (16 // 2))
        self.assertLessEqual(acc.layers["fc0"].cycles, 128)
        self.assertIn("fold: fc1 pinned at pe=2,simd=4",
                      "\n".join(acc.result.context.log))
        self.assertIn("pinned pe=2,simd=4", repr(acc.layers["fc1"]))

    def test_pin_sets_floor(self):
        acc = otf.load(model("mlp"))
        acc.layers["fc1"].fold(simd=4, pe=2)
        with self.assertRaises(BudgetError) as caught:
            acc.compile(device="vu9p", hdl="sv", target_cycles=64,
                        calibration=batch(Fixtures.MLP_SHAPE),
                        out_dir=Fixtures.build_dir("api_pin_floor"))
        message = str(caught.exception)
        self.assertIn("fc1 is pinned at pe=2,simd=4, 128 cycles", message)
        self.assertIn("acc.layers['fc1'].fold()", message)

    def test_partial_pin(self):
        acc = otf.load(model("cnn"))
        acc.layers["conv0"].fold(pe=4)
        acc.compile(device="vu9p", hdl="sv", target_cycles=100,
                    calibration=batch(Fixtures.CNN_SHAPE), strict=False,
                    out_dir=Fixtures.build_dir("api_pin_partial"))
        self.assertEqual(acc.layers["conv0"].folding["pe"], 4)
        self.assertEqual(acc.layers["conv0"].units, ["conv0_swu", "conv0"])

    def test_recompile(self):
        acc = compiled("mlp", device="icepi25", target_cycles=256)
        acc.layers["fc2"].fold(pe=2)
        with self.assertRaisesRegex(NotCompiledError, r"acc\.compile\(\) again"):
            acc.predict(batch(Fixtures.MLP_SHAPE))
        acc.compile()
        self.assertEqual(acc.layers["fc2"].folding["pe"], 2)
        self.assertEqual((acc.device.name, acc.result.folding.bottleneck_cycles),
                         ("icepi25", 256))
        acc.layers["fc2"].fold()
        acc.compile()
        self.assertNotIn("pinned", "\n".join(acc.result.context.log))

    def test_invalid_value(self):
        acc = otf.load(model("mlp"))
        acc.layers["fc1"].fold(simd=5)
        with self.assertRaises(ConfigError) as caught:
            acc.compile(hdl="sv", calibration=batch(Fixtures.MLP_SHAPE),
                        out_dir=Fixtures.build_dir("api_pin_bad"))
        self.assertIn("fc1 cannot fold to simd=5, valid values: 1, 2, 4, 8, 16, 32, 64",
                      str(caught.exception))

    def test_bad_factors(self):
        acc = otf.load(model("cnn"))
        with self.assertRaisesRegex(ConfigError,
                                    "pool0 \\(MaxPool\\) folds by pe, not simd"):
            acc.layers["pool0"].fold(simd=2)
        with self.assertRaisesRegex(ConfigError, "positive integer"):
            acc.layers["conv0"].fold(pe=0)
        with self.assertRaisesRegex(ConfigError, "positive integer"):
            acc.layers["conv0"].fold(pe=2.5)
        with self.assertRaisesRegex(ConfigError, "nothing to fold"):
            acc.layers["flat0"].fold(pe=2)
        with self.assertRaises(UnknownLayerError) as caught:
            acc.layers["conv9"]
        self.assertIsInstance(caught.exception, KeyError)
        self.assertEqual(str(caught.exception), "the model has no layer named 'conv9'. "
                                                "Layers: conv0, pool0, flat0, fc1")

    def test_pin_over_budget(self):
        acc = otf.load(model("mlp"))
        acc.layers["fc0"].fold(simd=32, pe=64)
        with self.assertRaises(BudgetError) as caught:
            acc.compile(device="icepi25", hdl="sv", calibration=batch(Fixtures.MLP_SHAPE),
                        out_dir=Fixtures.build_dir("api_pin_budget"))
        message = str(caught.exception)
        self.assertIn("folding fc0 to pe=64,simd=32 by hand", message)
        self.assertIn("dsp", message)


class Precision(unittest.TestCase):
    def test_int4(self):
        acc = otf.load(model("mlp"))
        acc.layers["fc1"].quantize(weights="int4")
        acc.compile(hdl="sv", calibration=batch(Fixtures.MLP_SHAPE),
                    out_dir=Fixtures.build_dir("api_int4"))
        units = {n.name: n for n in acc.result.hardware_graph.nodes}
        self.assertEqual(units["fc1"].weight_dtype.bits, 4)
        self.assertEqual((units["fc0"].weight_dtype.bits, units["fc2"].weight_dtype.bits),
                         (8, 8))
        weights = units["fc1"].weights
        self.assertTrue(-8 <= weights.min() and weights.max() <= 7)
        manifest = json.loads((acc.out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["quantization"]["layer_weights"], {"fc1": "INT4"})
        self.assertIn("weights int8 (fc1 int4)", summary_text(acc))

    def test_invalid(self):
        acc = otf.load(model("cnn"))
        with self.assertRaisesRegex(ConfigError, "'int2' to 'int8'"):
            acc.layers["fc1"].quantize(weights="int16")
        with self.assertRaisesRegex(ConfigError, "per layer"):
            acc.layers["fc1"].quantize(activations="int4")
        with self.assertRaisesRegex(ConfigError, "no weights"):
            acc.layers["pool0"].quantize(weights="int4")
        qdq = otf.load(example_model("qdq_mlp_uint8"))
        first = next(iter(qdq.layers))
        qdq.layers[first].quantize(weights="int4")
        with self.assertRaisesRegex(ConfigError, "quantized by the model itself"):
            qdq.compile(hdl="sv", out_dir=Fixtures.build_dir("api_qdq_int4"))


class Memory(unittest.TestCase):
    def test_override(self):
        acc = otf.load(model("mlp"))
        acc.layers["fc0"].memory("lut")
        acc.layers["fc2"].memory("block")
        self.assertIn("memory block", summary_text(acc))
        acc.compile(device="icepi25", hdl="sv", target_cycles=256,
                    calibration=batch(Fixtures.MLP_SHAPE),
                    out_dir=Fixtures.build_dir("api_memory"))
        units = {n.name: n for n in acc.result.hardware_graph.nodes}
        self.assertEqual([units[n].block_ram for n in ("fc0", "fc1", "fc2")],
                         [False, True, True])
        self.assertIn("weights in block RAM", repr(acc.layers["fc2"]))
        acc.layers["fc0"].memory()
        acc.compile()
        self.assertIn("weights in block RAM", repr(acc.layers["fc0"]))

    def test_errors(self):
        acc = otf.load(model("cnn"))
        with self.assertRaisesRegex(ConfigError, "'block', 'lut' or 'auto', not 'bram'"):
            acc.layers["fc1"].memory("bram")
        with self.assertRaisesRegex(ConfigError, "no weights"):
            acc.layers["pool0"].memory("block")
        acc.layers["fc1"].memory("block")
        with self.assertRaisesRegex(ConfigError, "needs mode='stream'"):
            acc.compile(mode="fixed")
        no_ram = Device("no_ram", "none", Resources(lut=10 ** 6, ff=10 ** 6, dsp=10 ** 4),
                        100.0, "DSP48E2")
        with self.assertRaisesRegex(ConfigError, "fc1 cannot go in block RAM because "
                                                 "no_ram has none"):
            acc.compile(device=no_ram, hdl="sv", calibration=batch(Fixtures.CNN_SHAPE),
                        out_dir=Fixtures.build_dir("api_memory_none"))
        mlp = otf.load(model("mlp"))
        mlp.layers["fc0"].fold(simd=32, pe=8).memory("block")
        with self.assertRaisesRegex(BudgetError, "block RAM for fc0 takes the design to "
                                                 "58 ebr, past the 56 on icepi25"):
            mlp.compile(device="icepi25", hdl="sv", calibration=batch(Fixtures.MLP_SHAPE),
                        out_dir=Fixtures.build_dir("api_memory_budget"))


class Errors(unittest.TestCase):
    def test_unsupported_op(self):
        with self.assertRaises(UnsupportedOpError) as caught:
            otf.load(custom_model("Softmax", axis=1))
        error = caught.exception
        self.assertEqual((error.op_type, error.node), ("Softmax", "custom"))
        message = str(error)
        for part in ("'Softmax'", "'custom'", "Conv", "Gemm", "MaxPool",
                     "custom_objects={'Softmax'"):
            self.assertIn(part, message)
        self.assertIsInstance(error, ValueError)

    def test_budget(self):
        acc = otf.load(model("mlp"))
        with self.assertRaises(BudgetError) as caught:
            acc.compile(device="icepi25", hdl="sv", target_cycles=64,
                        calibration=batch(Fixtures.MLP_SHAPE),
                        out_dir=Fixtures.build_dir("api_budget"))
        message = str(caught.exception)
        for part in ("target_cycles=64 is out of reach on icepi25",
                     "does not fit the dsp budget of icepi25", "256 cycles or more",
                     "strict=False", "a device with more dsp"):
            self.assertIn(part, message)
        self.assertIsNone(acc.result)
        acc.compile(target_cycles=64, strict=False)
        self.assertEqual(acc.result.folding.bottleneck_cycles, 256)

    def test_structural_floor(self):
        acc = otf.load(model("cnn"))
        with self.assertRaises(BudgetError) as caught:
            acc.compile(device="vu9p", hdl="sv", target_cycles=25,
                        calibration=batch(Fixtures.CNN_SHAPE),
                        out_dir=Fixtures.build_dir("api_floor"))
        message = str(caught.exception)
        self.assertIn("cannot fold further", message)
        self.assertNotIn("device with more", message)

    def test_targets(self):
        acc = otf.load(model("mlp"))
        with self.assertRaisesRegex(NotImplementedError, "latency"):
            acc.compile(target="latency")
        with self.assertRaisesRegex(ConfigError,
                                    "one of target, target_fps or target_cycles"):
            acc.compile(target="fit", target_cycles=64)
        with self.assertRaisesRegex(ConfigError, "unknown device 'vu13p'.*icepi25"):
            acc.compile(device="vu13p")
        with self.assertRaisesRegex(ConfigError, "between 0 and 1"):
            acc.compile(utilisation=80)

    def test_out_of_order(self):
        acc = otf.load(model("mlp"))
        x = batch(Fixtures.MLP_SHAPE)
        for call in (lambda: acc.predict(x), lambda: acc.evaluate(x),
                     lambda: acc.simulate()):
            with self.assertRaisesRegex(NotCompiledError,
                                        r"acc\.compile\(device=\.\.\.\)"):
                call()
        acc = compiled("mlp")
        with self.assertRaises(NotDeployedError) as caught:
            acc.predict(x, on="board")
        self.assertIn("acc.build(board='icepi_zero')", str(caught.exception))
        self.assertIn("acc.deploy(port=...)", str(caught.exception))
        with self.assertRaisesRegex(NotCompiledError, "acc.build"):
            acc.deploy()

    def test_input_shape(self):
        acc = compiled("cnn")
        with self.assertRaises(ConfigError) as caught:
            acc.predict(np.zeros((1, 12, 12)))
        self.assertIn("(N, 1, 12, 12)", str(caught.exception))
        self.assertIn("x[None]", str(caught.exception))
        with self.assertRaisesRegex(ConfigError, "8 inputs but 3 labels"):
            acc.evaluate(batch(Fixtures.CNN_SHAPE), [1, 2, 3])

    def test_missing_simulator(self):
        acc = compiled("mlp")
        real = shutil.which
        with mock.patch.object(api.shutil, "which",
                               lambda name: None if name == "verilator" else real(name)):
            with self.assertRaises(ToolchainError) as caught:
                acc.simulate()
        self.assertEqual(caught.exception.tool, "verilator")
        self.assertIn("sudo apt install verilator", str(caught.exception))

    def test_missing_board_flow(self):
        reason = Hdl.skip_reason("hardcaml")
        if reason:
            self.skipTest(reason)
        acc = compiled("mlp", hdl="hardcaml", device="icepi25", target_cycles=256)
        with mock.patch.object(IcepiZero, "missing_tools", return_value=["nextpnr-ecp5"]):
            with self.assertRaises(ToolchainError) as caught:
                acc.build()
        self.assertIn("nextpnr-ecp5", str(caught.exception))
        self.assertIn("OSS CAD Suite", str(caught.exception))

    def test_board_mismatch(self):
        acc = compiled("mlp")
        with self.assertRaisesRegex(ConfigError, "icepi_zero carries icepi25.*vu9p"):
            acc.build()
        acc = compiled("mlp", device="icepi25", target_cycles=256)
        with self.assertRaisesRegex(ConfigError, "needs the Hardcaml backend"):
            acc.build()
        with self.assertRaisesRegex(ConfigError, "unknown board 'de10'"):
            acc.build(board="de10")

    def test_family(self):
        for error in (BudgetError, ConfigError, ModelError, NotCompiledError,
                      NotDeployedError, ToolchainError, UnsupportedOpError):
            self.assertTrue(issubclass(error, otf.Onnx2FpgaError), error)


class Keras(unittest.TestCase):
    def test_from_keras(self):
        try:
            import keras
        except ImportError:
            with self.assertRaises(ToolchainError) as caught:
                otf.from_keras(object())
            self.assertIn("pip install keras", str(caught.exception))
            return
        inputs = keras.Input((32,))
        hidden = keras.layers.Dense(16, activation="relu")(inputs)
        net = keras.Model(inputs, keras.layers.Dense(10)(hidden), name="dense_net")
        try:
            acc = otf.from_keras(net)
        except ToolchainError as error:
            self.skipTest("this Keras cannot export ONNX here: %s" % error)
        x = np.random.default_rng(0).normal(0, 1, (8, 32)).astype(np.float32)
        acc.compile(hdl="sv", calibration=x, out_dir=Fixtures.build_dir("api_keras"))
        self.assertGreaterEqual(acc.evaluate(x)["agreement"], 0.75)


class Backend(unittest.TestCase):
    def test_environment(self):
        with environment(ONNX2FPGA_HDL="sv"):
            self.assertEqual(api.resolve_hdl(), "sv")
            acc = otf.load(model("mlp"))
            acc.compile(calibration=batch(Fixtures.MLP_SHAPE),
                        out_dir=Fixtures.build_dir("api_env_sv"))
            self.assertEqual(acc.hdl, "sv")

    def test_argument_wins_and_fallback(self):
        with environment(ONNX2FPGA_HDL="hardcaml"), without_hardcaml():
            self.assertEqual(api.resolve_hdl("sv"), "sv")
        with environment(ONNX2FPGA_HDL=None), without_hardcaml():
            self.assertEqual(api.resolve_hdl(), "sv")

    def test_unknown(self):
        with environment(ONNX2FPGA_HDL="vhdl"):
            with self.assertRaisesRegex(ConfigError, "ONNX2FPGA_HDL=vhdl.*sv, hardcaml"):
                api.resolve_hdl()

    def test_hardcaml_missing(self):
        with environment(ONNX2FPGA_HDL="hardcaml"), without_hardcaml():
            with self.assertRaises(ToolchainError) as caught:
                api.resolve_hdl()
        message = str(caught.exception)
        for part in ("ONNX2FPGA_HDL=hardcaml", "no Hardcaml project", "onnx2fpga doctor",
                     "hdl='sv'"):
            self.assertIn(part, message)

    def test_default_hardcaml(self):
        reason = Hdl.skip_reason("hardcaml")
        if reason:
            self.skipTest(reason)
        with environment(ONNX2FPGA_HDL=None):
            self.assertEqual(api.resolve_hdl(), "hardcaml")


class CustomLayer(unittest.TestCase):
    def load(self, layer, **attributes):
        op = {"Clip": "Clip", "Neg": "Neg"}[layer.__name__]
        return otf.load(custom_model(op, **attributes), custom_objects={op: layer},
                        name="custom_" + op.lower())

    def test_references(self):
        acc = self.load(Clip, min=0.0, max=0.5)
        x = batch((1, 32))
        acc.compile(hdl="sv", calibration=x, out_dir=Fixtures.build_dir("api_clip_ref"))
        self.assertIn("custom (Clip)", summary_text(acc))
        unit = next(n for n in acc.result.hardware_graph.nodes if n.name == "custom")
        layer = unit.layer
        self.assertEqual((layer.low, layer.high), (0, layer.grid.quantize(0.5)))
        float_graph = acc.result.float_graph
        floating = GraphRunner(float_graph).run({"x": x[:1].astype(np.float64)})
        self.assertLessEqual(floating["c"].max(), 0.5)
        self.assertGreaterEqual(acc.evaluate(x)["agreement"], 0.75)
        self.assertEqual(acc.predict(x).shape, (len(x), 10))

    def test_own_rtl_needs_sv(self):
        reason = Hdl.skip_reason("hardcaml")
        if reason:
            self.skipTest(reason)
        acc = self.load(Neg)
        with self.assertRaisesRegex(ConfigError, "units.ml.*hdl='sv'"):
            acc.compile(hdl="hardcaml", calibration=batch((1, 32)),
                        out_dir=Fixtures.build_dir("api_neg_hc"))

    def test_must_be_layer(self):
        with self.assertRaisesRegex(ConfigError, "HardwareLayer subclass"):
            otf.load(custom_model("Clip"), custom_objects={"Clip": np.clip})


class SimulationCases:
    """Bit exact in Verilator: golden frame, user inputs, pinned folding and custom layers, once per backend."""

    HDL = "sv"

    @classmethod
    def setUpClass(cls):
        for tool in ("verilator", "make", "g++"):
            if shutil.which(tool) is None:
                raise unittest.SkipTest("%s is not on PATH" % tool)
        reason = Hdl.skip_reason(cls.HDL)
        if reason:
            raise unittest.SkipTest(reason)

    def test_bit_exact(self):
        for name, (shape, _) in MODELS.items():
            with self.subTest(model=name):
                acc = compiled(name, hdl=self.HDL)
                golden = acc.simulate()
                self.assertTrue(golden.ok, golden.first.output)
                report = acc.simulate(batch(shape, count=4, seed=9))
                self.assertTrue(report.ok, report.first.output)
                self.assertEqual(report.frames, 4)
                self.assertIn("bit exact: 4 frames", repr(report))

    def test_pinned_backpressure(self):
        acc = otf.load(model("mlp"), name="mlp_pinned")
        acc.layers["fc1"].fold(simd=4, pe=2)
        acc.layers["fc2"].quantize(weights="int4")
        x = batch(Fixtures.MLP_SHAPE)
        acc.compile(device="icepi25", hdl=self.HDL, target_cycles=256, calibration=x,
                    out_dir=Fixtures.build_dir("api_pinned_" + self.HDL))
        report = acc.simulate(x[:3], sweep=True)
        self.assertTrue(report.ok, report)
        self.assertEqual(len(report.sweep), 15)

    def test_clip(self):
        acc = otf.load(custom_model("Clip", min=0.0, max=0.5),
                       custom_objects={"Clip": Clip}, name="clip_" + self.HDL)
        acc.layers["custom"].fold(pe=4)
        x = batch((1, 32))
        acc.compile(device="icepi25", hdl=self.HDL, target_cycles=64, calibration=x,
                    out_dir=Fixtures.build_dir("api_clip_" + self.HDL))
        self.assertEqual(acc.layers["custom"].folding, {"pe": 4})
        report = acc.simulate(x, sweep=True)
        self.assertTrue(report.ok, report)


class SvSimulation(SimulationCases, unittest.TestCase):
    HDL = "sv"

    def test_own_rtl(self):
        acc = otf.load(custom_model("Neg"), custom_objects={"Neg": Neg}, name="neg")
        x = batch((1, 32))
        acc.compile(device="vu9p", hdl="sv", target_cycles=64, calibration=x,
                    out_dir=Fixtures.build_dir("api_neg_sv"))
        self.assertTrue((acc.out_dir / "rtl" / "otf_neg.sv").is_file())
        report = acc.simulate(x, sweep=True)
        self.assertTrue(report.ok, report)


class HardcamlSimulation(SimulationCases, unittest.TestCase):
    HDL = "hardcaml"


class Board:
    """A stand in for a board on a pseudo terminal, answering each input frame with the integer model output."""

    def __init__(self, acc):
        import tty
        self.master, self.slave = os.openpty()
        tty.setraw(self.master)
        self.port = os.ttyname(self.slave)
        graph = acc.result.hardware_graph
        self.runner = GraphRunner(graph)
        self.names = graph.inputs[0], graph.outputs[0]
        self.source = StreamCodec(graph.tensor(self.names[0]))
        self.sink = StreamCodec(graph.tensor(self.names[1]))
        self.link = Link.load(acc.out_dir)
        self.frames = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)

    def answer(self, frame):
        link = self.link
        words = [int.from_bytes(frame[k:k + link.in_bytes], "little")
                 for k in range(0, len(frame), link.in_bytes)]
        codes = self.runner.run({self.names[0]: self.source.from_words(words)})[
            self.names[1]]
        return b"".join(word.to_bytes(link.out_bytes, "little")
                        for word in self.sink.to_image(codes).words)

    def serve(self):
        pending = b""
        while not self.stop.is_set():
            try:
                pending += os.read(self.master, 4096)
            except OSError:
                return
            while len(pending) >= self.link.frame_in:
                frame, pending = (pending[:self.link.frame_in],
                                  pending[self.link.frame_in:])
                self.frames += 1
                os.write(self.master, self.answer(frame))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        os.close(self.slave)
        os.close(self.master)
        self.thread.join(timeout=5)


class Deploy(unittest.TestCase):
    """build and deploy with the flow and loader mocked, and predict over a real serial line to the stand in."""

    @classmethod
    def setUpClass(cls):
        try:
            import serial  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pyserial is not installed")
        reason = Hdl.skip_reason("hardcaml")
        if reason:
            raise unittest.SkipTest(reason)

    def mocked(self):
        stack = contextlib.ExitStack()
        run = stack.enter_context(mock.patch.object(IcepiZero, "run"))
        stack.enter_context(mock.patch.object(IcepiZero, "missing_tools",
                                              return_value=[]))
        stack.enter_context(mock.patch.object(IcepiZero, "tool",
                                              return_value="/usr/bin/openFPGALoader"))
        return stack, run

    @staticmethod
    def write_board_json(acc):
        """What the board Makefile writes beside the bitstream."""
        folder = acc.out_dir / "icepi_zero"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "board.json").write_text(json.dumps(
            {"baud": 1_000_000, "rx_depth": 2048, "tx_depth": 2048, "mhz": 20,
             "clocks_per_bit": 20}))

    def test_predict_on_board(self):
        x = batch(Fixtures.MLP_SHAPE, count=12)
        acc = compiled("mlp", hdl="hardcaml", device="icepi25", target_cycles=256)
        stack, run = self.mocked()
        with stack:
            bitstream = acc.build(board="icepi_zero", mhz=24)
            self.assertEqual(bitstream.name, "v1.4.bit")
            self.assertEqual(acc.board.mhz, 24)
            self.write_board_json(acc)
            with Board(acc) as board:
                acc.deploy(port=board.port, timeout=5)
                self.assertEqual([c.args for c in run.call_args_list],
                                 [("bit",), ("prog",)])
                self.assertTrue(acc.deployed)
                self.assertIn("on %s" % board.port, repr(acc))
                on_board = acc.predict(x, quantized=True)
                self.assertEqual(board.frames, len(x))
                np.testing.assert_array_equal(
                    on_board, acc.predict(x, on="reference", quantized=True))
                np.testing.assert_allclose(acc.predict(x[:2]),
                                           acc.predict(x[:2], on="reference"))
                acc.disconnect()
        self.assertFalse(acc.deployed)
        acc.predict(x[:1])

    def test_build_compiles_first(self):
        acc = otf.load(model("mlp"))
        stack, run = self.mocked()
        with stack, warnings_caught():
            acc.build()
        self.assertEqual((acc.device.name, acc.hdl), ("icepi25", "hardcaml"))
        self.assertEqual(run.call_args_list[0].args, ("bit",))

    def test_no_loader(self):
        acc = compiled("mlp", hdl="hardcaml", device="icepi25", target_cycles=256)
        stack, _ = self.mocked()
        with stack:
            acc.build()
            with mock.patch.object(IcepiZero, "tool", return_value=None):
                with self.assertRaises(ToolchainError) as caught:
                    acc.deploy()
        self.assertIn("openFPGALoader", str(caught.exception))
        self.assertIn("deploy(program=False)", str(caught.exception))


class Bitstream(unittest.TestCase):
    """The real open flow, where yosys, nextpnr-ecp5 and ecppack are present."""

    @classmethod
    def setUpClass(cls):
        reason = Hdl.skip_reason("hardcaml")
        if reason:
            raise unittest.SkipTest(reason)
        missing = IcepiZero.missing_tools()
        if missing:
            raise unittest.SkipTest("%s not found" % ", ".join(missing))

    def test_build(self):
        acc = compiled("mlp", hdl="hardcaml", device="icepi25", target_cycles=256)
        with contextlib.redirect_stdout(io.StringIO()):
            bitstream = acc.build(board="icepi_zero", seeds=(1, 2, 3))
        self.assertTrue(bitstream.is_file())
        self.assertGreater(bitstream.stat().st_size, 100_000)
        report = acc.board.report()
        self.assertGreaterEqual(report["fmax_mhz"], 20)
        self.assertLessEqual(report["utilisation"]["MULT18X18D"][0], 28)


if __name__ == "__main__":
    unittest.main()
