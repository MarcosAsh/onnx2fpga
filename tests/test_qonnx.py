"""QONNX Quant nodes: a 4 bit MLP against qonnx's own executor, and the grids it states."""

import pathlib
import shutil
import unittest

import numpy as np

from support import ROOT, Fixtures, Hdl
from test_fixed_latency import prove

import onnx2fpga as otf
from onnx2fpga.board import IcepiZero
from onnx2fpga.errors import ModelError, UnsupportedOpError
from onnx2fpga.p1_ingest import OnnxModel
from onnx2fpga.p2_graph.datatype import IntType, QuantSpec
from onnx2fpga.p3_ops.hardware_ops import DenseUnit

HARDCAML = Hdl.skip_reason("hardcaml")
CORPUS = ROOT / "tests" / "fixtures" / "corpus"
DATA = np.load(CORPUS / "qonnx_mlp.npz")


def model():
    return OnnxModel.load(CORPUS / "qonnx_mlp.onnx")


def quant(proto, output):
    return next(node for node in proto.graph.node if node.output[0] == output)


def constant(proto, name, value):
    tensor = next(t for t in proto.graph.initializer if t.name == name)
    tensor.raw_data = np.asarray(value, np.float32).tobytes()


class Grids(unittest.TestCase):
    def test_unsigned_rebase(self):
        spec = QuantSpec(0.5, 0, IntType(4, False)).to_signed()
        self.assertEqual((spec.dtype.name, int(spec.zero_point[0])), ("INT4", -8))
        spec = QuantSpec(0.5, 3, IntType(8, False)).to_signed()
        self.assertEqual(int(spec.zero_point[0]), -125)

    def test_refused(self):
        cases = (("h0", "rounding_mode", b"FLOOR", "rounds by FLOOR"),
                 ("h0", "narrow", 1, "narrow range"))
        for output, key, value, text in cases:
            with self.subTest(key):
                proto = model()
                attribute = next(a for a in quant(proto, output).attribute if a.name == key)
                if isinstance(value, bytes):
                    attribute.s = value
                else:
                    attribute.i = value
                with self.assertRaisesRegex(UnsupportedOpError, text):
                    otf.load(proto)
        for name, value, text in (("h0_bits", 12.0, "12 bits, and activations take 2 to 8"),
                                  ("h0_bits", 3.5, "bit width that is not one integer"),
                                  ("h0_zero", 0.5, "zero point that is not an integer")):
            with self.subTest(name, value=value):
                proto = model()
                constant(proto, name, value)
                with self.assertRaisesRegex(UnsupportedOpError, text):
                    otf.load(proto)

    @unittest.skipIf(HARDCAML, HARDCAML)
    def test_missing_grid(self):
        proto = model()
        next(node for node in proto.graph.node if node.op_type == "Relu").op_type = "Sigmoid"
        with self.assertRaisesRegex(ModelError, "the model gives tensor 'm0' no grid"):
            otf.load(proto).compile(device="k26", mode="fixed",
                                    out_dir=Fixtures.build_dir("qonnx_missing"))


@unittest.skipIf(HARDCAML, HARDCAML)
class Mlp(unittest.TestCase):
    """8 bit input, 4 bit weights per channel and 4 bit unsigned activations, as Brevitas exports."""

    @classmethod
    def setUpClass(cls):
        cls.acc = otf.load(CORPUS / "qonnx_mlp.onnx")
        cls.acc.compile(device="k26", mode="fixed", out_dir=Fixtures.build_dir("qonnx_mlp"))

    def test_exact(self):
        got = self.acc.predict(DATA["x"]).reshape(DATA["y"].shape)
        np.testing.assert_allclose(got, DATA["y"], rtol=0, atol=1e-5)

    def test_grids(self):
        units = [unit for unit in self.acc.result.hardware_graph.topological_order()
                 if isinstance(unit, DenseUnit)]
        self.assertEqual([unit.weight_dtype.name for unit in units], ["INT4"] * 3)
        # The hidden layers are 4 wires a value; the ports stay 8 bits.
        self.assertEqual([(unit.act_dtype.bits, unit.out_dtype.bits) for unit in units],
                         [(8, 4), (4, 4), (4, 8)])
        self.assertEqual([(unit.requant.lower_clamp, unit.requant.upper_clamp)
                          for unit in units], [(-8, 7), (-8, 7), (-128, 127)])
        lines = []
        self.acc.summary(print_fn=lines.append)
        self.assertIn("weights int4, activations int8 on 4 bit grids, from the model's "
                      "Quant nodes", lines[0])

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_simulate(self):
        report = self.acc.simulate()
        self.assertTrue(report.ok, report.first.output)

    @unittest.skipIf(IcepiZero.tool("sby") is None, "sby not found")
    def test_proof(self):
        build, latency = pathlib.Path(self.acc.out_dir), self.acc.latency
        self.assertTrue(prove(build, 8 * 8, latency)[0])
        self.assertFalse(prove(build, 8 * 8, latency - 1)[0])


if __name__ == "__main__":
    unittest.main()
