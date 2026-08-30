"""The C++ kernels and the Python model must agree exactly.

Three implementations of the same arithmetic exist in this repository: the
numpy model in the compiler, the C++ kernels in runtime/, and the RTL. The
compiler checks itself against the RTL in simulation; this checks it against
the C++ side, which is what the host runtime and any large scale dataset sweep
will use.
"""

import shutil
import unittest

import numpy as np

from support import ROOT, Fixtures, run

from onnx2fpga.p2_graph.datatype import INT8, INT16, INT32
from onnx2fpga.p2_graph.graph import Graph
from onnx2fpga.p2_graph.tensor import Tensor
from onnx2fpga.p3_ops.hardware_ops import MatVecUnit
from onnx2fpga.p4_quantize.requantize import Requantizer


class MatVecFixture:
    """Writes a layer and its expected integer output for the C++ checker."""

    def __init__(self, mw, mh, vectors, seed, zero_point=0, out_dtype=INT8):
        rng = np.random.default_rng(seed)
        self.mw, self.mh, self.vectors = mw, mh, vectors
        self.zero_point = zero_point
        self.out_dtype = out_dtype
        self.weights = rng.integers(-127, 128, (mw, mh), dtype=np.int64)
        self.bias = rng.integers(-2000, 2000, mh, dtype=np.int64)
        self.requant = Requantizer.from_real_multipliers(
            rng.uniform(1e-3, 4e-2, mh), out_dtype, mult_bits=18,
            zero_point=zero_point)
        self.input = rng.integers(-128, 128, (vectors, mw), dtype=np.int64)
        self.expected = self._expected()

    def _expected(self):
        graph = Graph("fixture")
        graph.add_tensor(Tensor("x", (self.vectors, self.mw), INT8))
        graph.inputs.append("x")
        unit = MatVecUnit("m", ["x"], ["y"], self.weights, self.bias, self.requant,
                          INT8, self.out_dtype, INT8, vectors=self.vectors)
        graph.ensure_tensor("y")
        graph.add_node(unit)
        unit.infer(graph)
        return unit.execute(graph, {"x": self.input})[0]

    def write(self, path):
        def row(values):
            return " ".join(str(int(v)) for v in np.asarray(values).reshape(-1))

        path.write_text("\n".join([
            "%d %d %d %d %d %d %d" % (self.mw, self.mh, self.requant.shift,
                                      self.requant.lower_clamp,
                                      self.requant.upper_clamp, self.vectors,
                                      self.requant.zero_point),
            row(self.weights),
            row(self.bias),
            row(self.requant.broadcast_to(self.mh).multipliers),
            row(self.input),
            row(self.expected),
        ]) + "\n")
        return path


@unittest.skipIf(shutil.which("g++") is None, "no C++ compiler")
class CppContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        built = run(["make", "-s"], cwd=ROOT / "runtime")
        assert built.returncode == 0, built.stderr[-4000:]
        cls.checker = ROOT / "runtime" / "bin" / "check_matvec"

    def test_kernels_reproduce_the_python_model(self):
        cases = [(16, 8, 4, 0, 0), (64, 32, 3, 1, 0), (100, 10, 1, 2, 0),
                 (9, 4, 25, 3, 0), (32, 16, 4, 4, -37), (24, 12, 3, 5, 64)]
        for mw, mh, vectors, seed, zero_point in cases:
            with self.subTest(mw=mw, mh=mh, vectors=vectors, zp=zero_point):
                fixture = MatVecFixture(mw, mh, vectors, seed, zero_point)
                path = fixture.write(
                    Fixtures.build_dir("cpp") / ("matvec_%d.txt" % seed))
                result = run([str(self.checker), str(path)])
                self.assertEqual(result.returncode, 0,
                                 result.stdout + result.stderr)
                self.assertIn("PASS", result.stdout)

    def test_kernels_agree_at_a_wider_output(self):
        """A graph output need not be as narrow as the activations between
        layers, and int8 scores saturate where int16 ones do not. The C++ has
        to follow the Python out to the wider type, or the wide path is only
        checked against itself."""
        for dtype in (INT16, INT32):
            for mw, mh, vectors, seed in ((32, 16, 4, 11), (64, 10, 2, 12)):
                with self.subTest(dtype=dtype.name, mw=mw, mh=mh):
                    fixture = MatVecFixture(mw, mh, vectors, seed,
                                            out_dtype=dtype)
                    path = fixture.write(
                        Fixtures.build_dir("cpp")
                        / ("matvec_%s_%d.txt" % (dtype.name, seed)))
                    result = run([str(self.checker), str(path)])
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)
                    self.assertIn("PASS", result.stdout)

    def test_the_wider_output_actually_uses_the_range(self):
        """If every value still fit in an int8 the test above would pass
        without exercising anything."""
        wide = MatVecFixture(64, 10, 2, 12, out_dtype=INT16)
        self.assertTrue((np.abs(np.asarray(wide.expected)) > 127).any(),
                        "no value exceeded int8, so the width is untested")

    def test_a_deliberate_mismatch_is_caught(self):
        """The checker has to be able to fail, or it proves nothing."""
        fixture = MatVecFixture(16, 8, 2, 9)
        fixture.expected = fixture.expected.copy()
        fixture.expected[0, 0] = int(fixture.expected[0, 0]) ^ 0x7f
        path = fixture.write(Fixtures.build_dir("cpp") / "matvec_broken.txt")
        result = run([str(self.checker), str(path)])
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL", result.stderr)


if __name__ == "__main__":
    unittest.main()
