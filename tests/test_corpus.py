"""Real exported models: each compiles in fixed latency mode for k26, simulates bit exact and tracks ONNX Runtime."""

import functools
import shutil
import unittest
import warnings

import numpy as np

from support import ROOT, Fixtures, Hdl

import onnx2fpga as otf
from onnx2fpga.errors import UnsupportedOpError

CORPUS = ROOT / "tests" / "fixtures" / "corpus"
HARDCAML = Hdl.skip_reason("hardcaml")
MODELS = sorted(path.stem for path in CORPUS.glob("*.onnx"))
# Models ONNX Runtime quantized, whose grids the float reference must respect.
QUANTIZED = [name for name in MODELS if name.startswith("ort_")]
# Models that need an op from later work, and that op; none today.
EXPECTED_FAIL = {}
# Mean absolute error against ONNX Runtime, as a share of the output's standard deviation.
TOLERANCE = 0.025
# Share of labels that must agree with ONNX Runtime's.
AGREEMENT = 0.98


def stored(name):
    return np.load(CORPUS / ("%s.npz" % name))


class Corpus(unittest.TestCase):
    def test_expected_failures(self):
        for name, op in EXPECTED_FAIL.items():
            with self.subTest(name), self.assertRaisesRegex(UnsupportedOpError, op):
                otf.load(CORPUS / ("%s.onnx" % name)).compile(
                    device="k26", mode="fixed", calibration=stored(name)["calibration"])

    @unittest.skipIf(HARDCAML, HARDCAML)
    def test_tracks_onnxruntime(self):
        for name in sorted(set(MODELS) - set(EXPECTED_FAIL)):
            with self.subTest(name):
                data = stored(name)
                acc = compiled(name)
                got = acc.predict(data["x"]).reshape(data["y"].shape)
                if acc._labels:
                    self.assertGreaterEqual(np.mean(got == data["y"]), AGREEMENT)
                else:
                    error = np.mean(np.abs(got - data["y"])) / np.std(data["y"])
                    self.assertLess(error, TOLERANCE)

    @unittest.skipIf(HARDCAML, HARDCAML)
    def test_float_reference_takes_grids(self):
        # ORT drops a Relu whose output grid already clamps at zero.
        for name in QUANTIZED:
            with self.subTest(name):
                data = stored(name)
                acc = compiled(name)
                floating = np.concatenate([acc._float_reference({"x": x[None]})
                                           for x in data["x"]]).reshape(data["y"].shape)
                error = np.mean(np.abs(floating - data["y"])) / np.std(data["y"])
                self.assertLess(error, TOLERANCE)

    @unittest.skipIf(HARDCAML, HARDCAML)
    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_simulate(self):
        for name in sorted(set(MODELS) - set(EXPECTED_FAIL)):
            with self.subTest(name):
                report = compiled(name).simulate()
                self.assertTrue(report.ok, report.first.output[-2000:])
                self.assertEqual(report.frames, 10000)

    def test_fixtures_match_onnxruntime(self):
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        for name in MODELS:
            with self.subTest(name):
                data = stored(name)
                session = onnxruntime.InferenceSession(str(CORPUS / ("%s.onnx" % name)))
                feed = {session.get_inputs()[0].name: data["x"]}
                y = session.run([session.get_outputs()[0].name], feed)[0]
                np.testing.assert_allclose(np.reshape(y, data["y"].shape), data["y"],
                                           rtol=1e-5, atol=1e-6)


@functools.lru_cache(maxsize=None)
def compiled(name):
    acc = otf.load(CORPUS / ("%s.onnx" % name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc.compile(device="k26", mode="fixed", calibration=stored(name)["calibration"],
                    out_dir=Fixtures.build_dir("corpus_%s" % name))
    return acc


if __name__ == "__main__":
    unittest.main()
