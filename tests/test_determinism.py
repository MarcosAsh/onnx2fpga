"""Compiling the same model twice must produce byte identical files."""

import unittest

from support import Fixtures, Hdl

from onnx2fpga.compile import Compiler
from make_models import QdqModelFactory


def artifacts(build):
    return sorted(p for p in build.rglob("*")
                  if p.is_file() and "obj_dir" not in p.parts)


class DeterminismTest(unittest.TestCase):
    def _twice(self, name, build_model, samples=None, **options):
        results = []
        for run in ("a", "b"):
            build = Fixtures.build_dir("determinism_%s_%s" % (name, run))
            for stale in artifacts(build):
                stale.unlink()
            results.append(Compiler(device="vu9p", **options).compile(
                build_model(), samples, build))
        first, second = (r.build.root for r in results)
        left, right = artifacts(first), artifacts(second)
        self.assertEqual([p.relative_to(first) for p in left],
                         [p.relative_to(second) for p in right])
        for a, b in zip(left, right):
            self.assertEqual(a.read_bytes(), b.read_bytes(), "%s differs" % a.name)
        self.assertEqual(results[0].folding.render(), results[1].folding.render())
        return results

    def test_builds_are_identical(self):
        """Calibrated, quantized and convolutional models; the files include the golden vectors."""
        cases = (("float", lambda: Fixtures.factory().mlp(),
                  Fixtures.samples(Fixtures.MLP_SHAPE), 64),
                 ("qdq", lambda: QdqModelFactory(seed=7).mlp(), None, 64),
                 ("cnn", lambda: Fixtures.factory().cnn(),
                  Fixtures.samples(Fixtures.CNN_SHAPE), 128))
        for name, model, samples, target in cases:
            with self.subTest(model=name):
                self._twice(name, model, samples, target_cycles=target)

    @unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
    def test_hardcaml_builds_are_identical(self):
        """Generated net numbers must depend on the design alone."""
        cases = (("hc_cnn", lambda: Fixtures.factory().cnn(),
                  Fixtures.samples(Fixtures.CNN_SHAPE), 400),
                 ("hc_residual", lambda: Fixtures.factory().residual(),
                  Fixtures.samples((1, 16)), 64))
        for name, model, samples, target in cases:
            with self.subTest(model=name):
                results = self._twice(name, model, samples, target_cycles=target,
                                      hdl="hardcaml")
                self.assertTrue(results[0].build.path("rtl/top.v").is_file())

    def test_log_has_no_wall_clock(self):
        result = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("determinism_log"))
        for line in result.context.log:
            self.assertNotIn(" ms", line, line)
        self.assertTrue(result.context.timings)


if __name__ == "__main__":
    unittest.main()
