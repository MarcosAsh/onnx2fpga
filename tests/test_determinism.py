"""Compiling the same model twice must produce byte identical files."""

import filecmp
import hashlib
import unittest

from support import Fixtures, Hdl

from onnx2fpga.compile import Compiler
from make_models import QdqModelFactory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifacts(build):
    return sorted(p for p in build.rglob("*")
                  if p.is_file() and "obj_dir" not in p.parts)


class DeterminismTest(unittest.TestCase):
    def _twice(self, name, build_model, samples=None, **kwargs):
        outputs = []
        for run in ("a", "b"):
            build = Fixtures.build_dir("determinism_%s_%s" % (name, run))
            for stale in artifacts(build):
                stale.unlink()
            outputs.append(Compiler(device="vu9p", **kwargs).compile(
                build_model(), samples, build))
        return outputs

    def test_a_calibrated_model_compiles_identically(self):
        results = self._twice("float", lambda: Fixtures.factory().mlp(),
                              Fixtures.samples(Fixtures.MLP_SHAPE),
                              target_cycles=64)
        self._assert_identical("float", results)

    def test_a_quantized_model_compiles_identically(self):
        results = self._twice("qdq", lambda: QdqModelFactory(seed=7).mlp(),
                              target_cycles=64)
        self._assert_identical("qdq", results)

    def test_a_convolutional_model_compiles_identically(self):
        results = self._twice("cnn", lambda: Fixtures.factory().cnn(),
                              Fixtures.samples(Fixtures.CNN_SHAPE),
                              target_cycles=400)
        self._assert_identical("cnn", results)

    def _assert_identical(self, name, results):
        first, second = (r.build.root for r in results)
        left, right = artifacts(first), artifacts(second)
        self.assertEqual([p.relative_to(first) for p in left],
                         [p.relative_to(second) for p in right],
                         "%s: the two builds contain different files" % name)
        for a, b in zip(left, right):
            with self.subTest(artifact=a.name):
                self.assertEqual(digest(a), digest(b),
                                 "%s differs between two runs" % a.name)

    def test_the_pass_log_carries_no_wall_clock(self):
        """Timings belong beside the log, or the record of a compilation changes every run."""
        result = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir("determinism_log"))
        for line in result.context.log:
            self.assertNotIn(" ms", line, line)
        self.assertTrue(result.context.timings)

    def test_folding_decisions_are_stable(self):
        reports = []
        for run in ("a", "b"):
            result = Compiler(device="vu9p", target_cycles=128).compile(
                Fixtures.factory().cnn(), Fixtures.samples(Fixtures.CNN_SHAPE),
                Fixtures.build_dir("determinism_fold_" + run))
            reports.append(result.folding.render())
        self.assertEqual(reports[0], reports[1])

    @unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
    def test_the_hardcaml_backend_compiles_identically(self):
        """The generator numbers nets as it elaborates, so this pins that numbering depends on the design alone."""
        for name, model, samples, target in (
                ("hc_cnn", lambda: Fixtures.factory().cnn(),
                 Fixtures.samples(Fixtures.CNN_SHAPE), 400),
                ("hc_residual", lambda: Fixtures.factory().residual(),
                 Fixtures.samples((1, 16)), 64)):
            with self.subTest(model=name):
                results = self._twice(name, model, samples, target_cycles=target,
                                      hdl="hardcaml")
                self._assert_identical(name, results)
                self.assertTrue(results[0].build.path("rtl/top.v").is_file())

    def test_golden_vectors_are_stable(self):
        results = self._twice("golden", lambda: QdqModelFactory(seed=7).mlp(),
                              target_cycles=64)
        for relative in ("golden/input.hex", "golden/expected.hex"):
            self.assertTrue(filecmp.cmp(results[0].build.path(relative),
                                        results[1].build.path(relative),
                                        shallow=False), relative)


if __name__ == "__main__":
    unittest.main()
