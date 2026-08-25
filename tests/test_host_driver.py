"""The host driver has to be able to run what the compiler emits.

Nothing checked that before, and the gap showed: per-feature input scaling
made the manifest carry a list of scales, and the C++ manifest reader takes a
number. The compiler could produce a design its own runtime refused to load,
and the only way to find out was to try.
"""

import json
import shutil
import unittest

import numpy as np

from support import ROOT, Fixtures, run

from onnx2fpga.compile import Compiler

DRIVER = ROOT / "runtime" / "bin" / "otf_run"


def mixed_range_samples(count=48, features=32, seed=11):
    rng = np.random.default_rng(seed)
    spread = 10.0 ** rng.uniform(-3, 1, features)
    return [{"x": rng.normal(0, 1, (1, features)) * spread}
            for _ in range(count)]


@unittest.skipIf(shutil.which("g++") is None, "no C++ compiler")
@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class HostDriverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        built = run(["make", "-s"], cwd=ROOT / "runtime")
        assert built.returncode == 0, built.stderr[-3000:]
        cls.samples = mixed_range_samples()
        cls.builds = {}
        for per_feature in (False, True):
            build = Fixtures.build_dir("host_%s" % per_feature)
            Compiler(device="vu9p", target_cycles=64,
                     per_feature_input=per_feature).compile(
                Fixtures.factory().mlp(), cls.samples, build)
            made = run(["make", "-s"], cwd=build)
            assert made.returncode == 0, made.stderr[-3000:]
            cls.builds[per_feature] = build

    def drive(self, per_feature):
        return run([str(DRIVER), str(self.builds[per_feature])])

    def test_it_runs_a_design_on_one_scale(self):
        result = self.drive(False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("output", result.stdout)

    def test_it_runs_a_design_scaled_per_feature(self):
        """The regression. Before the manifest reader understood a list of
        scales this died with a json type mismatch, which is a fine way to
        fail and a poor thing to discover from a user."""
        result = self.drive(True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("output", result.stdout)

    def test_the_manifest_really_does_carry_a_list(self):
        """Otherwise the test above passes for the wrong reason."""
        manifest = json.loads(
            (self.builds[True] / "manifest.json").read_text())
        self.assertIsInstance(manifest["input"]["scale"], list)
        self.assertGreater(len(manifest["input"]["scale"]), 1)
        self.assertNotIsInstance(
            json.loads((self.builds[False] / "manifest.json").read_text())
            ["input"]["scale"], list)

    def test_the_two_designs_do_not_produce_the_same_answers(self):
        """They are quantized differently, so identical output would mean one
        of them is not being read."""
        flat = self.drive(False).stdout
        split = self.drive(True).stdout
        self.assertNotEqual(flat, split)


if __name__ == "__main__":
    unittest.main()
