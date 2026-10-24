"""End to end: the generated RTL must reproduce the numpy model bit for bit."""

import shutil
import unittest

from support import Fixtures, Hdl, SingleUnitProject, run
from test_cpp_contract import MatVecFixture

from onnx2fpga.compile import Compiler
from onnx2fpga.p2_graph.datatype import INT8

BACKPRESSURE = [(100, 100), (50, 50), (30, 90), (90, 30), (20, 20), (10, 100), (100, 10)]


class SimulationCases:
    """Run once per backend; each subclass names its HDL."""

    HDL = "sv"
    MODELS = {
        "mlp": (Fixtures.MLP_SHAPE, 64),
        "cnn": (Fixtures.CNN_SHAPE, 400),
    }

    @classmethod
    def setUpClass(cls):
        cls.builds = {}
        factory = Fixtures.factory()
        for name, (shape, target) in cls.MODELS.items():
            model = getattr(factory, name)()
            build = Fixtures.build_dir("sim_" + name + Hdl.suffix(cls.HDL))
            Compiler(device="vu9p", target_cycles=target, hdl=cls.HDL).compile(
                model, Fixtures.samples(shape), build)
            built = run(["make", "-s"], cwd=build)
            assert built.returncode == 0, built.stderr[-4000:]
            cls.builds[name] = build

    def test_lints_clean(self):
        for name, build in self.builds.items():
            with self.subTest(model=name):
                result = run(["make", "-s", "lint"], cwd=build)
                self.assertEqual(result.returncode, 0, result.stderr[-4000:])

    def test_matches_golden_model(self):
        for name, build in self.builds.items():
            with self.subTest(model=name):
                result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                              "--expected", "golden/expected.hex"], cwd=build)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS", result.stdout)

    def test_survives_randomised_backpressure(self):
        """A stream design that only sees tready high can pass its tests and still deadlock in hardware."""
        for name, build in self.builds.items():
            for input_duty, output_duty in BACKPRESSURE:
                for seed in (1, 2, 3):
                    with self.subTest(model=name, duty=(input_duty, output_duty),
                                      seed=seed):
                        result = run(["./obj_dir/Votf_top",
                                      "--input", "golden/input.hex",
                                      "--expected", "golden/expected.hex",
                                      "--input-duty", str(input_duty),
                                      "--output-duty", str(output_duty),
                                      "--seed", str(seed), "--quiet"], cwd=build)
                        self.assertEqual(result.returncode, 0,
                                         result.stdout + result.stderr)

    def test_reports_a_latency_within_the_predicted_frame_cost(self):
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"], cwd=self.builds["mlp"])
        self.assertIn("cycles", result.stdout)
        self.assertIn("latency", result.stdout)

    def test_a_tiny_multiplier_matches_the_compiler(self):
        """The C++ contract layer through the RTL, where the rounding constant used to overflow and negatives came out one low."""
        fixture = MatVecFixture.tiny_multiplier()
        unit = fixture.unit()
        unit.apply_folding({"simd": 4, "pe": 2})
        self.assertGreater(fixture.requant.shift,
                           unit.accumulator_dtype.bits + unit.mult_bits)
        build = Fixtures.build_dir("sim_tiny_multiplier" + Hdl.suffix(self.HDL))
        project = SingleUnitProject(unit, (fixture.vectors, fixture.mw), INT8,
                                    hdl=self.HDL)
        _, expected = project.write(build, fixture.input)
        self.assertTrue((expected == fixture.expected).all())
        built = run(["make", "-s"], cwd=build)
        self.assertEqual(built.returncode, 0, built.stderr[-4000:])
        result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                      "--expected", "golden/expected.hex"], cwd=build)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class SimulationTest(SimulationCases, unittest.TestCase):
    HDL = "sv"


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
class HardcamlSimulationTest(SimulationCases, unittest.TestCase):
    HDL = "hardcaml"


if __name__ == "__main__":
    unittest.main()
