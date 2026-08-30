"""The importable API: the whole flow without a subprocess of the caller's.

The point is not that these functions exist, it is that someone can get from a
path to a verified design without knowing that Compiler lives in
onnx2fpga.compile or that SimulationRunner wants a build directory somebody
else prepared.
"""

import pathlib
import shutil
import unittest

from support import Fixtures

import onnx2fpga

MODEL = pathlib.Path("examples/models/mlp.onnx")


class SurfaceTest(unittest.TestCase):
    def test_the_package_exports_the_flow(self):
        for name in ("load", "samples_for", "compile", "estimate", "simulate",
                     "devices", "device"):
            self.assertTrue(hasattr(onnx2fpga, name), name)

    def test_compile_is_the_function_not_the_module(self):
        """compile.py is a submodule, so the name could have been shadowed by
        it. Python binds the submodule during package import and the function
        is defined after, but that is worth pinning rather than remembering."""
        import onnx2fpga.compile  # noqa: F401
        self.assertTrue(callable(onnx2fpga.compile))

    def test_devices_lists_the_registry(self):
        from onnx2fpga.targets.device import Device
        self.assertEqual(onnx2fpga.devices(), Device.names())
        self.assertEqual(onnx2fpga.device("z7020").name, "z7020")


class LoadTest(unittest.TestCase):
    def test_it_takes_a_path(self):
        self.assertEqual(onnx2fpga.load(MODEL).graph_inputs(), ["x"])

    def test_it_takes_a_string(self):
        self.assertEqual(onnx2fpga.load(str(MODEL)).graph_inputs(), ["x"])

    def test_it_passes_a_model_through_untouched(self):
        model = onnx2fpga.load(MODEL)
        self.assertIs(onnx2fpga.load(model), model)

    def test_it_refuses_anything_else_by_name(self):
        with self.assertRaises(TypeError) as caught:
            onnx2fpga.load(42)
        self.assertIn("OnnxModel", str(caught.exception))


class FlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = onnx2fpga.load(MODEL)
        cls.samples = onnx2fpga.samples_for(cls.model, 16)

    def test_estimate_needs_no_samples_of_its_own(self):
        """A caller asking whether a model fits should not have to invent
        calibration data first."""
        estimate = onnx2fpga.estimate(MODEL, device="z7020", target_cycles=64)
        self.assertGreater(estimate.latency_cycles, 0)
        self.assertTrue(estimate.fits)

    def test_compile_returns_the_graphs_and_the_reports(self):
        result = onnx2fpga.compile(
            self.model, self.samples, out_dir=Fixtures.build_dir("api_compile"),
            device="vu9p", target_cycles=64)
        self.assertTrue(result.hardware_graph.nodes)
        self.assertIsNotNone(result.folding)
        self.assertIsNotNone(result.accuracy)
        self.assertIsNotNone(result.plan)

    def test_options_reach_the_compiler(self):
        wide = onnx2fpga.compile(
            self.model, self.samples, out_dir=Fixtures.build_dir("api_wide"),
            device="vu9p", target_cycles=64, act_bits=16, weight_bits=16)
        self.assertEqual(wide.plan.act_dtype.bits, 16)


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class SimulateTest(unittest.TestCase):
    def test_it_goes_from_a_path_to_a_verified_design(self):
        """The Done-when, in one call: no build directory to prepare, no make
        to invoke, no output to parse."""
        result = onnx2fpga.simulate(
            MODEL, out_dir=Fixtures.build_dir("api_simulate"),
            device="vu9p", target_cycles=64)
        self.assertTrue(result.ok)
        self.assertGreater(result.latency, 0)

    def test_a_failed_build_says_so_rather_than_returning_nothing(self):
        from onnx2fpga.simulate import SimulationRunner
        runner = SimulationRunner(Fixtures.build_dir("api_nobuild"))
        self.assertNotEqual(runner.build().returncode, 0)


if __name__ == "__main__":
    unittest.main()
