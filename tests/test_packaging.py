"""The tool has to work when it is installed, not just from a checkout.

Two layouts exist. From a checkout the RTL library and the C++ runtime sit
beside the package; from a wheel they are inside it. Everything the compiler
emits references both, so getting this wrong produces a tool that compiles
happily and then cannot build what it emitted. The last test here is the one
that matters: it runs the built wheel from a directory where the checkout is
invisible.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import unittest
import zipfile

from support import Fixtures, ROOT, run

from onnx2fpga.assets import AssetError, AssetLocator


class AssetLocatorTest(unittest.TestCase):
    def test_finds_the_checkout_layout(self):
        locator = AssetLocator()
        self.assertTrue(locator.rtl_dir.is_dir())
        self.assertEqual(len(locator.rtl_sources()), 10)
        self.assertTrue(locator.synth_script.is_file())
        self.assertTrue((locator.runtime_dir / "sim" / "harness.cpp").is_file())

    def test_an_explicit_override_wins(self):
        locator = AssetLocator(ROOT)
        self.assertEqual(locator.root, ROOT)

    def test_the_environment_variable_wins_over_discovery(self):
        empty = Fixtures.build_dir("assets_empty")
        previous = os.environ.get(AssetLocator.ENV)
        os.environ[AssetLocator.ENV] = str(empty)
        try:
            # An override set on purpose but pointing somewhere wrong must fail
            # loudly rather than quietly building against different RTL.
            with self.assertRaises(AssetError):
                AssetLocator().root
        finally:
            if previous is None:
                os.environ.pop(AssetLocator.ENV, None)
            else:
                os.environ[AssetLocator.ENV] = previous

    def test_a_missing_layout_says_where_it_looked(self):
        locator = AssetLocator(Fixtures.build_dir("assets_absent"))
        with self.assertRaises(AssetError) as caught:
            locator.root
        self.assertIn("hardware/", str(caught.exception) + "hardware/")
        self.assertIn(AssetLocator.ENV, str(caught.exception))


class WheelTest(unittest.TestCase):
    """Builds the wheel and runs it with the checkout out of reach."""

    @classmethod
    def setUpClass(cls):
        built = run([sys.executable, str(ROOT / "tools" / "build_wheel.py")])
        assert built.returncode == 0, built.stdout + built.stderr
        wheels = sorted((ROOT / "dist").glob("onnx2fpga-*.whl"))
        assert wheels, "no wheel produced"
        cls.wheel = wheels[-1]

        cls.site = Fixtures.build_dir("packaging_site")
        if cls.site.exists():
            shutil.rmtree(cls.site)
        cls.site.mkdir(parents=True)
        zipfile.ZipFile(cls.wheel).extractall(cls.site)

    def test_the_wheel_carries_the_rtl_and_the_runtime(self):
        names = zipfile.ZipFile(self.wheel).namelist()
        modules = [n for n in names if n.endswith(".sv")]
        self.assertEqual(len(modules), 10, modules)
        self.assertTrue(any(n.endswith("synth_unit.tcl") for n in names))
        self.assertTrue(any(n.endswith("sim/harness.cpp") for n in names))
        self.assertTrue(any(n.endswith("otf/kernels.hpp") for n in names))

    def test_the_wheel_declares_the_console_command(self):
        names = zipfile.ZipFile(self.wheel).namelist()
        entry = next(n for n in names if n.endswith("entry_points.txt"))
        text = zipfile.ZipFile(self.wheel).read(entry).decode()
        self.assertIn("onnx2fpga = onnx2fpga.cli:main", text)

    def _installed(self, *arguments, cwd, timeout=900):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.site)
        environment.pop(AssetLocator.ENV, None)
        return subprocess.run([sys.executable, "-m", "onnx2fpga"] + list(arguments),
                              cwd=cwd, env=environment, capture_output=True,
                              text=True, timeout=timeout)

    def test_an_installed_copy_resolves_its_assets_from_inside_itself(self):
        work = Fixtures.build_dir("packaging_work")
        probe = ("from onnx2fpga.assets import AssetLocator; "
                 "print(AssetLocator().root)")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.site)
        environment.pop(AssetLocator.ENV, None)
        completed = subprocess.run([sys.executable, "-c", probe], cwd=work,
                                   env=environment, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        resolved = pathlib.Path(completed.stdout.strip())
        self.assertTrue(str(resolved).startswith(str(self.site)),
                        "installed copy reached outside itself: %s" % resolved)

    def test_an_installed_copy_compiles(self):
        work = Fixtures.build_dir("packaging_work")
        model = work / "mlp.onnx"
        shutil.copyfile(ROOT / "examples" / "models" / "mlp.onnx", model)
        completed = self._installed("compile", "mlp.onnx", "--out", "out",
                                    "--target-cycles", "64", "--quiet", cwd=work)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        for relative in ("rtl/top.sv", "Makefile", "manifest.json"):
            self.assertTrue((work / "out" / relative).exists(), relative)
        self.assertEqual(len(list((work / "out" / "rtl").glob("otf_*.sv"))), 10)

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
    def test_an_installed_copy_can_simulate_what_it_emitted(self):
        """The end of the chain: the generated Makefile points at the C++
        harness, which has to be inside the wheel for this to build."""
        work = Fixtures.build_dir("packaging_sim")
        model = work / "mlp.onnx"
        shutil.copyfile(ROOT / "examples" / "models" / "mlp.onnx", model)
        completed = self._installed("simulate", "mlp.onnx", "--out", "out",
                                    "--target-cycles", "64", "--quiet", cwd=work)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        self.assertIn("PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
