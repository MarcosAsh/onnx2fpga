"""The tool must work when installed from a wheel, not just from a checkout."""

import os
import pathlib
import shutil
import subprocess
import sys
import unittest
import zipfile
from unittest import mock

from support import Fixtures, ROOT, run

from onnx2fpga import assets
from onnx2fpga.assets import AssetError, AssetLocator


class Locator(unittest.TestCase):
    def test_checkout(self):
        locator = AssetLocator()
        self.assertTrue(locator.rtl_dir.is_dir())
        self.assertEqual(len(locator.rtl_sources()), 10)
        self.assertTrue(locator.synth_script.is_file())
        self.assertTrue((locator.runtime_dir / "sim" / "harness.cpp").is_file())
        self.assertEqual(AssetLocator(ROOT).root, ROOT)

    def test_wrong_environment_fails(self):
        """A wrong override set on purpose must fail loudly, not quietly build against other RTL."""
        empty = Fixtures.build_dir("assets_empty")
        with mock.patch.dict(os.environ, {AssetLocator.ENV: str(empty)}):
            with self.assertRaises(AssetError):
                AssetLocator().root

    def test_stale_staged_copy(self):
        """Files left by make assets must not shadow a checkout's own, edited ones."""
        tree = Fixtures.build_dir("assets_stale")
        package = tree / "compiler" / "onnx2fpga"
        for root, text in ((tree, "fresh"), (package / "assets", "stale")):
            for marker in AssetLocator.MARKERS:
                (root / marker).parent.mkdir(parents=True, exist_ok=True)
                (root / marker).write_text(text)
        with mock.patch.object(assets, "__file__", str(package / "assets.py")):
            locator = AssetLocator()
            self.assertEqual(locator.root, tree)
            self.assertEqual((locator.runtime_dir / "sim" / "harness.cpp").read_text(), "fresh")

    def test_missing(self):
        with self.assertRaises(AssetError) as caught:
            AssetLocator(Fixtures.build_dir("assets_absent")).root
        self.assertIn("hardware/", str(caught.exception))
        self.assertIn(AssetLocator.ENV, str(caught.exception))


class Wheel(unittest.TestCase):
    """Builds the wheel and runs it with the checkout out of reach."""

    @classmethod
    def setUpClass(cls):
        built = run([sys.executable, str(ROOT / "tools" / "build_wheel.py")])
        assert built.returncode == 0, built.stdout + built.stderr
        wheels = sorted((ROOT / "dist").glob("onnx2fpga-*.whl"))
        assert wheels, "no wheel produced"
        cls.wheel = wheels[-1]
        cls.site = Fixtures.build_dir("packaging_site")
        shutil.rmtree(cls.site)
        cls.site.mkdir(parents=True)
        zipfile.ZipFile(cls.wheel).extractall(cls.site)

    def test_contents(self):
        archive = zipfile.ZipFile(self.wheel)
        names = archive.namelist()
        modules = [n for n in names if n.endswith(".sv")]
        self.assertEqual(len(modules), 10, modules)
        for suffix in ("synth_unit.tcl", "sim/harness.cpp", "otf/kernels.hpp"):
            self.assertTrue(any(n.endswith(suffix) for n in names), suffix)
        entry = next(n for n in names if n.endswith("entry_points.txt"))
        self.assertIn("onnx2fpga = onnx2fpga.cli:main", archive.read(entry).decode())

    def installed(self, *arguments, cwd):
        environment = dict(os.environ, PYTHONPATH=str(self.site))
        environment.pop(AssetLocator.ENV, None)
        return subprocess.run([sys.executable] + list(arguments), cwd=cwd,
                              env=environment, capture_output=True, text=True,
                              timeout=900)

    def test_assets_resolve_inside_the_install(self):
        completed = self.installed(
            "-c", "from onnx2fpga.assets import AssetLocator; print(AssetLocator().root)",
            cwd=Fixtures.build_dir("packaging_work"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        resolved = pathlib.Path(completed.stdout.strip())
        self.assertTrue(str(resolved).startswith(str(self.site)),
                        "installed copy reached outside itself: %s" % resolved)

    def run_command(self, command, folder):
        work = Fixtures.build_dir(folder)
        shutil.copyfile(ROOT / "examples" / "models" / "mlp.onnx", work / "mlp.onnx")
        completed = self.installed("-m", "onnx2fpga", command, "mlp.onnx", "--out", "out",
                                   "--target-cycles", "64", "--quiet", cwd=work)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return work / "out", completed

    def test_compile(self):
        out, _ = self.run_command("compile", "packaging_work")
        for relative in ("rtl/top.sv", "Makefile", "manifest.json"):
            self.assertTrue((out / relative).exists(), relative)
        self.assertEqual(len(list((out / "rtl").glob("otf_*.sv"))), 10)

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
    def test_simulate(self):
        """The generated Makefile points at the C++ harness, which must be inside the wheel."""
        _, completed = self.run_command("simulate", "packaging_sim")
        self.assertIn("PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
