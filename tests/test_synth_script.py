"""The Vivado script run under tclsh stubs: control flow, fmax arithmetic and the summary shape."""

import json
import os
import shutil
import subprocess
import unittest

from support import Fixtures, ROOT

from onnx2fpga.measure import SynthesisResult

STUB = ROOT / "tests" / "vivado_stub.tcl"
PERIOD = 5.0
SLACK = 0.842


def run_stub(name, implement=False, generics="MW=9 MH=4 SIMD=3 PE=4", clock=None):
    out = Fixtures.build_dir("synth_stub_" + name) / "synth" / "unit"
    if out.exists():
        shutil.rmtree(out)
    environment = dict(os.environ, OTF_PART="xc7z020-clg400-1", OTF_TOP="otf_mvau",
                       OTF_RTL="rtl/otf_requant.sv rtl/otf_mvau.sv",
                       OTF_GENERICS=generics, OTF_PERIOD=str(PERIOD), OTF_OUT=str(out),
                       OTF_IMPL="1" if implement else "0")
    if clock:
        environment["OTF_CLOCK"] = clock
    completed = subprocess.run(["tclsh", str(STUB)], cwd=out.parents[1], env=environment,
                               capture_output=True, text=True, timeout=120)
    calls = (out / "calls.txt").read_text().splitlines() if out.exists() else []
    return out, completed, calls


@unittest.skipIf(shutil.which("tclsh") is None, "tclsh not installed")
class SynthScriptTest(unittest.TestCase):
    def test_synthesis_run(self):
        out, completed, calls = run_stub("synth")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("OTF_SUMMARY", completed.stdout)

        result = SynthesisResult.load(out / "summary.json")
        measured = result.measured
        self.assertEqual((measured.lut, measured.ff, measured.dsp, measured.uram),
                         (412, 288, 6, 0))
        self.assertEqual(result.stage, "synth")
        self.assertAlmostEqual(result.fmax_mhz, round(1000.0 / (PERIOD - SLACK), 2), places=2)
        payload = json.loads((out / "summary.json").read_text())
        # Half width block RAMs count as half.
        self.assertEqual((payload["ramb36"], payload["ramb18"], payload["bram36"]), (1, 2, 2))

        xdc = (out / "clock.xdc").read_text()
        self.assertIn("create_clock", xdc)
        self.assertIn("-period 5.0", xdc)
        self.assertIn("[get_ports clk]", xdc)
        first = {step: next(i for i, c in enumerate(calls) if c.startswith(step))
                 for step in ("read_xdc", "synth_design")}
        self.assertLess(first["read_xdc"], first["synth_design"])
        synth = calls[first["synth_design"]]
        self.assertIn("-mode out_of_context", synth)
        self.assertIn("-top otf_mvau", synth)
        for pair in ("MW=9", "MH=4", "SIMD=3", "PE=4"):
            self.assertIn("-generic %s" % pair, synth)
        reads = [c for c in calls if c.startswith("read_verilog")]
        self.assertEqual(len(reads), 2)
        self.assertTrue(any("otf_mvau.sv" in c for c in reads))
        self.assertFalse(any(c.startswith("route_design") for c in calls))
        self.assertFalse((out / "utilization_impl.rpt").exists())

    def test_implementation_run(self):
        out, completed, calls = run_stub("impl", implement=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        for step in ("opt_design", "place_design", "phys_opt_design", "route_design"):
            self.assertTrue(any(c.startswith(step) for c in calls), step)
        self.assertEqual(SynthesisResult.load(out / "summary.json").stage, "impl")
        self.assertTrue((out / "utilization_impl.rpt").exists())

    def test_clock_port(self):
        # Hardcaml tops name the clock port clock.
        out, completed, _ = run_stub("clock", generics="", clock="clock")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("[get_ports clock]", (out / "clock.xdc").read_text())

    def test_no_generics(self):
        _, completed, calls = run_stub("nogenerics", generics="")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        synth = next(c for c in calls if c.startswith("synth_design"))
        self.assertNotIn("-generic", synth)


if __name__ == "__main__":
    unittest.main()
