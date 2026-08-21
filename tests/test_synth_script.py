"""The Vivado script, executed under stubs.

No Vivado exists on the development machine, so hardware/synth/synth_unit.tcl
would otherwise ship having never run. Sourcing it under tclsh with stand-ins
for the Vivado commands still checks its control flow, its fmax arithmetic and
the exact shape of the summary it writes, which is the interface the Python
driver reads. Whether Vivado itself agrees is what the first real run decides.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest

from support import Fixtures, ROOT

from onnx2fpga.measure import SynthesisResult  # noqa: E402

STUB = ROOT / "tests" / "vivado_stub.tcl"
PERIOD = 5.0
SLACK = 0.842


@unittest.skipIf(shutil.which("tclsh") is None, "tclsh not installed")
class SynthScriptTest(unittest.TestCase):
    def _run(self, implement=False, generics="MW=9 MH=4 SIMD=3 PE=4",
             timing_paths=True):
        work = Fixtures.build_dir("synth_stub_%s" % ("impl" if implement else "synth"))
        out = work / "synth" / "unit"
        if out.exists():
            shutil.rmtree(out)
        environment = dict(os.environ)
        environment.pop("OTF_STUB_NO_PATHS", None)
        if not timing_paths:
            environment["OTF_STUB_NO_PATHS"] = "1"
        environment.update({
            "OTF_PART": "xc7z020-clg400-1",
            "OTF_TOP": "otf_mvau",
            "OTF_RTL": "rtl/otf_requant.sv rtl/otf_mvau.sv",
            "OTF_GENERICS": generics,
            "OTF_PERIOD": str(PERIOD),
            "OTF_OUT": str(out),
            "OTF_IMPL": "1" if implement else "0",
        })
        completed = subprocess.run(["tclsh", str(STUB)], cwd=work, env=environment,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        return out, completed

    @property
    def _calls(self):
        return self.__dict__.setdefault("_cache", {})

    def calls_of(self, out):
        return (out / "calls.txt").read_text().splitlines()

    def test_the_script_runs_without_a_tcl_error(self):
        out, completed = self._run()
        self.assertTrue((out / "summary.json").exists())
        self.assertIn("OTF_SUMMARY", completed.stdout)

    def test_summary_is_exactly_what_the_python_side_reads(self):
        out, _ = self._run()
        result = SynthesisResult.load(out / "summary.json")
        self.assertEqual(result.measured.lut, 412)
        self.assertEqual(result.measured.ff, 288)
        self.assertEqual(result.measured.dsp, 6)
        self.assertEqual(result.measured.uram, 0)
        self.assertEqual(result.stage, "synth")

    def test_half_width_block_rams_count_as_half(self):
        out, _ = self._run()
        payload = json.loads((out / "summary.json").read_text())
        self.assertEqual(payload["ramb36"], 1)
        self.assertEqual(payload["ramb18"], 2)
        self.assertEqual(payload["bram36"], 1 + 0.5 * 2)

    def test_fmax_follows_from_the_period_and_the_slack(self):
        out, _ = self._run()
        result = SynthesisResult.load(out / "summary.json")
        expected = 1000.0 / (PERIOD - SLACK)
        self.assertAlmostEqual(result.fmax_mhz, round(expected, 2), places=2)
        self.assertIs(json.loads((out / "summary.json").read_text())["timed"], True)

    def test_a_unit_with_no_setup_path_is_unconstrained_not_zero_slack(self):
        """Reporting slack 0 would put fmax at exactly 1000/period, which reads
        as a unit that just met timing rather than one that was never timed."""
        out, _ = self._run(timing_paths=False)
        payload = json.loads((out / "summary.json").read_text())
        self.assertIs(payload["timed"], False)
        self.assertIsNone(payload["wns_ns"])
        self.assertIsNone(payload["fmax_mhz"])
        self.assertIsNone(SynthesisResult.load(out / "summary.json").fmax_mhz)
        self.assertNotEqual(payload["fmax_mhz"], 1000.0 / PERIOD)

    def test_an_unconstrained_unit_still_reports_its_resources(self):
        out, _ = self._run(timing_paths=False)
        result = SynthesisResult.load(out / "summary.json")
        self.assertEqual(result.measured.lut, 412)
        self.assertEqual(result.measured.dsp, 6)

    def test_clock_constraint_is_written_before_synthesis(self):
        out, _ = self._run()
        xdc = (out / "clock.xdc").read_text()
        self.assertIn("create_clock", xdc)
        self.assertIn("-period 5.0", xdc)
        calls = self.calls_of(out)
        self.assertLess(next(i for i, c in enumerate(calls) if c.startswith("read_xdc")),
                        next(i for i, c in enumerate(calls) if c.startswith("synth_design")))

    def test_synthesis_is_out_of_context_with_the_generics_applied(self):
        out, _ = self._run()
        synth = next(c for c in self.calls_of(out) if c.startswith("synth_design"))
        self.assertIn("-mode out_of_context", synth)
        self.assertIn("-top otf_mvau", synth)
        for pair in ("MW=9", "MH=4", "SIMD=3", "PE=4"):
            self.assertIn("-generic %s" % pair, synth)

    def test_every_source_is_read(self):
        out, _ = self._run()
        reads = [c for c in self.calls_of(out) if c.startswith("read_verilog")]
        self.assertEqual(len(reads), 2)
        self.assertTrue(any("otf_mvau.sv" in c for c in reads))

    def test_synthesis_only_run_does_not_place_or_route(self):
        out, _ = self._run(implement=False)
        calls = self.calls_of(out)
        self.assertFalse(any(c.startswith("route_design") for c in calls))
        self.assertFalse((out / "utilization_impl.rpt").exists())

    def test_implementation_run_places_routes_and_says_so(self):
        out, _ = self._run(implement=True)
        calls = self.calls_of(out)
        for step in ("opt_design", "place_design", "phys_opt_design", "route_design"):
            self.assertTrue(any(c.startswith(step) for c in calls), step)
        self.assertEqual(SynthesisResult.load(out / "summary.json").stage, "impl")
        self.assertTrue((out / "utilization_impl.rpt").exists())

    def test_a_unit_with_no_generics_still_works(self):
        out, _ = self._run(generics="")
        synth = next(c for c in self.calls_of(out) if c.startswith("synth_design"))
        self.assertNotIn("-generic", synth)


if __name__ == "__main__":
    unittest.main()
