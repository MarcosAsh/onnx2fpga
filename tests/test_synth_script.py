"""The Vivado script run under tclsh stubs: control flow, fmax arithmetic and the summary shape."""

import json
import os
import shutil
import subprocess
import unittest
from unittest import mock

from support import Fixtures, ROOT

from onnx2fpga.measure import SynthesisResult

STUB = ROOT / "tests" / "vivado_stub.tcl"
PERIOD = 5.0
SLACK = 0.842


def run_stub(name, implement=False, generics="MW=9 MH=4 SIMD=3 PE=4", clock=None,
             report=None, checkpoint=None, checkpoint_period=None, replicate_net=None):
    out = Fixtures.build_dir("synth_stub_" + name) / "synth" / "unit"
    if out.exists():
        shutil.rmtree(out)
    environment = dict(os.environ, OTF_PART="xc7z020-clg400-1", OTF_TOP="otf_mvau",
                       OTF_RTL="rtl/otf_requant.sv rtl/otf_mvau.sv",
                       OTF_GENERICS=generics, OTF_PERIOD=str(PERIOD), OTF_OUT=str(out),
                       OTF_IMPL="1" if implement else "0")
    if checkpoint:
        environment["OTF_CHECKPOINT"] = checkpoint
        environment.pop("OTF_RTL")
    if checkpoint_period is not None:
        environment["OTF_TEST_PERIOD"] = str(checkpoint_period)
    if replicate_net is not None:
        environment["OTF_REPLICATE_NET"] = replicate_net
    if report:
        environment["OTF_TEST_REPORT"] = str(report)
    if clock:
        environment["OTF_CLOCK"] = clock
    completed = subprocess.run(["tclsh", str(STUB)], cwd=out.parents[1], env=environment,
                               capture_output=True, text=True, timeout=120)
    calls = (out / "calls.txt").read_text().splitlines() if (out / "calls.txt").exists() else []
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

    def test_real_utilization(self):
        for implement in (False, True):
            with self.subTest(implement=implement):
                out, completed, _ = run_stub("real", implement=implement,
                                             report=ROOT / "tests" / "k26_utilization_synth.rpt")
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                measured = SynthesisResult.load(out / "summary.json").measured
                self.assertEqual((measured.lut, measured.ff), (860, 694))

    def test_dsp_utilization(self):
        out, completed, _ = run_stub("dsp", implement=True,
                                     report=ROOT / "tests" / "k26_brevitas_utilization.rpt")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        measured = SynthesisResult.load(out / "summary.json").measured
        self.assertEqual((measured.lut, measured.ff, measured.dsp), (6526, 3142, 47))

    def test_missing_utilization(self):
        report = Fixtures.build_dir("missing_utilization") / "report.rpt"
        report.write_text("| CLB LUTs* | 860 | 0 | 117120 | 0.73 |\n")
        out, completed, _ = run_stub("missing", report=report)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing utilization row", completed.stderr)
        self.assertFalse((out / "summary.json").exists())

    def test_implementation_run(self):
        out, completed, calls = run_stub("impl", implement=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        for step in ("opt_design", "place_design", "phys_opt_design", "route_design"):
            self.assertTrue(any(c.startswith(step) for c in calls), step)
        self.assertEqual(SynthesisResult.load(out / "summary.json").stage, "impl")
        self.assertTrue((out / "utilization_impl.rpt").exists())

    def test_checkpoint_refinement(self):
        out, completed, calls = run_stub("refine", checkpoint="routed.dcp")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(SynthesisResult.load(out / "summary.json").stage, "impl")
        self.assertTrue((out / "routed.dcp").exists())
        self.assertIn("open_checkpoint routed.dcp", calls)
        self.assertIn("phys_opt_design -directive AggressiveExplore", calls)
        self.assertIn("route_design -directive Explore", calls)
        self.assertFalse(any(c.startswith(("read_verilog", "read_xdc", "synth_design"))
                             for c in calls))

    def test_checkpoint_clock_mismatch(self):
        out, completed, _ = run_stub("bad_clock", checkpoint="routed.dcp", checkpoint_period=4)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("checkpoint period mismatch", completed.stderr)
        self.assertFalse((out / "summary.json").exists())

    def test_checkpoint_replication(self):
        _, completed, calls = run_stub("replicate", checkpoint="routed.dcp",
                                       replicate_net="book/control")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        unroute = calls.index("route_design -unroute")
        replicate = calls.index("phys_opt_design -force_replication_on_nets book/control")
        route = calls.index("route_design -directive Explore")
        self.assertLess(unroute, replicate)
        self.assertLess(replicate, route)
        self.assertNotIn("phys_opt_design -directive AggressiveExplore", calls)

    def test_replication_requires_checkpoint_and_net(self):
        for checkpoint, net, error in (
            (None, "book/control", "replication requires a checkpoint"),
            ("routed.dcp", "missing", "replication requires one exact net name"),
            ("routed.dcp", "book/*", "replication requires one exact net name"),
            ("routed.dcp", "*", "replication requires one exact net name"),
        ):
            with self.subTest(checkpoint=checkpoint, net=net):
                out, completed, _ = run_stub("bad_replication", checkpoint=checkpoint,
                                             replicate_net=net)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(error, completed.stderr)
                self.assertFalse((out / "summary.json").exists())

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

    def test_clock_name_is_literal(self):
        clock = 'clocking$clock'
        out, completed, _ = run_stub("literal_clock", generics="", clock=clock)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        script = ('proc get_ports {name} { return $name }\n'
                  'proc create_clock {args} { puts [lindex $args end] }\n'
                  + (out / "clock.xdc").read_text())
        result = subprocess.run(['tclsh'], input=script, capture_output=True, text=True)
        self.assertEqual(result.stderr, '')
        self.assertEqual(result.stdout.strip(), clock)

    def test_missing_clock_rejected(self):
        with mock.patch.dict(os.environ, OTF_TEST_MISSING_CLOCK="1"):
            out, completed, _ = run_stub("missing_clock")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing synthesis clock", completed.stderr)
        self.assertFalse((out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
