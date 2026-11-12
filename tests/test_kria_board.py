"""The k26 device, the KV260 and KR260 Vivado projects against the Vivado stub, and the timing report parser."""

import json
import os
import shutil
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from support import Fixtures, Hdl, ROOT

import onnx2fpga as otf
from onnx2fpga.board import BoardError, KV260, Kria
from onnx2fpga.errors import ConfigError, ToolchainError
from onnx2fpga.measure import TimingReport
from onnx2fpga.targets.device import Device

HARDCAML = Hdl.skip_reason("hardcaml")
# Made up in Vivado's layout, not measured.
CANNED = ROOT / "tests" / "kr260_timing_summary.rpt"


class K26(unittest.TestCase):
    def test_device(self):
        k26 = Device.get("k26")
        self.assertEqual(k26.part, "xck26-sfvc784-2LV-c")
        budget = k26.budget
        self.assertEqual((budget.lut, budget.ff, budget.dsp, budget.bram36, budget.uram),
                         (117120, 234240, 1248, 144, 64))
        self.assertEqual(k26.fmax_mhz, 322.265625)
        self.assertEqual((k26.macs_per_dsp(8, 8), k26.macs_per_dsp(4, 4)), (2.0, 4.0))

    def test_timing_report(self):
        report = TimingReport.load(CANNED)
        self.assertEqual((report.period_ns, report.wns_ns), (3.103, 0.137))
        self.assertAlmostEqual(report.fmax_mhz, 1000 / 2.966)
        self.assertTrue(report.met)
        with self.assertRaisesRegex(ValueError, "no timing for clock otf_clk"):
            TimingReport.parse("stub report")

    def test_generated_clock_report(self):
        report = TimingReport.load(ROOT / "tests" / "kr260_generated_timing_summary.rpt")
        self.assertEqual((report.period_ns, report.wns_ns), (3.103, 0.561))
        self.assertTrue(report.met)

    def test_synth_dry_run(self):
        completed = subprocess.run(
            [sys.executable, "-m", "onnx2fpga", "synth", "examples/models/mlp.onnx",
             "--device", "k26", "--dry-run",
             "--out", str(Fixtures.build_dir("synth_k26"))],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
            env=dict(os.environ, PYTHONPATH="compiler:tests:examples"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("on xck26-sfvc784-2LV-c", completed.stdout)
        self.assertIn("OTF_PERIOD=3.103", completed.stdout)


@unittest.skipIf(HARDCAML, HARDCAML)
@unittest.skipIf(shutil.which("tclsh") is None, "tclsh not installed")
class Flow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The example MLP fits the K26 only with fc1 on multipliers, so a smaller one.
        cls.acc = otf.load(Fixtures.factory().mlp((16, 32, 8)), name="mlp")
        x = np.random.default_rng(5).normal(0, 1, (8, 16)).astype(np.float32)
        cls.acc.compile(device="k26", mode="fixed", calibration=x,
                        out_dir=Fixtures.build_dir("kria_mlp"))
        cls.out = cls.acc.out_dir
        # A vivado on PATH that is really the stub, so nothing here starts a real build.
        cls.bin = Fixtures.build_dir("kria_fake_bin")
        vivado = cls.bin / "vivado"
        vivado.write_text('#!/bin/sh\nexec "%s" "$@"\n' % (ROOT / "tests" / "fake_vivado"))
        vivado.chmod(0o755)

    def build(self, board):
        for name in ("kv260", "kr260"):
            shutil.rmtree(self.out / name, ignore_errors=True)
        path = str(self.bin) + os.pathsep + os.environ["PATH"]
        with mock.patch.dict(os.environ, {"PATH": path}):
            bitstream = self.acc.build(board=board)
        calls = (self.out / board / "calls.txt").read_text().splitlines()
        return bitstream, calls

    def test_no_vivado(self):
        with mock.patch.object(Kria, "missing_tools", return_value=["vivado"]):
            with self.assertRaises(ToolchainError) as caught:
                self.acc.build(board="kr260")
        message = str(caught.exception)
        self.assertEqual(message.count("\n"), 0, message)
        self.assertIn("wrote the Vivado project", message)
        for name in ("build.tcl", "settings.tcl", "kria.xdc", "sfp.xdc", "kria_top.v",
                     "kria_shell.v"):
            self.assertTrue((self.out / "kr260" / name).is_file(), name)
        registers = json.loads((self.out / "manifest.json").read_text())["registers"]
        self.assertEqual((registers["base"], registers["span"]), (0xA0000000, 0x10000))
        self.assertEqual(registers["map"]["features"], {"offset": 0x1000, "access": "rw",
                                                        "words": 4})
        self.assertEqual(registers["map"]["latency"]["value"], 6)
        self.assertEqual(registers["map"]["prediction"]["words"], 2)
        self.assertEqual(registers["map"]["weights"]["access"], "reserved")

    def test_kr260(self):
        bitstream, calls = self.build("kr260")
        self.assertEqual(bitstream, self.out / "kr260" / "kr260.bit")
        self.assertTrue(bitstream.is_file())
        self.assertIn("create_project -force kr260 vivado -part xck26-sfvc784-2LV-c", calls)
        self.assertIn("add_files {kria_shell.v kria_top.v}", calls)
        clock = next(c for c in calls if c.startswith("create_generated_clock -name otf_clk"))
        self.assertNotIn("-period", clock)
        self.assertFalse(any(c.startswith("create_clock -name otf_clk") for c in calls))
        self.assertIn("create_clock -name sfp_mgt_refclk -period 6.400 clk", calls)
        self.assertIn("set_property IS_ENABLED false get_files", calls)
        self.assertTrue(any(c.startswith("assign_bd_address -offset 0xA0000000 -range 64K")
                            for c in calls))
        self.assertIn("launch_runs impl_1 -to_step write_bitstream -jobs 8", calls)
        self.assertIn("report_timing_summary -max_paths 10 -file timing_summary.rpt", calls)
        sfp = (self.out / "kr260" / "sfp.xdc").read_text()
        for pin, port in (("T2", "sfp_rx_p"), ("R4", "sfp_tx_p"), ("Y6", "sfp_mgt_refclk_p")):
            self.assertIn("PACKAGE_PIN %s [get_ports %s]" % (pin, port), sfp)
        shell = (self.out / "kr260" / "kria_shell.v").read_text()
        self.assertIn("otf_mmcm", shell)
        self.assertIn("module otf_top", shell)

    def test_kv260(self):
        _, calls = self.build("kv260")
        self.assertIn("create_project -force kv260 vivado -part xck26-sfvc784-2LV-c", calls)
        self.assertFalse((self.out / "kv260" / "sfp.xdc").exists())
        self.assertFalse(any("sfp" in c for c in calls))

    def test_invalid_timing_fails(self):
        failures = [
            ({"OTF_TEST_PRIMARY_CLOCK": "1"}, "MMCM generated clock"),
            ({"OTF_TEST_BOARD_PERIOD": "3.2"}, "unexpected otf_clk period"),
            ({"OTF_TEST_BAD_METHOD": "1"}, "critical methodology violations"),
        ]
        for check in ("no_clock", "unconstrained_internal_endpoints", "generated_clocks", "loops"):
            failures.append(({"OTF_TEST_BAD_CHECK": check}, "timing check failed"))
        for environment, error in failures:
            with self.subTest(environment=environment), mock.patch.dict(os.environ, environment):
                with self.assertRaisesRegex(BoardError, error):
                    self.build("kr260")
                self.assertFalse((self.out / "kr260" / "kr260.bit").exists())

    def test_failed_rebuild_removes_bitstream(self):
        bitstream, _ = self.build("kr260")
        path = str(self.bin) + os.pathsep + os.environ["PATH"]
        with mock.patch.dict(os.environ, {"PATH": path, "OTF_TEST_PRIMARY_CLOCK": "1"}):
            with self.assertRaisesRegex(BoardError, "MMCM generated clock"):
                self.acc.build(board="kr260")
        for suffix in (".bit", ".bin", ".xsa"):
            self.assertFalse(bitstream.with_suffix(suffix).exists())

    def test_failed_timing_summary(self):
        original = CANNED.read_text()
        row = next(line for line in original.splitlines() if line.strip().startswith("0.137"))
        report = Fixtures.build_dir("kria_bad_timing") / "report.rpt"
        for index, value, error in ((0, "-0.1", "setup"), (4, "-0.1", "hold"),
                                    (8, "-0.1", "pulse_width"), (2, "1", "setup"),
                                    (3, "0", "setup")):
            fields = row.split()
            fields[index] = value
            report.write_text(original.replace(row, " ".join(fields)))
            with self.subTest(index=index), mock.patch.dict(os.environ, {"OTF_TEST_TIMING_REPORT": str(report)}):
                with self.assertRaisesRegex(BoardError, error + " timing failed"):
                    self.build("kr260")
                self.assertFalse((self.out / "kr260" / "kr260.bit").exists())
        report.write_text("missing summary\n")
        with mock.patch.dict(os.environ, {"OTF_TEST_TIMING_REPORT": str(report)}):
            with self.assertRaisesRegex(BoardError, "missing design timing summary"):
                self.build("kr260")

    def test_summary(self):
        for board in ("kv260", "kr260"):
            shutil.rmtree(self.out / board, ignore_errors=True)
        lines = []
        self.acc.summary(print_fn=lines.append)
        self.assertNotIn("measured by Vivado", "\n".join(lines))
        (self.out / "kr260").mkdir(exist_ok=True)
        shutil.copyfile(CANNED, self.out / "kr260" / "timing_summary.rpt")
        lines = []
        self.acc.summary(print_fn=lines.append)
        text = "\n".join(lines)
        self.assertIn("Latency: 6 cycles, 18.6 ns at 322 MHz", text)
        self.assertIn("Clock: 322.3 MHz estimated, 337.2 MHz measured by Vivado "
                      "(WNS 0.137 ns, kr260/timing_summary.rpt)", text)
        shutil.rmtree(self.out / "kr260")

    @unittest.skipIf(shutil.which("verilator") is None, "verilator not on PATH")
    def test_lint(self):
        """kria_top.v must match the ports of kria_shell and of the block design wrapper."""
        build = KV260(self.out).write().parent
        completed = subprocess.run(
            ["verilator", "--lint-only", "-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSEDSIGNAL",
             "-Wno-UNUSEDPARAM", "-Wno-COMBDLY", "--top-module", "kria_top", "kria_top.v",
             "kria_shell.v", str(ROOT / "tests" / "kria_stubs.v")],
            cwd=build, capture_output=True, text=True, timeout=600)
        self.assertEqual(completed.returncode, 0, completed.stderr[-3000:])

    @unittest.skipIf(shutil.which("g++") is None, "g++ not on PATH")
    def test_runtime(self):
        """kria_run against a sparse file standing in for /dev/mem, holding a board that has already answered."""
        self.assertEqual(subprocess.run(["make", "-C", str(ROOT / "runtime")],
                                        capture_output=True).returncode, 0)
        KV260(self.out).write()
        registers = json.loads((self.out / "manifest.json").read_text())["registers"]
        offsets = {name: registers["base"] + entry["offset"]
                   for name, entry in registers["map"].items()}
        expected = (self.out / "golden" / "expected.hex").read_text().split()[0]
        memory = self.out / "fake_mem"
        with open(memory, "wb") as handle:
            handle.truncate(registers["base"] + registers["span"])
            for name, value in (("id", 0x4F544631), ("status", 1)):
                handle.seek(offsets[name])
                handle.write(value.to_bytes(4, "little"))
            handle.seek(offsets["prediction"])
            handle.write(int(expected, 16).to_bytes(8, "little"))

        def run():
            return subprocess.run([str(ROOT / "runtime" / "bin" / "kria_run"), str(self.out),
                                   "--mem", str(memory)], capture_output=True, text=True)

        completed = run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("1 vectors, 0 mismatched words", completed.stdout)
        with open(memory, "r+b") as handle:
            handle.seek(offsets["id"])
            handle.write(bytes(4))
        completed = run()
        self.assertEqual(completed.returncode, 1)
        self.assertIn("the id reads 0x0", completed.stderr)
        memory.unlink()

    def test_deploy(self):
        with self.assertRaisesRegex(NotImplementedError, "automatic Kria programming"):
            self.acc.deploy(host="kria.local")
        with self.assertRaisesRegex(ConfigError, "has frozen weights"):
            self.acc.load_weights(Fixtures.factory(8).mlp((16, 32, 8)))


if __name__ == "__main__":
    unittest.main()
