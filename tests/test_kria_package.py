"""Firmware consistency and overlay fixups for the Kria package."""

import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

from tools.package_kria import package, verify_build
from onnx2fpga.board import BoardError, KR260


@unittest.skipUnless(shutil.which("dtc") and shutil.which("fdtoverlay"), "device-tree tools not installed")
class PackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.board = self.root / "kr260"
        self.board.mkdir()
        self.destination = self.root / "firmware"
        self.manifest = {"registers": {"base": 0xA0000000, "span": 0x10000}}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        self.bit = (b"\x00\x09" + bytes.fromhex("0ff00ff00ff00ff000") + b"\x00\x01"
                    + b"".join(bytes([tag]) + b"\x00\x01\x00" for tag in b"abcd")
                    + b"e\x00\x00\x00\x07payload")
        (self.board / "kr260.bit").write_bytes(self.bit)
        (self.board / "kr260.bin").write_bytes(b"payload")
        for name in ("build.tcl", "check_timing.tcl", "settings.tcl", "kria.xdc",
                     "kria_top.v", "kria_shell.v", "timing_summary.rpt",
                     "utilization.rpt", "methodology.rpt"):
            (self.board / name).write_text("fixture\n")
        params = {"C_PL_CLK0_BUF": "TRUE", "PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ": "99.999001",
                  "C_MAXIGP0_DATA_WIDTH": "32", "C_MAXIGP1_DATA_WIDTH": "128",
                  "C_MAXIGP2_DATA_WIDTH": "32"}
        params.update({"C_SAXIGP%d_DATA_WIDTH" % i: "128" for i in range(7)})
        self.hwh = ('<EDKSYSTEM><MODULE MODTYPE="zynq_ultra_ps_e"><PARAMETERS>'
                    + ''.join('<PARAMETER NAME="%s" VALUE="%s"/>' % item for item in params.items())
                    + '</PARAMETERS><MEMRANGE INSTANCE="M_AXI" BASEVALUE="0xA0000000" '
                    'HIGHVALUE="0xA000FFFF"/></MODULE></EDKSYSTEM>')
        self.xsa()

    def xsa(self, bit=None):
        with zipfile.ZipFile(self.board / "kr260.xsa", "w") as archive:
            archive.writestr("kr260.bit", self.bit if bit is None else bit)
            archive.writestr("system.hwh", self.hwh)
        with mock.patch.object(KR260, "write"), mock.patch(
                "onnx2fpga.board.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "passed", "")):
            KR260(self.root).run()

    def run_package(self):
        return package(self.root, "kr260", self.destination)

    def test_overlay_applies(self):
        self.run_package()
        record = json.loads((self.destination / "package.json").read_text())
        self.assertEqual(record["ps_clock_hz"], 99999001)
        for name, digest in record["sha256"].items():
            self.assertEqual(hashlib.sha256((self.destination / name).read_bytes()).hexdigest(), digest)
        base = self.root / "base.dts"
        base.write_text('''/dts-v1/;
/ {
    #address-cells = <2>; #size-cells = <2>;
    fpga_full: fpga-full {};
    amba: amba { #address-cells = <2>; #size-cells = <2>; };
    zynqmp_clk: clocks { #clock-cells = <1>; };
    zynqmp_reset: resets { #reset-cells = <1>; };
};
''')
        subprocess.run(["dtc", "-@", "-O", "dtb", "-o", str(self.root / "base.dtb"), str(base)],
                       check=True, capture_output=True)
        merged = self.root / "merged.dtb"
        subprocess.run(["fdtoverlay", "-i", str(self.root / "base.dtb"), "-o", str(merged),
                        str(self.destination / "onnx2fpga.dtbo")], check=True, capture_output=True)
        def prop(node, name):
            return subprocess.check_output(["fdtget", "-t", "x", str(merged), node, name], text=True).strip()
        self.assertEqual(prop("/amba/onnx2fpga_clock", "assigned-clock-rates"), "5f5dd19")
        self.assertEqual(prop("/amba/onnx2fpga@a0000000", "reg"), "0 a0000000 0 10000")
        self.assertTrue(prop("/amba/onnx2fpga_afi", "config-afi").endswith("e 800 f 0"))

    def test_mismatched_artifacts_rejected(self):
        self.xsa(bit=b"other")
        with self.assertRaisesRegex(ValueError, "XSA bitstream"):
            self.run_package()
        self.xsa()
        (self.board / "kr260.bin").write_bytes(b"load")
        with self.assertRaisesRegex(ValueError, "binary payload"):
            self.run_package()
        self.assertFalse(self.destination.exists())

    def test_bad_map_and_clock_rejected(self):
        self.hwh = self.hwh.replace("0xA000FFFF", "0xA001FFFF")
        self.xsa()
        with self.assertRaisesRegex(ValueError, "window"):
            self.run_package()
        self.hwh = self.hwh.replace("99.999001", "200")
        self.xsa()
        with self.assertRaisesRegex(ValueError, "100 MHz"):
            self.run_package()

    def test_existing_destination_preserved(self):
        self.destination.mkdir()
        marker = self.destination / "keep"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_package()
        self.assertEqual(marker.read_text(), "keep")

    def test_stale_manifest_rejected(self):
        self.manifest["name"] = "another model"
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "build artifact changed: manifest.json"):
            self.run_package()
        self.assertFalse(self.destination.exists())

    def test_failed_rebuild_invalidates_record(self):
        with mock.patch.object(KR260, "write"), mock.patch(
                "onnx2fpga.board.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, "timing failed", "")):
            with self.assertRaises(BoardError):
                KR260(self.root).run()
        self.assertTrue((self.board / "kr260.bit").exists())
        with self.assertRaisesRegex(ValueError, "missing successful build record"):
            self.run_package()

    def test_inputs_changed_during_build(self):
        def finish(*args, **kwargs):
            (self.board / "kria_shell.v").write_text("changed\n")
            return subprocess.CompletedProcess([], 0, "passed", "")
        with mock.patch.object(KR260, "write"), mock.patch(
                "onnx2fpga.board.subprocess.run", side_effect=finish):
            with self.assertRaisesRegex(BoardError, "inputs changed"):
                KR260(self.root).run()
        self.assertFalse((self.board / "build.json").exists())

    def test_packaging_snapshot_is_bound(self):
        def changed(build, board, manifest):
            self.manifest["name"] = "new build"
            (self.root / "manifest.json").write_text(json.dumps(self.manifest))
            self.xsa()
            return verify_build(build, board, manifest)
        with mock.patch("tools.package_kria.verify_build", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed while packaging"):
                self.run_package()
        self.assertFalse(self.destination.exists())
