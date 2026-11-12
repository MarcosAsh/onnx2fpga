"""Require simulation evidence for the exact trading core before writing a board project."""

import hashlib
import json
import pathlib
import tempfile
import unittest

from onnx2fpga.board import BoardError
from onnx2fpga.trading.board import TradingKR260

ROOT = pathlib.Path(__file__).resolve().parents[1]


class Generator:
    project = ROOT / "hardware/hardcaml"

    def verilog(self, path, top):
        return "core" if top == "ddr-trading" else "shell"


class BoardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = pathlib.Path(self.temp.name)
        (self.out / "model").mkdir()
        values = {"tick_to_trade.json": b"{}", "model/design.json": b"{}",
                  "records.bin": bytes(16), "ddr_trading.v": b"core"}
        for name, value in values.items():
            (self.out / name).write_bytes(value)
        self.report = {"status": "passed", "runs": [{"bit_exact": True}], "latency_cycles": 22,
                       "sha256": {name: hashlib.sha256(value).hexdigest() for name, value in values.items()}}
        self.save()
        self.board = TradingKR260(self.out, generator=Generator())

    def save(self):
        (self.out / "report.json").write_text(json.dumps(self.report))

    def test_project_and_register_map(self):
        self.board.write()
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["kind"], "ddr_trading")
        self.assertEqual(manifest["registers"]["map"]["id"]["value"], 0x4F544432)
        self.assertEqual(manifest["registers"]["map"]["order"]["words"], 12)
        self.assertEqual(manifest["ddr"]["mapped_bytes"], 0x80000000)
        self.assertIn("set ddr 1", (self.out / "kr260/settings.tcl").read_text())
        self.assertTrue((self.out / "kr260/ddr_top.v").exists())

    def test_stale_artifacts_rejected(self):
        (self.out / "records.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(BoardError, "artifact changed"):
            self.board.write()
        self.assertFalse((self.out / "kr260").exists())

    def test_stale_rtl_rejected(self):
        self.report["sha256"]["ddr_trading.v"] = "old"
        self.save()
        with self.assertRaisesRegex(BoardError, "RTL changed"):
            self.board.write()

    def test_failed_replay_rejected(self):
        self.report["status"] = "failed"
        self.save()
        with self.assertRaisesRegex(BoardError, "passed replay"):
            self.board.write()

    def test_failed_run_rejected(self):
        self.report["runs"][0]["bit_exact"] = False
        self.save()
        with self.assertRaisesRegex(BoardError, "passed replay"):
            self.board.write()
