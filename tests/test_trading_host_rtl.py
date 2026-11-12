"""The host replay session drives generated AXI registers and a stalled DDR model."""

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from support import Fixtures, Hdl, ROOT
from test_trading_runtime import Buffer, manifest
from onnx2fpga.board import BoardError
from onnx2fpga.p6_emit.hardcaml import HardcamlGenerator
from onnx2fpga.register_transport import ProcessRegisters
from onnx2fpga.trading.board import OFFSETS
from onnx2fpga.trading.reference import Config, Pipeline
from onnx2fpga.trading.replay import Replay, WARNINGS
from onnx2fpga.trading.replay_records import records
from onnx2fpga.trading.runtime import ReplaySession


class SharedBuffer(Buffer):
    size = 1 << 20

    def __init__(self, path):
        super().__init__([])
        self.path = path
        self.path.write_bytes(bytes(self.size))

    def write(self, payload):
        super().write(payload)
        with self.path.open("r+b") as target:
            target.write(payload)


class HostRtlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = Hdl.skip_reason("hardcaml")
        if reason or not shutil.which("verilator"):
            raise unittest.SkipTest(reason or "verilator not installed")
        cls.out = Fixtures.build_dir("ddr_host_session")
        cls.replay = Replay(cls.out, messages=2000, seed=71, config=Config(order_hash="xor"))
        acc, model = cls.replay.model(device="k26", clock_mhz=322.265625)
        config = cls.out / "tick_to_trade.json"
        design = json.loads((Path(acc.out_dir) / "design.json").read_text())
        config.write_text(json.dumps(cls.replay.config.as_json(design)))
        rtl = subprocess.check_output([str(HardcamlGenerator().executable), "ddr-regs",
                                       str(config)], text=True)
        (cls.out / "ddr_regs.v").write_text(rtl)
        command = ["verilator", "--cc", "--exe", "--build", "--build-jobs", "0", *WARNINGS,
                   "-CFLAGS", "-O2 -std=c++17", "--top-module", "ddr_regs_sim", "ddr_regs.v",
                   str(ROOT / "runtime/sim/ddr_regs_harness.cpp"), "-o", "Vddr_regs_sim"]
        built = subprocess.run(command, cwd=cls.out, capture_output=True, text=True)
        if built.returncode:
            raise AssertionError(built.stdout[-2000:] + built.stderr[-4000:])
        cls.expected = Pipeline(cls.replay.config, model).run(cls.replay.framed)
        cls.payload = b"".join(records(cls.replay.beats))

    def connect(self, error=False):
        buffer = SharedBuffer(self.out / "ddr.bin")
        bus = ProcessRegisters([str(self.out / "obj_dir/Vddr_regs_sim"), str(buffer.path),
                                "65", str(int(error))], timeout=5)
        self.addCleanup(bus.close)
        return buffer, bus

    def test_decision_snapshot_and_restart(self):
        buffer, bus = self.connect()
        with ReplaySession(bus, buffer, manifest(latency=23), timeout=5) as session:
            for _ in range(2):
                result = session.run(self.payload)
                want = self.expected[-1]
                self.assertEqual(result["delivered"], len(self.payload) // 16)
                self.assertEqual(result["decisions"], len(self.expected))
                self.assertEqual(result["orders"], sum(bool(d.order) for d in self.expected))
                self.assertEqual((result["prediction"], result["action"], result["order"]),
                                 (want.prediction, want.action, want.order))
                self.assertEqual((result["t_out"] - result["t_last"]) & 0xFFFFFFFF, 23)
                self.assertTrue(buffer.cpu_owned)

    def test_timeout_abort_and_restart(self):
        buffer, bus = self.connect()
        with ReplaySession(bus, buffer, manifest(latency=23), timeout=5) as session:
            session.timeout = 0.001
            with self.assertRaisesRegex(BoardError, "timed out"):
                session.run(self.payload)
            self.assertFalse(buffer.cpu_owned)
            session.timeout = 5
            session.abort()
            self.assertTrue(buffer.cpu_owned)
            self.assertFalse(bus.read(OFFSETS["status"]) & 1)
            self.assertLess(bus.read(OFFSETS["delivered"]), len(self.payload) // 16)
            self.assertEqual(session.run(self.payload)["decisions"], len(self.expected))

    def test_ddr_error_is_drained(self):
        buffer, bus = self.connect(error=True)
        with ReplaySession(bus, buffer, manifest(latency=23), timeout=5) as session:
            with self.assertRaisesRegex(BoardError, "status"):
                session.run(self.payload)
            self.assertTrue(buffer.cpu_owned)
            self.assertTrue(session.needs_abort)
            self.assertFalse(bus.read(OFFSETS["status"]) & 1)
