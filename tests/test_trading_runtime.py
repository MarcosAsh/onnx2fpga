"""DDR buffer ownership survives timeouts, transport failures and abort drains."""

import unittest

from onnx2fpga.board import BoardError
from onnx2fpga.trading.board import OFFSETS
from onnx2fpga.trading.replay_records import records
from onnx2fpga.trading.runtime import ReplaySession


def manifest(latency=22):
    return {"kind": "ddr_trading", "registers": {"map": {
        name: {"offset": offset, "value": latency if name == "latency" else 0}
        for name, offset in OFFSETS.items()}},
        "ddr": {"record_bytes": 16, "base_alignment": 256, "address_bits": 40,
                "mapped_base": 0, "mapped_bytes": 0x80000000}}


class Buffer:
    address, size = 0x100000, 4096

    def __init__(self, events):
        self.events, self.cpu_owned = events, False

    def acquire(self):
        self.events.append("cpu")
        self.cpu_owned = True

    def write(self, payload):
        assert self.cpu_owned
        self.events.append("copy")

    def submit(self):
        assert self.cpu_owned
        self.events.append("device")
        self.cpu_owned = False


class Registers:
    def __init__(self, events):
        self.events = events
        self.values = {OFFSETS[name]: value for name, value in
                       (("id", 0x4F544432), ("version", 1), ("latency", 22),
                        ("status", 0), ("result", 0x5FF))}
        self.stall = False
        self.abort_stalls = False
        self.fail_start = False
        self.error = False

    def read(self, offset):
        value = self.values.get(offset, 0)
        if offset == OFFSETS["status"]:
            self.events.append("busy" if value & 1 else "idle")
        return value

    def write(self, offset, value):
        self.values[offset] = value
        if offset == OFFSETS["control"]:
            self.events.append("start" if value == 1 else "abort")
            if value == 1:
                self.values[OFFSETS["status"]] = 1 if self.stall else (6 if self.error else 2)
                self.values[OFFSETS["delivered"]] = self.values[OFFSETS["records"]]
                if self.fail_start:
                    raise OSError("lost response after start")
            elif not self.abort_stalls:
                self.values[OFFSETS["status"]] = 10


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.buffer = Buffer(self.events)
        self.regs = Registers(self.events)
        self.session = ReplaySession(self.regs, self.buffer, manifest(), timeout=0.002)
        self.payload = b"".join(records([(1, 255, False), (2, 1, True)]))

    def test_handoff_and_restart(self):
        self.session.open()
        self.assertEqual(self.events[-3:], ["abort", "idle", "cpu"])
        for _ in range(2):
            self.events.clear()
            result = self.session.run(self.payload)
            self.assertEqual(self.events, ["idle", "copy", "device", "start", "idle", "cpu"])
            self.assertEqual(result["delivered"], 2)
            self.assertEqual(result["prediction"], -1)
            self.assertEqual(result["action"], 1)
            self.assertEqual(len(result["order"]), 47)
        self.session.close()
        self.assertTrue(self.buffer.cpu_owned)

    def test_timeout_retains_device_ownership(self):
        self.session.open()
        self.regs.stall = True
        with self.assertRaisesRegex(BoardError, "timed out"):
            self.session.run(self.payload)
        self.assertFalse(self.buffer.cpu_owned)
        with self.assertRaisesRegex(BoardError, "needs open or abort"):
            self.session.run(self.payload)
        self.regs.abort_stalls = True
        with self.assertRaisesRegex(BoardError, "timed out"):
            self.session.close()
        self.assertFalse(self.buffer.cpu_owned)
        self.regs.abort_stalls = False
        self.session.close()
        self.assertTrue(self.buffer.cpu_owned)

    def test_ambiguous_start_requires_abort(self):
        self.session.open()
        self.regs.fail_start = True
        with self.assertRaises(OSError):
            self.session.run(self.payload)
        self.assertFalse(self.buffer.cpu_owned)
        self.assertTrue(self.session.needs_abort)
        self.session.abort()
        self.assertTrue(self.buffer.cpu_owned)

    def test_failed_hardware_requires_abort(self):
        self.session.open()
        self.regs.error = True
        with self.assertRaisesRegex(BoardError, "status"):
            self.session.run(self.payload)
        self.assertTrue(self.session.needs_abort)
        self.assertTrue(self.buffer.cpu_owned)

    def test_identity_before_mutation(self):
        self.regs.values[OFFSETS["id"]] = 0
        with self.assertRaisesRegex(BoardError, "id"):
            self.session.open()
        self.session.close()
        self.assertEqual(self.events, [])

    def test_invalid_records_before_handoff(self):
        self.session.open()
        for payload in (b"", b"x", bytes(16), self.payload[:16]):
            self.events.clear()
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.session.run(payload)
            self.assertEqual(self.events, [])

    def test_range_and_manifest(self):
        for address in (-256, 1, 0x80000000 - 256):
            self.buffer.address = address
            with self.subTest(address=address), self.assertRaises(BoardError):
                ReplaySession(self.regs, self.buffer, manifest())
        self.buffer.address = 0x100000
        bad = manifest()
        bad["ddr"]["mapped_bytes"] = 1 << 40
        with self.assertRaises(BoardError):
            ReplaySession(self.regs, self.buffer, bad)
