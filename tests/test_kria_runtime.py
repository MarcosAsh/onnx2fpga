"""Kria register transactions, timeout recovery and simulated hardware predictions."""

import unittest
from unittest.mock import Mock, patch

from onnx2fpga.board import BoardError, OFFSETS, registers
from onnx2fpga.kria_runtime import RegisterSession


def manifest():
    return {"input": {"elem_bits": 8, "elems_per_beat": 9, "beats": 1},
            "output": {"elem_bits": 8, "elems_per_beat": 5, "beats": 1},
            "registers": registers(72, 40, 6, 0xA0000000, 0x10000)}


class Transport:
    def __init__(self):
        self.words = {entry["offset"]: entry["value"]
                      for entry in manifest()["registers"]["map"].values() if "value" in entry}
        self.calls = []
        self.respond = True

    def read(self, offset):
        return self.words.get(offset, 0)

    def write(self, offset, value):
        self.calls.append((offset, value))
        self.words[offset] = value
        if offset == OFFSETS["control"]:
            self.words[OFFSETS["status"]] = int(value == 1 and self.respond)
            if value == 1:
                self.words[OFFSETS["prediction"]] = 0x12345678
                self.words[OFFSETS["prediction"] + 4] = 0x9A
                self.words[OFFSETS["cycles"]] = 6


class SessionTest(unittest.TestCase):
    def test_wide_vector_word_order(self):
        bus = Transport()
        with RegisterSession(bus, manifest()) as session:
            self.assertEqual(session.run([[0x123456789ABCDEF012]]), [[0x9A12345678]])
        writes = [value for offset, value in bus.calls if 0x1000 <= offset < 0x2000]
        self.assertEqual(writes, [0xBCDEF012, 0x3456789A, 0x12])

    def test_timeout_requires_reset(self):
        bus = Transport()
        with RegisterSession(bus, manifest(), timeout=.001) as session:
            bus.respond = False
            with patch("onnx2fpga.kria_runtime.time.monotonic", side_effect=[0, 1]):
                with self.assertRaisesRegex(BoardError, "timed out"):
                    session.run([[1]])
            bus.respond = True
            with self.assertRaisesRegex(BoardError, "needs open or reset"):
                session.run([[2]])
            session.reset()
            self.assertEqual(session.run([[2]]), [[0x9A12345678]])

    def test_wrong_bitstream_rejected(self):
        for field in ("id", "latency", "feature_bits", "prediction_bits", "capabilities"):
            bus = Transport()
            bus.words[OFFSETS[field]] = 0
            with self.subTest(field=field), self.assertRaises(BoardError):
                RegisterSession(bus, manifest()).open()
            self.assertEqual(bus.calls, [])

    def test_transport_failure_requires_reset(self):
        bus = Transport()
        with RegisterSession(bus, manifest()) as session:
            with patch.object(bus, "write", side_effect=OSError("disconnected")):
                with self.assertRaises(OSError):
                    session.run([[1]])
            with self.assertRaisesRegex(BoardError, "needs open or reset"):
                session.run([[2]])
            session.reset()
            self.assertEqual(session.run([[2]]), [[0x9A12345678]])

    def test_wrong_latency_requires_reset(self):
        bus = Transport()
        with RegisterSession(bus, manifest()) as session:
            read = bus.read
            with patch.object(bus, "read", side_effect=lambda offset:
                              7 if offset == OFFSETS["cycles"] else read(offset)):
                with self.assertRaisesRegex(BoardError, "took 7 cycles"):
                    session.run([[1]])
            self.assertTrue(session.needs_reset)
            session.reset()
            self.assertEqual(session.run([[2]]), [[0x9A12345678]])

    def test_invalid_frame_does_not_start(self):
        bus = Transport()
        with RegisterSession(bus, manifest()) as session:
            for frame in ([], [1, 2], [-1], [1 << 72]):
                with self.assertRaises(BoardError):
                    session.run([frame])
            self.assertEqual(bus.calls, [(OFFSETS["control"], 2)])


class RtlSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shutil
        from support import Hdl
        reason = Hdl.skip_reason("hardcaml")
        if reason or not shutil.which("verilator"):
            raise unittest.SkipTest(reason or "verilator not installed")

    def test_predict_reset_and_reload(self):
        import numpy as np
        import onnx2fpga as otf
        from support import Fixtures
        from onnx2fpga.board import KR260
        from onnx2fpga.simulate import SimulatedRegisters

        samples = np.random.default_rng(8).normal(0, 1, (16, 4)).astype(np.float32)
        acc = otf.load(Fixtures.factory().mlp((4, 8, 1)))
        acc.compile(device="k26", mode="fixed", weights="reloadable",
                    calibration=samples, out_dir=Fixtures.build_dir("kria_session"))
        acc.board = KR260(acc.out_dir)
        acc.board.write()
        with SimulatedRegisters(acc.out_dir) as bus:
            try:
                acc.deploy(registers=bus, program=False)
                np.testing.assert_array_equal(acc.predict(samples, quantized=True),
                                              acc.predict(samples, quantized=True, on="reference"))
                acc.load_weights(acc.model)
                state = tuple(bus.read(OFFSETS[name]) for name in
                              ("weights_bank", "weights_version", "weights_crc"))
                self.assertEqual(state[1], 1)
                bus.write(OFFSETS["control"], 1)
                acc.reset()
                self.assertEqual(bus.read(OFFSETS["status"]), 0)
                self.assertEqual(tuple(bus.read(OFFSETS[name]) for name in
                                       ("weights_bank", "weights_version", "weights_crc")), state)
                np.testing.assert_array_equal(acc.predict(samples, quantized=True),
                                              acc.predict(samples, quantized=True, on="reference"))
            finally:
                acc.disconnect()
            self.assertEqual(bus.read(OFFSETS["id"]), 0x4F544631)
            remote = Mock(read=bus.read, write=bus.write)
            with patch("onnx2fpga.api.connect_registers", return_value=remote) as connect:
                try:
                    acc.deploy(host="kria.local", program=False)
                    connect.assert_called_once()
                    np.testing.assert_array_equal(acc.predict(samples, quantized=True),
                                                  acc.predict(samples, quantized=True, on="reference"))
                finally:
                    acc.disconnect()
                remote.close.assert_called_once()
            remote = Mock(read=bus.read, write=bus.write)
            with patch("onnx2fpga.api.program_kria") as program, \
                    patch("onnx2fpga.api.connect_registers", return_value=remote):
                try:
                    acc.deploy(host="kria.local", firmware="verified-package")
                    program.assert_called_once_with("kria.local", "verified-package", "kr260",
                                                    acc.out_dir / "manifest.json")
                    np.testing.assert_array_equal(acc.predict(samples, quantized=True),
                                                  acc.predict(samples, quantized=True, on="reference"))
                finally:
                    acc.disconnect()
            with patch("onnx2fpga.api.program_kria", side_effect=BoardError("load failed")), \
                    patch("onnx2fpga.api.connect_registers") as connect:
                with self.assertRaisesRegex(BoardError, "load failed"):
                    acc.deploy(host="kria.local", firmware="verified-package")
                connect.assert_not_called()
                self.assertFalse(acc.deployed)
            wrong = Mock()
            wrong.read.return_value = 0
            with patch("onnx2fpga.api.connect_registers", return_value=wrong):
                with self.assertRaisesRegex(BoardError, "id reads"):
                    acc.deploy(host="kria.local", program=False)
                wrong.close.assert_called_once()
                self.assertFalse(acc.deployed)
