"""Exercise DMA mapping, exclusive ownership and sysfs handoff without a board."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from onnx2fpga.board import BoardError
from onnx2fpga.trading.dma import DmaBuffer


class DmaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.devices = self.root / "dev"
        self.attrs = self.root / "sys" / "udmabuf0"
        self.devices.mkdir()
        self.attrs.mkdir(parents=True)
        (self.devices / "udmabuf0").write_bytes(bytes(4096))
        for name, value in {"dev": "0:0", "phys_addr": "0x100000", "size": "4096",
                            "dma_coherent": "0"}.items():
            (self.attrs / name).write_text(value)

    def open(self):
        with patch("onnx2fpga.trading.dma.stat.S_ISCHR", return_value=True):
            return DmaBuffer(sysfs=self.root / "sys", devices=self.devices)

    def test_mapping_and_ownership(self):
        buffer = self.open()
        with self.assertRaises(BoardError):
            buffer.write(b"record")
        with self.assertRaises(BoardError):
            buffer.close()
        buffer.acquire()
        buffer.write(b"record")
        self.assertEqual((self.devices / "udmabuf0").read_bytes()[:6], b"record")
        buffer.submit()
        self.assertEqual((self.attrs / "sync_for_device").read_text(), "0x1005\n")
        with self.assertRaises(BoardError):
            buffer.write(b"overwrite")
        buffer.acquire()
        self.assertEqual((self.attrs / "sync_for_cpu").read_text(), "0x1005\n")
        buffer.close()
        buffer.close()
        with self.assertRaises(BoardError):
            buffer.acquire()

    def test_exclusive_lock(self):
        first = self.open()
        try:
            with self.assertRaises(BlockingIOError):
                self.open()
        finally:
            first.acquire()
            first.close()
        second = self.open()
        second.acquire()
        second.close()

    def test_failed_sync_keeps_buffer_unavailable(self):
        buffer = self.open()
        buffer.acquire()
        (self.attrs / "sync_for_device").mkdir()
        with self.assertRaises(OSError):
            buffer.submit()
        self.assertFalse(buffer.cpu_owned)
        with self.assertRaises(BoardError):
            buffer.close()
        buffer.acquire()
        buffer.close()

    def test_invalid_allocation_releases_lock(self):
        (self.attrs / "size").write_text("4095")
        with self.assertRaises(BoardError):
            self.open()
        (self.attrs / "size").write_text("4096")
        buffer = self.open()
        buffer.acquire()
        with self.assertRaises(BoardError):
            buffer.write(bytes(4097))
        buffer.close()

    def test_device_mismatch(self):
        (self.attrs / "dev").write_text("1:2")
        with self.assertRaisesRegex(BoardError, "match sysfs"):
            self.open()
        with self.assertRaises(ValueError):
            DmaBuffer("../udmabuf0")

    def test_coherent_allocation_rejected(self):
        (self.attrs / "dma_coherent").write_text("1")
        with self.assertRaisesRegex(BoardError, "noncoherent"):
            self.open()
