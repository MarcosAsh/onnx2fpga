"""Native register bounds and process timeout recovery, without physical memory."""

import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from onnx2fpga.board import BoardError
from onnx2fpga.register_transport import ProcessRegisters, connect

ROOT = pathlib.Path(__file__).resolve().parents[1]


class DeadlineTest(unittest.TestCase):
    def test_ssh_command(self):
        manifest = {"registers": {"base": 0xA0000000, "span": 0x10000}}
        with patch("onnx2fpga.register_transport.ProcessRegisters") as helper:
            connect("user@kria.local", manifest, timeout=2.5)
            helper.assert_called_once_with(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "--",
                 "user@kria.local", "kria_regs --base a0000000"], 2.5)
            for host in ("-oProxyCommand=x", "host; echo x", "$(hostname)", "host\n"):
                with self.assertRaises(ValueError):
                    connect(host, manifest)
            self.assertEqual(helper.call_count, 1)

    def test_startup_timeout(self):
        started = time.monotonic()
        with self.assertRaisesRegex(BoardError, "timed out"):
            ProcessRegisters([sys.executable, "-c", "import time; time.sleep(20)"], timeout=.1)
        self.assertLess(time.monotonic() - started, 3)

    def test_failed_exchange_closes_process(self):
        for code, reason in (
            ("import time; time.sleep(20)", "timed out"),
            ("print('bad', flush=True)", "invalid"),
            ("print('x' * 300, flush=True)", "too long"),
            ("pass", "disconnected"),
        ):
            script = "print('OTFREG1', flush=True); input(); " + code
            with self.subTest(reason=reason):
                with ProcessRegisters([sys.executable, "-c", script], timeout=.2) as regs:
                    process = regs.process
                    with self.assertRaisesRegex(BoardError, reason):
                        regs.read(0)
                    self.assertIsNotNone(process.poll())
                    with self.assertRaisesRegex(BoardError, "closed"):
                        regs.read(0)


@unittest.skipUnless(shutil.which("g++"), "g++ not installed")
class NativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make", "-C", str(ROOT / "runtime"), "bin/kria_regs"],
                       check=True, capture_output=True)
        cls.helper = str(ROOT / "runtime/bin/kria_regs")

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.memory = pathlib.Path(self.folder.name) / "registers"
        self.memory.write_bytes(bytes(0x10000))
        self.command = [self.helper, "--mem", str(self.memory), "--base", "0"]

    def test_word_access_and_bounds(self):
        with ProcessRegisters(self.command) as regs:
            regs.write(0x1000, 0x89ABCDEF)
            regs.write(0xFFFC, 0x12345678)
            self.assertEqual(regs.read(0x1000), 0x89ABCDEF)
            self.assertEqual(regs.read(0xFFFC), 0x12345678)
            for offset in (-4, 1, 0x10000):
                with self.assertRaises(ValueError):
                    regs.read(offset)
            for value in (-1, 1 << 32):
                with self.assertRaises(ValueError):
                    regs.write(0, value)
        self.assertEqual(self.memory.read_bytes()[0x1000:0x1004], b"\xef\xcd\xab\x89")

    def test_native_rejects_invalid_commands(self):
        for command in ("r 10000", "r 1", "w 0 100000000", "w 0 -1",
                        "w 0 1 extra", "r 0 extra", "r", "x 0"):
            with self.subTest(command=command):
                result = subprocess.run(self.command, input=command + "\n", text=True,
                                        capture_output=True, timeout=2)
                self.assertEqual(result.returncode, 1)
                self.assertTrue(result.stderr)
                self.assertEqual(self.memory.read_bytes(), bytes(0x10000))

    def test_short_file_and_bad_base(self):
        for arguments in (["--base", "1"], ["--base", "ffffffffffffffff"]):
            result = subprocess.run(self.command + arguments, capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 1)
        self.memory.write_bytes(bytes(4))
        with self.assertRaisesRegex(BoardError, "disconnected"):
            ProcessRegisters(self.command)
