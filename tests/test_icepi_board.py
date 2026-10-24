"""The Icepi Zero build short of the board: Verilator runs icepi_top.v against a serial host model."""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import threading
import tty
import unittest

import numpy as np

from support import Fixtures, Hdl, ROOT, run

from onnx2fpga.board import BoardError, IcepiZero
from onnx2fpga.compile import Compiler
from onnx2fpga.p6_emit.stream_io import StreamCodec
from onnx2fpga.reference.runner import GraphRunner

BOARD = ROOT / "hardware" / "boards" / "icepi_zero"


def load_host():
    spec = importlib.util.spec_from_file_location("icepi_host", BOARD / "host.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host = load_host()


def manifest(in_bits, in_per_beat, in_beats, out_bits, out_per_beat, out_beats):
    return {"input": {"elem_bits": in_bits, "elems_per_beat": in_per_beat,
                      "beats": in_beats},
            "output": {"elem_bits": out_bits, "elems_per_beat": out_per_beat,
                       "beats": out_beats}}


class Framing(unittest.TestCase):
    BOARD_SETTINGS = {"baud": 1_000_000, "rx_depth": 2048, "tx_depth": 2048}

    def test_byte_order(self):
        link = host.Link(manifest(8, 8, 4, 8, 1, 10), self.BOARD_SETTINGS)
        self.assertEqual(link.encode([0x0102030405060708]),
                         bytes([8, 7, 6, 5, 4, 3, 2, 1]))
        self.assertEqual(link.decode(bytes([0xcd, 0xe6, 0x07])), [0xcd, 0xe6, 0x07])

    def test_partial_bytes_round_up(self):
        link = host.Link(manifest(4, 3, 2, 3, 3, 5), self.BOARD_SETTINGS)
        self.assertEqual((link.in_bytes, link.out_bytes), (2, 2))
        self.assertEqual(link.decode(link.encode([0xabc, 0x123]) + b"\x7f"),
                         [0xabc, 0x123])

    def test_window(self):
        link = host.Link(manifest(8, 8, 4, 8, 1, 10), self.BOARD_SETTINGS)
        self.assertEqual((link.frame_in, link.frame_out, link.window), (32, 10, 64))
        wide_out = host.Link(manifest(8, 1, 4, 8, 4, 200), self.BOARD_SETTINGS)
        self.assertEqual(wide_out.window, 2048 // 800)

    def test_bad_settings(self):
        for mhz, baud in ((25, 1_000_000), (20, 3_000_000), (480, 200_000_000)):
            with self.subTest(mhz=mhz, baud=baud), self.assertRaises(BoardError):
                IcepiZero("build/unused", mhz=mhz, baud=baud)
        with self.assertRaises(BoardError):
            IcepiZero("build/unused", rev="v2.0")
        self.assertEqual(IcepiZero("build/unused").bitstream.name, "v1.4.bit")


class HostScript(unittest.TestCase):
    """host.py's run loop against a board stand in on a pseudo terminal, counting the most input bytes held."""

    @classmethod
    def setUpClass(cls):
        try:
            import serial  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pyserial is not installed")

    def build(self, rx_depth=96):
        out = Fixtures.build_dir("icepi_host")
        (out / "golden").mkdir(exist_ok=True)
        (out / "icepi_zero").mkdir(exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest(8, 8, 4, 8, 1, 10)))
        (out / "icepi_zero" / "board.json").write_text(json.dumps(
            {"baud": 1_000_000, "rx_depth": rx_depth, "tx_depth": 2048}))
        (out / "golden" / "input.hex").write_text(
            "".join("%016x\n" % (0x0101010101010101 * k) for k in range(4)))
        (out / "golden" / "expected.hex").write_text(
            "".join("%02x\n" % (0x10 + k) for k in range(10)))
        return out

    def serve(self, out, corrupt_frame=None, frames=40):
        master, slave = os.openpty()
        tty.setraw(master)
        link = host.Link.load(out)
        answer = bytes(0x10 + k for k in range(10))
        held = {"most": 0, "frames": 0}
        stop = threading.Event()

        def board():
            pending = b""
            while not stop.is_set():
                try:
                    pending += os.read(master, 4096)
                except OSError:
                    return
                held["most"] = max(held["most"], len(pending))
                while len(pending) >= link.frame_in:
                    pending = pending[link.frame_in:]
                    reply = bytearray(answer)
                    if held["frames"] == corrupt_frame:
                        reply[3] ^= 1
                    held["frames"] += 1
                    os.write(master, bytes(reply))

        thread = threading.Thread(target=board, daemon=True)
        thread.start()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                status = host.main([str(out), "--port", os.ttyname(slave),
                                    "--frames", str(frames), "--timeout", "5"])
        finally:
            stop.set()
            os.close(slave)
            os.close(master)
            thread.join(timeout=5)
        return status, held

    def test_run(self):
        status, held = self.serve(self.build())
        self.assertEqual(status, 0)
        self.assertEqual(held["frames"], 40)
        self.assertLessEqual(held["most"], 96)

    def test_wrong_frame(self):
        status, _ = self.serve(self.build(), corrupt_frame=3)
        self.assertEqual(status, 1)


class BoardSimulation(unittest.TestCase):
    """The generated board top, bit exact against the integer model."""

    FRAMES = 3

    @classmethod
    def setUpClass(cls):
        reason = Hdl.skip_reason("hardcaml")
        if reason:
            raise unittest.SkipTest(reason)
        for tool in ("verilator", "make"):
            if shutil.which(tool) is None:
                raise unittest.SkipTest("%s is not on PATH" % tool)

    def compile(self, name, model, shape, target):
        out = Fixtures.build_dir("icepi_" + name)
        result = Compiler(device="icepi25", target_cycles=target, hdl="hardcaml").compile(
            model, Fixtures.samples(shape), out)
        return result, out

    def frames(self, result, out):
        """The golden frame and a few random ones through the integer model, as host.py reads them."""
        graph = result.hardware_graph
        source, sink = graph.tensor(graph.inputs[0]), graph.tensor(graph.outputs[0])
        rng = np.random.default_rng(11)
        inputs = [rng.integers(source.dtype.min, source.dtype.max + 1, source.shape)
                  for _ in range(self.FRAMES - 1)]
        sent = host.read_hex(out / "golden" / "input.hex")
        expected = host.read_hex(out / "golden" / "expected.hex")
        for index, array in enumerate(inputs):
            produced = GraphRunner(graph).run({source.name: array})[sink.name]
            StreamCodec(source).to_image(array).write(out / ("frame%d_in.hex" % index))
            StreamCodec(sink).to_image(produced).write(out / ("frame%d_out.hex" % index))
            sent += host.read_hex(out / ("frame%d_in.hex" % index))
            expected += host.read_hex(out / ("frame%d_out.hex" % index))
        return sent, expected

    def simulate(self, out, link, sent, flags):
        send = out / "send.hex"
        send.write_text("".join("%02x\n" % byte for byte in link.encode(sent)))
        completed = run(["make", "--no-print-directory", "-f", str(BOARD / "Makefile"),
                         "sim",
                         "DESIGN=%s" % (out / "design.json"),
                         "BUILD=%s" % (out / "icepi_zero"), "SEND=%s" % send,
                         "SIM_FLAGS=--expect %d %s"
                         % (link.frame_out * self.FRAMES, flags)])
        self.assertEqual(completed.returncode, 0,
                         completed.stdout[-3000:] + completed.stderr[-3000:])
        summary = next((line for line in completed.stdout.splitlines()
                        if line.startswith("sent ")), completed.stdout[-2000:])
        received = bytes(int(line, 16) for line in
                         (out / "icepi_zero" / "received.hex").read_text().split())
        return summary, received

    def check(self, name, model, shape, target, flags):
        result, out = self.compile(name, model, shape, target)
        sent, expected = self.frames(result, out)
        link = host.Link(json.loads((out / "manifest.json").read_text()),
                         Framing.BOARD_SETTINGS)
        self.assertGreaterEqual(link.window, self.FRAMES)
        summary, received = self.simulate(out, link, sent, flags)
        self.assertIn("error_led 0 framing_errors 0", summary)
        self.assertEqual(len(received), link.frame_out * self.FRAMES, summary)
        self.assertEqual(link.decode(received), expected)
        settings = json.loads((out / "icepi_zero" / "board.json").read_text())
        self.assertEqual((settings["mhz"], settings["baud"], settings["clocks_per_bit"]),
                         (20, 1_000_000, 20))

    def test_makefile_settings(self):
        out = Fixtures.build_dir("icepi_settings")
        board = IcepiZero(out, rev="v1.2", mhz=24, uart="header")
        text = board.write().read_text()
        for line in ("REV := v1.2", "MHZ := 24", "UART := header",
                     "include %s" % (BOARD / "Makefile").resolve()):
            self.assertIn(line, text)
        self.assertEqual(board.bitstream.name, "v1.2-header.bit")

    def test_mlp(self):
        self.check("mlp", Fixtures.factory().mlp(), Fixtures.MLP_SHAPE, 256,
                   "--skew 2 --gap 3 --seed 1")

    def test_cnn(self):
        self.check("cnn", Fixtures.factory().cnn(), Fixtures.CNN_SHAPE, 400,
                   "--skew -2 --gap 2 --seed 2")


if __name__ == "__main__":
    unittest.main()
