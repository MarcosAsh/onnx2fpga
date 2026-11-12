"""Arty board generation and rejection of failed implementation checks."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from support import Fixtures, Hdl, ROOT
import onnx2fpga as otf
from onnx2fpga.board import ArtyA735, ArtyA7100, ArtyA7, BoardError
from onnx2fpga.cli import parser, run_bitstream
from onnx2fpga.errors import ConfigError
from onnx2fpga.p6_emit.stream_io import StreamCodec
from onnx2fpga.reference.runner import GraphRunner
from onnx2fpga.serial_link import Link


@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
class ArtyFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acc = otf.load(Fixtures.factory().mlp((4, 8, 1)))
        cls.acc.compile(device="a7_35t", mode="fixed",
                        calibration=np.zeros((8, 4), dtype=np.float32),
                        out_dir=Fixtures.build_dir("arty_mlp"))

    def fake_tool(self, name):
        return str(ROOT / "tests/fake_vivado") if name == "vivado" else None

    def test_build_and_link(self):
        with patch.object(ArtyA7, "tool", self.fake_tool):
            bitstream = self.acc.build(board="arty_a7_35t")
        self.assertTrue(bitstream.exists())
        board = self.acc.board
        calls = (board.build / "calls.txt").read_text().splitlines()
        route = next(i for i, c in enumerate(calls) if c.startswith("route_design"))
        write = next(i for i, c in enumerate(calls) if c.startswith("write_bitstream"))
        self.assertLess(route, write)
        link = Link.load(self.acc.out_dir, board.NAME)
        self.assertEqual((link.baud, link.in_beats, link.out_beats), (1_000_000, 1, 1))
        self.assertGreaterEqual(link.window, 1)
        self.assertIn("--board arty_a7_35t", board.host_command())

    def test_failed_checks_remove_stale_bitstream(self):
        board = ArtyA735(self.acc.out_dir)
        for key, value in (("OTF_TEST_BAD_CHECK", "loops"),
                           ("OTF_TEST_BAD_CHECK", "multiple_clock"),
                           ("OTF_TEST_BAD_METHOD", "1"),
                           ("OTF_TEST_BOARD_PERIOD", "20"),
                           ("OTF_TEST_BAD_SYNC", "1")):
            with self.subTest(check=key, value=value):
                board.build.mkdir(parents=True, exist_ok=True)
                board.bitstream.write_text("stale")
                with patch.object(ArtyA7, "tool", self.fake_tool), \
                        patch.dict(os.environ, {key: value}), self.assertRaises(BoardError):
                    board.run()
                self.assertFalse(board.bitstream.exists())

    def test_negative_hold_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.rpt"
            source = (ROOT / "tests/artix_timing_summary.rpt").read_text()
            path.write_text(source.replace("0.163", "-0.163"))
            board = ArtyA735(self.acc.out_dir)
            with patch.object(ArtyA7, "tool", self.fake_tool), \
                    patch.dict(os.environ, {"OTF_TEST_TIMING_REPORT": str(path)}), \
                    self.assertRaisesRegex(BoardError, "hold timing failed"):
                board.run()
            self.assertFalse(board.bitstream.exists())

    def test_device_mismatch(self):
        with self.assertRaises(ConfigError):
            self.acc.build(board=ArtyA7100(self.acc.out_dir))

    def test_deployed_prediction(self):
        try:
            import serial  # noqa: F401
        except ImportError:
            self.skipTest("pyserial not installed")
        from test_api import Board

        with patch.object(ArtyA7, "tool", self.fake_tool):
            self.acc.build(board="arty_a7_35t")
        x = np.random.default_rng(7).normal(0, 1, (8, 4)).astype(np.float32)
        with Board(self.acc) as peer:
            try:
                self.acc.deploy(port=peer.port, program=False)
                np.testing.assert_array_equal(
                    self.acc.predict(x, quantized=True),
                    self.acc.predict(x, on="reference", quantized=True))
                self.assertEqual(peer.frames, len(x))
            finally:
                self.acc.disconnect()


class ArtyCli(unittest.TestCase):
    def test_board_options(self):
        for board in ("arty_a7_35t", "arty_a7_100t"):
            args = parser().parse_args(["bitstream", "model.onnx", "--board", board])
            self.assertIsNone(args.device)
            args.mhz = 50
            with self.assertRaises(ConfigError):
                run_bitstream(args)


@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class ArtySerial(unittest.TestCase):
    def test_fixed_models_over_uart(self):
        for name, board_class in (("mlp_regressor", ArtyA735), ("xgb_small", ArtyA7100)):
            with self.subTest(model=name):
                corpus = ROOT / "tests/fixtures/corpus"
                with np.load(corpus / (name + ".npz")) as data:
                    calibration = data["calibration"]
                out = Fixtures.build_dir("arty_serial_" + name)
                acc = otf.load(corpus / (name + ".onnx"))
                acc.compile(device=board_class.DEVICE, mode="fixed",
                            calibration=calibration, out_dir=out)
                board = board_class(out)
                board.write()
                wrapper = board.build / "wrapper.v"
                wrapper.write_text("module icepi_top(input clk, button_n, rxd, "
                                   "output txd, output [1:0] led); "
                                   "arty_top dut(.clk(clk), .button(!button_n), "
                                   ".rxd(rxd), .txd(txd), .led(led)); endmodule\n")
                compiled = subprocess.run(
                    ["verilator", "--cc", "--exe", "--build", "--build-jobs", "0",
                     "-Wno-fatal", "--top-module", "icepi_top", "--Mdir", "obj",
                     str(wrapper.resolve()), str((board.build / "arty_top.v").resolve()),
                     str(ROOT / "hardware/boards/icepi_zero/sim/uart_bench.cpp")],
                    cwd=board.build, capture_output=True, text=True, timeout=300)
                self.assertEqual(compiled.returncode, 0, compiled.stderr[-3000:])
                graph = acc.result.hardware_graph
                source = graph.tensor(graph.inputs[0])
                sink = graph.tensor(graph.outputs[0])
                rng = np.random.default_rng(48)
                sent, expected = [], []
                for _ in range(8):
                    x = rng.integers(source.dtype.min, source.dtype.max + 1, source.shape)
                    y = GraphRunner(graph).run({source.name: x})[sink.name]
                    sent.extend(StreamCodec(source).to_image(x).words)
                    expected.extend(StreamCodec(sink).to_image(y).words)
                link = Link.load(out, board.NAME)
                self.assertGreaterEqual(link.window, 8)
                (board.build / "send.hex").write_text(
                    "".join("%02x\n" % byte for byte in link.encode(sent)))
                simulated = subprocess.run(
                    ["./obj/Vicepi_top", "--clocks-per-bit", "100", "--send", "send.hex",
                     "--receive", "received.hex", "--expect", str(link.frame_out * 8),
                     "--skew", "2", "--gap", "2", "--press"], cwd=board.build,
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(simulated.returncode, 0, simulated.stderr)
                self.assertIn("error_led 0 framing_errors 0", simulated.stdout)
                received = bytes.fromhex((board.build / "received.hex").read_text())
                self.assertEqual(link.decode(received), expected)
