"""The board command wires ownership cleanup and serializes the final snapshot."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_trading_runtime import Buffer, Registers, manifest
from tools import run_ddr
from onnx2fpga.trading.replay_records import records


class Transport(Registers):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.events.append("transport closed")


class OwnedBuffer(Buffer):
    def close(self):
        assert self.cpu_owned
        self.events.append("buffer closed")


class RunTest(unittest.TestCase):
    def test_local_replay(self):
        events = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            layout = manifest()
            layout["registers"]["base"] = 0xA0000000
            (root / "manifest.json").write_text(json.dumps(layout))
            (root / "records.bin").write_bytes(b"".join(records([(1, 255, True)])))
            with patch.object(run_ddr, "ProcessRegisters", return_value=Transport(events)) as helper, \
                 patch.object(run_ddr, "DmaBuffer", return_value=OwnedBuffer(events)), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(run_ddr.main([str(root / "manifest.json"),
                                                str(root / "records.bin")]), 0)
            helper.assert_called_once_with(["kria_regs", "--base", "a0000000"], timeout=30.0)
            self.assertEqual(json.loads(output.getvalue())["prediction"], -1)
            self.assertEqual(events[-2:], ["buffer closed", "transport closed"])
