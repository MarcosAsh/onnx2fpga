"""Full DDR replay against the software book, features and decisions."""

import shutil
import unittest

from support import Fixtures, Hdl
from onnx2fpga.trading.ddr_replay import DdrReplay
from onnx2fpga.trading.reference import Pipeline
from onnx2fpga.trading.replay import compare


@unittest.skipIf(Hdl.skip_reason("hardcaml") or not shutil.which("verilator"),
                 "Hardcaml and Verilator required")
class IntegratedReplay(unittest.TestCase):
    def test_stalls_and_restart(self):
        replay = DdrReplay(Fixtures.build_dir("ddr_trading"), messages=2000, seed=71)
        acc, model = replay.model(device="k26", clock_mhz=322.265625)
        replay.build(acc)
        expected = Pipeline(replay.config, model).run(replay.framed)
        latency = sum(cycles for _, cycles in replay.stages)
        self.assertEqual(latency, 22)
        for stall in (0, 65):
            with self.subTest(stall=stall):
                for lines in replay.run(gap=stall, seed=4, runs=2):
                    totals = compare(lines, expected, replay.config, latency, timed=False)
                    self.assertEqual(len(totals), len(expected))
        first, restarted = replay.run(gap=65, runs=2, abort_after=257)
        partial = sum(line.startswith("D ") for line in first)
        self.assertGreater(partial, 0)
        self.assertLess(partial, len(expected))
        compare(first, expected[:partial], replay.config, latency, timed=False)
        compare(restarted, expected, replay.config, latency, timed=False)
