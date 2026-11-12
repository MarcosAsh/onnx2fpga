"""Full DDR replay against the software book, features and decisions."""

import shutil
import unittest

from support import Fixtures, Hdl
from onnx2fpga.trading.ddr_replay import DdrReplay
from onnx2fpga.trading import itch
from onnx2fpga.trading.reference import Config, Pipeline, frame
from onnx2fpga.trading.replay import compare


@unittest.skipIf(Hdl.skip_reason("hardcaml") or not shutil.which("verilator"),
                 "Hardcaml and Verilator required")
class IntegratedReplay(unittest.TestCase):
    def test_hashed_collisions_and_same_slot_replace(self):
        messages = [itch.add_order(2, 0, 1, "B", 100, "BRVO", 10000),
                    itch.add_order(2, 1, 2, "S", 100, "BRVO", 10100),
                    itch.add_order(2, 2, 257, "B", 30, "BRVO", 10000),
                    itch.add_order(2, 3, 256, "B", 50, "BRVO", 10000),
                    itch.order_delete(2, 4, 256),
                    itch.add_order(2, 4, (1 << 63) | 129, "B", 50, "BRVO", 10000),
                    itch.order_delete(2, 4, (1 << 63) | 129),
                    itch.order_replace(2, 5, 257, 514, 20, 10000),
                    itch.order_delete(2, 6, 257),
                    itch.order_executed(2, 7, 514, 5, 1)]
        config = Config(order_bits=8, order_hash="xor")
        reference = Pipeline(config, lambda vector: 0)
        reference.run(frame(itch.packets(messages)))
        self.assertEqual(reference.book.collisions, 2)
        self.assertEqual(reference.book.sides[0][0], [[10000, 115]])
        messages += itch.Generator(71).messages(2000)
        replay = DdrReplay(Fixtures.build_dir("ddr_trading_hashed"), config=config,
                           source_messages=messages)
        acc, model = replay.model(device="k26", clock_mhz=322.265625)
        replay.build(acc)
        expected = Pipeline(config, model).run(replay.framed)
        latency = sum(cycles for _, cycles in replay.stages)
        for lines in replay.run(gap=65, seed=71, runs=2, stream=True):
            self.assertNotIsInstance(lines, list)
            compare(lines, expected, config, latency, timed=False)
            self.assertTrue(any(line.startswith("cycles ") for line in lines))

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
                    self.assertIsInstance(lines, list)
                    totals = compare(lines, expected, replay.config, latency, timed=False)
                    self.assertEqual(len(totals), len(expected))
        first, restarted = replay.run(gap=65, runs=2, abort_after=257)
        partial = sum(line.startswith("D ") for line in first)
        self.assertGreater(partial, 0)
        self.assertLess(partial, len(expected))
        compare(first, expected[:partial], replay.config, latency, timed=False)
        compare(restarted, expected, replay.config, latency, timed=False)
