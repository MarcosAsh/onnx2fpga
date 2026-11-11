"""Tick to trade: the synthetic ITCH stream, the RTL replayed against the software model, and the proofs."""

import json
import shutil
import subprocess
import unittest

from support import Fixtures, Hdl, ROOT

from onnx2fpga.board import IcepiZero
from onnx2fpga.trading import formal, itch
from onnx2fpga.trading.reference import BUY, SELL, Config, Pipeline, frame
from onnx2fpga.trading.replay import Replay, compare, generate

HARDCAML = Hdl.skip_reason("hardcaml")
VERILATOR = None if shutil.which("verilator") else "verilator not on PATH"
SBY = IcepiZero.tool("sby")

# Parsed but never built: the proofs only generate single units.
DESIGN = {"format": 1, "top": "otf_top", "mode": "fixed", "latency": 3, "input": "x",
          "output": "y", "streams": [{"name": "x", "width": 32}, {"name": "y", "width": 8}],
          "nodes": [{"name": "fc0", "instance": "u_fc0", "kind": "otf_dense",
                     "params": {"MW": 4, "MH": 1, "ACT_BITS": 8, "WGT_BITS": 8,
                                "ACC_BITS": 18, "OUT_BITS": 8, "MULT_BITS": 8, "SHIFT": 6,
                                "OUT_MIN": -128, "OUT_MAX": 127, "OUT_ZP": 0,
                                "CONSTANT": 1, "RELOADABLE": 0},
                     "memories": {"weights": {"elem_bits": 8, "lanes": 4,
                                              "values": [1, 2, 3, 4]},
                                  "bias": {"elem_bits": 18, "lanes": 1, "values": [0]},
                                  "mult": {"elem_bits": 8, "lanes": 1, "values": [64]}},
                     "inputs": ["x"], "outputs": ["y"]}]}


class Stream(unittest.TestCase):
    def test_deterministic(self):
        first = itch.packets(itch.Generator(3).messages(2000), 3)
        self.assertEqual(first, itch.packets(itch.Generator(3).messages(2000), 3))
        self.assertNotEqual(first, itch.packets(itch.Generator(4).messages(2000), 4))

    def test_framing(self):
        messages = itch.Generator(5).messages(5000)
        self.assertTrue(all(len(m) == itch.LENGTHS[chr(m[0])] for m in messages))
        packets = itch.packets(messages, 5)
        self.assertEqual([m.data for m in frame(packets)], messages)
        beats = itch.beats(packets)
        stream = b"".join(data.to_bytes(8, "little")[:bin(keep).count("1")]
                          for data, keep, _ in beats)
        self.assertEqual(stream, b"".join(packets))
        self.assertEqual(sum(last for _, _, last in beats), len(packets))

    def test_book(self):
        decisions = Pipeline(Config(), lambda vector: 0).run(
            frame(itch.packets(itch.Generator(6).messages(20000), 6)))
        for decision in decisions:
            _, bids, asks, _ = decision.snapshot
            live_bids = [price for price, shares in bids if shares]
            live_asks = [price for price, shares in asks if shares]
            self.assertEqual(live_bids, sorted(set(live_bids), reverse=True))
            self.assertEqual(live_asks, sorted(set(live_asks)))
            if live_bids and live_asks:
                self.assertLess(live_bids[0], live_asks[0])


@unittest.skipIf(HARDCAML or VERILATOR, HARDCAML or VERILATOR)
class Rtl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = Replay(Fixtures.build_dir("tick_to_trade"), messages=100000)
        cls.accelerator, model = cls.replay.model()
        cls.replay.build(cls.accelerator)
        cls.pipeline = Pipeline(cls.replay.config, model)
        cls.decisions = cls.pipeline.run(cls.replay.framed)
        cls.latency = sum(cycles for _, cycles in cls.replay.stages)

    def test_latency(self):
        self.assertEqual(self.replay.stages, [("moldudp64", 1), ("itch_parser", 1),
                                              ("order_book", 3), ("features", 3),
                                              ("model", 6), ("decision", 1)])
        self.assertEqual(self.accelerator.latency, 6)

    def test_replay(self):
        totals = compare(self.replay.run(), self.decisions, self.replay.config, self.latency)
        self.assertGreater(len(self.decisions), 60000)
        actions = [decision.action for decision in self.decisions]
        self.assertGreater(actions.count(BUY), 1000)
        self.assertGreater(actions.count(SELL), 1000)
        self.assertGreater(self.pipeline.book.collisions, 0)
        self.assertGreater(self.pipeline.book.forgotten, 0)
        self.assertEqual((totals.min(), totals.max()), (self.latency + 2, self.latency + 5))

    def test_idle_cycles(self):
        compare(self.replay.run(gap=30, seed=2), self.decisions, self.replay.config,
                self.latency, timed=False)


@unittest.skipIf(HARDCAML or VERILATOR, HARDCAML or VERILATOR)
class K26(unittest.TestCase):
    def test_replay(self):
        replay = Replay(Fixtures.build_dir("tick_to_trade_k26"), messages=2000)
        acc, model = replay.model(device="k26", clock_mhz=322.265625)
        replay.build(acc)
        self.assertEqual(acc.latency, 12)
        latency = sum(cycles for _, cycles in replay.stages)
        self.assertEqual(latency, 21)
        decisions = Pipeline(replay.config, model).run(replay.framed)
        for gap in (0, 30):
            compare(replay.run(gap=gap), decisions, replay.config, latency, timed=gap == 0)


@unittest.skipIf(HARDCAML or IcepiZero.tool("yosys") is None, HARDCAML or "yosys not found")
class BookEquivalence(unittest.TestCase):
    def test_update(self):
        build = Fixtures.build_dir("book_equivalence")
        config = build / "config.json"
        config.write_text(json.dumps(Config().as_json(DESIGN)))
        before = ROOT / "tests" / "fixtures" / "book_side_baseline.v"
        shutil.copyfile(before, build / "before.v")
        after = generate(config, "-unit", "book_side").split("endmodule", 1)[0] + "endmodule\n"
        (build / "after.v").write_text(after.replace("module book_side (", "module book_after ("))
        script = ("read_verilog before.v after.v; "
                  "miter -equiv -flatten book_before book_after miter; prep -top miter; "
                  "sat -verify -prove trigger 0")
        result = subprocess.run([IcepiZero.tool("yosys"), "-Q", "-T", "-p", script],
                                cwd=build, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr)
        self.assertIn("SUCCESS", result.stdout)

    def test_pipeline(self):
        build = Fixtures.build_dir("book_pipeline_equivalence")
        config = build / "config.json"
        config.write_text(json.dumps(Config().as_json(DESIGN)))
        shutil.copyfile(ROOT / "tests" / "fixtures" / "book_side_baseline.v", build / "before.v")
        after = generate(config, "-unit", "book_side_pipelined")
        (build / "after.v").write_text(after.replace("module dut (", "module book_after ("))
        (build / "reference.v").write_text("""
module reference (
    input clock, clear,
    input [255:0] book$levels, input book$ask, input [63:0] book$opposite,
    input book$remove$valid, input [31:0] book$remove$price, book$remove$shares,
    input book$insert$valid, input [31:0] book$insert$price, book$insert$shares,
    output reg [255:0] updated, output accepted
);
    wire [255:0] next;
    book_before original (
        .levels(book$levels), .ask(book$ask), .opposite(book$opposite),
        .remove$valid(book$remove$valid), .remove$price(book$remove$price),
        .remove$shares(book$remove$shares), .insert$valid(book$insert$valid),
        .insert$price(book$insert$price), .insert$shares(book$insert$shares),
        .updated(next), .accepted(accepted));
    always @(posedge clock) updated <= next;
endmodule
""")
        script = ("read_verilog before.v reference.v after.v; "
                  "miter -equiv -flatten reference book_after miter; prep -top miter; "
                  "sat -verify -seq 2 -prove-skip 1 -prove trigger 0")
        result = subprocess.run([IcepiZero.tool("yosys"), "-Q", "-T", "-p", script],
                                cwd=build, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr)
        self.assertIn("SUCCESS", result.stdout)


@unittest.skipIf(HARDCAML, HARDCAML)
@unittest.skipIf(SBY is None, "sby not found")
class Proofs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = Fixtures.build_dir("tick_to_trade_formal")
        cls.config = cls.build / "config.json"
        cls.config.write_text(json.dumps(Config().as_json(DESIGN)))

    def check(self, unit, check):
        ok, log = formal.prove(self.build / unit, generate(self.config, "-unit", unit), check,
                               SBY)
        self.assertTrue(ok, log[-3000:])
        self.assertIn("successful proof by k-induction", log)
        self.assertIn("reached cover statement", log)

    def test_framer(self):
        self.check("moldudp64_step", formal.SERIAL)

    def test_book_side(self):
        self.check("book_side", formal.book_check(Config().levels))


if __name__ == "__main__":
    unittest.main()
