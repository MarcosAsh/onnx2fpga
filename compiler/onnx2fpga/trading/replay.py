"""Replays synthetic ITCH traffic through the tick to trade RTL and checks it against the software model."""

import argparse
import json
import pathlib
import re
import subprocess
import sys

import numpy as np

from ..api import load
from ..assets import AssetLocator
from ..p1_ingest import ModelBuilder
from ..p6_emit.hardcaml import HardcamlGenerator
from ..reference.runner import GraphRunner
from . import itch
from .reference import Config, Pipeline, frame

WARNINGS = ["-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSEDPARAM", "-Wno-COMBDLY",
            "-Wno-UNUSEDSIGNAL"]
CLOCK_MHZ = 322.265625


class ReplayError(Exception):
    pass


def tiny_mlp(seed=7, hidden=8):
    """Four features in, one score out."""
    rng = np.random.default_rng(seed)
    return (ModelBuilder("t2t").add_input("x", (1, 4))
            .add_initializer("w0", rng.uniform(-1, 1, (4, hidden)).astype(np.float32))
            .add_initializer("b0", rng.uniform(-0.1, 0.1, hidden).astype(np.float32))
            .add_initializer("w1", rng.uniform(-1, 1, (hidden, 1)).astype(np.float32))
            .add_initializer("b1", np.zeros(1, np.float32))
            .add_node("Gemm", ["x", "w0", "b0"], ["h_pre"], name="fc0")
            .add_node("Relu", ["h_pre"], ["h"], name="relu0")
            .add_node("Gemm", ["h", "w1", "b1"], ["y"], name="fc1")
            .add_output("y", (1, 1)).build())


class IntegerModel:
    """The compiled model's integer reference on int8 feature codes, which is what the RTL computes."""

    def __init__(self, accelerator):
        graph = accelerator.result.hardware_graph
        self.runner = GraphRunner(graph)
        self.input, self.output = graph.inputs[0], graph.outputs[0]
        self.cache = {}

    def __call__(self, vector):
        if vector not in self.cache:
            codes = np.array([vector], dtype=np.int64)
            outputs = self.runner.run({self.input: codes})
            self.cache[vector] = int(outputs[self.output].reshape(-1)[0])
        return self.cache[vector]


def generate(config_path, *flags):
    """Runs the Hardcaml generator's tick-to-trade command and returns what it prints."""
    completed = subprocess.run([str(HardcamlGenerator().executable), "tick-to-trade",
                                str(config_path), *flags], capture_output=True, text=True)
    if completed.returncode != 0:
        raise ReplayError("the generator rejected %s:\n%s" % (config_path,
                                                             completed.stderr[-4000:]))
    return completed.stdout


class Replay:
    def __init__(self, out_dir, messages=100000, seed=1, config=None):
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.config = config or Config()
        self.messages = itch.Generator(seed).messages(messages)
        self.packets = itch.packets(self.messages, seed)
        self.beats = itch.beats(self.packets)
        self.framed = frame(self.packets)

    def model(self):
        """Compiles the MLP in fixed mode on the stream's features; thresholds at its 10th and 90th percentile."""
        books = Pipeline(self.config, lambda vector: 0).run(self.framed)
        vectors = [decision.features for decision in books]
        accelerator = load(tiny_mlp(), name="t2t_model")
        accelerator.compile(device="vu9p", mode="fixed", hdl="hardcaml",
                            calibration=np.array(vectors[:4096], np.float32),
                            out_dir=self.out / "model")
        model = IntegerModel(accelerator)
        predictions = np.array([model(vector) for vector in vectors])
        self.config.sell_threshold = int(np.percentile(predictions, 10))
        self.config.buy_threshold = max(int(np.percentile(predictions, 90)),
                                        self.config.sell_threshold + 1)
        return accelerator, model

    def build(self, accelerator):
        """Generates the RTL and a Verilator model with the replay harness."""
        design = json.loads((pathlib.Path(accelerator.out_dir) / "design.json").read_text())
        config_path = self.out / "tick_to_trade.json"
        config_path.write_text(json.dumps(self.config.as_json(design), indent=1))
        (self.out / "tick_to_trade.v").write_text(generate(config_path))
        self.stages = [(name, int(cycles)) for name, cycles
                       in re.findall(r"\((\w+) (\d+)\)", generate(config_path, "-stages"))]
        runtime = AssetLocator().runtime_dir
        completed = subprocess.run(
            ["verilator", "--cc", "--exe", "--build", "-j", "0", *WARNINGS,
             "-CFLAGS", "-O2 -std=c++17 -I%s" % (runtime / "include"),
             "--top-module", "tick_to_trade", "tick_to_trade.v",
             str(runtime / "sim" / "tick_harness.cpp"), "-o", "Vtick_to_trade"],
            cwd=self.out, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ReplayError("Verilator could not build the replay:\n" + completed.stderr[-4000:])

    def run(self, gap=0, seed=1):
        """Runs the RTL over the stream and returns its output lines."""
        itch.write_beats(self.out / "beats.bin", self.beats)
        result = self.out / ("rtl_gap%d.txt" % gap)
        completed = subprocess.run(
            ["./obj_dir/Vtick_to_trade", "--beats", "beats.bin", "--out", result.name,
             "--gap", str(gap), "--seed", str(seed)],
            cwd=self.out, capture_output=True, text=True, timeout=3600)
        if completed.returncode != 0:
            raise ReplayError("the replay failed:\n" + completed.stderr[-4000:])
        return result.read_text().splitlines()


def _levels(text, count):
    word = int(text, 16)
    return tuple(((word >> (64 * i)) & 0xFFFFFFFF, (word >> (64 * i + 32)) & 0xFFFFFFFF)
                 for i in range(count))


def _signed(value, bits):
    return value - (1 << bits) if value >> (bits - 1) else value


def compare(lines, decisions, config, latency, timed=True):
    """Checks every RTL output against the model and returns the first byte to decision cycles."""
    books = [line.split() for line in lines if line.startswith("B ")]
    vectors = [line.split() for line in lines if line.startswith("F ")]
    outs = [line.split() for line in lines if line.startswith("D ")]
    beat0 = next(int(line.split()[1]) for line in lines if line.startswith("beat0"))
    if not len(books) == len(vectors) == len(outs) == len(decisions):
        raise ReplayError("the RTL made %d books, %d feature vectors and %d decisions for %d "
                          "tracked messages" % (len(books), len(vectors), len(outs),
                                                len(decisions)))
    totals, offset = [], None
    for index, (book, vector, out, want) in enumerate(zip(books, vectors, outs, decisions)):
        got_book = (int(book[1]), _levels(book[2], config.levels), _levels(book[3], config.levels),
                    _signed(int(book[4], 16), 20))
        word = int(vector[1], 16)
        got_vector = tuple(_signed((word >> (8 * i)) & 0xFF, 8) for i in range(4))
        cycle, action, prediction, order_valid = (int(out[1]), int(out[2]),
                                                  _signed(int(out[3], 16), 8), int(out[4]))
        order = int(out[5], 16).to_bytes(47, "little") if order_valid else b""
        t_first, t_last, t_out = (int(value) for value in out[6:9])
        got = (got_book, got_vector, action, prediction, order)
        expected = (want.snapshot, want.features, want.action, want.prediction, want.order)
        if got != expected:
            raise ReplayError("decision %d differs, for message %s:\n  rtl   %s\n  model %s"
                              % (index, want.message.data.hex(), got, expected))
        if t_out - t_last != latency:
            raise ReplayError("decision %d took %d cycles from its last byte, not %d"
                              % (index, t_out - t_last, latency))
        offset = t_out - cycle if offset is None else offset
        if timed and (t_first, t_last) != (beat0 + want.message.first_beat + offset,
                                           beat0 + want.message.last_beat + offset):
            raise ReplayError("decision %d is stamped %d to %d, not beats %d to %d"
                              % (index, t_first, t_last, want.message.first_beat,
                                 want.message.last_beat))
        totals.append(t_out - t_first)
    return np.array(totals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--messages", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gap", type=int, default=0, help="percent of idle cycles between beats")
    parser.add_argument("--out", default="build/tick_to_trade")
    args = parser.parse_args(argv)

    replay = Replay(args.out, args.messages, args.seed)
    accelerator, model = replay.model()
    replay.build(accelerator)
    pipeline = Pipeline(replay.config, model)
    decisions = pipeline.run(replay.framed)
    latency = sum(cycles for _, cycles in replay.stages)
    totals = compare(replay.run(args.gap, args.seed), decisions, replay.config, latency,
                     timed=args.gap == 0)
    actions = [decision.action for decision in decisions]
    print("%d messages in %d packets, %d beats; %d tracked messages, all identical"
          % (len(replay.messages), len(replay.packets), len(replay.beats), len(decisions)))
    print("orders: %d buy, %d sell; %d adds dropped on a full slot, %d levels forgotten"
          % (actions.count(1), actions.count(2), pipeline.book.collisions,
             pipeline.book.forgotten))
    print("last byte to decision: %d cycles, %.1f ns at %.2f MHz, every message (%s)"
          % (latency, latency * 1000 / CLOCK_MHZ, CLOCK_MHZ,
             ", ".join("%s %d" % stage for stage in replay.stages)))
    print("first byte to decision: min %d, median %d, p99 %d, max %d cycles"
          % (totals.min(), np.median(totals), np.percentile(totals, 99), totals.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
