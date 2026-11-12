#!/usr/bin/env python3
"""Check DDR-backed trading replay against the software model without a board."""

import argparse
import gzip
import hashlib
import json
import pathlib
import subprocess

from onnx2fpga.trading.ddr_replay import DdrReplay
from onnx2fpga.trading.reference import Pipeline
from onnx2fpga.trading.reference import Config
from onnx2fpga.trading.recorded import select
from onnx2fpga.trading.replay import compare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("build/ddr_trading"))
    parser.add_argument("--messages", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--itch", type=pathlib.Path, help="recorded BinaryFILE, optionally gzip compressed")
    parser.add_argument("--symbol", default="AAPL", help="stock directory symbol for recorded replay")
    parser.add_argument("--stalls", type=int, nargs="+", default=[0, 65])
    args = parser.parse_args(argv)
    if args.messages < 100 or any(not 0 <= stall < 100 for stall in args.stalls):
        parser.error("use at least 100 messages and stall percentages between 0 and 99")
    args.out.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parents[1]
    report = {"status": "running", "scope": "Verilator AXI memory model; no physical DDR or board",
              "messages_requested": args.messages, "seed": args.seed,
              "target_clock_mhz": 322.265625, "runs": []}
    report["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("runtime/sim/ddr_tick_harness.cpp", "hardware/hardcaml/src/ddr_trading.ml",
                     "hardware/hardcaml/src/ddr_replay.ml", "tools/replay_ddr.py",
                     "compiler/onnx2fpga/trading/recorded.py", "compiler/onnx2fpga/trading/itch.py",
                     "compiler/onnx2fpga/trading/replay.py", "compiler/onnx2fpga/trading/ddr_replay.py")}
    path = args.out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        source_messages, config = None, None
        if args.itch:
            opener = gzip.open if args.itch.suffix == ".gz" else open
            with opener(args.itch, "rb") as stream:
                source_messages, record = select(stream, args.symbol, args.messages)
            digest = hashlib.sha256()
            with args.itch.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            record.update(file=args.itch.name, file_sha256=digest.hexdigest(),
                          framing="MoldUDP64 packets rebuilt from selected raw messages")
            report["recorded_source"] = record
            config = Config(locates=(record["locate"],), names=(args.symbol,))
        replay = DdrReplay(args.out, messages=args.messages, seed=args.seed,
                           config=config, source_messages=source_messages)
        acc, model = replay.model(device="k26", clock_mhz=322.265625)
        acc.model.save(args.out / "model/model.onnx")
        replay.build(acc)
        reference = Pipeline(replay.config, model)
        expected = reference.run(replay.framed)
        report["reference_book"] = {"levels": replay.config.levels,
                                    "order_slots": 1 << replay.config.order_bits,
                                    "collisions": reference.book.collisions,
                                    "forgotten_levels": reference.book.forgotten}
        latency = sum(cycles for _, cycles in replay.stages)
        report.update(records=len(replay.beats), decisions_per_run=len(expected),
                      latency_cycles=latency, stages=replay.stages,
                      verilator=subprocess.check_output(["verilator", "--version"], text=True).strip())
        for stall in args.stalls:
            sections = replay.run(gap=stall, seed=args.seed, runs=2)
            for index, lines in enumerate(sections):
                compare(lines, expected, replay.config, latency, timed=False)
                cycles = int(next(line.split()[1] for line in lines if line.startswith("cycles ")))
                report["runs"].append({"stall_percent": stall, "restart": bool(index),
                                       "cycles": cycles, "decisions": len(expected), "bit_exact": True})
            (args.out / ("results_stall%d.txt" % stall)).write_text(
                (args.out / "ddr_results.txt").read_text())
        report["sha256"] = {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                            for name in ("records.bin", "ddr_trading.v", "tick_to_trade.json",
                                         "model/model.onnx", "model/design.json")}
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
