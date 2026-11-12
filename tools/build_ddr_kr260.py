#!/usr/bin/env python3
"""Build the complete KR260 DDR trading design after a successful replay simulation."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))

from onnx2fpga.trading.board import TradingKR260  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="output directory from tools/replay_ddr.py")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    board = TradingKR260(args.replay, jobs=args.jobs)
    if args.generate_only:
        print(board.write())
    else:
        print(board.run())
        print(board.bitstream)


if __name__ == "__main__":
    main()
