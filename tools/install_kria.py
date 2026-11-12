#!/usr/bin/env python3
"""Install a verified firmware package on a Kria Linux board."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))

from onnx2fpga.kria_install import install  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--board", choices=("kr260", "kv260"), required=True)
    parser.add_argument("--program", action="store_true",
                        help="unload the current application and load this package")
    args = parser.parse_args()
    print(install(args.package, args.board, program=args.program))


if __name__ == "__main__":
    main()
