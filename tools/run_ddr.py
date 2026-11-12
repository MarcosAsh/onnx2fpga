#!/usr/bin/env python3
"""Replay packed DDR records on an already programmed Kria board."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))

from onnx2fpga.register_transport import ProcessRegisters  # noqa: E402
from onnx2fpga.trading.dma import DmaBuffer  # noqa: E402
from onnx2fpga.trading.runtime import ReplaySession  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("--buffer", default="udmabuf0")
    parser.add_argument("--helper", default="kria_regs")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("kind") != "ddr_trading":
        parser.error("expected DDR trading manifest")
    base = manifest["registers"]["base"]
    if not isinstance(base, int) or base < 0 or base % 4096:
        parser.error("invalid register base")
    payload = args.records.read_bytes()
    with ProcessRegisters([args.helper, "--base", "%x" % base], timeout=args.timeout) as bus:
        buffer = DmaBuffer(args.buffer)
        try:
            with ReplaySession(bus, buffer, manifest, timeout=args.timeout) as session:
                result = session.run(payload)
        finally:
            # A failed abort leaves the persistent allocation reserved for recovery.
            if buffer.cpu_owned:
                buffer.close()
    result["order"] = result["order"].hex()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
