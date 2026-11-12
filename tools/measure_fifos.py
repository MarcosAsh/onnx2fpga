"""Measure stream FIFO resources in both RTL backends with Vivado synthesis."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))

from onnx2fpga.measure import Subject, VivadoRunner  # noqa: E402
from onnx2fpga.p3_ops.hardware_ops import StreamFifo  # noqa: E402
from onnx2fpga.p6_emit.hardcaml import HardcamlGenerator  # noqa: E402
from onnx2fpga.targets.device import Device  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="a7_35t")
    parser.add_argument("--depths", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--vivado", default="vivado")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "fifo_resources")
    args = parser.parse_args()
    if args.width < 1 or any(depth < 1 for depth in args.depths):
        parser.error("width and depths must be positive")
    device = Device.get(args.device)
    report = {"status": "running", "device": args.device,
              "scope": "synthesis only; no routed timing", "runs": []}
    for depth in args.depths:
        for backend in ("sv", "hardcaml"):
            out = (args.out / (backend + "_" + str(depth))).resolve()
            out.mkdir(parents=True, exist_ok=True)
            node = StreamFifo("fifo", ["x"], ["y"], depth, args.width)
            params = {"WIDTH": args.width, "DEPTH": depth}
            if backend == "sv":
                source, top, clock = ROOT / "hardware/rtl/otf_fifo.sv", "otf_fifo", "clk"
            else:
                design = {"format": 1, "top": "otf_top", "mode": "stream",
                          "input": "x", "output": "y",
                          "streams": [{"name": name, "width": args.width} for name in ("x", "y")],
                          "nodes": [{"name": "fifo", "instance": "fifo", "kind": "otf_fifo",
                                     "params": params, "inputs": ["x"], "outputs": ["y"],
                                     "memories": {}}]}
                path = out / "design.json"
                path.write_text(json.dumps(design))
                source, top, clock = out / "fifo.v", "fifo", "clock"
                source.write_bytes(subprocess.check_output(
                    [str(HardcamlGenerator().executable), "units", str(path)]))
                params = {}
            predicted = node.resources(device)
            subject = Subject("fifo", top, params, predicted, 0)
            runner = VivadoRunner(device.part, 10, executable=args.vivado, clock=clock)
            result = runner.run(subject, [source], out, "vivado")
            row = {"depth": depth, "width": args.width, "backend": backend,
                   "predicted": vars(predicted), "measured": result.payload,
                   "rtl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "synthesis_script_sha256": hashlib.sha256(runner.script.read_bytes()).hexdigest()}
            report["runs"].append(row)
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    report["status"] = "passed"
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
