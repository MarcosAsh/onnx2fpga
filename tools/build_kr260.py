"""Simulate an ONNX model and build the complete KR260 design in Vivado."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "compiler"), str(ROOT / "examples")]

import onnx2fpga as otf  # noqa: E402
from make_models import ModelFactory  # noqa: E402
from onnx2fpga.board import KR260  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "kr260_smoke")
    parser.add_argument("--model", type=Path, help="ONNX model, defaults to a seeded 4-8-1 MLP")
    parser.add_argument("--samples", type=Path, help="NumPy .npy calibration and simulation inputs")
    parser.add_argument("--generate-only", action="store_true",
                        help="simulate and write the Vivado project without running Vivado")
    args = parser.parse_args()
    if (args.model is None) != (args.samples is None):
        parser.error("--model and --samples must be supplied together")
    if args.model is None:
        model = ModelFactory(7).mlp((4, 8, 1))
        samples = np.random.default_rng(5).normal(0, 1, (64, 4)).astype(np.float32)
    else:
        model = args.model
        samples = np.load(args.samples, allow_pickle=False)
        if not isinstance(samples, np.ndarray) or samples.ndim < 2 or len(samples) == 0:
            parser.error("--samples must contain a nonempty batch in .npy format")
    acc = otf.load(model, name="kr260_smoke")
    args.out.mkdir(parents=True, exist_ok=True)
    acc.model.save(args.out / "model.onnx")
    acc.compile(device="k26", mode="fixed", clock_mhz=322.265625,
                calibration=samples, out_dir=args.out)
    report = acc.simulate(samples[:16])
    print(report, flush=True)
    if not report.ok:
        raise SystemExit("simulation failed")
    if args.generate_only:
        print(KR260(acc.out_dir).write())
    else:
        print(acc.build(board="kr260"))
        acc.summary()


if __name__ == "__main__":
    main()
