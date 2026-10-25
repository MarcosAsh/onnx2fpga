"""Predicted fmax against nextpnr-ecp5 for the example models at several clocks, or the k26 figures at the trading clock."""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "compiler"), str(ROOT / "tests"), str(ROOT / "examples")]

import onnx2fpga as otf  # noqa: E402
from onnx2fpga.errors import BudgetError  # noqa: E402
from onnx2fpga.measure import YosysRunner  # noqa: E402
from onnx2fpga.trading.replay import CLOCK_MHZ, tiny_mlp  # noqa: E402
from support import Fixtures  # noqa: E402

TREES = ROOT / "tests" / "fixtures"


def models(device):
    rng = np.random.default_rng(0)
    if device == "icepi25":
        yield "4-8-1 MLP", otf.load(tiny_mlp(), name="t2t"), rng.normal(0, 1, (64, 4))
        yield "16-8-4 MLP", otf.load(Fixtures.factory().mlp((16, 8, 4)), name="m16"), \
            rng.normal(0, 1, (64, 16))
        yield "100 tree classifier", otf.load(TREES / "xgb_classifier.onnx"), None
        return
    yield "example MLP", otf.load(Fixtures.factory().mlp(), name="mlp"), \
        rng.normal(0, 1, (64, 32))
    yield "100 tree regressor", otf.load(TREES / "xgb_regressor.onnx"), None
    yield "4-8-1 MLP", otf.load(tiny_mlp(), name="t2t"), rng.normal(0, 1, (64, 4))


def stages(acc):
    return " ".join("/".join(str(n) for n in unit.stages.values())
                    for unit in acc.result.hardware_graph.topological_order())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("icepi25", "k26"))
    parser.add_argument("--clocks", default="50,80,100,125")
    parser.add_argument("--out", default=str(ROOT / "build" / "fmax"))
    args = parser.parse_args()
    clocks = [CLOCK_MHZ] if args.device == "k26" else [float(c) for c in args.clocks.split(",")]
    print("| Model | Asked MHz | Stages | Latency | Predicted MHz | nextpnr MHz | Error | Met |")
    print("|---|---|---|---|---|---|---|---|")
    for name, acc, samples in models(args.device):
        for clock in clocks:
            out = pathlib.Path(args.out) / ("%s_%s_%g" % (acc.name, args.device, clock))
            try:
                # The example MLP needs 88 percent of the k26 LUTs, past the default 80.
                acc.compile(device=args.device, mode="fixed", clock_mhz=clock, out_dir=out,
                            calibration=samples, utilisation=0.9)
            except BudgetError as error:
                print("| %s | %g | %s | | | | | |" % (name, clock, error))
                continue
            predicted = acc.result.fmax_mhz
            cells = [name, ("%f" % clock).rstrip("0").rstrip("."), stages(acc),
                     "%d, %.1f ns" % (acc.latency, acc.latency * 1e3 / clock),
                     "%.1f" % predicted]
            if args.device == "icepi25":
                _, fmax = YosysRunner(acc.device, implement=True, freq_mhz=clock).run(out)
                cells += ["%.1f" % fmax, "%+.0f%%" % (100 * (predicted / fmax - 1)),
                          "yes" if fmax >= clock else "no"]
            else:
                cells += ["", "", ""]
            print("| %s |" % " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
