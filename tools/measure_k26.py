"""Vivado out of context on every corpus model in fixed mode at the trading clock, measured against predicted."""

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "compiler")]

import onnx2fpga as otf  # noqa: E402
from onnx2fpga.errors import BudgetError  # noqa: E402
from onnx2fpga.measure import Subject, VivadoRunner  # noqa: E402
from onnx2fpga.targets.resources import Resources  # noqa: E402
from onnx2fpga.trading.replay import CLOCK_MHZ  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "corpus"


def compiled(path, out):
    acc = otf.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc.compile(device="k26", mode="fixed", clock_mhz=CLOCK_MHZ, out_dir=out,
                    calibration=np.load(path.with_suffix(".npz"))["calibration"])
    return acc


def subject(acc):
    units = acc.result.hardware_graph.nodes
    predicted = sum((unit.resources(acc.device) for unit in units), Resources())
    return Subject(acc.name, "otf_top", {}, predicted, acc.latency)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="corpus names, all of them by default")
    parser.add_argument("--vivado", default="vivado")
    parser.add_argument("--synth-only", action="store_true",
                        help="stop after synthesis, which is faster and less true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "build" / "measure_k26")
    args = parser.parse_args()
    names = args.models or sorted(path.stem for path in CORPUS.glob("*.onnx"))
    runner = VivadoRunner("xck26-sfvc784-2LV-c", round(1000.0 / CLOCK_MHZ, 3),
                          not args.synth_only, args.vivado, clock="clock")
    if not args.dry_run and not runner.available:
        print("cannot find %r; source settings64.sh or pass --vivado" % args.vivado,
              file=sys.stderr)
        return 2
    rows = []
    print("| Model | Cycles | LUT | Vivado LUT | FF | Vivado FF | DSP | Predicted MHz "
          "| WNS ns | Vivado MHz |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        out = args.out / name
        try:
            acc = compiled(CORPUS / ("%s.onnx" % name), out)
        except BudgetError as error:
            print("| %s | %s |" % (name, error), flush=True)
            continue
        unit = subject(acc)
        sources = [pathlib.Path("rtl") / "top.v"]
        if args.dry_run:
            environment = runner.environment(unit, sources, "vivado")
            print("cd %s && %s %s" % (out, " ".join("%s='%s'" % pair
                                                    for pair in environment.items()),
                                      " ".join(runner.command())))
            continue
        result = runner.run(unit, sources, out, "vivado")
        measured, payload = result.measured, result.payload
        print("| %s | %d | %d | %d | %d | %d | %d | %.1f | %.3f | %.1f |"
              % (name, acc.latency, unit.predicted.lut, measured.lut, unit.predicted.ff,
                 measured.ff, measured.dsp, acc.result.fmax_mhz, payload["wns_ns"],
                 result.fmax_mhz), flush=True)
        rows.append({"model": name, "cycles": acc.latency,
                     "predicted": {"lut": unit.predicted.lut, "ff": unit.predicted.ff,
                                   "dsp": unit.predicted.dsp,
                                   "fmax_mhz": acc.result.fmax_mhz},
                     "measured": payload})
    if rows:
        (args.out / "measured.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
