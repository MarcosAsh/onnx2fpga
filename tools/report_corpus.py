"""Check corpus coverage and record numerical and simulated FPGA performance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))

import onnx2fpga as otf  # noqa: E402
from onnx2fpga.p1_ingest import OnnxModel  # noqa: E402
from onnx2fpga.measure import Subject, VivadoRunner, YosysRunner, routed_timing  # noqa: E402
from onnx2fpga.targets.device import Device  # noqa: E402
from onnx2fpga.targets.resources import Resources  # noqa: E402

CORPUS = ROOT / "tests" / "fixtures" / "corpus"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args):
    if not shutil.which(args[0]):
        return None
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    return (result.stdout.strip() or result.stderr.strip()) if result.returncode == 0 else None


def metadata():
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": command("git", "rev-parse", "HEAD"),
        "worktree_status": command("git", "status", "--porcelain"),
        "reporter_sha256": digest(Path(__file__)),
        "fixture_generator_sha256": digest(ROOT / "tests/fixtures/make_corpus.py"),
        "hardcaml_lock_sha256": digest(ROOT / "hardware/hardcaml/onnx2fpga_hardcaml.opam.locked"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "onnx2fpga": otf.__version__, "verilator": command("verilator", "--version"),
                     "dune": command("dune", "--version"),
                     "ocaml": command("ocamlopt", "-version")},
    }


def model_info(model):
    return {
        "producer": model.proto.producer_name,
        "producer_version": model.proto.producer_version,
        "ir_version": model.proto.ir_version,
        "opsets": {entry.domain: entry.version for entry in model.proto.opset_import},
        "inputs": [{"name": info.name, "shape": model.value_shape(info.name),
                    "dtype": getattr(info.type and info.type.tensor_type, "elem_type", None)}
                   for info in model.graph.input if info.name in model.graph_inputs()],
        "operators": [{"domain": node.domain, "op_type": node.op_type,
                       "attributes": sorted(a.name for a in node.attribute)}
                      for node in model.nodes()],
    }


def compare(got, expected, labels):
    got = np.asarray(got).reshape(expected.shape)
    if not np.all(np.isfinite(got)) or not np.all(np.isfinite(expected)):
        return {"passed": False, "error": "non-finite predictions or reference values"}
    if labels:
        agreement = float(np.mean(got == expected))
        return {"passed": agreement >= 0.98, "agreement": agreement,
                "minimum_agreement": 0.98}
    error = np.abs(got - expected)
    mae, std = float(np.mean(error)), float(np.std(expected))
    return {"passed": mae < 0.025 * std if std else mae == 0,
            "mae": mae, "max_absolute_error": float(np.max(error)),
            "reference_std": std, "normalized_mae": mae / std if std else None,
            "maximum_normalized_mae": 0.025}


def implement(acc, out, clock, mode="fixed"):
    if acc.device.nextpnr_args:
        runner = YosysRunner(acc.device, implement=True, freq_mhz=clock)
        resources, _ = runner.run(out)
        routed = json.loads((out / "synth/route.json").read_text())
        clocks = routed["fmax"]
        if any(not math.isfinite(entry[key]) for entry in clocks.values()
               for key in ("achieved", "constraint")):
            raise ValueError("non-finite nextpnr frequency")
        passed = len(clocks) == 1 and all(
            entry["constraint"] == clock
            and entry["achieved"] >= clock for entry in clocks.values())
        files = ("top.json", "route.json", "commands.json", "yosys.log", "nextpnr.log")
        return {"measured": {"part": acc.device.part, "stage": "impl", **vars(resources)},
                "timing": {"passed": passed, "clocks": clocks,
                           "scope": "nextpnr setup estimate; hold, pulse width and board I/O unassessed"},
                "artifacts_sha256": {name: digest(out / "synth" / name) for name in files}}
    runner = VivadoRunner(acc.device.part, 1000 / clock, implement=True,
                          clock="clock" if mode == "fixed" else "clk")
    result = runner.run(Subject(acc.name, "otf_top", {}, Resources(), acc.latency),
                        ["rtl/top.v"], out, "vivado")
    if result.stage != "impl" or result.payload["part"] != acc.device.part:
        raise ValueError("implementation target mismatch")
    timing = routed_timing((out / "vivado/timing_impl.rpt").read_text())
    files = ("summary.json", "timing_impl.rpt", "utilization_impl.rpt", "routed.dcp")
    return {"measured": result.payload, "timing": timing,
            "script_sha256": digest(runner.script),
            "artifacts_sha256": {name: digest(out / "vivado" / name) for name in files}}


def report_model(path, args):
    row = {"model": path.stem, "status": "failed", "warnings": [],
           "stages": {name: "not_run" for name in
                      ("inspection", "compilation", "reference", "simulation",
                       "implementation", "physical_board")}}
    stage = "inspection"
    try:
        row["model_sha256"] = digest(path)
        model = OnnxModel.load(path)
        row["onnx"] = model_info(model)
        check = otf.check(model, device=args.device, mode=args.mode, clock_mhz=args.clock_mhz)
        row["compatibility"] = check.to_dict()
        row["stages"][stage] = "passed" if check.ok else "blocked_or_unknown"
        if not check.ok:
            return row
        if args.inspect_only:
            row["status"] = "inspected"
            return row
        fixture = path.with_suffix(".npz")
        row["fixture_sha256"] = digest(fixture)
        with np.load(fixture, allow_pickle=False) as data:
            samples, expected, calibration = data["x"], data["y"], data["calibration"]
        if not len(samples) or not len(calibration):
            raise ValueError("empty evaluation or calibration batch")
        row["samples"] = {"evaluation": len(samples), "calibration": len(calibration),
                          "simulation": min(args.frames, len(samples))}
        row["reference_source"] = ("stored QONNX outputs" if path.stem.startswith(
            ("qonnx_", "brevitas_")) else "stored ONNX Runtime outputs")
        stage = "compilation"
        acc = otf.load(model, name=path.stem)
        out = args.out / path.stem
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acc.compile(device=args.device, mode=args.mode, clock_mhz=args.clock_mhz,
                        hdl="hardcaml", calibration=calibration, out_dir=out)
        row["warnings"] = [str(w.message) for w in caught]
        manifest = json.loads((out / "manifest.json").read_text())
        clock = acc.result.clock_mhz or acc.device.fmax_mhz
        interval = manifest["cycles_per_frame"]
        row["estimates"] = {
            "resources": manifest["resources"], "latency_cycles": acc.latency,
            "initiation_interval_cycles": interval, "clock_assumption_mhz": clock,
            "latency_ns_at_assumed_clock": acc.latency * 1000 / clock if acc.latency is not None else None,
            "frames_per_second_at_assumed_clock": clock * 1e6 / interval,
            "predicted_fmax_mhz": acc.result.fmax_mhz,
        }
        row["generated_sha256"] = {str(p.relative_to(out)): digest(p)
                                   for p in sorted((out / "rtl").glob("*")) if p.is_file()}
        row["manifest_sha256"] = digest(out / "manifest.json")
        row["stages"][stage] = "passed"
        stage = "reference"
        row["numerical_comparison"] = compare(acc.predict(samples), expected, acc._labels)
        row["stages"][stage] = "passed" if row["numerical_comparison"]["passed"] else "failed"
        stage = "simulation"
        simulation = acc.simulate(samples[:args.frames])
        row["simulation"] = {"bit_exact": bool(simulation.ok), "frames": simulation.frames}
        row["stages"][stage] = "passed" if simulation.ok else "failed"
        if not simulation.ok:
            row["error"] = simulation.first.output[-2000:]
        if args.implement and simulation.ok and row["numerical_comparison"]["passed"]:
            stage = "implementation"
            row["implementation"] = implement(acc, out, clock, args.mode)
            row["stages"][stage] = "passed" if row["implementation"]["timing"]["passed"] else "failed"
        required = ["inspection", "compilation", "reference", "simulation"]
        if args.implement:
            required.append("implementation")
        row["status"] = "passed" if all(row["stages"][s] == "passed" for s in
                                        required) else "failed"
    except (otf.Onnx2FpgaError, OSError, ValueError, KeyError, RuntimeError) as error:
        row["stages"][stage] = "failed"
        row["error"] = str(error)
    return row


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="corpus model names; all by default")
    parser.add_argument("--device", choices=Device.names(), default="k26")
    parser.add_argument("--mode", choices=("stream", "fixed"), default="fixed")
    parser.add_argument("--clock-mhz", type=float)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--implement", action="store_true", help="run device place and route")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "corpus_report")
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.implement and (args.inspect_only or args.device == "sim"):
        parser.error("--implement requires compilation on a physical device target")
    names = args.models or sorted(path.stem for path in CORPUS.glob("*.onnx"))
    known = {path.stem for path in CORPUS.glob("*.onnx")}
    if set(names) - known:
        parser.error("unknown corpus models: %s" % ", ".join(sorted(set(names) - known)))
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "environment": {},
              "configuration": {"device": args.device, "mode": args.mode,
                                "clock_mhz": args.clock_mhz, "frames": args.frames,
                                "implement": args.implement,
                                "inspect_only": args.inspect_only, "models": names},
              "complete": False, "models": []}
    destination = args.out / "report.json"
    write_report(destination, report)
    report["environment"] = metadata()
    if args.implement:
        versions = report["environment"]["versions"]
        if Device.get(args.device).nextpnr_args:
            versions["yosys"] = command("yosys", "-V")
            versions["nextpnr"] = command("nextpnr-ecp5", "--version")
        else:
            versions["vivado"] = command("vivado", "-version")
    write_report(destination, report)
    for name in names:
        row = report_model(CORPUS / (name + ".onnx"), args)
        report["models"].append(row)
        write_report(destination, report)
        print("%s: %s" % (name, row["status"]), flush=True)
    report["complete"] = True
    write_report(destination, report)
    print(destination)
    return 1 if any(row["status"] == "failed" for row in report["models"]) else 0


if __name__ == "__main__":
    sys.exit(main())
