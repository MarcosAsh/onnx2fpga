"""Measure what the generated units actually cost, and calibrate the model.

Every resource and timing figure the compiler reports comes from a hand written
estimator that has never been checked against synthesis. This compiles a model,
synthesises each of its units out of context, and prints the measured cost
beside the predicted one.

    onnx2fpga synth model.onnx --device z7020
    onnx2fpga synth model.onnx --device z7020 --impl
    onnx2fpga synth model.onnx --dry-run

The dry run prints the exact commands without needing Vivado, which is also how
this is tested where no Vivado exists.
"""

import json
import os
import pathlib
import shutil
import subprocess

import numpy as np

from .assets import AssetLocator
from .compile import Compiler
from .p1_ingest import OnnxModel
from .targets.device import Device
from .targets.resources import Resources


class Subject:
    """One unit to characterise: its module, its parameters, and what the
    compiler predicted it would cost."""

    def __init__(self, name, module, generics, predicted, cycles):
        self.name = name
        self.module = module
        self.generics = dict(generics)
        self.predicted = predicted
        self.cycles = cycles

    @classmethod
    def from_node(cls, node, graph, device):
        return cls(node.name, node.module, node.sv_params(graph),
                   node.resources(device), node.cycles)

    def generic_arguments(self):
        """An empty string parameter is an unset file name. Passing it would
        make Vivado try to read a file called nothing, so it is dropped and the
        module keeps its default."""
        return " ".join("%s=%s" % (key, value)
                        for key, value in sorted(self.generics.items())
                        if not (isinstance(value, str) and not value))

    def __repr__(self):
        return "Subject(%s, %s)" % (self.name, self.module)


class SynthesisResult:
    FIELDS = ("lut", "ff", "dsp", "bram36", "uram")

    def __init__(self, payload):
        self.payload = dict(payload)

    @classmethod
    def load(cls, path):
        return cls(json.loads(pathlib.Path(path).read_text()))

    @property
    def measured(self):
        return Resources(**{f: float(self.payload.get(f, 0)) for f in self.FIELDS})

    @property
    def fmax_mhz(self):
        return float(self.payload.get("fmax_mhz", 0.0))

    @property
    def stage(self):
        return self.payload.get("stage", "synth")


class VivadoRunner:
    def __init__(self, part, period_ns=5.0, implement=False, executable="vivado",
                 assets=None):
        self.part = part
        self.period_ns = period_ns
        self.implement = implement
        self.executable = executable
        self.assets = assets or AssetLocator()

    @property
    def script(self):
        return self.assets.synth_script

    @property
    def available(self):
        return (shutil.which(self.executable) is not None
                or os.access(self.executable, os.X_OK))

    def environment(self, subject, sources, out_dir):
        return {
            "OTF_PART": self.part,
            "OTF_TOP": subject.module,
            "OTF_RTL": " ".join(str(s) for s in sources),
            "OTF_GENERICS": subject.generic_arguments(),
            "OTF_PERIOD": str(self.period_ns),
            "OTF_OUT": str(out_dir),
            "OTF_IMPL": "1" if self.implement else "0",
        }

    def command(self):
        return [self.executable, "-mode", "batch", "-nojournal", "-nolog",
                "-source", str(self.script)]

    def run(self, subject, sources, work_dir, out_dir):
        environment = dict(os.environ)
        environment.update(self.environment(subject, sources, out_dir))
        completed = subprocess.run(self.command(), cwd=work_dir, env=environment,
                                   capture_output=True, text=True)
        summary = pathlib.Path(work_dir) / out_dir / "summary.json"
        if completed.returncode != 0 or not summary.exists():
            raise RuntimeError("vivado failed for %s\n%s"
                               % (subject.name, completed.stdout[-3000:]))
        return SynthesisResult.load(summary)


class Comparison:
    def __init__(self, subject, result):
        self.subject = subject
        self.result = result

    def error(self, field):
        predicted = getattr(self.subject.predicted, field)
        measured = getattr(self.result.measured, field)
        if measured == 0:
            return None if predicted == 0 else float("inf")
        return predicted / measured - 1.0

    def row(self):
        cells = []
        for field in SynthesisResult.FIELDS:
            predicted = getattr(self.subject.predicted, field)
            measured = getattr(self.result.measured, field)
            ratio = self.error(field)
            marker = "  n/a" if ratio is None else (
                " inf" if ratio == float("inf") else "%+5.0f%%" % (100 * ratio))
            cells.append("%7.0f %7.0f %6s" % (predicted, measured, marker))
        return "%-24s %-14s %s  %7.1f" % (
            self.subject.name, self.subject.module, " ".join(cells),
            self.result.fmax_mhz)


class Characterizer:
    def __init__(self, runner, device):
        self.runner = runner
        self.device = device

    def subjects(self, graph):
        seen, subjects = set(), []
        for node in graph.topological_order():
            key = (node.module, tuple(sorted(node.sv_params(graph).items())))
            if key in seen:
                continue
            seen.add(key)
            subjects.append(Subject.from_node(node, graph, self.device))
        return subjects

    def sources(self, build_dir):
        """Relative to the build directory, because that is where Vivado runs
        so that the $readmemh paths in the netlist resolve."""
        root = pathlib.Path(build_dir)
        return [path.relative_to(root) for path in sorted((root / "rtl").glob("otf_*.sv"))]

    def describe(self, subject, build_dir):
        environment = self.runner.environment(
            subject, self.sources(build_dir), "synth/" + subject.name)
        lines = ["cd %s" % build_dir]
        for key, value in environment.items():
            lines.append("  %s=%s \\" % (key, _quote(value)))
        lines.append("  " + " ".join(self.runner.command()))
        return "\n".join(lines)

    def characterize(self, subjects, build_dir):
        comparisons = []
        for subject in subjects:
            result = self.runner.run(subject, self.sources(build_dir),
                                     build_dir, "synth/" + subject.name)
            comparisons.append(Comparison(subject, result))
        return comparisons


def calibration_samples(model, count=16, seed=0):
    """A quantized model states its own scales; a float one has to be measured,
    and for characterisation the data only has to have the right shape."""
    name = model.graph_inputs()[0]
    shape = tuple(1 if isinstance(d, str) else int(d)
                  for d in model.value_shape(name))
    rng = np.random.default_rng(seed)
    return [{name: rng.normal(0, 1, shape).astype(np.float32)} for _ in range(count)]


def _quote(value):
    return '"%s"' % value if " " in str(value) else str(value)


def render(comparisons):
    header = "%-24s %-14s %s  %7s" % (
        "unit", "module",
        " ".join("%22s" % ("%s pred meas err" % f)
                 for f in SynthesisResult.FIELDS),
        "fmax")
    lines = [header, "-" * len(header)]
    lines.extend(c.row() for c in comparisons)
    if comparisons:
        worst = min(c.result.fmax_mhz for c in comparisons)
        lines.append("")
        lines.append("slowest unit runs at %.1f MHz, which is the honest fmax "
                     "for this design" % worst)
    return "\n".join(lines)
