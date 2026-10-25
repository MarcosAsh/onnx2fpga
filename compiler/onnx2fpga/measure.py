"""Measures what the generated units actually cost against the compiler estimates."""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

from .assets import AssetLocator
from .targets.resources import Resources


class Subject:
    """One unit to characterise: its module, parameters and predicted cost."""

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
        """An empty string parameter is an unset file name and is dropped so the module keeps its default."""
        return " ".join("%s=%s" % (key, value)
                        for key, value in sorted(self.generics.items())
                        if not (isinstance(value, str) and not value))


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
        """Relative to the build directory, where Vivado runs, so the $readmemh paths resolve."""
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


class TimingReport:
    """One clock's period and worst setup slack from a Vivado report_timing_summary file."""

    def __init__(self, clock, period_ns, wns_ns):
        self.clock = clock
        self.period_ns = period_ns
        self.wns_ns = wns_ns

    @property
    def fmax_mhz(self):
        return 1000.0 / (self.period_ns - self.wns_ns)

    @property
    def met(self):
        return self.wns_ns >= 0

    @classmethod
    def parse(cls, text, clock="otf_clk"):
        name = re.escape(clock)
        period = re.search(r"^\s*%s\s+\{[^}]*\}\s+([\d.]+)" % name, text, re.MULTILINE)
        intra = text.split("Intra Clock Table", 1)[-1]
        slack = re.search(r"^%s\s+(-?[\d.]+)" % name, intra, re.MULTILINE)
        if "Intra Clock Table" not in text or not period or not slack:
            raise ValueError("no timing for clock %s in the report" % clock)
        return cls(clock, float(period.group(1)), float(slack.group(1)))

    @classmethod
    def load(cls, path, clock="otf_clk"):
        return cls.parse(pathlib.Path(path).read_text(), clock)


def calibration_samples(model, count=16, seed=0):
    """A quantized model states its scales; a float one needs data only of the right shape."""
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
        lines.append("slowest unit runs at %.1f MHz" % worst)
    return "\n".join(lines)


class YosysRunner:
    """The open ECP5 flow (yosys synth_ecp5, then nextpnr-ecp5) on the Hardcaml rtl/top.v, which is plain Verilog."""

    FIELDS = ("lut", "ff", "dsp", "ebr")
    CELLS = {"TRELLIS_COMB": "lut", "TRELLIS_FF": "ff", "MULT18X18D": "dsp",
             "DP16KD": "ebr"}

    def __init__(self, device, implement=False, yosys="yosys", nextpnr="nextpnr-ecp5",
                 freq_mhz=None):
        self.device = device
        self.implement = implement
        self.freq_mhz = freq_mhz or device.fmax_mhz
        self.yosys = yosys
        self.nextpnr = nextpnr

    @property
    def available(self):
        return all(shutil.which(tool) for tool in (self.yosys, self.nextpnr))

    def commands(self, top="otf_top"):
        synth = [self.yosys, "-q", "-p",
                 "read_verilog rtl/top.v; synth_ecp5 -top %s -json synth/top.json" % top]
        place = [self.nextpnr] + list(self.device.nextpnr_args) + [
            "--json", "synth/top.json", "--lpf-allow-unconstrained",
            "--freq", "%g" % self.freq_mhz, "--timing-allow-fail"]
        if not self.implement:
            place.append("--pack-only")
        return [synth, place]

    def run(self, build_dir):
        """Measured resources and, after place and route, fmax in MHz."""
        (pathlib.Path(build_dir) / "synth").mkdir(exist_ok=True)
        log = ""
        for command in self.commands():
            completed = subprocess.run(command, cwd=build_dir, capture_output=True,
                                       text=True)
            log = completed.stdout + completed.stderr
            if completed.returncode != 0:
                raise RuntimeError("%s failed\n%s" % (command[0], log[-3000:]))
        return self.parse(log)

    @classmethod
    def parse(cls, log):
        measured = Resources()
        for cell, field in cls.CELLS.items():
            found = re.search(r"%s:\s+(\d+)/" % cell, log)
            if found:
                setattr(measured, field, int(found.group(1)))
        fmax = re.findall(r"Max frequency for clock .*?: ([\d.]+) MHz", log)
        return measured, float(fmax[-1]) if fmax else None

    @classmethod
    def render(cls, predicted, measured, fmax, device):
        lines = ["%-6s %9s %9s %7s %9s" % ("", "estimate", "measured", "error", "budget")]
        for field in cls.FIELDS:
            guess, real = getattr(predicted, field), getattr(measured, field)
            error = "%+6.0f%%" % (100 * (guess / real - 1)) if real else "    n/a"
            lines.append("%-6s %9.0f %9d %7s %9d" % (field, guess, real, error,
                                                     getattr(device.budget, field)))
        if fmax is not None:
            lines.append("fmax   %.1f MHz after place and route (the model assumes %.0f)"
                         % (fmax, device.fmax_mhz))
        return "\n".join(lines)


def open_flow(result, build_dir, device, implement=False, dry_run=False):
    """`onnx2fpga synth` for a device built with yosys and nextpnr."""
    runner = YosysRunner(device, implement)
    if dry_run:
        print("cd %s" % build_dir)
        for command in runner.commands():
            print("  " + " ".join(_quote(part) for part in command))
        return 0
    if not runner.available:
        print("needs %s and %s on PATH, or use --dry-run"
              % (runner.yosys, runner.nextpnr), file=sys.stderr)
        return 2
    measured, fmax = runner.run(build_dir)
    # The whole netlist, so buffers and width converters added after folding are counted.
    predicted = sum((node.resources(device) for node in result.hardware_graph.nodes),
                    Resources.zero())
    print(YosysRunner.render(predicted, measured, fmax, device))
    return 0
