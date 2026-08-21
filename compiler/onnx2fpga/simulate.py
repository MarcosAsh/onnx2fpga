"""Builds and runs a generated project under Verilator.

The build directory already carries a Makefile that knows how to do this, so
this drives that rather than reimplementing it, and adds the part a shell loop
is bad at: sweeping backpressure and reporting which settings held.
"""

import json
import pathlib
import re
import subprocess


def nanoseconds(cycles, fmax_mhz):
    """Cycles are what the harness counts. Nanoseconds are what anyone
    deploying this has a budget in, and the conversion needs a clock the
    harness does not know about, so it happens here."""
    if cycles is None or not fmax_mhz:
        return None
    return 1000.0 * cycles / fmax_mhz


class SimulationResult:
    CYCLES = re.compile(r"^cycles\s+(\d+)", re.M)
    LATENCY = re.compile(r"^latency\s+(\d+)", re.M)
    LATENCY_LINE = re.compile(r"^latency\s+\d+ cycles to first output$", re.M)

    def __init__(self, duty, seed, returncode, output, fmax_mhz=None):
        self.duty = duty
        self.seed = seed
        self.returncode = returncode
        self.output = output
        self.fmax_mhz = fmax_mhz

    @property
    def ok(self):
        return self.returncode == 0

    def _number(self, pattern):
        found = pattern.search(self.output)
        return int(found.group(1)) if found else None

    @property
    def cycles(self):
        return self._number(self.CYCLES)

    @property
    def latency(self):
        return self._number(self.LATENCY)

    @property
    def latency_ns(self):
        return nanoseconds(self.latency, self.fmax_mhz)

    def annotated_output(self):
        """The harness output with nanoseconds spliced into the latency line.

        The harness counts clock edges and has no idea what clock the design
        was compiled for, so the conversion belongs on this side. Splicing it
        into the existing line rather than adding one keeps the number next to
        the cycles it came from."""
        if self.latency_ns is None:
            return self.output
        return self.LATENCY_LINE.sub(
            lambda found: "%s, %.0f ns at %.0f MHz" % (
                found.group(0), self.latency_ns, self.fmax_mhz),
            self.output, count=1)

    def render(self):
        state = "pass" if self.ok else "FAIL"
        detail = []
        if self.cycles is not None:
            detail.append("%d cycles" % self.cycles)
        if self.latency is not None:
            detail.append("latency %d" % self.latency
                          if self.latency_ns is None else
                          "latency %d (%.0f ns)" % (self.latency, self.latency_ns))
        return "  input %3d%%  output %3d%%  seed %d  %s%s" % (
            self.duty[0], self.duty[1], self.seed, state,
            ("  " + "  ".join(detail)) if detail else "")


class LatencySummary:
    """The spread of first-output latency across a set of runs.

    Reported as a range over a stated number of runs rather than as a tail
    percentile. A sweep is a handful of runs, and a p99.9 computed from fifteen
    samples is exactly the sort of number this project exists not to quote.

    The low duty settings stall the design on purpose, so most of the spread
    here is the testbench holding tvalid or tready down rather than the design
    varying. The unstalled run is the intrinsic latency; the rest show how it
    degrades when the source or the sink cannot keep up.
    """

    def __init__(self, results, fmax_mhz=None):
        self.samples = sorted(r.latency for r in results
                              if r.ok and r.latency is not None)
        self.fmax_mhz = fmax_mhz

    def __bool__(self):
        return bool(self.samples)

    __nonzero__ = __bool__

    @property
    def count(self):
        return len(self.samples)

    @property
    def best(self):
        return self.samples[0]

    @property
    def worst(self):
        return self.samples[-1]

    @property
    def median(self):
        return self.samples[(len(self.samples) - 1) // 2]

    @property
    def spread(self):
        return self.worst - self.best

    def render(self):
        if not self.samples:
            return "latency: no run reported one"
        clock = ("" if self.fmax_mhz is None
                 else ", ns at %.0f MHz" % self.fmax_mhz)
        lines = ["latency over %d passing run%s%s"
                 % (self.count, "" if self.count == 1 else "s", clock)]
        for label, cycles in (("best", self.best), ("median", self.median),
                              ("worst", self.worst), ("spread", self.spread)):
            ns = nanoseconds(cycles, self.fmax_mhz)
            lines.append("  %-7s %6d cycles%s"
                         % (label, cycles, "" if ns is None else "  %8.0f ns" % ns))
        return "\n".join(lines)


class SimulationRunner:
    DEFAULT_SWEEP = ((100, 100), (50, 50), (20, 20), (100, 10), (10, 100))

    def __init__(self, build_dir, top="otf_top"):
        self.build_dir = pathlib.Path(build_dir)
        self.top = top
        self._fmax = False

    @property
    def binary(self):
        return self.build_dir / "obj_dir" / ("V" + self.top)

    @property
    def fmax_mhz(self):
        """The clock the design was compiled for, taken from the manifest so a
        build directory is enough. None if there is no manifest, in which case
        everything stays in cycles."""
        if self._fmax is False:
            path = self.build_dir / "manifest.json"
            self._fmax = (json.loads(path.read_text()).get("fmax_mhz")
                          if path.exists() else None)
        return self._fmax

    def build(self):
        return subprocess.run(["make", "-s"], cwd=self.build_dir,
                              capture_output=True, text=True)

    def once(self, duty=(100, 100), seed=1, quiet=False, timeout=600):
        command = ["./obj_dir/V" + self.top,
                   "--input", "golden/input.hex",
                   "--expected", "golden/expected.hex",
                   "--input-duty", str(duty[0]),
                   "--output-duty", str(duty[1]),
                   "--seed", str(seed)]
        if quiet:
            command.append("--quiet")
        completed = subprocess.run(command, cwd=self.build_dir, capture_output=True,
                                   text=True, timeout=timeout)
        return SimulationResult(duty, seed, completed.returncode,
                                completed.stdout + completed.stderr,
                                fmax_mhz=self.fmax_mhz)

    def sweep(self, duties=None, seeds=(1, 2, 3)):
        results = []
        for duty in duties or self.DEFAULT_SWEEP:
            for seed in seeds:
                results.append(self.once(duty, seed, quiet=True))
        return results

    def summarise(self, results):
        return LatencySummary(results, self.fmax_mhz)
