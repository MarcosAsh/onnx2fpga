"""Builds and runs a generated project under Verilator.

The build directory already carries a Makefile that knows how to do this, so
this drives that rather than reimplementing it, and adds the part a shell loop
is bad at: sweeping backpressure and reporting which settings held.
"""

import pathlib
import re
import subprocess


class SimulationResult:
    CYCLES = re.compile(r"^cycles\s+(\d+)", re.M)
    LATENCY = re.compile(r"^latency\s+(\d+)", re.M)

    def __init__(self, duty, seed, returncode, output):
        self.duty = duty
        self.seed = seed
        self.returncode = returncode
        self.output = output

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

    def render(self):
        state = "pass" if self.ok else "FAIL"
        detail = "" if self.cycles is None else "  %d cycles" % self.cycles
        return "  input %3d%%  output %3d%%  seed %d  %s%s" % (
            self.duty[0], self.duty[1], self.seed, state, detail)


class SimulationRunner:
    DEFAULT_SWEEP = ((100, 100), (50, 50), (20, 20), (100, 10), (10, 100))

    def __init__(self, build_dir, top="otf_top"):
        self.build_dir = pathlib.Path(build_dir)
        self.top = top

    @property
    def binary(self):
        return self.build_dir / "obj_dir" / ("V" + self.top)

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
                                completed.stdout + completed.stderr)

    def sweep(self, duties=None, seeds=(1, 2, 3)):
        results = []
        for duty in duties or self.DEFAULT_SWEEP:
            for seed in seeds:
                results.append(self.once(duty, seed, quiet=True))
        return results
