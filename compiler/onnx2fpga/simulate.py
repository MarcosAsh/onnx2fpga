"""Builds and runs a generated project under Verilator, sweeping backpressure."""

import pathlib
import re
import subprocess

from .assets import AssetLocator
from .errors import SimulationError
from .p6_emit.hardcaml import HardcamlGenerator


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
    """`vectors` names the stimulus and expected output relative to the build directory."""

    SWEEP = ((100, 100), (50, 50), (20, 20), (100, 10), (10, 100))
    GOLDEN = ("golden/input.hex", "golden/expected.hex")

    def __init__(self, build_dir, top="otf_top", vectors=GOLDEN):
        self.build_dir = pathlib.Path(build_dir)
        self.top = top
        self.vectors = tuple(str(v) for v in vectors)

    def build(self):
        return subprocess.run(["make", "-s"], cwd=self.build_dir,
                              capture_output=True, text=True)

    def _duty(self, duty):
        return ["--input-duty", str(duty[0]), "--output-duty", str(duty[1])]

    def once(self, duty=(100, 100), seed=1, quiet=False):
        command = (["./obj_dir/V" + self.top, "--input", self.vectors[0],
                    "--expected", self.vectors[1]]
                   + self._duty(duty) + ["--seed", str(seed)])
        if quiet:
            command.append("--quiet")
        completed = subprocess.run(command, cwd=self.build_dir, capture_output=True,
                                   text=True, timeout=600)
        return SimulationResult(duty, seed, completed.returncode,
                                completed.stdout + completed.stderr)

    def sweep(self, seeds=(1, 2, 3)):
        return [self.once(duty, seed, quiet=True)
                for duty in self.SWEEP for seed in seeds]


class FixedSimulationRunner(SimulationRunner):
    """Runs a fixed latency build; the sweep thins features_valid instead of adding backpressure, and `reload` holds the harness flags for a second bank."""

    SWEEP = ((50, 100), (10, 100))

    def __init__(self, build_dir, latency, top="otf_top", vectors=SimulationRunner.GOLDEN,
                 reload=()):
        SimulationRunner.__init__(self, build_dir, top, vectors)
        self.latency = latency
        self.reload = list(reload)

    def _duty(self, duty):
        return ["--latency", str(self.latency), "--input-duty", str(duty[0])] + self.reload


class SimulatedRegisters:
    """kria_regs of a reloadable build under Verilator, one AXI4-Lite transfer at a time over a pipe, standing in for a board's register window."""

    def __init__(self, build_dir, assets=None):
        build_dir = pathlib.Path(build_dir)
        folder = build_dir / "registers"
        folder.mkdir(exist_ok=True)
        (folder / "kria_regs.v").write_text(
            HardcamlGenerator().verilog((build_dir / "design.json").resolve(), "kria-regs"))
        harness = AssetLocator(assets).runtime_dir / "sim" / "regs_harness.cpp"
        built = subprocess.run(
            ["verilator", "--cc", "--exe", "--build", "-j", "0", "-Wno-fatal",
             "--top-module", "kria_regs", "-CFLAGS", "-O1 -std=c++17", "kria_regs.v",
             str(harness), "-o", "Vkria_regs"],
            cwd=folder, capture_output=True, text=True)
        if built.returncode != 0:
            raise SimulationError("Verilator could not build the register file in %s:\n%s"
                                  % (folder, built.stderr[-4000:]), built.stdout, built.stderr)
        self.process = subprocess.Popen([str(folder / "obj_dir" / "Vkria_regs")],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True)

    def write(self, offset, value):
        self.process.stdin.write("w %x %x\n" % (offset, value))

    def read(self, offset):
        self.process.stdin.write("r %x\n" % offset)
        self.process.stdin.flush()
        return int(self.process.stdout.readline(), 16)

    def close(self):
        self.process.stdin.close()
        self.process.wait()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
