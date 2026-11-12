"""Simulates fabric DDR replay through the complete trading pipeline."""

import json
import pathlib
import re
import subprocess

from ..assets import AssetLocator
from ..p6_emit.hardcaml import HardcamlGenerator
from .replay import Replay, ReplayError, WARNINGS, generate
from .replay_records import records


class _Section:
    def __init__(self, path, start, end):
        self.path, self.start, self.end = path, start, end

    def __iter__(self):
        with self.path.open("rb") as stream:
            stream.seek(self.start)
            while stream.tell() < self.end:
                yield stream.readline().decode("ascii").rstrip("\n")


class DdrReplay(Replay):
    def build(self, accelerator):
        design = json.loads((pathlib.Path(accelerator.out_dir) / "design.json").read_text())
        config = self.out / "tick_to_trade.json"
        config.write_text(json.dumps(self.config.as_json(design), indent=1))
        generated = subprocess.run([str(HardcamlGenerator().executable), "ddr-trading", str(config)],
                                   capture_output=True, text=True, check=True)
        (self.out / "ddr_trading.v").write_text(generated.stdout)
        self.stages = [(name, int(cycles)) for name, cycles in
                       re.findall(r"\((\w+) (\d+)\)", generate(config, "-stages"))]
        runtime = AssetLocator().runtime_dir
        completed = subprocess.run(
            ["verilator", "--cc", "--exe", "--build", "--build-jobs", "0", *WARNINGS,
             "-CFLAGS", "-O2 -std=c++17 -I%s" % (runtime / "include"),
             "--top-module", "ddr_trading", "ddr_trading.v",
             str(runtime / "sim/ddr_tick_harness.cpp"), "-o", "Vddr_trading"],
            cwd=self.out, capture_output=True, text=True)
        if completed.returncode:
            raise ReplayError("Verilator could not build DDR replay:\n" + completed.stderr[-4000:])

    def run(self, gap=0, seed=1, runs=2, abort_after=None, stream=False):
        """Streamed sections reread the result file and must be consumed before the next run."""
        with (self.out / "records.bin").open("wb") as record_file:
            for record in records(self.beats):
                record_file.write(record)
        output = self.out / "ddr_results.txt"
        command = ["./obj_dir/Vddr_trading", "records.bin", output.name,
                   str(runs), str(seed), str(gap)]
        if abort_after is not None:
            command.append(str(abort_after))
        result = subprocess.run(command, cwd=self.out,
                                capture_output=True, text=True, timeout=3600)
        if result.returncode:
            raise ReplayError("DDR replay failed:\n" + result.stderr[-4000:])
        sections = []
        start = None
        with output.open("rb") as result_file:
            while True:
                offset = result_file.tell()
                line = result_file.readline()
                if line.startswith(b"run ") or not line:
                    if start is not None:
                        sections.append(_Section(output, start, offset))
                    start = result_file.tell()
                if not line:
                    break
        if len(sections) != runs:
            raise ReplayError("DDR replay returned the wrong number of runs")
        return sections if stream else [list(section) for section in sections]
