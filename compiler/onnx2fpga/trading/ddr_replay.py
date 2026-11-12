"""Simulates fabric DDR replay through the complete trading pipeline."""

import json
import pathlib
import re
import subprocess

from ..assets import AssetLocator
from ..p6_emit.hardcaml import HardcamlGenerator
from .replay import Replay, ReplayError, WARNINGS, generate
from .replay_records import records


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

    def run(self, gap=0, seed=1, runs=2, abort_after=None):
        with (self.out / "records.bin").open("wb") as stream:
            for record in records(self.beats):
                stream.write(record)
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
        for line in output.read_text().splitlines():
            if line.startswith("run "):
                sections.append([])
            else:
                sections[-1].append(line)
        if len(sections) != runs:
            raise ReplayError("DDR replay returned the wrong number of runs")
        return sections
