"""KR260 board project for the simulated DDR trading design."""

import hashlib
import json
import shutil

from ..board import BoardError, KR260

OFFSETS = {"id": 0, "version": 4, "control": 0x10, "status": 0x14,
           "base_low": 0x20, "base_high": 0x24, "records": 0x28,
           "delivered": 0x30, "decisions": 0x34, "orders": 0x38,
           "cycles": 0x3C, "latency": 0x40, "result": 0x44, "t_first": 0x48,
           "t_last": 0x4C, "t_out": 0x50, "order": 0x100}


class TradingKR260(KR260):
    def write(self):
        report = json.loads((self.out / "report.json").read_text())
        if (report.get("status") != "passed" or not report.get("runs")
                or any(run.get("bit_exact") is not True for run in report["runs"])):
            raise BoardError("DDR trading build requires a passed replay report")
        for name in ("tick_to_trade.json", "model/design.json", "records.bin"):
            actual = hashlib.sha256((self.out / name).read_bytes()).hexdigest()
            if report.get("sha256", {}).get(name) != actual:
                raise BoardError("replay artifact changed: %s" % name)
        config = (self.out / "tick_to_trade.json").resolve()
        rtl = self.generator.verilog(config, "ddr-trading")
        if hashlib.sha256(rtl.encode()).hexdigest() != report.get("sha256", {}).get("ddr_trading.v"):
            raise BoardError("trading RTL changed; rerun the DDR replay before building")
        self.build.mkdir(parents=True, exist_ok=True)
        (self.build / "ddr_shell.v").write_text(
            self.generator.verilog(config, "ddr-shell"))
        for name in ("build.tcl", "check_timing.tcl", "kria.xdc", "ddr_top.v"):
            shutil.copyfile(self.directory / name, self.build / name)
        (self.build / "settings.tcl").write_text(
            "set board kr260\nset part %s\nset base 0x%08X\nset jobs %d\nset sfp 0\nset ddr 1\n"
            % (self.PART, self.BASE, self.jobs))
        mapping = {name: {"offset": offset} for name, offset in OFFSETS.items()}
        mapping["id"]["value"] = 0x4F544432
        mapping["version"]["value"] = 1
        mapping["latency"]["value"] = report["latency_cycles"]
        mapping["order"]["words"] = 12
        manifest = {"kind": "ddr_trading", "board": "kr260", "clock_mhz": 322.265625,
                    "registers": {"base": self.BASE, "span": self.SPAN, "map": mapping},
                    "ddr": {"address_bits": 40, "base_alignment": 256, "record_bytes": 16,
                            "mapped_base": 0, "mapped_bytes": 0x80000000},
                    "replay_report_sha256": hashlib.sha256((self.out / "report.json").read_bytes()).hexdigest()}
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return self.build / "build.tcl"
