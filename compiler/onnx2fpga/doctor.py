"""Reports what this machine can do, and what each missing piece would unlock.

The tool degrades in stages rather than all at once: compiling needs nothing
but Python and numpy, simulating needs Verilator and a C++ compiler, and
measuring real resources needs Vivado. This says which stage you are at.
"""

import os
import pathlib
import shutil
import subprocess

from .assets import AssetError, AssetLocator

# Vivado still links against the ncurses 5 ABI, which Ubuntu dropped after
# 24.04. Symlinking the 6 ABI over it is the standard workaround.
LEGACY_NCURSES = "libtinfo.so.5"
LIB_DIR = pathlib.Path("/usr/lib/x86_64-linux-gnu")

VIVADO_ROOTS = ("/opt/Xilinx", "/tools/Xilinx", "~/Xilinx", "~/xilinx",
                "/usr/local/Xilinx")


class Check:
    def __init__(self, name, ok, detail, unlocks=None, remedy=None, required=True):
        self.name = name
        self.ok = ok
        self.detail = detail
        self.unlocks = unlocks
        self.remedy = remedy
        self.required = required

    def render(self):
        mark = "ok  " if self.ok else ("MISS" if self.required else "----")
        line = "  %s  %-16s %s" % (mark, self.name, self.detail)
        if not self.ok and self.unlocks:
            line += "\n        needed for: " + self.unlocks
        if not self.ok and self.remedy:
            line += "\n        fix: " + self.remedy
        return line


class Doctor:
    def __init__(self, vivado="vivado"):
        self.vivado = vivado

    def _tool(self, executable, name, unlocks, remedy, required=True, version=None):
        found = shutil.which(executable)
        detail = found or "not on PATH"
        if found and version:
            detail = "%s  (%s)" % (found, version(found))
        return Check(name, bool(found), detail, unlocks, remedy, required)

    @staticmethod
    def _first_line(path, *args):
        try:
            out = subprocess.run([path] + list(args), capture_output=True,
                                 text=True, timeout=30)
            return (out.stdout + out.stderr).strip().splitlines()[0]
        except (OSError, IndexError, subprocess.SubprocessError):
            return "version unknown"

    def core(self):
        checks = []
        try:
            import numpy
            checks.append(Check("numpy", True, numpy.__version__))
        except ImportError:
            checks.append(Check("numpy", False, "not importable",
                                "everything", "pip install numpy"))
        try:
            root = AssetLocator().root
            checks.append(Check("rtl and runtime", True, str(root)))
        except AssetError as error:
            checks.append(Check("rtl and runtime", False, str(error).splitlines()[0],
                                "emitting a build directory",
                                "set OTF_ROOT to the checkout"))
        return checks

    def simulation(self):
        return [
            self._tool("verilator", "verilator", "simulating a compiled design",
                       "sudo apt install verilator  (version 5 or newer)",
                       version=lambda p: self._first_line(p, "--version")),
            self._tool("g++", "c++ compiler", "the simulation harness and kernels",
                       "sudo apt install g++",
                       version=lambda p: self._first_line(p, "--version")),
            self._tool("make", "make", "building a generated project",
                       "sudo apt install make"),
            self._tool("tclsh", "tclsh", "exercising the synthesis script offline",
                       "sudo apt install tcl", required=False),
        ]

    def synthesis(self):
        checks = [self._tool(self.vivado, "vivado",
                             "measured resources and timing",
                             "free ML Standard edition from AMD",
                             required=False,
                             version=lambda p: self._first_line(p, "-version"))]
        if not shutil.which(self.vivado):
            checks.extend(self._vivado_prerequisites())
        return checks

    def _vivado_prerequisites(self):
        checks = []
        target = LIB_DIR / LEGACY_NCURSES
        if LIB_DIR.is_dir() and not target.exists():
            modern = sorted(LIB_DIR.glob("libtinfo.so.6*"))
            remedy = ("sudo ln -sf %s %s" % (modern[0], target) if modern
                      else "install libtinfo6 then symlink it to .so.5")
            checks.append(Check(LEGACY_NCURSES, False,
                                "absent; Vivado links the ncurses 5 ABI",
                                "installing Vivado on Ubuntu 24.04 or newer",
                                remedy, required=False))
        found = self.installed_vivados()
        if found:
            checks.append(Check("vivado on disk", True, str(found[0])))
        return checks

    def installed_vivados(self):
        found = []
        for root in VIVADO_ROOTS:
            base = pathlib.Path(os.path.expanduser(root)) / "Vivado"
            if not base.is_dir():
                continue
            for release in sorted(base.iterdir()):
                binary = release / "bin" / "vivado"
                if binary.exists():
                    found.append(binary)
        return found

    def report(self):
        sections = (("compiling", self.core()),
                    ("simulating", self.simulation()),
                    ("measuring", self.synthesis()))
        blocking = 0
        for title, checks in sections:
            print(title)
            for check in checks:
                print(check.render())
                if check.required and not check.ok:
                    blocking += 1
            print()

        found = self.installed_vivados()
        if found and not shutil.which(self.vivado):
            print("vivado is installed but not on PATH. Either:")
            print("  source %s/settings64.sh" % found[0].parents[1])
            print("  onnx2fpga synth model.onnx --vivado %s" % found[0])
        return 1 if blocking else 0
