"""Finds the hand written RTL and the C++ runtime, whichever layout we are in.

The compiler emits a build directory that references both: it copies the module
library in, and the generated Makefile points at the simulation harness. Where
those live depends on how the tool was obtained. Running from a checkout they
sit beside the package; installed from a wheel they are inside it. Setting
OTF_ROOT overrides both, which is what a packager or a test needs.
"""

import os
import pathlib


class AssetError(Exception):
    pass


class AssetLocator:
    ENV = "OTF_ROOT"
    RTL = pathlib.Path("hardware") / "rtl"
    RUNTIME = pathlib.Path("runtime")
    SYNTH = pathlib.Path("hardware") / "synth" / "synth_unit.tcl"
    MARKERS = (RTL / "otf_mvau.sv", RUNTIME / "sim" / "harness.cpp")

    def __init__(self, override=None):
        self._override = pathlib.Path(override) if override else None
        self._root = None

    def candidates(self):
        """An explicit override, from the constructor or the environment, is
        the only candidate. Falling back from one that was set on purpose but
        points somewhere wrong would hide the mistake behind a build that
        quietly used different RTL."""
        if self._override is not None:
            yield self._override
            return
        environment = os.environ.get(self.ENV)
        if environment:
            yield pathlib.Path(environment)
            return
        here = pathlib.Path(__file__).resolve().parent
        yield here / "assets"          # installed from a wheel
        yield here.parents[1]          # running from a checkout

    @staticmethod
    def holds_assets(root):
        return all((root / marker).is_file() for marker in AssetLocator.MARKERS)

    @property
    def root(self):
        if self._root is None:
            for candidate in self.candidates():
                if self.holds_assets(candidate):
                    self._root = candidate
                    break
            else:
                raise AssetError(
                    "cannot find the RTL library and C++ runtime. Looked in %s. "
                    "Set %s to the directory holding hardware/ and runtime/."
                    % (", ".join(str(c) for c in self.candidates()), self.ENV))
        return self._root

    @property
    def rtl_dir(self):
        return self.root / self.RTL

    @property
    def runtime_dir(self):
        return self.root / self.RUNTIME

    @property
    def synth_script(self):
        return self.root / self.SYNTH

    def rtl_sources(self):
        return sorted(self.rtl_dir.glob("otf_*.sv"))

    def __repr__(self):
        try:
            return "AssetLocator(%s)" % self.root
        except AssetError:
            return "AssetLocator(unresolved)"
