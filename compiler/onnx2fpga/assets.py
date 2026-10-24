"""Finds the hand written RTL and C++ runtime in a checkout or an installed wheel; OTF_ROOT overrides."""

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
        """An explicit override is the only candidate, so a wrong one fails instead of silently using other RTL."""
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

    @property
    def root(self):
        if self._root is None:
            for candidate in self.candidates():
                if all((candidate / marker).is_file() for marker in self.MARKERS):
                    self._root = candidate
                    break
            else:
                raise AssetError("cannot find hardware/ and runtime/ in %s; set %s"
                                 % (", ".join(map(str, self.candidates())), self.ENV))
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
