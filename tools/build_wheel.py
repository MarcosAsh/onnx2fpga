"""Builds a wheel using setuptools' own PEP 517 backend.

`python -m build` is the usual frontend but it is a separate package that is
often absent. The backend that does the work ships with setuptools, so calling
it directly needs nothing extra.

The RTL library and the C++ runtime live at the top level, which is where a
checkout finds them. A wheel has to carry its own copy, so they are staged
inside the package first. That happens here rather than in the Makefile so
that building a wheel is correct on its own: staging it separately made a
clean checkout produce a wheel with no hardware in it.
"""

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "compiler" / "onnx2fpga" / "assets"

# Build products, not sources. `runtime/bin` and the object files beside it are
# whatever the last `make runtime` left behind.
LEAVINGS = shutil.ignore_patterns("bin", "obj_dir", "__pycache__",
                                  "*.o", "*.d", "*.a")


def stage():
    """Copy hardware/ and runtime/ into the package."""
    shutil.rmtree(ASSETS, ignore_errors=True)
    (ASSETS / "hardware").mkdir(parents=True)
    for name in ("rtl", "synth"):
        shutil.copytree(ROOT / "hardware" / name, ASSETS / "hardware" / name)
    shutil.copytree(ROOT / "runtime", ASSETS / "runtime", ignore=LEAVINGS)
    staged = sum(1 for path in ASSETS.rglob("*") if path.is_file())
    print("staged %d files" % staged)
    return staged


def main():
    stage()
    if "--stage-only" in sys.argv:
        return 0

    sys.path.insert(0, str(ROOT))
    from setuptools import build_meta

    target = ROOT / "dist"
    target.mkdir(exist_ok=True)
    name = build_meta.build_wheel(str(target))
    built = target / name
    print("built %s (%.1f KB)" % (built, built.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
