"""Builds a wheel using setuptools' own PEP 517 backend.

`python -m build` is the usual frontend but it is a separate package that is
often absent. The backend that does the work ships with setuptools, so calling
it directly needs nothing extra.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
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
