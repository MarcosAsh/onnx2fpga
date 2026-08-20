"""Runs the test suite in a shuffled order.

Tests that pass only in the order they were written are a slow leak: one of
them is leaving state behind. Shuffling with a recorded seed turns that from
something noticed months later into something a run fails on, reproducibly.
"""

import argparse
import random
import sys
import unittest

from support import ROOT  # noqa: F401  sets up sys.path


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--verbosity", type=int, default=1)
    args = parser.parse_args(argv)

    discovered = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), pattern=args.pattern)
    tests = list(flatten(discovered))
    random.Random(args.seed).shuffle(tests)

    runner = unittest.TextTestRunner(verbosity=args.verbosity)
    result = runner.run(unittest.TestSuite(tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
