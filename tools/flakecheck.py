"""Runs the test suite repeatedly, shuffled, to catch flaky tests.

Two things make a suite unreliable, and both hide in a single ordered run.
Tests that depend on the order they run in, and code that depends on Python's
per process string hash randomisation. This varies both and fails if any run
disagrees with the others.

    python tools/flakecheck.py                       everything, three times
    python tools/flakecheck.py --repeat 10 --fast    unit tests only, ten times
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "shuffled_runner.py"

FAST = ("test_ingest.py", "test_devices.py", "test_numerics.py", "test_pipeline.py",
        "test_latency.py", "test_refold.py", "test_accuracy.py", "test_rejections.py", "test_estimate.py", "test_worked_example.py", "test_api.py", "test_reweight.py", "test_determinism.py", "test_quantized_import.py",
        "test_characterize.py", "test_synth_script.py")


class Run:
    def __init__(self, index, seed, pattern, ok, seconds, output):
        self.index = index
        self.seed = seed
        self.pattern = pattern
        self.ok = ok
        self.seconds = seconds
        self.output = output

    @property
    def reproduce(self):
        return ("PYTHONHASHSEED=%d PYTHONPATH=compiler:tests:examples "
                "python3 tests/shuffled_runner.py --seed %d --pattern '%s'"
                % (self.seed, self.seed, self.pattern))


class FlakeCheck:
    def __init__(self, repeat=3, patterns=("test_*.py",), verbosity=0):
        self.repeat = repeat
        self.patterns = tuple(patterns)
        self.verbosity = verbosity

    def environment(self, seed):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = str(seed)
        environment["PYTHONPATH"] = "compiler:tests:examples"
        return environment

    def once(self, index, seed, pattern):
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(RUNNER), "--seed", str(seed),
             "--pattern", pattern, "--verbosity", str(self.verbosity)],
            cwd=ROOT, env=self.environment(seed), capture_output=True, text=True)
        return Run(index, seed, pattern, completed.returncode == 0,
                   time.perf_counter() - started,
                   completed.stdout + completed.stderr)

    def run(self):
        runs = []
        for index in range(self.repeat):
            for pattern in self.patterns:
                seed = 1000 + index
                run = self.once(index, seed, pattern)
                runs.append(run)
                print("run %d  seed %-5d %-24s %-6s %6.1fs"
                      % (index + 1, seed, pattern,
                         "ok" if run.ok else "FAILED", run.seconds))
                if not run.ok:
                    print(run.output[-4000:])
        return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--fast", action="store_true",
                        help="skip the Verilator suites, which dominate the clock")
    parser.add_argument("--pattern", default=None)
    args = parser.parse_args(argv)

    if args.pattern:
        patterns = (args.pattern,)
    elif args.fast:
        patterns = FAST
    else:
        patterns = ("test_*.py",)

    checker = FlakeCheck(args.repeat, patterns)
    runs = checker.run()
    failures = [run for run in runs if not run.ok]

    print()
    if failures:
        print("%d of %d runs failed. Reproduce the first with:"
              % (len(failures), len(runs)))
        print("  " + failures[0].reproduce)
        return 1
    print("%d runs, %d orderings, all agreed" % (len(runs), args.repeat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
