"""Runs the test suite repeatedly, shuffled, to catch order dependent and hash seed dependent tests."""

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "shuffled_runner.py"

FAST = ("test_ingest.py", "test_numerics.py", "test_pipeline.py",
        "test_determinism.py", "test_quantized_import.py",
        "test_characterize.py", "test_synth_script.py", "test_hardcaml_backend.py")


def run_once(seed, pattern):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH="compiler:tests:examples")
    started = time.perf_counter()
    done = subprocess.run(
        [sys.executable, str(RUNNER), "--seed", str(seed), "--pattern", pattern,
         "--verbosity", "0"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    return done.returncode == 0, time.perf_counter() - started, done.stdout + done.stderr


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

    failed = []
    for index in range(args.repeat):
        seed = 1000 + index
        for pattern in patterns:
            ok, seconds, output = run_once(seed, pattern)
            print("run %d  seed %-5d %-24s %-6s %6.1fs"
                  % (index + 1, seed, pattern, "ok" if ok else "FAILED", seconds))
            if not ok:
                print(output[-4000:])
                failed.append((seed, pattern))

    total = args.repeat * len(patterns)
    print()
    if failed:
        seed, pattern = failed[0]
        print("%d of %d runs failed. Reproduce the first with:" % (len(failed), total))
        print("  PYTHONHASHSEED=%d PYTHONPATH=compiler:tests:examples "
              "python3 tests/shuffled_runner.py --seed %d --pattern '%s'"
              % (seed, seed, pattern))
        return 1
    print("%d runs, %d orderings, all agreed" % (total, args.repeat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
