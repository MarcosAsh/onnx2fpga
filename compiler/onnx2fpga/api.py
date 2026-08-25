"""The importable form of the command line tool.

Every capability here existed already, behind a module path a caller had to
know: onnx2fpga.compile.Compiler, onnx2fpga.estimate.Estimate, and a
SimulationRunner that had to be pointed at a build directory the caller was
responsible for producing. That is a fine shape for the CLI to be built on and
a poor one to hand somebody in a notebook.

These are functions that take a model and return a value. Nothing prints,
nothing shells out on the caller's behalf, and everything returned is the same
object the CLI renders, so a notebook can ask a further question of it rather
than parsing text back out.
"""

import pathlib

from .compile import Compiler
from .measure import calibration_samples
from .p1_ingest import OnnxModel
from .simulate import SimulationRunner
from .targets.device import Device


def load(model):
    """An OnnxModel from whatever the caller has: a path, a string, or a model
    that was already loaded or built in memory."""
    if isinstance(model, OnnxModel):
        return model
    if isinstance(model, (str, pathlib.Path)):
        return OnnxModel.load(model)
    raise TypeError("expected a path or an OnnxModel, got %s"
                    % type(model).__name__)


def samples_for(model, count=16, seed=0):
    """Random inputs of the right shape, for when there is no real data yet.

    Enough to calibrate a shape check against and not enough to claim anything
    about accuracy; the accuracy report says what it was measured on for that
    reason.
    """
    return calibration_samples(load(model), count, seed)


def compile(model, samples=None, out_dir="build/out", **options):
    """Compile a model and write a build directory. Returns the result, which
    carries the graphs, the folding report and the accuracy report."""
    model = load(model)
    return Compiler(**options).compile(model, samples, out_dir)


def estimate(model, samples=None, **options):
    """Will it fit, and how fast, without writing anything."""
    model = load(model)
    return Compiler(**options).estimate(
        model, samples if samples is not None else samples_for(model, 8))


def simulate(model, samples=None, out_dir="build/sim", duty=(100, 100),
             seed=1, **options):
    """Compile, build with Verilator, and run one frame against the golden
    vectors. Returns the SimulationResult; `.ok` is whether it matched.

    Verilator is a separate program, so this does start one, but the caller
    does not have to: no build directory to prepare, no make to invoke, no
    output to parse.
    """
    model = load(model)
    if samples is None:
        samples = samples_for(model, 8)
    compile(model, samples, out_dir, **options)
    runner = SimulationRunner(out_dir, top=options.get("top_name", "otf_top"))
    built = runner.build()
    if built.returncode != 0:
        raise RuntimeError("verilator build failed:\n%s" % built.stderr[-4000:])
    return runner.once(duty=duty, seed=seed, quiet=True)


def devices():
    """The device models the scheduler can budget against."""
    return Device.names()


def device(name):
    return Device.get(name)
