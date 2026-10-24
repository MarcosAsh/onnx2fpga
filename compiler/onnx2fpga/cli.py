"""Command line front end; compile, simulate and bitstream are thin layers over onnx2fpga.api."""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

from .api import load
from .board import IcepiZero
from .doctor import Doctor
from .errors import Onnx2FpgaError, SimulationError, ToolchainError
from .measure import (Characterizer, SynthesisResult, VivadoRunner, calibration_samples,
                      open_flow, render)
from .p1_ingest import OnnxModel
from .p2_graph.onnx_importer import OnnxImporter
from .p6_emit.hardcaml import HardcamlError
from .p6_emit.project import ProjectWriter
from .targets.device import Device


def add_hdl_option(parser):
    parser.add_argument("--hdl", default="sv", choices=ProjectWriter.HDLS,
                        help="sv instantiates the hand written library; hardcaml "
                             "elaborates the same units in Hardcaml, which needs "
                             "dune and the hardcaml opam switch")


def add_model_options(parser):
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--device", default="vu9p", choices=Device.names())
    parser.add_argument("--target-cycles", type=int, default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")


def random_calibration(acc, args):
    samples = calibration_samples(acc.model, args.samples, args.seed)
    return np.concatenate([next(iter(feed.values())) for feed in samples])


def print_folding(acc):
    print()
    print(acc.result.folding.render())
    print()


def build(args, default_out):
    """A Hardcaml build gets its own default directory so backends never mix sources."""
    acc = load(args.model)
    hdl = getattr(args, "hdl", "sv")
    if hdl != "sv":
        default_out = "%s_%s" % (default_out, hdl)
    out = args.out or pathlib.Path("build") / default_out
    acc.compile(device=args.device, target_cycles=args.target_cycles, hdl=hdl,
                out_dir=out, verbose=not args.quiet, strict=False,
                calibration=random_calibration(acc, args))
    return acc, out


def run_compile(args):
    """strict=False because an unmet target is reported in the log and the balanced design written anyway."""
    acc = load(args.model)
    calibration = (np.load(args.calibration) if args.calibration
                   else random_calibration(acc, args))
    acc.compile(device=args.device, target_cycles=args.target_cycles,
                utilisation=args.utilisation, mult_bits=args.mult_bits,
                top=args.top, hdl=args.hdl, out_dir=args.out,
                calibration=calibration, verbose=not args.quiet, strict=False)
    if not args.quiet:
        print_folding(acc)
        print("wrote %s" % args.out)
        print("run   cd %s && make run" % args.out)
    return 0


def run_simulate(args):
    acc, _ = build(args, args.model.stem)
    if not args.quiet:
        print_folding(acc)
    try:
        report = acc.simulate(sweep=args.sweep)
    except SimulationError as error:
        print(error.stdout[-4000:], file=sys.stderr)
        print(error.stderr[-4000:], file=sys.stderr)
        return 1
    print(report.first.output.strip())
    if not report.first.ok:
        return 1
    if not args.sweep:
        return 0
    print("\nbackpressure sweep")
    for entry in report.sweep:
        print(entry.render())
    failed = [entry for entry in report.sweep if not entry.ok]
    print("\n%d of %d settings passed" % (len(report.sweep) - len(failed),
                                          len(report.sweep)))
    return 1 if failed else 0


def run_synth(args):
    args.quiet = True
    device = Device.get(args.device)
    if device.nextpnr_args:
        args.hdl = "hardcaml"
        acc, out = build(args, args.model.stem + "_synth")
        return open_flow(acc.result, out, device, args.impl, args.dry_run)
    acc, out = build(args, args.model.stem + "_synth")
    period = args.period or round(1000.0 / device.fmax_mhz, 3)
    runner = VivadoRunner(args.part or device.part, period, args.impl,
                          executable=args.vivado)
    driver = Characterizer(runner, device)
    subjects = driver.subjects(acc.result.hardware_graph)

    if args.dry_run:
        print("%d units to characterise on %s\n" % (len(subjects), runner.part))
        for subject in subjects:
            print(driver.describe(subject, out))
            print()
        return 0
    if not runner.available:
        print("cannot find %r; source settings64.sh or pass --vivado" % runner.executable,
              file=sys.stderr)
        return 2

    comparisons = driver.characterize(subjects, out)
    print(render(comparisons))
    if args.measurements:
        args.measurements.write_text(json.dumps(
            [{"unit": c.subject.name, "module": c.subject.module,
              "generics": c.subject.generics, "cycles": c.subject.cycles,
              "predicted": {f: getattr(c.subject.predicted, f)
                            for f in SynthesisResult.FIELDS},
              "measured": c.result.payload}
             for c in comparisons], indent=2) + "\n")
    return 0


def run_bitstream(args):
    options = {"rev": args.rev, "seeds": [int(s) for s in args.seeds.split()],
               "uart": args.uart}
    if args.mhz:
        options["mhz"] = args.mhz
    if args.baud:
        options["baud"] = args.baud
    args.hdl = "hardcaml"
    args.out = args.out or pathlib.Path("build") / ("%s_%s" % (args.model.stem,
                                                               args.board))
    board = IcepiZero(args.out, **options)
    acc, _ = build(args, args.model.stem)
    if not args.quiet:
        print_folding(acc)
        print("running yosys, nextpnr-ecp5 and ecppack in %s" % board.build)
    acc.build(board=board)
    print(board.render_report())
    print()
    print("bitstream %s" % board.bitstream)
    print("load      make -C %s prog" % board.build)
    print("run       %s" % board.host_command())
    return 0


def run_devices(args):
    print("%-8s %-26s %9s %9s %13s %7s  %s"
          % ("name", "part", "lut", "dsp", "block ram", "source", "notes"))
    for name in Device.names():
        device = Device.get(name)
        ram = "%d %s" % (getattr(device.budget, device.block_ram), device.block_ram)
        print("%-8s %-26s %9d %9d %13s %7s  %s"
              % (name, device.part, device.budget.lut, device.budget.dsp,
                 ram, device.provenance, device.notes))
    return 0


def run_inspect(args):
    model = OnnxModel.load(args.model)
    print("producer %s %s, opset %d, ir_version %d"
          % (model.proto.producer_name, model.proto.producer_version,
             model.opset, model.proto.ir_version))
    for name in model.graph_inputs():
        print("input  %-24s %s" % (name, model.value_shape(name)))
    for name in model.graph_outputs():
        print("output %-24s %s" % (name, model.value_shape(name)))
    print()
    print(OnnxImporter(model).run().summary())
    return 0


def parser():
    top = argparse.ArgumentParser(
        prog="onnx2fpga",
        description="Compile ONNX models into streaming dataflow FPGA designs.")
    commands = top.add_subparsers(dest="command", required=True)

    def command(name, help, run):
        sub = commands.add_parser(name, help=help)
        sub.set_defaults(run=run)
        return sub

    sub = command("doctor", "report what this machine can compile, simulate and measure",
                  lambda args: Doctor(args.vivado).report())
    sub.add_argument("--vivado", default=os.environ.get("VIVADO", "vivado"))

    sub = command("inspect", "print what the compiler sees in an ONNX file", run_inspect)
    sub.add_argument("model", type=pathlib.Path)

    sub = command("compile", "compile an ONNX model into a build directory", run_compile)
    sub.add_argument("model", type=pathlib.Path)
    sub.add_argument("--out", type=pathlib.Path, default=pathlib.Path("build/out"))
    sub.add_argument("--device", default="vu9p", choices=Device.names())
    sub.add_argument("--target-cycles", type=int, default=None,
                     help="stop folding once every stage is this fast")
    sub.add_argument("--utilisation", type=float, default=0.80,
                     help="fraction of the device the design may occupy")
    sub.add_argument("--mult-bits", type=int, default=18,
                     help="width of the requantisation multiplier")
    sub.add_argument("--calibration", type=pathlib.Path, default=None,
                     help="npy file of calibration samples; random if omitted")
    sub.add_argument("--samples", type=int, default=32)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--top", default="otf_top")
    add_hdl_option(sub)
    sub.add_argument("--quiet", action="store_true")

    sub = command("simulate",
                  "compile a model and check the generated RTL against the golden model",
                  run_simulate)
    add_model_options(sub)
    add_hdl_option(sub)
    sub.add_argument("--sweep", action="store_true",
                     help="also run a randomised backpressure sweep, which "
                          "is where stream designs actually fail")

    sub = command("synth",
                  "synthesise each unit and compare measured cost against the model",
                  run_synth)
    add_model_options(sub)
    sub.add_argument("--part", default=None, help="override the device's part")
    sub.add_argument("--period", type=float, default=None,
                     help="target clock period in ns, the device's clock by default")
    sub.add_argument("--impl", action="store_true",
                     help="place and route as well, which is slower and true")
    sub.add_argument("--dry-run", action="store_true",
                     help="print the commands instead of running them")
    sub.add_argument("--vivado", default=os.environ.get("VIVADO", "vivado"),
                     help="path to vivado, which is usually not on PATH "
                          "until settings64.sh has been sourced")
    sub.add_argument("--measurements", type=pathlib.Path, default=None,
                     help="write the measurements as json")

    command("devices", "list the device models the scheduler can budget against",
            run_devices)

    sub = command("bitstream",
                  "compile a model and build a bitstream that runs it on a board",
                  run_bitstream)
    add_model_options(sub)
    sub.add_argument("--board", default=IcepiZero.NAME, choices=(IcepiZero.NAME,))
    sub.add_argument("--rev", default="v1.4", help="board revision, from the silkscreen")
    sub.add_argument("--mhz", type=int, default=None,
                     help="clock from the PLL, a divisor of 480 (default 20)")
    sub.add_argument("--baud", type=int, default=None,
                     help="serial rate to the host (default 1000000)")
    sub.add_argument("--seeds", default="1 2 3 4 5 6 7 8",
                     help="placement seeds to try until one meets timing")
    sub.add_argument("--uart", default="usb", choices=("usb", "header"),
                     help="usb: the board's own USB serial port; header: pins "
                          "8 (TX) and 10 (RX) for an external adapter")
    sub.set_defaults(device="icepi25")
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except HardcamlError as error:
        print("%s (run `onnx2fpga doctor`, or use --hdl sv)" % error, file=sys.stderr)
        return 2
    except Onnx2FpgaError as error:
        print(error, file=sys.stderr)
        return 2 if isinstance(error, ToolchainError) else 1
