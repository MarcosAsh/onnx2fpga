"""Command line front end; compile, simulate and bitstream are thin layers over onnx2fpga.api."""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

from .api import load
from .board import BOARDS, BoardError
from .doctor import Doctor
from .errors import Onnx2FpgaError, SimulationError, ToolchainError
from .measure import Characterizer, VivadoRunner, calibration_samples, open_flow, render
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


class Command:
    name = "command"
    help = ""

    def configure(self, parser):
        pass

    def run(self, args):
        raise NotImplementedError


class CompileCommand(Command):
    name = "compile"
    help = "compile an ONNX model into a build directory"

    def configure(self, parser):
        parser.add_argument("model", type=pathlib.Path)
        parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("build/out"))
        parser.add_argument("--device", default="vu9p", choices=Device.names())
        parser.add_argument("--target-cycles", type=int, default=None,
                            help="stop folding once every stage is this fast")
        parser.add_argument("--utilisation", type=float, default=0.80,
                            help="fraction of the device the design may occupy")
        parser.add_argument("--mult-bits", type=int, default=18,
                            help="width of the requantisation multiplier")
        parser.add_argument("--calibration", type=pathlib.Path, default=None,
                            help="npy file of calibration samples; random if omitted")
        parser.add_argument("--samples", type=int, default=32)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--top", default="otf_top")
        add_hdl_option(parser)
        parser.add_argument("--quiet", action="store_true")

    def run(self, args):
        """strict=False because an unmet target is reported in the log and the balanced design written anyway."""
        acc = load(args.model)
        acc.compile(device=args.device, target_cycles=args.target_cycles,
                    utilisation=args.utilisation, mult_bits=args.mult_bits,
                    top=args.top, hdl=args.hdl, out_dir=args.out,
                    calibration=batch(self._samples(acc.model, args)),
                    verbose=not args.quiet, strict=False)
        if not args.quiet:
            print()
            print(acc.result.folding.render())
            print()
            print("wrote %s" % args.out)
            print("run   cd %s && make run" % args.out)
        return 0

    @staticmethod
    def _samples(model, args):
        if args.calibration:
            name = model.graph_inputs()[0]
            data = np.load(args.calibration)
            return [{name: sample[None, ...]} for sample in data]
        return calibration_samples(model, args.samples, args.seed)


class ModelCommand(Command):
    """Shared options for anything that starts from an ONNX file."""

    def configure(self, parser):
        parser.add_argument("model", type=pathlib.Path)
        parser.add_argument("--out", type=pathlib.Path, default=None)
        parser.add_argument("--device", default="vu9p", choices=Device.names())
        parser.add_argument("--target-cycles", type=int, default=None)
        parser.add_argument("--samples", type=int, default=32)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--quiet", action="store_true")

    def build(self, args, default_out):
        """A Hardcaml build gets its own default directory so backends never mix sources."""
        acc = load(args.model)
        hdl = getattr(args, "hdl", "sv")
        if hdl != "sv":
            default_out = "%s_%s" % (default_out, hdl)
        out = args.out or pathlib.Path("build") / default_out
        acc.compile(device=args.device, target_cycles=args.target_cycles, hdl=hdl,
                    out_dir=out, verbose=not args.quiet, strict=False,
                    calibration=batch(calibration_samples(acc.model, args.samples,
                                                          args.seed)))
        return acc, out


class SimulateCommand(ModelCommand):
    name = "simulate"
    help = "compile a model and check the generated RTL against the golden model"

    def configure(self, parser):
        ModelCommand.configure(self, parser)
        add_hdl_option(parser)
        parser.add_argument("--sweep", action="store_true",
                            help="also run a randomised backpressure sweep, which "
                                 "is where stream designs actually fail")

    def run(self, args):
        acc, out = self.build(args, args.model.stem)
        if not args.quiet:
            print()
            print(acc.result.folding.render())
            print()

        try:
            report = acc.simulate(sweep=args.sweep)
        except SimulationError as error:
            print(error.stdout[-4000:], file=sys.stderr)
            print(error.stderr[-4000:], file=sys.stderr)
            return 1

        first = report.first
        print(first.output.strip())
        if not first.ok:
            return 1
        if not args.sweep:
            return 0

        print("\nbackpressure sweep")
        results = report.sweep
        for entry in results:
            print(entry.render())
        failed = [entry for entry in results if not entry.ok]
        print("\n%d of %d settings passed" % (len(results) - len(failed), len(results)))
        return 1 if failed else 0


class SynthCommand(ModelCommand):
    name = "synth"
    help = "synthesise each unit and compare measured cost against the model"

    def configure(self, parser):
        ModelCommand.configure(self, parser)
        parser.add_argument("--part", default=None, help="override the device's part")
        parser.add_argument("--period", type=float, default=5.0,
                            help="target clock period in ns")
        parser.add_argument("--impl", action="store_true",
                            help="place and route as well, which is slower and true")
        parser.add_argument("--dry-run", action="store_true",
                            help="print the commands instead of running them")
        parser.add_argument("--vivado", default=os.environ.get("VIVADO", "vivado"),
                            help="path to vivado, which is usually not on PATH "
                                 "until settings64.sh has been sourced")
        parser.add_argument("--measurements", type=pathlib.Path, default=None,
                            help="write the measurements as json")

    def run(self, args):
        args.quiet = True
        device = Device.get(args.device)
        if device.nextpnr_args:
            args.hdl = "hardcaml"
            acc, out = self.build(args, args.model.stem + "_synth")
            return open_flow(acc.result, out, device, args.impl, args.dry_run)
        acc, out = self.build(args, args.model.stem + "_synth")
        result = acc.result
        runner = VivadoRunner(args.part or device.part, args.period, args.impl,
                              executable=args.vivado)
        driver = Characterizer(runner, device)
        subjects = driver.subjects(result.hardware_graph)

        if args.dry_run:
            print("%d units to characterise on %s\n" % (len(subjects), runner.part))
            for subject in subjects:
                print(driver.describe(subject, out))
                print()
            return 0

        if not runner.available:
            print("cannot find %r. Source settings64.sh, pass --vivado with the "
                  "full path, or use --dry-run. Run `onnx2fpga doctor` for more."
                  % runner.executable, file=sys.stderr)
            return 2

        comparisons = driver.characterize(subjects, out)
        print(render(comparisons))
        if args.measurements:
            args.measurements.write_text(json.dumps(
                [{"unit": c.subject.name, "module": c.subject.module,
                  "generics": c.subject.generics, "cycles": c.subject.cycles,
                  "predicted": {f: getattr(c.subject.predicted, f)
                                for f in SYNTH_FIELDS},
                  "measured": c.result.payload}
                 for c in comparisons], indent=2) + "\n")
        return 0


class BitstreamCommand(ModelCommand):
    name = "bitstream"
    help = "compile a model and build a bitstream that runs it on a board"

    def configure(self, parser):
        ModelCommand.configure(self, parser)
        parser.add_argument("--board", default="icepi_zero", choices=sorted(BOARDS))
        parser.add_argument("--rev", default="v1.4",
                            help="board revision, from the silkscreen")
        parser.add_argument("--mhz", type=int, default=None,
                            help="clock from the PLL, a divisor of 480 (default 20)")
        parser.add_argument("--baud", type=int, default=None,
                            help="serial rate to the host (default 1000000)")
        parser.add_argument("--seeds", default="1 2 3 4 5 6 7 8",
                            help="placement seeds to try until one meets timing")
        parser.add_argument("--uart", default="usb", choices=("usb", "header"),
                            help="usb: the board's own USB serial port; header: pins "
                                 "8 (TX) and 10 (RX) for an external adapter")
        parser.set_defaults(device="icepi25")

    def run(self, args):
        options = {"rev": args.rev, "seeds": [int(s) for s in args.seeds.split()],
                   "uart": args.uart}
        if args.mhz:
            options["mhz"] = args.mhz
        if args.baud:
            options["baud"] = args.baud
        args.hdl = "hardcaml"
        args.out = args.out or pathlib.Path("build") / ("%s_%s" % (args.model.stem,
                                                                   args.board))
        board = BOARDS[args.board](args.out, **options)
        acc, _ = self.build(args, args.model.stem)
        if not args.quiet:
            print()
            print(acc.result.folding.render())
            print()
            print("running yosys, nextpnr-ecp5 and ecppack in %s" % board.build)
        acc.build(board=board)
        print(board.render_report())
        print()
        print("bitstream %s" % board.bitstream)
        print("load      make -C %s prog" % board.build)
        print("run       %s" % board.host_command())
        return 0


class DoctorCommand(Command):
    name = "doctor"
    help = "report what this machine can compile, simulate and measure"

    def configure(self, parser):
        parser.add_argument("--vivado", default=os.environ.get("VIVADO", "vivado"))

    def run(self, args):
        return Doctor(args.vivado).report()


class DevicesCommand(Command):
    name = "devices"
    help = "list the device models the scheduler can budget against"

    def run(self, args):
        print("%-8s %-26s %9s %9s %13s %7s  %s"
              % ("name", "part", "lut", "dsp", "block ram", "source", "notes"))
        for name in Device.names():
            device = Device.get(name)
            ram = "%d %s" % (getattr(device.budget, device.block_ram), device.block_ram)
            print("%-8s %-26s %9d %9d %13s %7s  %s"
                  % (name, device.part, device.budget.lut, device.budget.dsp,
                     ram, device.provenance, device.notes))
        return 0


class InspectCommand(Command):
    name = "inspect"
    help = "print what the compiler sees in an ONNX file"

    def configure(self, parser):
        parser.add_argument("model", type=pathlib.Path)

    def run(self, args):
        model = OnnxModel.load(args.model)
        print("producer %s %s, opset %d, ir_version %d"
              % (model.proto.producer_name, model.proto.producer_version,
                 model.opset, model.proto.ir_version))
        for name in model.graph_inputs():
            print("input  %-24s %s" % (name, model.value_shape(name)))
        for name in model.graph_outputs():
            print("output %-24s %s" % (name, model.value_shape(name)))
        print()
        graph = OnnxImporter(model).run()
        print(graph.summary())
        return 0


class CommandLine:
    COMMANDS = (DoctorCommand(), InspectCommand(), CompileCommand(),
                SimulateCommand(), SynthCommand(), DevicesCommand(),
                BitstreamCommand())

    def build_parser(self):
        parser = argparse.ArgumentParser(
            prog="onnx2fpga",
            description="Compile ONNX models into streaming dataflow FPGA designs.")
        subparsers = parser.add_subparsers(dest="command", required=True)
        for command in self.COMMANDS:
            sub = subparsers.add_parser(command.name, help=command.help)
            command.configure(sub)
            sub.set_defaults(handler=command)
        return parser

    def main(self, argv=None):
        args = self.build_parser().parse_args(argv)
        try:
            return args.handler.run(args)
        except HardcamlError as error:
            print("%s\nRun `onnx2fpga doctor` for more, or use --hdl sv." % error,
                  file=sys.stderr)
            return 2
        except BoardError as error:
            print(error, file=sys.stderr)
            return 1
        except Onnx2FpgaError as error:
            print(error, file=sys.stderr)
            return 2 if isinstance(error, ToolchainError) else 1


SYNTH_FIELDS = ("lut", "ff", "dsp", "bram36", "uram")


def batch(samples):
    """The API takes calibration as one (N, ...) batch; these are N single-sample feeds concatenated back."""
    return np.concatenate([next(iter(feed.values())) for feed in samples])


def main(argv=None):
    return CommandLine().main(argv)
