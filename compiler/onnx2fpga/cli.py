"""Command line front end.

    onnx2fpga doctor                          what this machine can do
    onnx2fpga inspect model.onnx              what the compiler sees
    onnx2fpga compile model.onnx --out build/m
    onnx2fpga simulate model.onnx             compile, then prove it in Verilator
    onnx2fpga synth model.onnx --device z7020 measured resources and timing
    onnx2fpga devices                         target parts and their budgets

Also reachable as `python -m onnx2fpga` from a checkout, with no install.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

from .compile import Compiler
from .doctor import Doctor
from .measure import Characterizer, VivadoRunner, calibration_samples, render
from .p1_ingest import OnnxModel
from .p2_graph.graph import GraphError
from .p2_graph.onnx_importer import OnnxImporter
from .simulate import SimulationRunner
from .targets.device import Device


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
        parser.add_argument("--unroll", action="store_true",
                            help="do not fold at all: every unit at its widest, "
                                 "for the lowest latency the device will hold")
        parser.add_argument("--output-bits", type=int, default=None,
                            choices=(8, 16, 32),
                            help="width of the graph's output; wider keeps "
                                 "scores int8 would saturate")
        parser.add_argument("--act-bits", type=int, default=8, choices=(8, 16),
                            help="width of the activations between layers; "
                                 "int16 for models int8 cannot hold")
        parser.add_argument("--weight-bits", type=int, default=8, choices=(8, 16),
                            help="width the weights are quantized to; widen it "
                                 "with --act-bits, not on its own")
        parser.add_argument("--per-feature-input", action="store_true",
                            help="fit a scale per input column instead of one "
                                 "for the whole vector; for feature vectors "
                                 "whose columns differ in magnitude")
        parser.add_argument("--utilisation", type=float, default=0.80,
                            help="fraction of the device the design may occupy")
        parser.add_argument("--mult-bits", type=int, default=18,
                            help="width of the requantisation multiplier")
        parser.add_argument("--calibration", type=pathlib.Path, default=None,
                            help="npy file of calibration samples; random if omitted")
        parser.add_argument("--samples", type=int, default=32)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--top", default="otf_top")
        parser.add_argument("--quiet", action="store_true")

    def run(self, args):
        model = OnnxModel.load(args.model)
        samples = self._samples(model, args)
        compiler = Compiler(device=args.device, target_cycles=args.target_cycles,
                            utilisation_limit=args.utilisation,
                            mult_bits=args.mult_bits, top_name=args.top,
                            verbose=not args.quiet, unroll=args.unroll,
                            output_bits=args.output_bits,
                            act_bits=args.act_bits,
                            weight_bits=args.weight_bits,
                            per_feature_input=args.per_feature_input)
        result = compiler.compile(model, samples, args.out)
        if not args.quiet:
            print()
            print(result.folding.render())
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
        parser.add_argument("--unroll", action="store_true",
                            help="do not fold at all: every unit at its widest")
        parser.add_argument("--samples", type=int, default=32)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--quiet", action="store_true")

    def build(self, args, default_out):
        model = OnnxModel.load(args.model)
        out = args.out or pathlib.Path("build") / default_out
        compiler = Compiler(device=args.device, target_cycles=args.target_cycles,
                            verbose=not args.quiet, unroll=args.unroll)
        result = compiler.compile(model, calibration_samples(
            model, args.samples, args.seed), out)
        return result, out


class SimulateCommand(ModelCommand):
    name = "simulate"
    help = "compile a model and check the generated RTL against the golden model"

    def configure(self, parser):
        ModelCommand.configure(self, parser)
        parser.add_argument("--sweep", action="store_true",
                            help="also run a randomised backpressure sweep, which "
                                 "is where stream designs actually fail")

    def run(self, args):
        result, out = self.build(args, args.model.stem)
        if not args.quiet:
            print()
            print(result.folding.render())
            print()

        runner = SimulationRunner(out)
        built = runner.build()
        if built.returncode != 0:
            print(built.stdout[-4000:], file=sys.stderr)
            print(built.stderr[-4000:], file=sys.stderr)
            return 1

        first = runner.once()
        print(first.annotated_output().strip())
        if not first.ok:
            return 1
        if not args.sweep:
            return 0

        print("\nbackpressure sweep")
        results = runner.sweep()
        for entry in results:
            print(entry.render())
        failed = [entry for entry in results if not entry.ok]
        print("\n%d of %d settings passed" % (len(results) - len(failed), len(results)))
        summary = runner.summarise(results)
        if summary:
            print()
            print(summary.render())
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
        result, out = self.build(args, args.model.stem + "_synth")
        device = Device.get(args.device)
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
        print("%-8s %-26s %9s %9s %8s %7s  %s"
              % ("name", "part", "lut", "dsp", "bram36", "source", "notes"))
        for name in Device.names():
            device = Device.get(name)
            print("%-8s %-26s %9d %9d %8d %7s  %s"
                  % (name, device.part, device.budget.lut, device.budget.dsp,
                     device.budget.bram36, device.provenance, device.notes))
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
                SimulateCommand(), SynthCommand(), DevicesCommand())

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
        except GraphError as refused:
            # A refused graph is an answer, not a crash. The passes raise it
            # when they can prove the design is not buildable as asked, and the
            # message already says what to change, so a traceback here would
            # bury it under frames nobody outside this repo can read.
            print("onnx2fpga: %s" % refused, file=sys.stderr)
            return 1


SYNTH_FIELDS = ("lut", "ff", "dsp", "bram36", "uram")


def main(argv=None):
    return CommandLine().main(argv)
