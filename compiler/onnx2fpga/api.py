"""The Python API: load a model, compile it, and get an accelerator back."""

import collections.abc
import contextlib
import math
import numbers
import os
import pathlib
import re
import shutil
import tempfile
import warnings

import numpy as np

from . import reload
from .board import BOARDS, Kria
from .compile import CompilationResult, Compiler
from .doctor import TOOLS
from .errors import (BudgetError, ConfigError, ModelError, NotCompiledError,
                     NotDeployedError, SimulationError, ToolchainError,
                     UnknownLayerError)
from .hardware_layer import CustomOp, custom_passes
from .measure import TimingReport, calibration_samples
from .p1_ingest import OnnxModel
from .p1_ingest.protobuf_codec import WireError
from .p2_graph.datatype import DataType, IntType
from .p2_graph.onnx_importer import OnnxImporter
from .p3_ops.elementwise import EltwiseUnit, TableUnit
from .p3_ops.hardware_ops import (DenseUnit, DuplicateUnit, StreamFifo, TreeEnsembleUnit,
                                  WidthConverter)
from .p3_ops.onnx_ops import ConvOp, FlattenOp, GemmOp
from .p3_ops.trees import TreeEnsembleOp
from .p4_quantize.lower import FuseRelu
from .p4_quantize.trees import tree_op
from .p5_schedule.fixed import slowest
from .p6_emit.hardcaml import HardcamlError, HardcamlGenerator
from .p6_emit.project import ProjectWriter
from .p6_emit.stream_io import StreamCodec
from .pipeline import CompilerContext
from .reference.runner import GraphRunner
from .serial_link import Link, Session
from .simulate import FixedSimulationRunner, SimulationRunner
from .targets.device import Device

HDL_ENV = "ONNX2FPGA_HDL"
DEFAULT_DEVICE = "vu9p"
DEFAULT_UTILISATION = 0.80
DEFAULT_CALIBRATION = 32
FIXED_VECTORS = 10000


def load(model, custom_objects=None, name=None):
    """An Accelerator for an ONNX model given as a path or an OnnxModel; custom_objects maps an op type to its HardwareLayer."""
    if isinstance(model, OnnxModel):
        return Accelerator(model, name=name, custom_objects=custom_objects)
    path = pathlib.Path(model)
    if not path.is_file():
        raise FileNotFoundError("no ONNX model at %s" % path)
    try:
        onnx = OnnxModel.load(path)
    except (WireError, ValueError, IndexError, NotImplementedError) as error:
        raise ModelError("cannot read %s: %s" % (path, error)) from error
    return Accelerator(onnx, name=name or path.stem, custom_objects=custom_objects)


def from_keras(model, path=None, custom_objects=None):
    """An Accelerator for a Keras model, exported through its own ONNX export."""
    try:
        import keras  # noqa: F401
    except ImportError as error:
        raise ToolchainError("from_keras needs Keras 3", tool="keras",
                             remedy="pip install keras") from error
    if not hasattr(model, "export"):
        raise ConfigError("from_keras takes a Keras model, not %s" % type(model).__name__)
    with contextlib.ExitStack() as stack:
        if path is None:
            folder = stack.enter_context(tempfile.TemporaryDirectory())
            path = pathlib.Path(folder) / ("%s.onnx" % (model.name or "model"))
        path = pathlib.Path(path)
        try:
            model.export(str(path), format="onnx")
        except (ImportError, NotImplementedError, TypeError, ValueError) as error:
            raise ToolchainError("Keras could not export %r to ONNX: %s"
                                 % (model.name, error), tool="keras",
                                 remedy="install tf2onnx or onnx, or export the model "
                                        "yourself and call otf.load") from error
        return load(path, custom_objects=custom_objects, name=model.name)


def resolve_hdl(hdl=None):
    """The backend to use: the argument, else ONNX2FPGA_HDL, else Hardcaml if it can be built, else SystemVerilog."""
    source = "hdl=%r" % hdl
    if hdl is None and os.environ.get(HDL_ENV):
        hdl, source = os.environ[HDL_ENV], "%s=%s" % (HDL_ENV, os.environ[HDL_ENV])
    if hdl is None:
        return "hardcaml" if _hardcaml_builds() else "sv"
    chosen = str(hdl).strip().lower()
    if chosen not in ProjectWriter.HDLS:
        raise ConfigError("unknown backend %s, expected one of %s"
                          % (source, ", ".join(ProjectWriter.HDLS)))
    if chosen == "hardcaml":
        reason = HardcamlGenerator().unavailable()
        if reason:
            raise ToolchainError("%s needs the Hardcaml generator: %s" % (source, reason),
                                 tool="hardcaml",
                                 remedy="run `onnx2fpga doctor`, or use hdl='sv'")
    return chosen


def _hardcaml_builds():
    """Whether defaulting to Hardcaml is safe; a project that is present but fails to build is a bug, not a fallback."""
    generator = HardcamlGenerator()
    if generator.unavailable():
        return False
    try:
        generator.executable
    except HardcamlError as error:
        warnings.warn("the Hardcaml generator does not build, using sv: %s"
                      % str(error).splitlines()[0])
        return False
    return True


def _require_tools(names, step):
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise ToolchainError(
            "%s needs %s, which %s not on PATH"
            % (step, " and ".join(missing), "is" if len(missing) == 1 else "are"),
            tool=missing[0], remedy="; ".join(TOOLS[name][1] for name in missing))


@contextlib.contextmanager
def _translated():
    """NotImplementedError from the pipeline is a fact about the user model."""
    try:
        yield
    except NotImplementedError as error:
        raise ModelError("cannot lower this model yet: %s" % error) from error


class Accelerator:
    """A model on its way to an FPGA. Made by otf.load or otf.from_keras."""

    def __init__(self, model, name=None, custom_objects=None):
        self.model = model
        self.name = name or model.graph.name or "model"
        self.custom_objects = dict(custom_objects or {})
        handlers, _ = custom_passes(self.custom_objects)
        with _translated():
            graph = OnnxImporter(model, handlers).run()
        # Relu is fused into its producer before lowering, so only the remaining layers can be folded.
        FuseRelu().run(graph, CompilerContext(Device.get("sim")))
        self._float_view = graph
        self.layers = Layers(self, graph)
        self.result = None
        self.settings = {}
        self._requested = {}
        self.board = None
        self.session = None
        self._revision = 0
        self._compiled_revision = None
        self._compiled_graph = None

    @property
    def input_name(self):
        return self.model.graph_inputs()[0]

    @property
    def input_shape(self):
        """The declared input shape, batch axis first, in the model's layout."""
        return tuple(1 if isinstance(d, str) else int(d)
                     for d in self.model.value_shape(self.input_name))

    @property
    def device(self):
        return self.result.context.device if self.result else None

    @property
    def hdl(self):
        return self.settings.get("hdl")

    @property
    def out_dir(self):
        return self.settings.get("out_dir")

    @property
    def mode(self):
        return self.settings.get("mode")

    @property
    def latency(self):
        """Cycles from features_valid to prediction_valid in fixed mode; None before compile or when streaming."""
        return self.result.latency if self.result else None

    @property
    def deployed(self):
        return self.session is not None

    def _changed(self):
        self._revision += 1

    def _require_compiled(self, step):
        if self.result is None:
            raise NotCompiledError("%s needs a compiled design: call "
                                   "acc.compile(device=...) first" % step)
        if self._compiled_revision != self._revision:
            raise NotCompiledError("the layers changed since the last compile: call "
                                   "acc.compile() again before %s" % step)

    def compile(self, device=None, target=None, target_fps=None, target_cycles=None,
                hdl=None, calibration=None, utilisation=None, out_dir=None,
                strict=True, verbose=False, mult_bits=None, top=None, mode=None,
                weights=None, clock_mhz=None):
        """Quantizes, folds and emits the design for `device`; arguments left out keep the previous compile value."""
        previous = self._requested
        mode = mode or previous.get("mode", "stream")
        if mode not in ("stream", "fixed"):
            raise ConfigError("mode is 'stream' or 'fixed', not %r" % (mode,))
        weights = weights or previous.get("weights", "frozen")
        if weights not in ("frozen", "reloadable"):
            raise ConfigError("weights is 'frozen' or 'reloadable', not %r" % (weights,))
        if weights == "reloadable":
            self._check_reloadable(mode)
        device = self._device(device if device is not None
                              else previous.get("device", DEFAULT_DEVICE))
        if clock_mhz is None and mode == "fixed":
            clock_mhz = previous.get("clock_mhz")
        if clock_mhz is not None and mode != "fixed":
            raise ConfigError("clock_mhz needs mode='fixed'")
        if clock_mhz is not None and (isinstance(clock_mhz, bool) or not isinstance(
                clock_mhz, numbers.Real) or clock_mhz <= 0):
            raise ConfigError("clock_mhz must be a positive number, not %r" % (clock_mhz,))
        if mode == "fixed":
            self._check_fixed(target, target_fps, target_cycles)
            goal = ("fit", None, None)
            hdl = _fixed_hdl(hdl)
        else:
            constant = [name for name, kind in self.layers.arithmetic().items()
                        if kind == "constant"]
            if constant:
                raise ConfigError("%s is folded in mode='stream', so arithmetic('constant') "
                                  "needs mode='fixed'" % constant[0])
            goal = self._target(device, target, target_fps, target_cycles,
                                previous.get("target", ("fit", None)))
            hdl = resolve_hdl(hdl) if hdl is not None or "hdl" not in previous \
                else previous["hdl"]
        utilisation = self._utilisation(utilisation if utilisation is not None
                                        else previous.get("utilisation",
                                                          DEFAULT_UTILISATION))
        if calibration is not None:
            samples = self._feeds(calibration, "calibration")
        elif "samples" in previous:
            samples = previous["samples"]
        else:
            samples = self._default_samples()
        if out_dir is None and (previous.get("hdl"), previous.get("mode")) == (hdl, mode):
            out_dir = previous.get("out_dir")
        if out_dir is None:
            suffix = "fixed" if mode == "fixed" else hdl
            out_dir = pathlib.Path("build") / (self.name if suffix == "sv"
                                               else "%s_%s" % (self.name, suffix))
        mult_bits = mult_bits or previous.get("mult_bits", 18)
        top = top or previous.get("top", "otf_top")
        cycles = goal[2]
        settings = {"device": device.name, "target": goal[:2], "hdl": hdl,
                    "utilisation": utilisation, "samples": samples,
                    "out_dir": pathlib.Path(out_dir), "mult_bits": mult_bits, "top": top,
                    "mode": mode, "weights": weights, "clock_mhz": clock_mhz}
        self._requested = settings

        compiler = Compiler(device=device, target_cycles=cycles,
                            utilisation_limit=utilisation, mult_bits=mult_bits,
                            top_name=top, verbose=verbose, hdl=hdl,
                            folding=self.layers.pins(),
                            weight_dtypes=self.layers.weight_dtypes(),
                            arithmetic=self.layers.arithmetic(),
                             memory=self.layers.memory(),
                             custom_objects=self.custom_objects, mode=mode, weights=weights,
                             clock_mhz=clock_mhz)
        with _translated():
            result = compiler.compile(self.model, samples, out_dir)
        if strict and cycles is not None:
            self._check_target(result, goal, utilisation)

        self.disconnect()
        self.board = None
        self.result = result
        self._compiled_graph = result.hardware_graph
        self.settings = settings
        self._compiled_revision = self._revision
        return self

    def _check_reloadable(self, mode):
        if mode != "fixed":
            raise ConfigError("weights='reloadable' needs mode='fixed'")
        if tree_op(self._float_view):
            raise ConfigError("weights='reloadable' does not support tree ensembles yet")
        constant = [name for name, kind in self.layers.arithmetic().items()
                    if kind == "constant"]
        if constant:
            raise ConfigError("%s is set to arithmetic('constant'), which needs "
                              "weights='frozen'" % constant[0])

    def _check_fixed(self, target, target_fps, target_cycles):
        given = [key for key, value in (("target", target), ("target_fps", target_fps),
                                        ("target_cycles", target_cycles))
                 if value is not None]
        if given:
            raise ConfigError("mode='fixed' does not take %s" % given[0])
        pins = self.layers.pins()
        if pins:
            raise ConfigError("mode='fixed' cannot use the fold() pins on %s"
                              % ", ".join(sorted(pins)))
        memory = self.layers.memory()
        if memory:
            raise ConfigError("mode='fixed' keeps weights in logic, so memory() on %s "
                              "needs mode='stream'" % ", ".join(sorted(memory)))

    @staticmethod
    def _device(name):
        if isinstance(name, Device):
            return name
        try:
            return Device.get(name)
        except KeyError:
            raise ConfigError("unknown device %r, expected one of %s"
                              % (name, ", ".join(Device.names()))) from None

    @staticmethod
    def _target(device, target, target_fps, target_cycles, previous):
        """(kind, value, cycles a frame or None for as fast as fits)."""
        given = [(key, value) for key, value in (("target", target),
                                                 ("target_fps", target_fps),
                                                 ("target_cycles", target_cycles))
                 if value is not None]
        if len(given) > 1:
            raise ConfigError("pass one of target, target_fps or target_cycles, not %s"
                              % " and ".join("%s=%r" % item for item in given))
        kind, value = previous if not given else given[0]
        if kind in ("target", "fit"):
            if value in (None, "fit"):
                return ("fit", None, None)
            if value == "latency":
                raise NotImplementedError("target='latency' is not implemented yet")
            raise ConfigError("unknown target %r, expected 'fit', target_fps or "
                              "target_cycles" % (value,))
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or value <= 0:
            raise ConfigError("%s must be a positive number, not %r" % (kind, value))
        if kind in ("target_fps", "fps"):
            cycles = max(1, int(math.floor(device.fmax_mhz * 1e6 / value)))
            return ("fps", value, cycles)
        return ("cycles", int(value), int(value))

    @staticmethod
    def _utilisation(value):
        if not 0 < value <= 1:
            raise ConfigError("utilisation must be between 0 and 1, not %r" % (value,))
        return value

    def _check_target(self, result, goal, utilisation):
        reached = result.folding.bottleneck_cycles
        cycles = goal[2]
        if reached <= cycles:
            return
        device = result.context.device
        asked = ("target_fps=%g, %d cycles a frame at %.0f MHz"
                 % (goal[1], cycles, device.fmax_mhz) if goal[0] == "fps"
                 else "target_cycles=%d" % cycles)
        reasons = [line for line in result.context.log
                   if line.startswith("fold:") and "bottleneck" not in line
                   and "pinned at" not in line]
        slow_pins = [node for node in result.hardware_graph.nodes
                     if node.pinned_folding and node.cycles > cycles]
        reasons += ["fold: %s is pinned at %s, %d cycles"
                    % (node.name, _folding(node.folding), node.cycles)
                    for node in slow_pins]
        # A stage that cannot fold further is a floor no device lifts; otherwise more of the binding resource helps.
        binding = set(re.findall(r"the (\w+) budget", "\n".join(reasons)))
        options = ["%d cycles or more" % reached, "target='fit'", "strict=False"]
        options += ["acc.layers[%r].fold()" % node.name for node in slow_pins]
        if any(device.limits(utilisation).get(field, 1.0) < 1.0 for field in binding):
            options.append("a utilisation above %.2f" % utilisation)
        larger = [name for name in Device.names() if name not in (device.name, "sim")
                  and all(getattr(Device.get(name).budget, field)
                          > getattr(device.budget, field) for field in binding)]
        if binding and larger:
            options.append("a device with more %s (%s)"
                           % (" and ".join(sorted(binding)), ", ".join(larger)))
        raise BudgetError("%s is out of reach on %s, the closest design takes %d cycles"
                          "\n%s\nTry: %s" % (asked, device.name, reached,
                                             "\n".join(reasons), ", ".join(options)))

    def _feeds(self, x, purpose):
        x = np.asarray(x)
        expected = self.input_shape[1:]
        if x.ndim == len(expected) + 1 and tuple(x.shape[1:]) == expected:
            return [{self.input_name: x[k:k + 1]} for k in range(x.shape[0])]
        hint = " (missing batch axis? use x[None])" if tuple(x.shape) == expected else ""
        raise ConfigError("%s takes a batch shaped (N, %s), not %s%s"
                          % (purpose, ", ".join(str(d) for d in expected),
                             tuple(x.shape), hint))

    def _default_samples(self):
        """Trees take their feature grids from the thresholds, so only other models warn."""
        if not self._float_view.annotations and not tree_op(self._float_view):
            warnings.warn("calibrating %s on random inputs; pass "
                          "calibration=x_train[:64] for accurate scales" % self.name,
                          stacklevel=3)
        return calibration_samples(self.model, DEFAULT_CALIBRATION, 0)

    def summary(self, print_fn=None):
        """A table of the layers, with folding, cost and bottleneck against the device budget once compiled."""
        if self.result is None:
            lines = self._layer_summary()
        elif self.mode == "fixed":
            lines = self._fixed_summary()
        else:
            lines = self._compiled_summary()
        (print_fn or print)("\n".join(lines))

    def _layer_summary(self):
        widths = (30, 18, 9, 22)
        rule = sum(widths)
        lines = ['Accelerator: "%s" (not compiled)' % self.name, "_" * rule,
                 _row(("Layer (type)", "Output shape", "Params", "Overrides"), widths,
                      (False, False, True, False)), "=" * rule]
        total = 0
        for layer in self.layers.values():
            total += layer.params
            lines.append(_row(("%s (%s)" % (layer.name, layer.kind),
                               _shape(layer.output_shape), layer.params,
                               layer._overrides()), widths, (False, False, True, False)))
        if any(len(layer.output_shape or ()) == 4 for layer in self.layers.values()):
            lines.append(" Shapes are channel-last, the order the hardware streams "
                         "them in.")
        lines += ["=" * rule, " Total params: %d" % total,
                  " Not compiled yet: acc.compile(device=...) folds it onto a device.",
                  "_" * rule]
        return lines

    def _compiled_summary(self):
        graph, device = self.result.hardware_graph, self.device
        ram = device.block_ram
        widths = (30, 16, 9, 9, 7, 7)
        right = (False, False, True, True, True, True)
        rule = sum(widths)
        units, plumbing = [], []
        for node in graph.topological_order():
            kind = (plumbing if isinstance(node, (StreamFifo, WidthConverter,
                                                  DuplicateUnit)) else units)
            kind.append((node, node.resources(device)))
        bottleneck = max(node.cycles for node, _ in units + plumbing)
        total = _sum(cost for _, cost in units + plumbing)

        lines = ['Accelerator: "%s" on %s (%s), hdl %s, target %s'
                 % (self.name, device.name, device.part, self.hdl,
                    self._describe_target()), "_" * rule,
                 _row(("Unit (kind)", "Folding", "Cycles", "LUT", "DSP", ram.upper()),
                      widths, right), "=" * rule]
        for node, cost in units:
            lines.append(_row(("%s (%s)" % (node.name, _kind(node)),
                               _folding(node.folding), node.cycles, int(cost.lut),
                               _number(cost.dsp), _number(getattr(cost, ram))),
                              widths, right)
                         + ("  <- bottleneck" if node.cycles == bottleneck else ""))
        if plumbing:
            counts = {}
            for node, _ in plumbing:
                counts[_kind(node)] = counts.get(_kind(node), 0) + 1
            cost = _sum(c for _, c in plumbing)
            slowest = max(node.cycles for node, _ in plumbing)
            label = "streams (%s)" % ", ".join("%d %s" % (n, k.lower())
                                               for k, n in sorted(counts.items()))
            lines.append(_row((label, "", slowest, int(cost.lut), _number(cost.dsp),
                               _number(getattr(cost, ram))), widths, right)
                         + ("  <- bottleneck" if slowest == bottleneck else ""))
        budget = device.budget
        lines += ["=" * rule,
                  _row(("Total", "", "", int(total.lut), _number(total.dsp),
                        _number(getattr(total, ram))), widths, right),
                  _row(("Budget on %s" % device.name, "", "", budget.lut, budget.dsp,
                        getattr(budget, ram)), widths, right),
                  _row(("Used", "", "", _percent(total.lut, budget.lut),
                        _percent(total.dsp, budget.dsp),
                        _percent(getattr(total, ram), getattr(budget, ram))),
                       widths, right),
                  " Throughput: %d cycles a frame, %.1f frames/s at %.0f MHz"
                  % (bottleneck, device.fmax_mhz * 1e6 / bottleneck, device.fmax_mhz)]
        lines += self._measured_clock()
        lines += [" Quantization: %s" % self._describe_quantization(),
                  " Build: %s" % self.out_dir, "_" * rule]
        return lines

    def _fixed_summary(self):
        graph, device = self.result.hardware_graph, self.device
        widths = (30, 12, 12, 12, 9, 9, 7, 9, 7)
        right = (False,) + (True,) * 8
        rule = sum(widths)
        units = graph.topological_order()
        trees = isinstance(units[0], TreeEnsembleUnit)
        delays = device.delays
        lines = ['Accelerator: "%s" on %s (%s), hdl %s, fixed latency'
                 % (self.name, device.name, device.part, self.hdl), "_" * rule,
                 _row(("Layer (kind)", "Trees/depth" if trees else "In -> out",
                       "" if trees else "Arithmetic", "Comparators" if trees else "Multipliers",
                       "Latency", "Stages", "Fmax", "LUT", "DSP"), widths, right), "=" * rule]
        costs = [unit.resources(device) for unit in units]
        parts = [unit.comparators if isinstance(unit, TreeEnsembleUnit) else unit.multipliers
                 for unit in units]
        for unit, cost, count in zip(units, costs, parts):
            fmax = "%.0f" % (1000.0 / slowest(unit, delays)[0]) if delays else ""
            kind = "+".join(self.layers[name].kind
                            for name in unit.attrs.get("merged", []) + [unit.name])
            lines.append(_row(("%s (%s)" % (unit.name, kind), *_fixed_shape(unit), count,
                               unit.latency, "/".join(str(n) for n in unit.stages.values()),
                               fmax, int(cost.lut), _number(cost.dsp)), widths, right))
        total, budget = _sum(costs), device.budget
        latency, clock = self.latency, self.result.clock_mhz
        saved = sum(unit.weights.size - unit.multipliers for unit in units
                    if isinstance(unit, DenseUnit))
        fmax = self.result.fmax_mhz
        lines += ["=" * rule,
                  _row(("Total", "", "", sum(parts), latency, "",
                        "%.0f" % fmax if fmax else "", int(total.lut), _number(total.dsp)),
                       widths, right),
                  _row(("Budget on %s" % device.name, "", "", "", "", "", "", budget.lut,
                        budget.dsp), widths, right),
                  _row(("Used", "", "", "", "", "", "", _percent(total.lut, budget.lut),
                        _percent(total.dsp, budget.dsp)), widths, right),
                  " Stages: registers added %s%s" % (
                      "in the leaf select/adder tree" if trees
                      else "at the multiplies/adder tree/rescale",
                      ", or the multiplies/rescale of an elementwise unit"
                      if any(isinstance(unit, EltwiseUnit) for unit in units) else ""),
                  " Latency: %d cycles, %.1f ns at %.0f MHz, with a new input every cycle"
                  % (latency, latency * 1e3 / clock, clock)]
        if fmax:
            lines.append(" Clock: %s, %.1f MHz predicted by the %s delay model (%s)"
                         % ("%s MHz asked" % self.settings["clock_mhz"]
                            if self.settings.get("clock_mhz") else "no clock_mhz asked",
                            fmax, device.name, delays.provenance))
        if saved:
            lines.append(" Constant arithmetic: %d multipliers saved" % saved)
        words = reload.image(graph)
        if words:
            lines.append(" Weights: reloadable, two banks of %d words, CRC 0x%08x loaded"
                         % (len(words), reload.crc(words)))
        lines += self._measured_clock()
        lines += [" Quantization: %s" % self._describe_quantization(),
                  " Build: %s" % self.out_dir, "_" * rule]
        return lines

    def _measured_clock(self):
        """Vivado's fmax beside the estimate, once a board build has left a timing report."""
        for path in sorted(self.out_dir.glob("*/timing_summary.rpt")):
            try:
                report = TimingReport.load(path)
            except ValueError:
                continue
            return [" Clock: %.1f MHz estimated, %.1f MHz measured by Vivado (WNS %.3f ns, %s)"
                    % (self.result.clock_mhz or self.device.fmax_mhz, report.fmax_mhz, report.wns_ns,
                       path.relative_to(self.out_dir))]
        return []

    def _describe_target(self):
        kind, value = self.settings["target"]
        return {"fit": "fit", "fps": "%g fps" % (value or 0),
                "cycles": "%s cycles" % value}[kind]

    def _describe_quantization(self):
        plan = self.result.plan
        if plan.source == "thresholds":
            return ("features %s on per feature grids from the thresholds, leaves %s"
                    % (plan.act_dtype.name.lower(), plan.weight_dtype.name.lower()))
        weights = plan.weight_dtype.name.lower()
        if plan.layer_weight_dtypes:
            weights += " (%s)" % ", ".join(
                "%s %s" % (name, dtype.name.lower())
                for name, dtype in sorted(plan.layer_weight_dtypes.items()))
        return "weights %s, activations %s, from %s" % (
            weights, plan.act_dtype.name.lower(),
            "the model's QDQ nodes" if plan.from_model else "calibration")

    def evaluate(self, x, y=None):
        """Accuracy cost of quantization: the integer model against the float one, plus top-1 agreement."""
        self._require_compiled("evaluate")
        feeds = self._feeds(x, "evaluate")
        integer = self._dequantize(self._reference(feeds)).reshape(len(feeds), -1)
        floating = np.concatenate([self._float_reference(feed)
                                   for feed in feeds]).reshape(len(feeds), -1)
        metrics = {}
        # One output is a label for a classifier and a value for a regressor, so argmax means nothing.
        if integer.shape[1] == 1 and not self._labels:
            metrics["mean_abs_error"] = float(np.mean(np.abs(integer - floating)))
            return metrics
        picked = np.argmax(integer, axis=1) if integer.shape[1] > 1 else integer[:, 0]
        wanted = np.argmax(floating, axis=1) if integer.shape[1] > 1 else floating[:, 0]
        if y is not None:
            labels = np.asarray(y)
            labels = labels if labels.ndim == 1 else np.argmax(
                labels.reshape(len(labels), -1), axis=1)
            if len(labels) != len(feeds):
                raise ConfigError("evaluate got %d inputs but %d labels"
                                  % (len(feeds), len(labels)))
            metrics["accuracy"] = float(np.mean(picked == labels))
            metrics["float_accuracy"] = float(np.mean(wanted == labels))
        metrics["agreement"] = float(np.mean(picked == wanted))
        return metrics

    @property
    def _labels(self):
        graph = self.result.hardware_graph
        return graph.tensor(graph.outputs[0]).dtype.bits == 1

    def predict(self, x, on=None, quantized=False):
        """Outputs for a batch of inputs, from the reference model until deployed and from the board after."""
        self._require_compiled("predict")
        if on is None:
            on = "board" if self.deployed else "reference"
        if on not in ("reference", "board"):
            raise ConfigError("predict runs on='reference' or on='board', not %r" % (on,))
        if on == "board" and not self.deployed:
            raise NotDeployedError("predict(on='board') needs "
                                   "acc.build(board='icepi_zero') and "
                                   "acc.deploy(port=...) first")
        feeds = self._feeds(x, "predict")
        codes = self._reference(feeds) if on == "reference" else self._on_board(feeds)
        return codes if quantized else self._dequantize(codes)

    def _quantize(self, feed):
        graph, plan = self.result.hardware_graph, self.result.plan
        name = graph.inputs[0]
        raw = np.asarray(feed[name], dtype=np.float64)
        if name in graph.layouts:
            raw = graph.layouts[name].to_internal(raw)
        return plan.quantize(name, raw)

    def _reference(self, feeds):
        graph = self.result.hardware_graph
        runner = GraphRunner(graph)
        return np.concatenate([runner.run({graph.inputs[0]: self._quantize(feed)})[
            graph.outputs[0]] for feed in feeds])

    def _float_reference(self, feed):
        graph = self.result.float_graph
        name = graph.inputs[0]
        raw = np.asarray(next(iter(feed.values())), dtype=np.float64)
        if name in graph.layouts:
            raw = graph.layouts[name].to_internal(raw)
        return GraphRunner(graph).run({name: raw})[graph.outputs[0]]

    def _dequantize(self, codes):
        plan, name = self.result.plan, self.result.hardware_graph.outputs[0]
        return plan.scale(name) * (np.asarray(codes, dtype=np.float64)
                                   - plan.zero_point(name))

    def _on_board(self, feeds):
        graph = self.result.hardware_graph
        source = StreamCodec(graph.tensor(graph.inputs[0]))
        sink = StreamCodec(graph.tensor(graph.outputs[0]))
        frames = [source.to_image(self._quantize(feed)).words for feed in feeds]
        return np.concatenate([sink.from_words(words)
                               for words in self.session.run(frames)])

    def simulate(self, x=None, sweep=False, reload=None, commits=None):
        """Builds the design with Verilator and checks every output against the integer reference; in fixed mode `reload` fills the idle bank and `commits` flips it."""
        self._require_compiled("simulate")
        _require_tools(("verilator", "g++", "make"), "simulate")
        out, top = self.out_dir, self.settings["top"]
        if self.mode == "fixed":
            inputs = self._fixed_inputs(x)
            flags = [] if reload is None else self._reload_flags(reload, commits, inputs)
            runner = FixedSimulationRunner(out, self.latency, top,
                                           self._write_vectors(inputs), flags)
            frames = len(inputs)
        else:
            vectors = SimulationRunner.GOLDEN
            if x is not None:
                feeds = self._feeds(x, "simulate")
                vectors = self._write_vectors([self._quantize(feed) for feed in feeds])
            runner = SimulationRunner(out, top, vectors)
            frames = len(x) if x is not None else 1
        built = runner.build()
        if built.returncode != 0:
            raise SimulationError("Verilator could not build %s:\n%s\n%s"
                                  % (out, built.stdout[-4000:], built.stderr[-4000:]),
                                  built.stdout, built.stderr)
        first = runner.once()
        runs = runner.sweep() if sweep and first.ok else []
        return SimulationReport(first, runs, frames)

    def _fixed_inputs(self, x):
        """Quantized inputs for x, or FIXED_VECTORS random vectors covering the whole input grid."""
        if x is not None:
            return [self._quantize(feed) for feed in self._feeds(x, "simulate")]
        tensor = self.result.hardware_graph.tensor(self.result.hardware_graph.inputs[0])
        dtype = self.result.plan.act_dtype
        rng = np.random.default_rng(0)
        return list(rng.integers(dtype.min, dtype.max + 1,
                                 (FIXED_VECTORS,) + tuple(tensor.shape)))

    def _reload_flags(self, model, commits, inputs):
        """Harness flags for a second bank holding `model`, committed on each of `commits`."""
        _, graph = self._retarget(model)
        words = reload.image(graph)
        if commits is None:
            commits = [max(len(words), len(inputs) // 2)]
        early = [cycle for cycle in commits if cycle < len(words)]
        if early:
            raise ConfigError("a commit at cycle %d comes before the %d cycles that write the "
                              "idle bank" % (early[0], len(words)))
        (self.out_dir / "golden" / "x_weights.hex").write_text(
            "".join("%08x\n" % word for word in words))
        _, expected = self._write_vectors(inputs, graph, "x_expected_new.hex")
        compiled = reload.crc(reload.image(self._compiled_graph))
        return ["--weights", "golden/x_weights.hex", "--expected-new", expected,
                "--commits", ",".join(str(cycle) for cycle in sorted(set(commits))),
                "--crc", "%x,%x" % (compiled, reload.crc(words))]

    def _write_vectors(self, inputs, graph=None, expected_name="x_expected.hex"):
        """Stimulus and expected answers for the harness, from `graph` or the compiled design."""
        graph = graph or self._compiled_graph
        source = StreamCodec(graph.tensor(graph.inputs[0]))
        sink = StreamCodec(graph.tensor(graph.outputs[0]))
        runner = GraphRunner(graph)
        sent, expected = [], []
        for quantized in inputs:
            sent.append(source.to_image(quantized).render())
            expected.append(sink.to_image(
                runner.run({graph.inputs[0]: quantized})[graph.outputs[0]]).render())
        golden = self.out_dir / "golden"
        (golden / "x_input.hex").write_text("".join(sent))
        (golden / expected_name).write_text("".join(expected))
        return ("golden/x_input.hex", "golden/" + expected_name)

    def build(self, board="icepi_zero", verbose=False, **options):
        """A bitstream for `board`, an Icepi Zero with the open flow or a Kria with Vivado; compiles first if nothing is compiled yet."""
        if isinstance(board, str) and board not in BOARDS:
            raise ConfigError("unknown board %r, expected one of %s"
                              % (board, ", ".join(BOARDS)))
        board_class = BOARDS[board] if isinstance(board, str) else type(board)
        kria = issubclass(board_class, Kria)
        if self.result is None:
            if kria:
                self.compile(device=board_class.DEVICE, mode="fixed")
            else:
                self.compile(device=board_class.DEVICE, hdl="hardcaml")
        self._require_compiled("build")
        name = board_class.NAME
        wanted = "fixed" if kria else "stream"
        if self.mode != wanted:
            raise ConfigError("build(board=%r) needs mode=%r" % (name, wanted))
        if self.device.name != board_class.DEVICE:
            raise ConfigError("%s carries %s, but this design was compiled for %s"
                              % (name, board_class.DEVICE, self.device.name))
        if self.hdl != "hardcaml":
            raise ConfigError("build(board=%r) needs the Hardcaml backend" % name)
        if isinstance(board, str):
            board = board_class(self.out_dir, **options)
        elif pathlib.Path(board.out).resolve() != self.out_dir.resolve():
            raise ConfigError("the board writes to %s but the design is in %s"
                              % (board.out, self.out_dir))
        missing = board_class.missing_tools()
        if missing and kria:
            board.write()
            raise ToolchainError("build(board=%r) wrote the Vivado project to %s but needs "
                                 "vivado on PATH to run it" % (name, board.build),
                                 tool="vivado")
        if missing:
            raise ToolchainError("build(board=%r) needs %s, which is not on PATH"
                                 % (name, " and ".join(missing)),
                                 tool=missing[0], remedy=TOOLS[missing[0]][1])
        if kria:
            board.run()
        else:
            board.run("bit")
        if verbose:
            print("\n".join(self._measured_clock()) if kria else board.render_report())
        self.board = board
        return board.bitstream

    def deploy(self, port=None, program=True, timeout=2.0, host=None):
        """Loads the bitstream into the board SRAM with openFPGALoader and opens its serial port."""
        if host is not None or isinstance(self.board, Kria):
            raise NotImplementedError("deploying to a Kria board needs the Kria runtime, "
                                      "which is not written yet")
        if self.board is None:
            raise NotCompiledError("deploy needs acc.build(board='icepi_zero') first")
        self._require_compiled("deploy")
        if program:
            if self.board.tool("openFPGALoader") is None:
                raise ToolchainError("deploy needs openFPGALoader, which is not on PATH",
                                     tool="openFPGALoader",
                                     remedy="install it, or load the bitstream yourself "
                                            "and call deploy(program=False)")
            self.board.run("prog")
        try:
            import serial  # noqa: F401
        except ImportError as error:
            raise ToolchainError("deploy needs pyserial", tool="pyserial",
                                 remedy=TOOLS["pyserial"][1]) from error
        self.disconnect()
        link = Link.load(self.out_dir, self.board.NAME)
        if link.window < 1:
            raise ConfigError("a frame of %d bytes in and %d out does not fit the board "
                              "buffers of %d and %d" % (link.frame_in, link.frame_out,
                                                        link.rx_depth, link.tx_depth))
        self.session = Session(link, port, timeout).open()
        return self

    def load_weights(self, model, registers=None):
        """Checks `model` against the compiled architecture and grid, writes it through `registers` into the idle bank, commits and confirms; predict then uses it."""
        self._require_compiled("load_weights")
        float_graph, graph = self._retarget(model)
        if registers is None:
            raise NotDeployedError("load_weights needs registers=, such as "
                                   "SimulatedRegisters, until the Kria runtime can deploy "
                                   "a board")
        reload.write_bank(registers, reload.image(graph))
        self.result = CompilationResult(float_graph, graph, self.result.context,
                                        self.result.build)
        return self

    def _retarget(self, model):
        if self.settings.get("weights") != "reloadable":
            raise ConfigError("%s has frozen weights; compile it with weights='reloadable' "
                              "to load new ones" % self.name)
        onnx = model if isinstance(model, OnnxModel) else load(model).model
        with _translated():
            return reload.retarget(onnx, self.result, self.settings["samples"])

    def disconnect(self):
        """Closes the serial port; predict goes back to the reference model."""
        if self.session is not None:
            self.session.close()
            self.session = None

    def __repr__(self):
        if self.result is None:
            return '<Accelerator "%s": %d layers, not compiled>' % (self.name,
                                                                   len(self.layers))
        if self.mode == "fixed":
            return '<Accelerator "%s": %s, hdl %s, fixed latency %d cycles>' % (
                self.name, self.device.name, self.hdl, self.latency)
        where = ", on %s" % self.session.port if self.deployed else ""
        return '<Accelerator "%s": %s, hdl %s, %d cycles a frame%s>' % (
            self.name, self.device.name, self.hdl,
            self.result.folding.bottleneck_cycles, where)


class SimulationReport:
    """What a simulate() run found: `first` at full throughput, `sweep` under backpressure."""

    def __init__(self, first, sweep, frames):
        self.first = first
        self.sweep = list(sweep)
        self.frames = frames

    @property
    def ok(self):
        return self.first.ok and all(run.ok for run in self.sweep)

    @property
    def cycles(self):
        return self.first.cycles

    @property
    def latency(self):
        return self.first.latency

    def __repr__(self):
        verdict = "bit exact" if self.ok else "MISMATCH"
        detail = "" if self.cycles is None else ", %d cycles" % self.cycles
        swept = ("" if not self.sweep else ", %d of %d sweep runs passed"
                 % (sum(run.ok for run in self.sweep), len(self.sweep)))
        return "<SimulationReport %s: %d frame%s%s%s>" % (
            verdict, self.frames, "" if self.frames == 1 else "s", detail, swept)


class Layer:
    """One layer of the model as the compiler sees it, after Relu fusion."""

    def __init__(self, accelerator, node, graph):
        self._accelerator = accelerator
        self.name = node.name
        self.kind = node.op_type + ("+Relu" if node.attrs.get("relu") else "")
        self.output_shape = graph.tensor(node.outputs[0]).shape
        self.params = sum(graph.tensor(name).numel or 0 for name in node.inputs
                          if graph.tensor(name).is_constant)
        if isinstance(node, TreeEnsembleOp):
            self.params = node.params
        if isinstance(node, (GemmOp, ConvOp)):
            self.factors = ("simd", "pe")
        elif isinstance(node, (FlattenOp, TreeEnsembleOp)):
            self.factors = ()
        else:
            self.factors = ("pe",)
        self.has_weights = isinstance(node, (GemmOp, ConvOp))
        self.custom = isinstance(node, CustomOp)
        self.pinned = {}
        self.weight_dtype = None
        self.arithmetic_choice = "auto"
        self.memory_choice = "auto"

    def fold(self, **factors):
        """Pins how parallel this layer is (simd, pe), overriding the allocator; none hands it back."""
        if not self.factors:
            raise ConfigError("%s (%s) has nothing to fold" % (self.name, self.kind))
        unknown = sorted(set(factors) - set(self.factors))
        if unknown:
            raise ConfigError("%s (%s) folds by %s, not %s"
                              % (self.name, self.kind, " and ".join(self.factors),
                                 ", ".join(unknown)))
        for key, value in factors.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) \
                    or value < 1:
                raise ConfigError("%s=%r for %s must be a positive integer"
                                  % (key, value, self.name))
        self.pinned = {key: int(value) for key, value in factors.items()}
        self._accelerator._changed()
        return self

    def quantize(self, weights=None, activations=None):
        """Sets the weight precision of this layer, "int2" to "int8"; none restores the default."""
        if activations is not None:
            raise ConfigError("activation precision cannot be set per layer")
        if weights is not None and not self.has_weights:
            raise ConfigError("%s (%s) has no weights" % (self.name, self.kind))
        self.weight_dtype = None if weights is None else _weight_dtype(weights, self.name)
        self._accelerator._changed()
        return self

    def arithmetic(self, kind="auto"):
        """How fixed latency mode multiplies by this layer's weights: "constant", "multiplier" or "auto"."""
        if kind not in ("constant", "multiplier", "auto"):
            raise ConfigError("arithmetic is 'constant', 'multiplier' or 'auto', not %r"
                              % (kind,))
        if not self.has_weights:
            raise ConfigError("%s (%s) has no weights" % (self.name, self.kind))
        self.arithmetic_choice = kind
        self._accelerator._changed()
        return self

    def memory(self, kind="auto"):
        """Where a streaming design keeps this layer's weights: "block" RAM, "lut" or "auto"."""
        if kind not in ("block", "lut", "auto"):
            raise ConfigError("memory is 'block', 'lut' or 'auto', not %r" % (kind,))
        if not self.has_weights:
            raise ConfigError("%s (%s) has no weights" % (self.name, self.kind))
        self.memory_choice = kind
        self._accelerator._changed()
        return self

    @property
    def folding(self):
        """The compiled folding of this layer's unit, or None before compile."""
        unit = self._unit()
        return dict(unit.folding) if unit is not None else None

    @property
    def cycles(self):
        unit = self._unit()
        return unit.cycles if unit is not None else None

    @property
    def latency(self):
        """Cycles this layer adds in a fixed latency design, else None."""
        return getattr(self._unit(), "latency", None)

    @property
    def units(self):
        """Names of the hardware units this layer became."""
        result = self._accelerator.result
        if result is None:
            return []
        return [node.name for node in result.hardware_graph.topological_order()
                if node.name in (self.name, self.name + "_swu")]

    def _unit(self):
        result = self._accelerator.result
        if result is None:
            return None
        return next((node for node in result.hardware_graph.nodes
                     if node.name == self.name), None)

    def _overrides(self):
        parts = []
        if self.pinned:
            parts.append("fold " + _folding(self.pinned))
        if self.weight_dtype is not None:
            parts.append("weights " + self.weight_dtype.name.lower())
        if self.arithmetic_choice != "auto":
            parts.append("arithmetic " + self.arithmetic_choice)
        if self.memory_choice != "auto":
            parts.append("memory " + self.memory_choice)
        return ", ".join(parts)

    def __repr__(self):
        parts = []
        folding = self.folding
        if folding:
            parts.append("folding " + _folding(folding))
        if self.pinned:
            parts.append("pinned " + _folding(self.pinned))
        if self.weight_dtype is not None:
            parts.append("weights " + self.weight_dtype.name.lower())
        arithmetic = getattr(self._unit(), "arithmetic", None)
        if arithmetic:
            parts.append("arithmetic " + arithmetic)
        if getattr(self._unit(), "block_ram", False):
            parts.append("weights in block RAM")
        return "<Layer %s (%s)%s>" % (self.name, self.kind,
                                      ": " + ", ".join(parts) if parts else "")


class Layers(collections.abc.Mapping):
    """The model's layers by name, in pipeline order."""

    def __init__(self, accelerator, graph):
        self._layers = {node.name: Layer(accelerator, node, graph)
                        for node in graph.topological_order()}

    def __getitem__(self, name):
        try:
            return self._layers[name]
        except KeyError:
            raise UnknownLayerError("the model has no layer named %r. Layers: %s"
                                    % (name, ", ".join(self._layers))) from None

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def pins(self):
        return {name: dict(layer.pinned) for name, layer in self._layers.items()
                if layer.pinned}

    def weight_dtypes(self):
        return {name: layer.weight_dtype for name, layer in self._layers.items()
                if layer.weight_dtype is not None}

    def arithmetic(self):
        return {name: layer.arithmetic_choice for name, layer in self._layers.items()
                if layer.arithmetic_choice != "auto"}

    def memory(self):
        return {name: layer.memory_choice for name, layer in self._layers.items()
                if layer.memory_choice != "auto"}

    def __repr__(self):
        return "<Layers %s>" % ", ".join(self._layers)


def _fixed_hdl(hdl):
    chosen = hdl if hdl is not None else os.environ.get(HDL_ENV) or "hardcaml"
    if str(chosen).strip().lower() == "sv":
        raise ConfigError("mode='fixed' needs hdl='hardcaml'")
    return resolve_hdl(chosen)


def _weight_dtype(text, layer):
    found = re.fullmatch(r"int(\d+)", str(text).strip().lower())
    if not found or not 2 <= int(found.group(1)) <= 8:
        raise ConfigError("weights=%r for %s, expected 'int2' to 'int8'" % (text, layer))
    bits = int(found.group(1))
    return DataType._registry.get("INT%d" % bits) or IntType(bits, True)


def _fixed_shape(unit):
    """The shape and arithmetic columns of a fixed latency unit."""
    if isinstance(unit, TreeEnsembleUnit):
        return "%d/%d" % (unit.ensemble.trees, unit.ensemble.depth), ""
    if isinstance(unit, DenseUnit):
        return "%d -> %d" % (unit.mw, unit.mh), unit.arithmetic
    lanes = "%d -> %d" % (unit.lanes, unit.lanes)
    return lanes, "table" if isinstance(unit, TableUnit) else unit.op


def _row(cells, widths, right):
    out = []
    for cell, width, align in zip(cells, widths, right):
        text = str(cell)
        out.append(text.rjust(width - 1) if align else text.ljust(width - 1))
    return " " + " ".join(out).rstrip()


def _shape(shape):
    return "(" + ", ".join(str(d) for d in shape) + ")" if shape else "?"


def _folding(folding):
    return ",".join("%s=%d" % item for item in sorted(folding.items())) or "-"


def _kind(node):
    name = type(node).__name__
    if hasattr(node, "layer"):
        return type(node.layer).__name__
    short = name[:-4] if name.endswith("Unit") else name
    return {"StreamFifo": "Fifo", "WidthConverter": "Dwc",
            "DuplicateUnit": "Fork"}.get(name, short)


def _number(value):
    return ("%d" % value) if float(value).is_integer() else ("%.1f" % value)


def _percent(used, available):
    return "%.1f%%" % (100.0 * used / available) if available else "-"


def _sum(costs):
    total = None
    for cost in costs:
        total = cost if total is None else total + cost
    return total
