"""Reloadable weights: a new model on a compiled design's grid, its bank image, and the register writes that load it."""

import copy
import zlib

import numpy as np

from .board import OFFSETS, BoardError
from .errors import ConfigError
from .p2_graph.onnx_importer import OnnxImporter
from .p3_ops.elementwise import ElementwiseOp, EltwiseUnit, TableUnit
from .p3_ops.reductions import ReduceOp, SoftmaxOp
from .p3_ops.hardware_ops import MatVecUnit
from .p3_ops.onnx_ops import AddOp, ReluOp
from .p4_quantize.calibrate import Calibrator
from .p4_quantize.lower import FuseRelu, LowerToHardware
from .p4_quantize.plan import QuantizationPlan
from .p5_schedule.fixed import check_fixed_ops, compose_tables
from .pipeline import CompilerContext


def reloadable_units(graph):
    return [unit for unit in graph.topological_order() if getattr(unit, "reloadable", False)]


def image(graph):
    """One bank, layer after layer in pipeline order, one masked value per word."""
    return [word for unit in reloadable_units(graph) for word in unit.image()]


def crc(words):
    """zlib's CRC-32 of the words as little endian bytes, which the hardware computes as they are written."""
    return zlib.crc32(b"".join(int(word).to_bytes(4, "little") for word in words))


def layout(graph):
    """The bank layout for manifest.json: offsets and counts in words from the start of the window."""
    layers, offset = [], 0
    for unit in reloadable_units(graph):
        layer = {"name": unit.name, "inputs": unit.mw, "outputs": unit.mh,
                 "shift": unit.requant.shift}
        for field, count, bits in (("weights", unit.mw * unit.mh, unit.weight_dtype.bits),
                                   ("bias", unit.mh, unit.accumulator_dtype.bits),
                                   ("mult", unit.mh, unit.mult_bits)):
            layer[field] = {"offset": offset, "count": count, "bits": bits}
            offset += count
        layers.append(layer)
    words = image(graph)
    return {"mode": "reloadable", "words": len(words), "crc": crc(words),
            "order": "weights w[k][c] at c * inputs + k, then a bias and a multiplier per "
                     "output channel, one value per 32 bit word masked to its bits",
            "layers": layers}


def retarget(model, result, samples):
    """The float and hardware graphs of `model` on the grid of the compiled `result`; ConfigError names the first mismatch."""
    device = result.context.device
    graph = OnnxImporter(model).run()
    FuseRelu().run(graph, CompilerContext(device))
    check_fixed_ops(graph)
    _same_architecture(result.float_graph, graph)
    plan = _plan(graph, result, samples)
    lowered = LowerToHardware(plan, fixed=True).run(copy.deepcopy(graph),
                                                    CompilerContext(device))
    compose_tables(lowered)
    _same_frozen(result.hardware_graph, lowered)
    retrained = copy.deepcopy(result.hardware_graph)
    news = [node for node in lowered.topological_order() if isinstance(node, MatVecUnit)]
    for unit, new in zip(reloadable_units(retrained), news):
        _fit(unit, new)
    return graph, retrained


def _same_architecture(compiled, new):
    def shape(graph, name):
        return tuple(graph.tensor(name).shape or ())

    old_input, new_input = shape(compiled, compiled.inputs[0]), shape(new, new.inputs[0])
    if old_input != new_input:
        raise ConfigError("the input is shaped %s in the compiled model but %s in the new one"
                          % (old_input, new_input))
    old_layers, new_layers = compiled.topological_order(), new.topological_order()
    if len(old_layers) != len(new_layers):
        raise ConfigError("the compiled model has %d layers but the new one has %d"
                          % (len(old_layers), len(new_layers)))
    for old, layer in zip(old_layers, new_layers):
        kinds = [node.op_type + ("+Relu" if node.attrs.get("relu") else "")
                 for node in (old, layer)]
        if kinds[0] != kinds[1]:
            raise ConfigError("%s is %s in the compiled model but %s in the new one"
                              % (old.name, kinds[0], kinds[1]))
        for role, old_name, name in zip(("input", "weights", "bias"), old.inputs, layer.inputs):
            shapes = shape(compiled, old_name), shape(new, name)
            if shapes[0] != shapes[1]:
                raise ConfigError("%s has %s shaped %s in the compiled model but %s in the "
                                  "new one" % (old.name, role, shapes[0], shapes[1]))


def _plan(graph, result, samples):
    """Calibrates a float model with its input and output grids pinned; a QDQ model must state the same ones."""
    compiled, old = result.plan, result.float_graph
    ends = [(side, getattr(old, side)[0], getattr(graph, side)[0])
            for side in ("inputs", "outputs")]
    if compiled.from_model:
        if not graph.annotations:
            raise ConfigError("the compiled model is quantized by its QDQ nodes but the new "
                              "one has none")
        plan = QuantizationPlan.from_annotations(graph.annotations, compiled.mult_bits)
        for side, old_name, name in ends:
            was, now = compiled.spec(old_name), plan.spec(name)
            if not (np.allclose(was.scale, now.scale, rtol=1e-9, atol=0)
                    and np.array_equal(was.zero_point, now.zero_point)):
                raise ConfigError("the new model's %s has scale %g and zero point %d where "
                                  "the compiled one has %g and %d"
                                  % (side[:-1], now.scale.flat[0], now.zero_point.flat[0],
                                     was.scale.flat[0], was.zero_point.flat[0]))
    else:
        name = graph.inputs[0]
        feeds = [{name: graph.layouts[name].to_internal(array) if name in graph.layouts
                  else array} for feed in samples for array in feed.values()]
        plan = Calibrator(graph).plan(feeds, mult_bits=compiled.mult_bits)
        for _, old_name, name in ends:
            plan.specs[name] = compiled.spec(old_name)
        # Tables and elementwise units are frozen, so the grids around them must stay.
        for before, layer in zip(old.topological_order(), graph.topological_order()):
            if isinstance(layer, (ElementwiseOp, ReluOp, AddOp, ReduceOp, SoftmaxOp)):
                for old_name, name in zip(before.inputs + before.outputs,
                                          layer.inputs + layer.outputs):
                    if compiled.has(old_name):
                        plan.specs[name] = compiled.spec(old_name)
    plan.layer_weight_dtypes = {
        layer.name: compiled.layer_weight_dtypes[before.name]
        for before, layer in zip(old.topological_order(), graph.topological_order())
        if before.name in compiled.layer_weight_dtypes}
    return plan


def _same_frozen(compiled, lowered):
    """Raises unless every table and elementwise unit comes out as it was compiled."""
    def frozen(graph):
        return [node for node in graph.topological_order()
                if isinstance(node, (TableUnit, EltwiseUnit))]

    def values(unit):
        if isinstance(unit, TableUnit):
            return unit.rows().tolist()
        return (unit.op, [np.asarray(s).tolist() for s in unit.sources], unit.zero_points,
                unit.mults, unit.mult, unit.shift, unit.out_zero_point,
                [None if table is None else np.asarray(table).tolist() for table in unit.tables])

    for old, new in zip(frozen(compiled), frozen(lowered)):
        if values(old) != values(new):
            raise ConfigError("%s is frozen, and the new model changes it" % old.name)


def _fit(unit, new):
    """Puts the new layer's values into `unit`, at its shift, if they fit its widths."""
    if new.weight_dtype.bits != unit.weight_dtype.bits:
        raise ConfigError("%s has %d bit weights in the new model but %d bit in the compiled one"
                          % (unit.name, new.weight_dtype.bits, unit.weight_dtype.bits))
    requant = new.requant.broadcast_to(unit.mh)
    if (requant.zero_point, requant.lower_clamp) != (unit.requant.zero_point,
                                                     unit.requant.lower_clamp):
        raise ConfigError("%s's output zero point is %d in the new model but %d in the "
                          "compiled one" % (unit.name, requant.zero_point,
                                            unit.requant.zero_point))
    requant = requant.at_shift(unit.requant.shift)
    largest = int(np.abs(requant.multipliers).max())
    if largest >= 1 << (unit.mult_bits - 1):
        raise ConfigError("%s needs a rescale multiplier of %d, past the %d bits the design holds"
                          % (unit.name, largest, unit.mult_bits))
    held = unit.accumulator_dtype.bits
    unit.weights, unit.bias, unit.requant = new.weights, new.bias, requant
    needed = MatVecUnit.accumulator_dtype.fget(unit).bits
    if needed > held:
        raise ConfigError("%s needs a %d bit accumulator, past the %d bits the design holds"
                          % (unit.name, needed, held))


def write_bank(registers, words):
    """Writes the idle bank, checks its CRC before committing, commits, and checks the bank, version and CRC; returns the version."""
    expected = crc(words)
    held = registers.read(OFFSETS["weights_words"])
    if held != len(words):
        raise BoardError("the board holds %d words a bank but this design has %d, so it runs "
                         "another bitstream" % (held, len(words)))
    bank = registers.read(OFFSETS["weights_bank"])
    version = registers.read(OFFSETS["weights_version"])
    for index, word in enumerate(words):
        registers.write(OFFSETS["weights"] + 4 * index, word)
    idle = registers.read(OFFSETS["weights_idle_crc"])
    if idle != expected:
        raise BoardError("the idle bank reads CRC 0x%08x after writing 0x%08x, so it was not "
                         "committed" % (idle, expected))
    registers.write(OFFSETS["weights_commit"], 1)
    state = tuple(registers.read(OFFSETS[name])
                  for name in ("weights_bank", "weights_version", "weights_crc"))
    wanted = (1 - bank, (version + 1) & 0xFFFFFFFF, expected)
    if state != wanted:
        raise BoardError("after the commit the board reads bank %d, version %d and CRC 0x%08x "
                         "instead of bank %d, version %d and CRC 0x%08x" % (state + wanted))
    return state[1]
