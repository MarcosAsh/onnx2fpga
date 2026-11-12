"""Import checks without calibration, hardware generation or external tools."""

import copy
from dataclasses import asdict, dataclass, field
import math

from .errors import ConfigError, ModelError
from .p1_ingest import OnnxModel, TensorProto
from .p1_ingest.onnx_model import tensor_to_numpy
from .p1_ingest.protobuf_codec import WireError
from .p2_graph.constants import FOLD
from .p2_graph.graph import Graph, GraphError
from .p2_graph.onnx_importer import BUILTIN, QLINEAR, OnnxImporter
from .p4_quantize.elementwise import check_stream_ops
from .p4_quantize.trees import tree_op
from .p5_schedule.fixed import check_fixed_ops
from .targets.device import Device


DOMAINS = {
    "ai.onnx.ml": {"LinearRegressor", "LinearClassifier", "Scaler", "Binarizer",
                   "Normalizer", "TreeEnsemble", "TreeEnsembleRegressor",
                   "TreeEnsembleClassifier", "ArrayFeatureExtractor", "ZipMap"},
    "com.microsoft": set(QLINEAR) | {"QLinearConcat"},
    "qonnx.custom_op.general": {"Quant", "IntQuant"},
    "finn.custom_op.general": {"Quant", "IntQuant"},
}
IMPORT_ERRORS = (ModelError, GraphError, ValueError, KeyError, IndexError,
                 NotImplementedError)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    node_index: int = None
    node: str = None
    op_type: str = None
    domain: str = None
    opset: int = None


@dataclass
class CompatibilityReport:
    device: str
    mode: str
    findings: list = field(default_factory=list)
    unchecked: tuple = ("full ONNX schema and opset validation", "quantization and accuracy",
                        "hardware lowering and resource fit", "backend tool availability",
                        "routed timing", "physical board operation")

    @property
    def ok(self):
        return not any(f.severity in ("error", "unknown") for f in self.findings)

    def to_dict(self):
        return dict(ok=self.ok, device=self.device, mode=self.mode,
                    findings=[asdict(f) for f in self.findings], unchecked=list(self.unchecked))

    def render(self):
        result = "ready to attempt compilation" if self.ok else "blocked or incomplete"
        lines = ["%s (%s, %s)" % (result, self.device, self.mode)]
        for f in self.findings:
            where = "node %d %r: " % (f.node_index, f.node) if f.node_index is not None else ""
            lines.append("%s [%s] %s%s" % (f.severity, f.code, where, f.message))
        lines.append("not checked: " + "; ".join(self.unchecked))
        return "\n".join(lines)


class _Importer(OnnxImporter):
    current = None

    def __init__(self, model):
        super().__init__(model)
        self.origins = {}

    def emit(self, node):
        super().emit(node)
        self.origins[id(node)] = self.current

    def visit(self, node):
        self.current = node
        super().visit(node)
        self.current = None

    def materialize(self, name):
        previous = self.current
        if previous is None:
            self.current = next((n for n in self.model.nodes() if name in n.output), None)
        result = super().materialize(name)
        self.current = previous
        return result


def check(model, device="vu9p", mode="stream", clock_mhz=None):
    """Report built-in import compatibility; a passing report does not prove a build fits."""
    try:
        target = Device.get(device)
    except KeyError as error:
        raise ConfigError(str(error)) from error
    if mode not in ("stream", "fixed"):
        raise ConfigError("mode must be 'stream' or 'fixed'")
    report = CompatibilityReport(target.name, mode)
    if clock_mhz is not None:
        if isinstance(clock_mhz, bool) or not isinstance(clock_mhz, (int, float)) or not (
                math.isfinite(clock_mhz) and clock_mhz > 0):
            raise ConfigError("clock_mhz must be a positive finite number")
        if mode != "fixed":
            report.findings.append(Finding("error", "clock_mode", "clock_mhz needs mode='fixed'"))
        elif target.delays is None:
            report.findings.append(Finding("error", "clock_model", "%s has no delay model" % device))
    if not isinstance(model, OnnxModel):
        try:
            model = OnnxModel.load(model)
        except (OSError, WireError, ValueError, IndexError, NotImplementedError) as error:
            report.findings.append(Finding("error", "model_read", str(error)))
            return report

    nodes = model.nodes()
    indices = {id(node): i for i, node in enumerate(nodes)}
    opsets = {entry.domain: int(entry.version) for entry in model.proto.opset_import}
    blocked, removed = set(), set()

    def finding(severity, code, message, node=None):
        report.findings.append(Finding(
            severity, code, message, indices[id(node)] if node is not None else None,
            node.name if node is not None else None, node.op_type if node is not None else None,
            node.domain if node is not None else None,
            opsets.get(node.domain) if node is not None else None))

    def remove(node):
        removed.add(id(node))
        blocked.update(name for name in node.output if name)

    inputs = model.graph_inputs()
    if len(inputs) != 1:
        finding("unknown", "input_count", "hardware interfaces need checking for %d inputs" % len(inputs))
    if not model.graph_outputs():
        finding("error", "outputs", "model has no graph outputs")
    for info in model.graph.input:
        if info.name not in inputs:
            continue
        tensor = info.type and info.type.tensor_type
        if tensor is None or tensor.elem_type not in TensorProto.NUMPY_DTYPE:
            finding("error", "input_type", "input %r has an unsupported tensor type" % info.name)
            blocked.add(info.name)
        elif tensor.elem_type not in (TensorProto.FLOAT, TensorProto.DOUBLE,
                                      TensorProto.FLOAT16, TensorProto.INT8, TensorProto.UINT8):
            finding("unknown", "input_type", "input %r integer/boolean semantics need checking" % info.name)
            blocked.add(info.name)
        if tensor is None or tensor.shape is None:
            finding("unknown", "input_shape", "input %r has no static shape" % info.name)
            blocked.add(info.name)
            continue
        for axis, dim in enumerate(tensor.shape.dim):
            if dim.dim_param or not dim.dim_value:
                if axis == 0:
                    finding("warning", "batch_shape", "input %r uses batch size 1 during import" % info.name)
                else:
                    finding("unknown", "input_shape", "input %r dimension %d is not static" % (info.name, axis))
                    blocked.add(info.name)
    for tensor in model.graph.initializer:
        try:
            tensor_to_numpy(tensor)
        except IMPORT_ERRORS as error:
            finding("error", "initializer", "initializer %r: %s" % (tensor.name, error))
            blocked.add(tensor.name)
    for node in nodes:
        if node.domain not in ("", "ai.onnx") and node.op_type not in DOMAINS.get(node.domain, ()):
            finding("error", "operator_domain", "unsupported operator %s::%s" % (node.domain, node.op_type), node)
            remove(node)
        elif node.op_type not in BUILTIN and node.op_type not in FOLD:
            finding("error", "operator", "unsupported operator %r" % node.op_type, node)
            remove(node)

    while True:
        # Failed handlers can modify earlier layers, so retry from clean state.
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if id(node) not in removed and any(name in blocked for name in node.input if name):
                    finding("unknown", "dependency", "input depends on a blocked or unassessed value", node)
                    remove(node)
                    changed = True
        proto = copy.copy(model.proto)
        proto.graph = copy.copy(model.graph)
        proto.graph.node = [node for node in nodes if id(node) not in removed]
        proto.graph.input = [info for info in model.graph.input if info.name not in blocked]
        proto.graph.initializer = [t for t in model.graph.initializer if t.name not in blocked]
        proto.graph.output = [info for info in model.graph.output if info.name not in blocked]
        importer = _Importer(OnnxModel(proto))
        try:
            graph = importer.run()
            graph.validate()
        except IMPORT_ERRORS as error:
            node = importer.current
            finding("error" if node is not None else "unknown", "import", str(error), node)
            if node is None:
                break
            remove(node)
            continue
        for node in graph.nodes:
            single = Graph(graph.name)
            single.inputs = list(node.inputs)
            single.add_node(node)
            try:
                if mode == "fixed":
                    check_fixed_ops(single)
                elif tree_op(single):
                    raise ConfigError("a tree ensemble compiles only with mode='fixed'")
                else:
                    check_stream_ops(single)
            except ConfigError as error:
                finding("error", "execution_mode", str(error), importer.origins.get(id(node)))
        break
    return report
