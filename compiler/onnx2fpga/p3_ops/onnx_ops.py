"""Float-domain operators imported from ONNX.

Internal layout is channel-last (NHWC, weights KH.KW.Cin.Cout). Channels are
the innermost axis so that folding across parallel lanes is a contiguous slice
of every stream beat.
"""

import numpy as np

from ..p2_graph.datatype import FLOAT32
from ..p2_graph.node import Node


class FrontendNode(Node):
    def _out(self, graph, index=0):
        return graph.ensure_tensor(self.outputs[index])


class Im2Col:
    """Shared window extraction for Conv and Pool reference execution."""

    def __init__(self, kernel, strides, pads, dilations):
        self.kernel = tuple(kernel)
        self.strides = tuple(strides)
        self.pads = tuple(pads)
        self.dilations = tuple(dilations)

    def output_extent(self, in_h, in_w):
        kh, kw = self.kernel
        eff_h = (kh - 1) * self.dilations[0] + 1
        eff_w = (kw - 1) * self.dilations[1] + 1
        out_h = (in_h + self.pads[0] + self.pads[2] - eff_h) // self.strides[0] + 1
        out_w = (in_w + self.pads[1] + self.pads[3] - eff_w) // self.strides[1] + 1
        return out_h, out_w

    def apply(self, x, pad_value=0):
        n, in_h, in_w, c = x.shape
        top, left, bottom, right = self.pads
        if any(self.pads):
            x = np.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)),
                       constant_values=pad_value)
        kh, kw = self.kernel
        out_h, out_w = self.output_extent(in_h, in_w)
        cols = np.empty((n, out_h, out_w, kh, kw, c), dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                rows = slice(i * self.dilations[0],
                             i * self.dilations[0] + out_h * self.strides[0],
                             self.strides[0])
                columns = slice(j * self.dilations[1],
                                j * self.dilations[1] + out_w * self.strides[1],
                                self.strides[1])
                cols[:, :, :, i, j, :] = x[:, rows, columns, :]
        return cols.reshape(n, out_h, out_w, kh * kw * c)


class GemmOp(FrontendNode):
    """y = x @ W + b with W already normalised to (in_features, out_features)."""

    op_type = "Gemm"

    def infer(self, graph):
        x = graph.tensor(self.inputs[0])
        w = graph.tensor(self.inputs[1])
        out = self._out(graph)
        out.shape = (x.shape[0], w.shape[1])
        out.dtype = FLOAT32

    def execute(self, graph, values):
        x = values[self.inputs[0]]
        w = graph.tensor(self.inputs[1]).data
        y = x @ w
        if len(self.inputs) > 2:
            y = y + graph.tensor(self.inputs[2]).data
        return [np.maximum(y, 0) if self.attrs.get("relu") else y]


class ConvOp(FrontendNode):
    op_type = "Conv"

    def window(self):
        return Im2Col(self.attrs["kernel_shape"], self.attrs["strides"],
                      self.attrs["pads"], self.attrs["dilations"])

    def infer(self, graph):
        x = graph.tensor(self.inputs[0])
        w = graph.tensor(self.inputs[1])
        out_h, out_w = self.window().output_extent(x.shape[1], x.shape[2])
        out = self._out(graph)
        out.shape = (x.shape[0], out_h, out_w, w.shape[3])
        out.dtype = FLOAT32

    def execute(self, graph, values):
        x = values[self.inputs[0]]
        w = graph.tensor(self.inputs[1]).data
        cols = self.window().apply(x)
        kh, kw, cin, cout = w.shape
        y = cols @ w.reshape(kh * kw * cin, cout)
        if len(self.inputs) > 2:
            y = y + graph.tensor(self.inputs[2]).data
        return [np.maximum(y, 0) if self.attrs.get("relu") else y]


class ReluOp(FrontendNode):
    op_type = "Relu"

    def infer(self, graph):
        source = graph.tensor(self.inputs[0])
        out = self._out(graph)
        out.shape, out.dtype = source.shape, source.dtype

    def execute(self, graph, values):
        return [np.maximum(values[self.inputs[0]], 0)]


class AddOp(FrontendNode):
    op_type = "Add"

    def infer(self, graph):
        left = graph.tensor(self.inputs[0])
        right = graph.tensor(self.inputs[1])
        out = self._out(graph)
        out.shape = left.shape if left.numel >= (right.numel or 0) else right.shape
        out.dtype = FLOAT32

    def execute(self, graph, values):
        operands = [values[name] if name in values else graph.tensor(name).data
                    for name in self.inputs]
        return [operands[0] + operands[1]]


class MaxPoolOp(FrontendNode):
    op_type = "MaxPool"

    def window(self):
        return Im2Col(self.attrs["kernel_shape"], self.attrs["strides"],
                      self.attrs["pads"], self.attrs.get("dilations", (1, 1)))

    def infer(self, graph):
        x = graph.tensor(self.inputs[0])
        out_h, out_w = self.window().output_extent(x.shape[1], x.shape[2])
        out = self._out(graph)
        out.shape = (x.shape[0], out_h, out_w, x.shape[3])
        out.dtype = x.dtype

    def execute(self, graph, values):
        x = values[self.inputs[0]]
        window = self.window()
        cols = window.apply(x, pad_value=np.finfo(np.float32).min)
        n, out_h, out_w, _ = cols.shape
        channels = x.shape[3]
        kh, kw = window.kernel
        return [cols.reshape(n, out_h, out_w, kh * kw, channels).max(axis=3)]


class FlattenOp(FrontendNode):
    """Stream no-op: channel-last data already arrives in flattened order."""

    op_type = "Flatten"

    def infer(self, graph):
        x = graph.tensor(self.inputs[0])
        out = self._out(graph)
        out.shape = (x.shape[0], x.numel // x.shape[0])
        out.dtype = x.dtype

    def execute(self, graph, values):
        x = values[self.inputs[0]]
        return [x.reshape(x.shape[0], -1)]
