"""The base graph node, and the memories a hardware node initialises."""

from abc import ABC, abstractmethod


class Node(ABC):
    op_type = "Node"

    def __init__(self, name, inputs, outputs, **attrs):
        self.name = name
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.attrs = dict(attrs)

    @abstractmethod
    def infer(self, graph):
        """Fill in shape, dtype and quantization of this node's output tensors."""

    @abstractmethod
    def execute(self, graph, values):
        """Numpy reference. `values` maps tensor name to array; returns a list."""

    def stream_inputs(self, graph):
        return [name for name in self.inputs if not graph.tensor(name).is_constant]

    def __repr__(self):
        return "%s(%s: %s -> %s)" % (type(self).__name__, self.name,
                                     ",".join(self.inputs), ",".join(self.outputs))


class MemorySpec:
    """An initialised memory; `packed` puts the last axis of `array` in one word."""

    def __init__(self, array, elem_bits, packed=False):
        self.array = array
        self.elem_bits = int(elem_bits)
        self.packed = bool(packed)
