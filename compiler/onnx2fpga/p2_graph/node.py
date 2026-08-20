"""Node hierarchy.

Two families live here. Frontend nodes mirror ONNX operators and only know how
to infer shapes and run a numpy reference. Hardware nodes additionally know
their folding factors, cycle count, resource cost and SystemVerilog form. The
lowering passes rewrite the first family into the second.
"""

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
    """One initialised memory: the values, how wide each element is, and
    whether the last axis is packed into a single word."""

    def __init__(self, array, elem_bits, packed=False):
        self.array = array
        self.elem_bits = int(elem_bits)
        self.packed = bool(packed)

    @property
    def word_bits(self):
        if not self.packed:
            return self.elem_bits
        return self.elem_bits * int(self.array.shape[-1])


class HwNode(Node):
    """A node that maps to exactly one streaming SystemVerilog module."""

    module = None

    def __init__(self, name, inputs, outputs, **attrs):
        Node.__init__(self, name, inputs, outputs, **attrs)
        self.folding = dict(self.default_folding())

    def default_folding(self):
        return {}

    def folding_options(self):
        """Legal folding assignments, cheapest first. Each entry is a dict."""
        return [{}]

    def apply_folding(self, choice):
        self.folding.update(choice)

    @property
    @abstractmethod
    def cycles(self):
        """Cycles to process one frame once the pipeline is primed."""

    @abstractmethod
    def resources(self, device):
        """Resources object estimating the cost on the given device."""

    @abstractmethod
    def sv_params(self, graph):
        """Parameter overrides for the SystemVerilog instance."""

    def memories(self, graph):
        """Named MemorySpec entries to emit as $readmemh initialisation files."""
        return {}

    def sv_input_prefixes(self):
        return ["s"]

    def sv_output_prefixes(self):
        return ["m"]

    def sv_connections(self, graph, nets):
        """Port map for this node's instance. Overridden by units whose RTL
        ports are not the plain single-in single-out stream pair."""
        wiring = {"clk": "clk", "rst_n": "rst_n"}
        for prefix, name in zip(self.sv_input_prefixes(), self.stream_inputs(graph)):
            wiring["%s_tdata" % prefix] = nets.data(name)
            wiring["%s_tvalid" % prefix] = nets.valid(name)
            wiring["%s_tready" % prefix] = nets.ready(name)
        for prefix, name in zip(self.sv_output_prefixes(), self.outputs):
            wiring["%s_tdata" % prefix] = nets.data(name)
            wiring["%s_tvalid" % prefix] = nets.valid(name)
            wiring["%s_tready" % prefix] = nets.ready(name)
        return wiring

    def memory_paths(self):
        return self.attrs.get("mem_paths", {})

    def input_beat_elems(self, graph):
        return [1] * len(self.stream_inputs(graph))

    def output_beat_elems(self, graph):
        return [1] * len(self.outputs)
