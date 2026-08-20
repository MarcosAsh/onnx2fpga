"""The compiler's working graph: tensors, nodes, and structural edits."""

from .tensor import Tensor


class GraphError(Exception):
    pass


class Graph:
    def __init__(self, name="graph"):
        self.name = name
        self._tensors = {}
        self._nodes = []
        self.inputs = []
        self.outputs = []
        self.layouts = {}
        self.annotations = {}

    def add_tensor(self, tensor):
        if tensor.name in self._tensors:
            raise GraphError("duplicate tensor %r" % tensor.name)
        self._tensors[tensor.name] = tensor
        return tensor

    def tensor_names(self):
        return list(self._tensors)

    def tensor(self, name):
        try:
            return self._tensors[name]
        except KeyError:
            raise GraphError("unknown tensor %r" % name)

    def has_tensor(self, name):
        return name in self._tensors

    def ensure_tensor(self, name, **kwargs):
        if name not in self._tensors:
            self.add_tensor(Tensor(name, **kwargs))
        return self._tensors[name]

    def fresh_name(self, stem):
        candidate, index = stem, 0
        while candidate in self._tensors or any(n.name == candidate for n in self._nodes):
            index += 1
            candidate = "%s_%d" % (stem, index)
        return candidate

    @property
    def nodes(self):
        return list(self._nodes)

    def add_node(self, node):
        self._nodes.append(node)
        return node

    def remove_node(self, node):
        self._nodes.remove(node)

    def replace_node(self, node, replacements):
        index = self._nodes.index(node)
        self._nodes[index:index + 1] = list(replacements)

    def insert_after(self, node, new_node):
        self._nodes.insert(self._nodes.index(node) + 1, new_node)

    def producer(self, tensor_name):
        for node in self._nodes:
            if tensor_name in node.outputs:
                return node
        return None

    def consumers(self, tensor_name):
        return [n for n in self._nodes if tensor_name in n.inputs]

    def rewire(self, old_name, new_name, only=None):
        for node in self._nodes:
            if only is not None and node not in only:
                continue
            node.inputs = [new_name if i == old_name else i for i in node.inputs]
        self.outputs = [new_name if o == old_name else o for o in self.outputs]

    def topological_order(self):
        pending = list(self._nodes)
        ready = {name for name, t in self._tensors.items() if t.is_constant}
        ready.update(self.inputs)
        ordered = []
        while pending:
            progressed = False
            for node in list(pending):
                if all(name in ready for name in node.inputs):
                    ordered.append(node)
                    ready.update(node.outputs)
                    pending.remove(node)
                    progressed = True
            if not progressed:
                stuck = ", ".join(n.name for n in pending)
                raise GraphError("cycle or missing input among: %s" % stuck)
        self._nodes = ordered
        return list(ordered)

    def infer(self):
        for node in self.topological_order():
            node.infer(self)
        return self

    def validate(self):
        for node in self._nodes:
            for name in node.inputs + node.outputs:
                self.tensor(name)
        for name in self.inputs + self.outputs:
            self.tensor(name)
        self.topological_order()
        return self

    def summary(self):
        lines = ["graph %s" % self.name]
        for node in self.topological_order():
            shapes = " ".join("%s%s" % (o, self.tensor(o).shape or "?") for o in node.outputs)
            lines.append("  %-22s %-16s %s" % (node.name, type(node).__name__, shapes))
        return "\n".join(lines)
