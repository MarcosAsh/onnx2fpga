"""Tree ensembles as flat node arrays, and the float operator ONNX's tree ensembles import to."""

import numpy as np

from ..p2_graph.datatype import FLOAT32
from .onnx_ops import FrontendNode

MODES = ("BRANCH_LEQ", "BRANCH_LT", "BRANCH_GTE", "BRANCH_GT", "BRANCH_EQ", "BRANCH_NEQ",
         "LEAF")
LEQ, LT, GTE, GT, EQ, NEQ, LEAF = range(len(MODES))


class Ensemble:
    """Every node of every tree in one set of arrays; a leaf is its own child and holds a value."""

    def __init__(self, roots, feature, threshold, mode, if_true, if_false, value):
        self.roots = np.asarray(roots, dtype=np.int64)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold)
        self.mode = np.asarray(mode, dtype=np.int64)
        self.if_true = np.asarray(if_true, dtype=np.int64)
        self.if_false = np.asarray(if_false, dtype=np.int64)
        self.value = np.asarray(value)

    @property
    def trees(self):
        return len(self.roots)

    @property
    def nodes(self):
        return len(self.mode)

    @property
    def splits(self):
        return int(np.sum(self.mode != LEAF))

    @property
    def depth(self):
        depth, level = 0, self.roots
        while np.any(self.mode[level] != LEAF):
            level = level[self.mode[level] != LEAF]
            level = np.concatenate([self.if_true[level], self.if_false[level]])
            depth += 1
        return depth

    def tree_leaves(self):
        """The leaf nodes of each tree."""
        found = []
        for root in self.roots:
            leaves, pending = [], [root]
            while pending:
                node = pending.pop()
                if self.mode[node] == LEAF:
                    leaves.append(node)
                else:
                    pending += [self.if_true[node], self.if_false[node]]
            found.append(np.array(leaves))
        return found

    def leaves(self, x):
        """The leaf each tree reaches for each row of x, shaped (rows, trees)."""
        x = np.asarray(x).reshape(len(x), -1)
        node = np.repeat(self.roots[None, :], len(x), axis=0)
        for _ in range(self.depth):
            value = np.take_along_axis(x, self.feature[node], axis=1)
            threshold = self.threshold[node]
            tests = (value <= threshold, value < threshold, value >= threshold,
                     value > threshold, value == threshold, value != threshold)
            taken = np.choose(np.minimum(self.mode[node], NEQ), tests)
            node = np.where(taken, self.if_true[node], self.if_false[node])
        return node

    def leaf_values(self, x):
        return self.value[self.leaves(x)]


class TreeEnsembleOp(FrontendNode):
    """A sum of trees plus `base`; with a `decision` threshold the output is the label, margin > decision."""

    op_type = "TreeEnsemble"

    def __init__(self, name, inputs, outputs, ensemble, base, decision=None, op_type=None):
        FrontendNode.__init__(self, name, inputs, outputs)
        self.ensemble = ensemble
        self.base = float(base)
        self.decision = decision
        self.op_type = op_type or self.op_type

    @property
    def params(self):
        return self.ensemble.nodes

    def infer(self, graph):
        x = graph.tensor(self.inputs[0])
        out = self._out(graph)
        out.shape = (x.shape[0], 1)
        out.dtype = FLOAT32

    def margin(self, x):
        return self.ensemble.leaf_values(x).sum(axis=1) + self.base

    def execute(self, graph, values):
        margin = self.margin(np.asarray(values[self.inputs[0]], dtype=np.float64))
        if self.decision is not None:
            margin = (margin > self.decision).astype(np.float64)
        return [margin.reshape(-1, 1)]
