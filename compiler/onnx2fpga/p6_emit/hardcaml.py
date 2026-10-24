"""The Hardcaml backend: the netlist as design.json, turned into Verilog by OCaml."""

import os
import pathlib
import re
import shutil
import subprocess

from ..assets import AssetError, AssetLocator
from ..errors import Onnx2FpgaError
from .systemverilog import identifier


def design(graph, name="otf_top", latency=None):
    """The scheduled hardware graph as the generator reads it, in a fixed key order."""
    nodes = graph.topological_order()
    streams, seen = [], set()
    for tensor in [graph.inputs[0]] + [out for node in nodes for out in node.outputs]:
        if tensor not in seen:
            seen.add(tensor)
            streams.append({"name": tensor, "width": graph.tensor(tensor).stream_width})
    described = {"format": 1, "top": name}
    if latency is not None:
        described.update(mode="fixed", latency=latency)
    described.update({
        "input": graph.inputs[0],
        "output": graph.outputs[0],
        "streams": streams,
        "nodes": [_node(graph, node) for node in nodes],
    })
    return described


def _node(graph, node):
    params = {key: int(value) for key, value in node.sv_params(graph).items()
              if not isinstance(value, str)}
    return {
        "name": node.name,
        "instance": "u_" + identifier(node.name),
        "kind": node.module,
        "params": params,
        "memories": {key: _memory(spec) for key, spec in node.memories(graph).items()},
        "inputs": node.stream_inputs(graph),
        "outputs": list(node.outputs),
    }


def _memory(spec):
    """Masked to the element width, as $readmemh would load the hex image."""
    mask = (1 << spec.elem_bits) - 1
    return {"elem_bits": spec.elem_bits,
            "lanes": int(spec.array.shape[-1]) if spec.packed else 1,
            "values": [int(v) & mask for v in spec.array.reshape(-1).tolist()]}


class HardcamlError(Onnx2FpgaError):
    pass


class HardcamlGenerator:
    """Builds and runs the generator in the Hardcaml project at OTF_HARDCAML, the assets or the checkout."""

    ENV = "OTF_HARDCAML"
    PROJECT = pathlib.Path("hardware") / "hardcaml"
    EXECUTABLE = pathlib.Path("_build") / "default" / "bin" / "generate.exe"
    MISSING_LIBRARY = re.compile(r'Library "([^"]+)" not found')
    BUILD_TIMEOUT = 900
    _built = {}

    def __init__(self, assets=None, dune="dune"):
        self.assets = assets if isinstance(assets, AssetLocator) else AssetLocator(assets)
        self.dune = dune

    def candidates(self):
        environment = os.environ.get(self.ENV)
        if environment:
            yield pathlib.Path(environment)
            return
        try:
            yield self.assets.root / self.PROJECT
        except AssetError:
            pass
        yield pathlib.Path(__file__).resolve().parents[3] / self.PROJECT

    @property
    def project(self):
        for candidate in self.candidates():
            if (candidate / "dune-project").is_file():
                return candidate
        return None

    def unavailable(self):
        """Why the backend cannot run, or None; a build error counts as broken, not unavailable."""
        project = self.project
        if project is None:
            return ("no Hardcaml project in %s"
                    % ", ".join(str(c) for c in self.candidates()))
        if shutil.which(self.dune) is None:
            return "dune is not on PATH"
        returncode, errors = self._build(project)
        missing = sorted(set(self.MISSING_LIBRARY.findall(errors)))
        if returncode != 0 and missing:
            return "the active opam switch has no %s" % ", ".join(missing)
        return None

    @property
    def executable(self):
        reason = self.unavailable()
        if reason:
            raise HardcamlError(reason)
        returncode, errors = self._build(self.project)
        if returncode != 0:
            raise HardcamlError("dune could not build the generator:\n" + errors[-4000:])
        return self.project / self.EXECUTABLE

    def _build(self, project):
        key = (self.dune, str(project.resolve()))
        if key not in self._built:
            try:
                completed = subprocess.run(
                    [self.dune, "build", "--root", str(project), "--display", "quiet",
                     "./bin/generate.exe"],
                    capture_output=True, text=True, timeout=self.BUILD_TIMEOUT)
                self._built[key] = (completed.returncode, completed.stderr.strip())
            except subprocess.TimeoutExpired:
                self._built[key] = (-1, "dune build timed out after %ds" % self.BUILD_TIMEOUT)
        return self._built[key]

    def verilog(self, design_path, top="design"):
        """Verilog from the generator's `top` command, design or kria."""
        completed = subprocess.run([str(self.executable), top, str(design_path)],
                                   capture_output=True, text=True)
        if completed.returncode != 0:
            raise HardcamlError("the Hardcaml generator rejected %s:\n%s"
                                % (design_path, completed.stderr.strip()[-4000:]))
        return completed.stdout
