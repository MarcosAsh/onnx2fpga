"""The Hardcaml backend: the same netlist written as design.json and elaborated to Verilog in OCaml."""

import json
import os
import pathlib
import re
import shutil
import subprocess

from ..assets import AssetError, AssetLocator
from ..errors import Onnx2FpgaError


class DesignDescription:
    """The scheduled hardware graph as the generator reads it."""

    FORMAT = 1

    def __init__(self, graph, name="otf_top", latency=None):
        self.graph = graph
        self.name = name
        self.latency = latency

    def build(self):
        graph = self.graph
        streams, seen = [], set()
        for name in [graph.inputs[0]] + [out for node in graph.topological_order()
                                         for out in node.outputs]:
            if name not in seen:
                seen.add(name)
                streams.append({"name": name, "width": self._width(name)})
        design = {"format": self.FORMAT, "top": self.name}
        if self.latency is not None:
            design.update(mode="fixed", latency=self.latency)
        design.update({
            "input": graph.inputs[0],
            "output": graph.outputs[0],
            "streams": streams,
            "nodes": [self._node(node) for node in graph.topological_order()],
        })
        return design

    def render(self):
        """Key order is insertion order and values are ints or strings, so output bytes are deterministic."""
        return json.dumps(self.build(), indent=1) + "\n"

    def _node(self, node):
        graph = self.graph
        params = {key: int(value) for key, value in node.sv_params(graph).items()
                  if not isinstance(value, str)}
        return {
            "name": node.name,
            "instance": "u_" + self._identifier(node.name),
            "kind": node.module,
            "params": params,
            "memories": {key: self._memory(spec)
                         for key, spec in node.memories(graph).items()},
            "inputs": node.stream_inputs(graph),
            "outputs": list(node.outputs),
        }

    @staticmethod
    def _memory(spec):
        """Masked to the element width like the hex images, matching what $readmemh would load."""
        lanes = int(spec.array.shape[-1]) if spec.packed else 1
        mask = (1 << spec.elem_bits) - 1
        flat = [int(v) & mask for v in spec.array.reshape(-1).tolist()]
        return {"elem_bits": spec.elem_bits, "lanes": lanes, "values": flat}

    def _width(self, tensor_name):
        tensor = self.graph.tensor(tensor_name)
        return tensor.elems_per_beat * tensor.dtype.bits

    @staticmethod
    def _identifier(name):
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


class HardcamlError(Onnx2FpgaError):
    pass


class HardcamlGenerator:
    """Builds and runs hardware/hardcaml/bin/generate.exe, found beside the RTL library, in the checkout, or at OTF_HARDCAML."""

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
        """None if the backend can run, else why not; a project that fails to build is broken, not unavailable."""
        project = self.project
        if project is None:
            return ("no Hardcaml project in %s"
                    % ", ".join(str(c) for c in self.candidates()))
        if shutil.which(self.dune) is None:
            return "dune is not on PATH; activate the opam switch with hardcaml"
        returncode, errors = self._build(project)
        missing = sorted(set(self.MISSING_LIBRARY.findall(errors)))
        if returncode != 0 and missing:
            return "the active opam switch has no %s" % ", ".join(missing)
        return None

    def available(self):
        return self.unavailable() is None

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
                self._built[key] = (-1, "dune build timed out after %ds; is another "
                                        "dune holding %s/_build?"
                                    % (self.BUILD_TIMEOUT, project))
        return self._built[key]

    def verilog(self, design_path):
        completed = subprocess.run([str(self.executable), "design", str(design_path)],
                                   capture_output=True, text=True)
        if completed.returncode != 0:
            raise HardcamlError("the Hardcaml generator rejected %s:\n%s"
                                % (design_path, completed.stderr.strip()[-4000:]))
        return completed.stdout


class HardcamlEmitter:
    """Writes design.json and the Verilog the generator makes of it."""

    def __init__(self, graph, name="otf_top", assets=None, latency=None):
        self.graph = graph
        self.name = name
        self.latency = latency
        self.generator = HardcamlGenerator(assets)

    def write(self, build):
        design = build.write("design.json", DesignDescription(
            self.graph, self.name, self.latency).render())
        build.write("rtl/top.v", self.generator.verilog(design.resolve()))
        return build
