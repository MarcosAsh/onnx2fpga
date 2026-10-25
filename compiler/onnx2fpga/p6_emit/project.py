"""Writes a self-contained build directory: rtl, mem, golden, manifest.json and a Makefile."""

import json
import pathlib
import shutil

from ..assets import AssetLocator
from ..errors import ConfigError
from ..p3_ops.hardware_ops import DenseUnit
from ..reload import layout, reloadable_units
from .hardcaml import HardcamlGenerator, design
from .memories import HexImage
from .systemverilog import identifier, render_top


class BuildDirectory:
    def __init__(self, root, subdirs):
        self.root = pathlib.Path(root)
        for sub in subdirs:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def path(self, *parts):
        return self.root.joinpath(*parts)

    def write(self, relative, text):
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target


class ProjectWriter:
    HDLS = ("sv", "hardcaml")

    def __init__(self, graph, context, name="otf_top", assets=None, hdl="sv"):
        if hdl not in self.HDLS:
            raise ValueError("unknown hdl %r, expected one of %s"
                             % (hdl, ", ".join(self.HDLS)))
        self.graph = graph
        self.context = context
        self.name = name
        self.assets = assets if isinstance(assets, AssetLocator) else AssetLocator(assets)
        self.hdl = hdl
        self.latency = context.artifacts.get("latency")

    def write(self, out_dir):
        self._check_modules()
        nodes = self.graph.topological_order()
        if self.hdl == "hardcaml":
            build = BuildDirectory(out_dir, ("rtl", "golden"))
            path = build.write("design.json", json.dumps(
                design(self.graph, self.name, self.latency), indent=1) + "\n")
            build.write("rtl/top.v", HardcamlGenerator(self.assets).verilog(path.resolve()))
        else:
            build = BuildDirectory(out_dir, ("rtl", "mem", "golden"))
            for source in self.assets.rtl_sources():
                shutil.copyfile(source, build.path("rtl", source.name))
            for node in nodes:
                for filename, text in sorted(node.rtl_files().items()):
                    build.write("rtl/" + filename, text)
            for node in nodes:
                self._write_memories(node, build)
            build.write("rtl/top.sv", render_top(self.graph, self.name))
        build.write("Makefile", self._makefile())
        build.write("manifest.json", json.dumps(self._manifest(), indent=2) + "\n")
        return build

    def _check_modules(self):
        """Every unit must be a library module or bring its own source."""
        library = {source.stem for source in self.assets.rtl_sources()}
        for node in self.graph.topological_order():
            if node.module in library or (self.hdl == "hardcaml"
                                           and getattr(node, "hardcaml_only", False)):
                continue
            if self.hdl == "hardcaml":
                raise ConfigError("%s uses %r, which has no Hardcaml unit in units.ml; "
                                  "compile it with hdl='sv'" % (node.name, node.module))
            if "%s.sv" % node.module not in node.rtl_files():
                raise ConfigError("%s uses %r, which is not in the RTL library and has "
                                  "no rtl source" % (node.name, node.module))

    def _write_memories(self, node, build):
        paths = {}
        for key, spec in node.memories(self.graph).items():
            filename = "%s_%s.hex" % (identifier(node.name), key)
            array = spec.array if spec.packed else spec.array.reshape(-1, 1)
            HexImage.packed(array, spec.elem_bits).write(build.path("mem", filename))
            paths[key] = "mem/" + filename
        if paths:
            node.attrs["mem_paths"] = paths

    def _manifest(self):
        folding = self.context.artifacts.get("folding")
        plan = self.context.artifacts.get("plan")
        device = self.context.device
        source = self.graph.tensor(self.graph.inputs[0])
        sink = self.graph.tensor(self.graph.outputs[0])
        inputs = port(source)
        if source.name in self.graph.layouts:
            inputs.update(self.graph.layouts[source.name].describe())
        manifest = {
            "name": self.name,
            "device": device.name,
            "part": device.part,
            "fmax_mhz": device.fmax_mhz,
            "input": inputs,
            "output": port(sink),
            "nodes": [{"name": name, "kind": kind, "folding": fold, "cycles": cycles}
                      for name, kind, fold, cycles, _ in
                      (folding.rows if folding else [])],
        }
        if self.latency is not None:
            manifest["mode"] = "fixed"
            manifest["latency"] = self.latency
            for row, node in zip(manifest["nodes"], self.graph.topological_order()):
                row["latency"] = node.latency
                if isinstance(node, DenseUnit):
                    row["arithmetic"] = node.arithmetic
            manifest["weights"] = (layout(self.graph) if reloadable_units(self.graph)
                                   else {"mode": "frozen"})
        if folding:
            manifest["cycles_per_frame"] = folding.bottleneck_cycles
            manifest["estimated_fps"] = round(folding.frames_per_second, 2)
            manifest["resources"] = {f: round(getattr(folding.total, f), 1)
                                     for f in folding.total.FIELDS}
        if plan:
            for side, tensor in (("input", source.name), ("output", sink.name)):
                spec = plan.spec(tensor)
                per_feature = spec.scale.size > 1
                manifest[side]["scale"] = (spec.scale.tolist() if per_feature
                                           else plan.scale(tensor))
                manifest[side]["zero_point"] = (spec.zero_point.tolist() if per_feature
                                                else plan.zero_point(tensor))
                manifest[side]["symmetric"] = spec.is_symmetric
                if spec.rebased:
                    manifest[side]["note"] = (
                        "the model declared this port unsigned; it is carried on "
                        "the signed grid, which the scale and zero point above "
                        "already account for")
            manifest["quantization"] = {
                "source": plan.source,
                "activations": plan.act_dtype.name,
                "weights": plan.weight_dtype.name,
            }
            if plan.layer_weight_dtypes:
                manifest["quantization"]["layer_weights"] = {
                    name: dtype.name
                    for name, dtype in sorted(plan.layer_weight_dtypes.items())}
        return manifest

    def _makefile(self):
        """Hardcaml output trips -Wall on unread bits and case multiplexers, so those are waived."""
        hardcaml = self.hdl == "hardcaml"
        fixed = self.latency is not None
        reload = " -DOTF_RELOAD" if reloadable_units(self.graph) else ""
        return MAKEFILE_TEMPLATE % {
            "top": self.name,
            "root": self.assets.root,
            "harness": "fixed_harness.cpp" if fixed else "harness.cpp",
            "run_flags": " --latency %d" % self.latency if fixed else "",
            "sources": "rtl/top.v" if hardcaml else "$(wildcard rtl/*.sv)",
            "waivers": " -Wno-COMBDLY -Wno-UNUSEDSIGNAL" if hardcaml else "",
            "defines": reload,
        }


def port(tensor):
    return {
        "tensor": tensor.name,
        "shape": list(tensor.shape or []),
        "dtype": tensor.dtype.name,
        "elem_bits": tensor.dtype.bits,
        "elems_per_beat": tensor.elems_per_beat,
        "beats": tensor.beats,
        "layout": "channel-last",
    }


MAKEFILE_TEMPLATE = """# Generated by onnx2fpga.
VERILATOR ?= verilator
OTF_ROOT  := %(root)s
TOP       := %(top)s
SOURCES   := %(sources)s
HARNESS   := $(OTF_ROOT)/runtime/sim/%(harness)s
WARNINGS  := -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM%(waivers)s
CXXFLAGS  := -O2 -std=c++17 -I$(OTF_ROOT)/runtime/include%(defines)s

.PHONY: all run lint clean

all: obj_dir/V$(TOP)

obj_dir/V$(TOP): $(SOURCES) $(HARNESS)
\t$(VERILATOR) --cc --exe --build -j 0 $(WARNINGS) \\
\t  -CFLAGS "$(CXXFLAGS)" --top-module $(TOP) \\
\t  $(SOURCES) $(HARNESS) -o V$(TOP)

run: obj_dir/V$(TOP)
\t./obj_dir/V$(TOP) --input golden/input.hex --expected golden/expected.hex%(run_flags)s

lint:
\t$(VERILATOR) --lint-only $(WARNINGS) --top-module $(TOP) $(SOURCES)

clean:
\trm -rf obj_dir
"""
