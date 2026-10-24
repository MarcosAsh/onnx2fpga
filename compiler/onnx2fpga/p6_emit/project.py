"""Writes a self-contained build directory: rtl, mem, golden, manifest.json and a Makefile."""

import json
import pathlib
import shutil

from ..assets import AssetLocator
from ..errors import ConfigError
from .hardcaml import HardcamlEmitter
from .memories import HexImage
from .systemverilog import NetlistBuilder


class BuildDirectory:
    def __init__(self, root):
        self.root = pathlib.Path(root)

    def prepare(self, subdirs=("rtl", "mem", "golden")):
        for sub in subdirs:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

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

    def write(self, out_dir):
        self._check_modules()
        if self.hdl == "hardcaml":
            build = BuildDirectory(out_dir).prepare(("rtl", "golden"))
            HardcamlEmitter(self.graph, self.name, self.assets, self.latency).write(build)
        else:
            build = BuildDirectory(out_dir).prepare()
            self._copy_library(build)
            self._write_unit_sources(build)
            self._write_memories(build)
            build.write("rtl/top.sv",
                        NetlistBuilder(self.graph, self.name).build().render())
        build.write("Makefile", self._makefile())
        build.write("manifest.json", json.dumps(self._manifest(), indent=2) + "\n")
        return build

    @property
    def latency(self):
        return self.context.artifacts.get("latency")

    def _check_modules(self):
        """Every unit must be a library module unless it brought its own source; checking here names the unit."""
        library = {source.stem for source in self.assets.rtl_sources()}
        for node in self.graph.topological_order():
            if node.module in library or (self.hdl == "hardcaml"
                                           and getattr(node, "hardcaml_only", False)):
                continue
            if self.hdl == "hardcaml":
                raise ConfigError(
                    "%s is implemented by %r, which the Hardcaml generator does not "
                    "have: it elaborates only the library units (%s), chosen in "
                    "hardware/hardcaml/src/units.ml. Compile with hdl='sv', or add a "
                    "Hardcaml unit for it there"
                    % (node.name, node.module, ", ".join(sorted(library))))
            if "%s.sv" % node.module not in node.rtl_files():
                raise ConfigError(
                    "%s is implemented by %r, which is not in the RTL library (%s). "
                    "Give its SystemVerilog source as the layer's rtl attribute"
                    % (node.name, node.module, ", ".join(sorted(library))))

    def _copy_library(self, build):
        for source in self.assets.rtl_sources():
            shutil.copyfile(source, build.path("rtl", source.name))

    def _write_unit_sources(self, build):
        for node in self.graph.topological_order():
            for filename, text in sorted(node.rtl_files().items()):
                build.write("rtl/" + filename, text)

    def _write_memories(self, build):
        for node in self.graph.topological_order():
            specs = node.memories(self.graph)
            if not specs:
                continue
            paths = {}
            for key, spec in specs.items():
                filename = "%s_%s.hex" % (self._safe(node.name), key)
                image = (HexImage.packed(spec.array, spec.elem_bits) if spec.packed
                         else HexImage.scalar(spec.array, spec.elem_bits))
                image.write(build.path("mem", filename))
                paths[key] = "mem/" + filename
            node.attrs["mem_paths"] = paths

    def _manifest(self):
        folding = self.context.artifacts.get("folding")
        plan = self.context.artifacts.get("plan")
        source = self.graph.tensor(self.graph.inputs[0])
        sink = self.graph.tensor(self.graph.outputs[0])
        manifest = {
            "name": self.name,
            "device": self.context.device.name,
            "part": self.context.device.part,
            "fmax_mhz": self.context.device.fmax_mhz,
            "input": self._port(source, self.graph.layouts.get(source.name)),
            "output": self._port(sink),
            "nodes": [{"name": name, "kind": kind, "folding": fold, "cycles": cycles}
                      for name, kind, fold, cycles, _ in
                      (folding.rows if folding else [])],
        }
        if self.latency is not None:
            manifest["mode"] = "fixed"
            manifest["latency"] = self.latency
            for row, node in zip(manifest["nodes"], self.graph.topological_order()):
                row["latency"] = node.latency
        if folding:
            manifest["cycles_per_frame"] = folding.bottleneck_cycles
            manifest["estimated_fps"] = round(folding.frames_per_second, 2)
            manifest["resources"] = {f: round(getattr(folding.total, f), 1)
                                     for f in folding.total.FIELDS}
        if plan:
            for side, tensor in (("input", self.graph.inputs[0]),
                                 ("output", self.graph.outputs[0])):
                manifest[side]["scale"] = plan.scale(tensor)
                manifest[side]["zero_point"] = plan.zero_point(tensor)
                manifest[side]["symmetric"] = plan.spec(tensor).is_symmetric
                if plan.spec(tensor).rebased:
                    manifest[side]["note"] = (
                        "the model declared this port unsigned; it is carried on "
                        "the signed grid, which the scale and zero point above "
                        "already account for")
            manifest["quantization"] = {
                "source": "model" if plan.from_model else "calibration",
                "activations": plan.act_dtype.name,
                "weights": plan.weight_dtype.name,
            }
            if plan.layer_weight_dtypes:
                manifest["quantization"]["layer_weights"] = {
                    name: dtype.name
                    for name, dtype in sorted(plan.layer_weight_dtypes.items())}
        return manifest

    @staticmethod
    def _port(tensor, layout=None):
        described = {
            "tensor": tensor.name,
            "shape": list(tensor.shape or []),
            "dtype": tensor.dtype.name,
            "elem_bits": tensor.dtype.bits,
            "elems_per_beat": tensor.elems_per_beat,
            "beats": tensor.beats,
            "layout": "channel-last",
        }
        if layout is not None:
            described.update(layout.describe())
        return described

    def _makefile(self):
        """Hardcaml output trips -Wall on unread top bits and case-based multiplexers, so those lints are waived."""
        hardcaml = self.hdl == "hardcaml"
        fixed = self.latency is not None
        return MAKEFILE_TEMPLATE % {
            "top": self.name,
            "root": self.assets.root,
            "harness": "fixed_harness.cpp" if fixed else "harness.cpp",
            "run_flags": " --latency %d" % self.latency if fixed else "",
            "sources": "rtl/top.v" if hardcaml else "$(wildcard rtl/*.sv)",
            "waivers": HARDCAML_WAIVERS if hardcaml else "",
        }

    @staticmethod
    def _safe(name):
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


MAKEFILE_TEMPLATE = """# Generated by onnx2fpga.
VERILATOR ?= verilator
OTF_ROOT  := %(root)s
TOP       := %(top)s
SOURCES   := %(sources)s
HARNESS   := $(OTF_ROOT)/runtime/sim/%(harness)s
WARNINGS  := -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM%(waivers)s
CXXFLAGS  := -O2 -std=c++17 -I$(OTF_ROOT)/runtime/include

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

HARDCAML_WAIVERS = " -Wno-COMBDLY -Wno-UNUSEDSIGNAL"
