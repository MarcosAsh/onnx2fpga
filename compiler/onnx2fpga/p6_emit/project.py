"""Writes a self-contained build directory.

Layout produced:

    <out>/rtl/          top.sv plus the hand-written module library
    <out>/mem/          $readmemh images for weights, biases and multipliers
    <out>/golden/       stimulus and expected output for the simulation harness
    <out>/manifest.json what the host runtime needs to talk to the design
    <out>/Makefile      builds and runs the Verilator harness
"""

import json
import pathlib
import shutil

from ..assets import AssetLocator
from .memories import HexImage
from .systemverilog import NetlistBuilder


class BuildDirectory:
    def __init__(self, root):
        self.root = pathlib.Path(root)

    def prepare(self):
        for sub in ("rtl", "mem", "golden"):
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
    def __init__(self, graph, context, name="otf_top", assets=None):
        self.graph = graph
        self.context = context
        self.name = name
        self.assets = assets if isinstance(assets, AssetLocator) else AssetLocator(assets)

    def write(self, out_dir):
        build = BuildDirectory(out_dir).prepare()
        self._copy_library(build)
        self._write_memories(build)
        build.write("rtl/top.sv", NetlistBuilder(self.graph, self.name).build().render())
        build.write("Makefile", self._makefile())
        build.write("manifest.json", json.dumps(self._manifest(), indent=2) + "\n")
        return build

    def _copy_library(self, build):
        for source in self.assets.rtl_sources():
            shutil.copyfile(source, build.path("rtl", source.name))

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
        return MAKEFILE_TEMPLATE % {
            "top": self.name,
            "root": self.assets.root,
        }

    @staticmethod
    def _safe(name):
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


MAKEFILE_TEMPLATE = """# Generated by onnx2fpga.
VERILATOR ?= verilator
OTF_ROOT  := %(root)s
TOP       := %(top)s
SOURCES   := $(wildcard rtl/*.sv)
HARNESS   := $(OTF_ROOT)/runtime/sim/harness.cpp
WARNINGS  := -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM
CXXFLAGS  := -O2 -std=c++17 -I$(OTF_ROOT)/runtime/include

.PHONY: all run lint clean

all: obj_dir/V$(TOP)

obj_dir/V$(TOP): $(SOURCES) $(HARNESS)
\t$(VERILATOR) --cc --exe --build -j 0 $(WARNINGS) \\
\t  -CFLAGS "$(CXXFLAGS)" --top-module $(TOP) \\
\t  $(SOURCES) $(HARNESS) -o V$(TOP)

run: obj_dir/V$(TOP)
\t./obj_dir/V$(TOP) --input golden/input.hex --expected golden/expected.hex

lint:
\t$(VERILATOR) --lint-only $(WARNINGS) --top-module $(TOP) $(SOURCES)

clean:
\trm -rf obj_dir
"""
