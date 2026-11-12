#!/usr/bin/env python3
"""Package a routed Kria register design for an AMD Linux FPGA-manager image."""

import argparse
import hashlib
import io
import json
import pathlib
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal


def bit_payload(data):
    """Extracts the length-delimited payload from a Vivado .bit file."""
    cursor = 0

    def take(size):
        nonlocal cursor
        value = data[cursor:cursor + size]
        if len(value) != size:
            raise ValueError("truncated bitstream")
        cursor += size
        return value

    def length(size):
        return int.from_bytes(take(size), "big")

    if take(length(2)) != bytes.fromhex("0ff00ff00ff00ff000") or take(2) != b"\x00\x01":
        raise ValueError("invalid bitstream header")
    for tag in b"abcd":
        if take(1) != bytes([tag]):
            raise ValueError("invalid bitstream metadata")
        take(length(2))
    if take(1) != b"e":
        raise ValueError("missing bitstream payload")
    payload = take(length(4))
    if not payload or cursor != len(data):
        raise ValueError("invalid bitstream payload length")
    return payload


def hardware(xsa, manifest):
    with zipfile.ZipFile(xsa) as archive:
        root = ET.fromstring(archive.read("system.hwh"))
    ps = root.find(".//MODULE[@MODTYPE='zynq_ultra_ps_e']")
    if ps is None:
        raise ValueError("XSA has no Zynq UltraScale+ PS")
    params = {p.attrib["NAME"]: p.attrib["VALUE"]
              for p in ps.findall("./PARAMETERS/PARAMETER") if "VALUE" in p.attrib}
    if params["C_PL_CLK0_BUF"].upper() != "TRUE":
        raise ValueError("XSA does not enable PL0")
    rate = int(Decimal(params["PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ"]) * 1000000)
    if not 99900000 <= rate <= 100100000:
        raise ValueError("Kria inference shell requires a nominal 100 MHz PS clock")
    layout = manifest["registers"]
    matches = [m for m in ps.findall(".//MEMRANGE")
               if m.attrib.get("INSTANCE") == "M_AXI"]
    if len(matches) != 1:
        raise ValueError("XSA must expose one M_AXI register window")
    region = matches[0].attrib
    base, high = int(region["BASEVALUE"], 16), int(region["HIGHVALUE"], 16)
    if (base != layout["base"] or high - base + 1 != layout["span"]
            or layout["span"] != 0x10000 or base > 0xFFFFFFFF):
        raise ValueError("XSA register window disagrees with manifest")
    slave = {128: 0, 64: 1, 32: 2}
    master = {32: 0, 64: 1, 128: 2}
    afi = []
    for index in range(7):
        width = int(params["C_SAXIGP%d_DATA_WIDTH" % index])
        afi.extend([(2 * index, slave[width]), (2 * index + 1, slave[width])])
    widths = [int(params["C_MAXIGP%d_DATA_WIDTH" % i]) for i in range(3)]
    if widths[0] != 32:
        raise ValueError("Kria register shell requires a 32-bit PS master")
    afi.extend([(14, (master[widths[0]] << 8) | (master[widths[1]] << 10)),
                (15, master[widths[2]] << 8)])
    return rate, base, afi


def overlay(name, rate, base, afi):
    config = ", ".join("<%d 0x%x>" % pair for pair in afi)
    return '''/dts-v1/;
/plugin/;

&fpga_full {
    firmware-name = "%s.bit.bin";
    resets = <&zynqmp_reset 116>, <&zynqmp_reset 117>,
             <&zynqmp_reset 118>, <&zynqmp_reset 119>;
};

&amba {
    #address-cells = <2>;
    #size-cells = <2>;
    onnx2fpga_clock {
        compatible = "xlnx,fclk";
        clocks = <&zynqmp_clk 71>;
        assigned-clocks = <&zynqmp_clk 71>;
        assigned-clock-rates = <%d>;
    };
    onnx2fpga_afi {
        compatible = "xlnx,afi-fpga";
        config-afi = %s;
        resets = <&zynqmp_reset 116>, <&zynqmp_reset 117>,
                 <&zynqmp_reset 118>, <&zynqmp_reset 119>;
    };
    onnx2fpga@%x {
        compatible = "onnx2fpga,registers";
        reg = <0 0x%x 0 0x10000>;
    };
};
''' % (name, rate, config, base, base)


def verify_build(build, board, manifest):
    path = build / board / "build.json"
    if not path.is_file():
        raise ValueError("missing successful build record; rerun the board build")
    record = json.loads(path.read_text())
    if record.get("board") != board:
        raise ValueError("build record names another board")
    prefix = "ddr" if manifest.get("kind") == "ddr_trading" else "kria"
    required = {
        "inputs": {"manifest.json"} | {board + "/" + name for name in
                    ("build.tcl", "check_timing.tcl", "settings.tcl", "kria.xdc",
                     prefix + "_top.v", prefix + "_shell.v")},
        "outputs": {board + "/" + board + suffix for suffix in (".bit", ".bin", ".xsa")}
                   | {board + "/" + name for name in
                      ("timing_summary.rpt", "utilization.rpt", "methodology.rpt")},
    }
    for section, names in required.items():
        hashes = record.get(section, {})
        if not names <= hashes.keys():
            raise ValueError("incomplete build record")
        for name, digest in hashes.items():
            artifact = build / name
            if not artifact.resolve().is_relative_to(build.resolve()):
                raise ValueError("build record path escapes the build directory")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                raise ValueError("build artifact changed: %s" % name)
    return record


def package(build, board, destination, name="onnx2fpga", dtc="dtc"):
    build, destination = pathlib.Path(build), pathlib.Path(destination)
    if board not in ("kr260", "kv260"):
        raise ValueError("expected a Kria board")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name):
        raise ValueError("invalid firmware package name")
    if destination.exists():
        raise ValueError("package destination already exists")
    manifest_data = (build / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    folder = build / board
    xsa = folder / (board + ".xsa")
    xsa_data = xsa.read_bytes()
    bit = (folder / (board + ".bit")).read_bytes()
    binary = (folder / (board + ".bin")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(xsa_data)) as archive:
        if archive.read(board + ".bit") != bit:
            raise ValueError("XSA bitstream differs from board bitstream")
    if bit_payload(bit) != binary:
        raise ValueError("binary payload differs from board bitstream")
    receipt = verify_build(build, board, manifest)
    snapshots = {"manifest.json": manifest_data, board + "/" + board + ".bit": bit,
                 board + "/" + board + ".bin": binary, board + "/" + board + ".xsa": xsa_data}
    hashes = {**receipt["inputs"], **receipt["outputs"]}
    if any(hashlib.sha256(data).hexdigest() != hashes[name] for name, data in snapshots.items()):
        raise ValueError("build artifacts changed while packaging")
    rate, base, afi = hardware(io.BytesIO(xsa_data), manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        out = pathlib.Path(temporary) / name
        out.mkdir()
        (out / (name + ".bit.bin")).write_bytes(binary)
        (out / (name + ".dts")).write_text(overlay(name, rate, base, afi))
        subprocess.run([dtc, "-@", "-I", "dts", "-O", "dtb", "-o",
                        str(out / (name + ".dtbo")), str(out / (name + ".dts"))],
                       check=True, capture_output=True, text=True)
        (out / "shell.json").write_text(json.dumps({"shell_type": "XRT_FLAT", "num_slots": "1"}) + "\n")
        (out / "manifest.json").write_bytes(manifest_data)
        (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
        record = {"board": board, "ps_clock_hz": rate, "register_base": base,
                  "xsa_sha256": hashlib.sha256(xsa_data).hexdigest(),
                  "physical_validation": "not_run",
                  "requires": ["fpga_full, amba, zynqmp_clk and zynqmp_reset DT symbols",
                               "xlnx,fclk and xlnx,afi-fpga kernel drivers", "xmutil/dfx-mgr"],
                  "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(out.iterdir())}}
        (out / "package.json").write_text(json.dumps(record, indent=2) + "\n")
        out.rename(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    parser.add_argument("--board", choices=("kr260", "kv260"), default="kr260")
    parser.add_argument("--name", default="onnx2fpga")
    parser.add_argument("--dtc", default="dtc")
    args = parser.parse_args()
    print(package(args.build, args.board, args.destination, args.name, args.dtc))
