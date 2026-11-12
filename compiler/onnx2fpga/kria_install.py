"""Install a verified firmware package on a Kria Linux board."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def read_package(source, board):
    source = Path(source)
    data = (source / "package.json").read_bytes()
    record = json.loads(data)
    name = record.get("name", "")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name):
        raise ValueError("invalid firmware package name")
    if board not in ("kr260", "kv260") or record.get("board") != board:
        raise ValueError("package does not match the selected board")
    required = {name + suffix for suffix in (".bit.bin", ".dtbo", ".dts")}
    required |= {"shell.json", "manifest.json", "build.json"}
    hashes = record.get("sha256", {})
    if set(hashes) != required:
        raise ValueError("incomplete firmware package")
    files = {"package.json": data}
    for filename, digest in hashes.items():
        path = source / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("invalid package file: %s" % filename)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("package file changed: %s" % filename)
        files[filename] = content
    receipt = json.loads(files["build.json"])
    if receipt.get("board") != board:
        raise ValueError("build record names another board")
    if (receipt.get("inputs", {}).get("manifest.json") != hashes["manifest.json"]
            or receipt.get("outputs", {}).get(board + "/" + board + ".bin")
            != hashes[name + ".bit.bin"]):
        raise ValueError("package differs from successful build")
    return name, files


def preflight(symbols="/proc/device-tree/__symbols__", xmutil="xmutil"):
    for name in ("fpga_full", "amba", "zynqmp_clk", "zynqmp_reset"):
        if not (Path(symbols) / name).is_file():
            raise ValueError("missing device-tree symbol: %s" % name)
    command = shutil.which(xmutil)
    if command is None:
        raise ValueError("xmutil is not installed")
    subprocess.run([command, "listapps"], check=True)
    return command


def load(name, command):
    subprocess.run([command, "unloadapp"], check=True)
    subprocess.run([command, "loadapp", name], check=True)


def install(source, board, root="/lib/firmware/xilinx", *, program=False,
            symbols="/proc/device-tree/__symbols__", xmutil="xmutil", reuse=False):
    name, files = read_package(source, board)
    command = preflight(symbols, xmutil) if program else None
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = root / name
        if destination.exists() or destination.is_symlink():
            if not reuse or destination.is_symlink():
                raise ValueError("firmware application already exists: %s" % destination)
            _, installed = read_package(destination, board)
            if installed != files:
                raise ValueError("installed firmware differs from the requested package")
            if program:
                load(name, command)
            return destination
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            staged = Path(temporary) / name
            staged.mkdir(mode=0o755)
            for filename, content in files.items():
                target = staged / filename
                target.write_bytes(content)
                target.chmod(0o644)
            staged.rename(destination)
        if program:
            load(name, command)
    finally:
        os.close(lock)
    return destination


