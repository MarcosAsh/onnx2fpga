"""Firmware installation and programming without board access."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import shlex
import tempfile
import unittest
from unittest import mock

from onnx2fpga.kria_install import install
from onnx2fpga.kria_deploy import program, RECEIVE
from onnx2fpga.board import BoardError


class InstallTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.source, self.root, self.symbols = (base / p for p in ("source", "firmware", "symbols"))
        self.source.mkdir()
        self.symbols.mkdir()
        for name in ("fpga_full", "amba", "zynqmp_clk", "zynqmp_reset"):
            (self.symbols / name).write_bytes(b"/fixture\x00")
        files = {"demo.bit.bin": b"payload", "demo.dtbo": b"overlay", "demo.dts": b"source",
                 "manifest.json": b"{}", "shell.json": b'{"shell_type": "XRT_FLAT", "num_slots": "1"}'}
        digest = lambda content: hashlib.sha256(content).hexdigest()
        receipt = {"board": "kr260", "inputs": {"manifest.json": digest(files["manifest.json"])},
                   "outputs": {"kr260/kr260.bin": digest(files["demo.bit.bin"])}}
        files["build.json"] = json.dumps(receipt).encode()
        for name, content in files.items():
            (self.source / name).write_bytes(content)
        self.record = {"name": "demo", "board": "kr260",
                       "sha256": {name: digest(content) for name, content in files.items()}}
        self.write_record()

    def write_record(self):
        (self.source / "package.json").write_text(json.dumps(self.record))

    def install(self, **kwargs):
        return install(self.source, "kr260", self.root, symbols=self.symbols, **kwargs)

    def test_copy_without_programming(self):
        with mock.patch("onnx2fpga.kria_install.subprocess.run") as run:
            destination = self.install()
        run.assert_not_called()
        for path in self.source.iterdir():
            self.assertEqual((destination / path.name).read_bytes(), path.read_bytes())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.install()

    def test_modified_package_leaves_no_installation(self):
        (self.source / "demo.bit.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "file changed"):
            self.install()
        self.assertFalse(self.root.exists())

    def test_build_binding(self):
        (self.source / "manifest.json").write_bytes(b'{"different": true}')
        self.record["sha256"]["manifest.json"] = hashlib.sha256(
            (self.source / "manifest.json").read_bytes()).hexdigest()
        self.write_record()
        with self.assertRaisesRegex(ValueError, "successful build"):
            self.install()

    def test_wrong_board_and_paths(self):
        with self.assertRaisesRegex(ValueError, "selected board"):
            install(self.source, "kv260", self.root)
        self.record["name"] = "../demo"
        self.write_record()
        with self.assertRaisesRegex(ValueError, "package name"):
            self.install()

    def test_symlink_rejected(self):
        path = self.source / "demo.dtbo"
        path.unlink()
        path.symlink_to(self.source / "demo.dts")
        with self.assertRaisesRegex(ValueError, "invalid package file"):
            self.install()

    def test_program_sequence(self):
        with mock.patch("onnx2fpga.kria_install.shutil.which", return_value="/usr/bin/xmutil"), \
                mock.patch("onnx2fpga.kria_install.subprocess.run") as run:
            self.install(program=True)
        self.assertEqual(run.call_args_list, [
            mock.call(["/usr/bin/xmutil", "listapps"], check=True),
            mock.call(["/usr/bin/xmutil", "unloadapp"], check=True),
            mock.call(["/usr/bin/xmutil", "loadapp", "demo"], check=True)])

    def test_preflight_before_installation(self):
        (self.symbols / "amba").unlink()
        with mock.patch("onnx2fpga.kria_install.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "missing device-tree symbol"):
                self.install(program=True)
        run.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_failed_unload_stops_programming(self):
        with mock.patch("onnx2fpga.kria_install.shutil.which", return_value="xmutil"), \
                mock.patch("onnx2fpga.kria_install.subprocess.run", side_effect=[
                    subprocess.CompletedProcess([], 0), subprocess.CalledProcessError(1, "xmutil")]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.install(program=True)
        self.assertEqual(run.call_count, 2)
        self.assertTrue((self.root / "demo" / "package.json").is_file())

    def test_missing_xmutil(self):
        with mock.patch("onnx2fpga.kria_install.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "xmutil is not installed"):
                self.install(program=True)
        self.assertFalse(self.root.exists())

    def test_reuse_requires_identical_package(self):
        self.install()
        self.install(reuse=True)
        self.record["description"] = "another package"
        self.write_record()
        with self.assertRaisesRegex(ValueError, "differs"):
            self.install(reuse=True)

    def test_remote_package_roundtrip(self):
        commands = []

        def run(command, **kwargs):
            if command[0] == "ssh":
                self.assertEqual(command[1:7], ["-o", "BatchMode=yes", "-o",
                                              "ConnectTimeout=300", "--", "kria.local"])
                words = shlex.split(command[-1])
                self.assertEqual(words[:4], ["sudo", "-n", "python3", "-c"])
                source = words[4]
                self.assertTrue(source.endswith(RECEIVE))
                namespace = {}
                exec(source[:-len(RECEIVE)], namespace)
                remote_install = namespace["install"]
                namespace["install"] = lambda source, board, **options: remote_install(
                    source, board, root=self.root, symbols=self.symbols, **options)
                with mock.patch("sys.stdin", io.StringIO(kwargs["input"])):
                    exec(RECEIVE, namespace)
            else:
                commands.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch("onnx2fpga.kria_deploy.subprocess.run", side_effect=run), \
                mock.patch("onnx2fpga.kria_install.shutil.which", return_value="xmutil"):
            for _ in range(2):
                program("kria.local", self.source, "kr260", self.source / "manifest.json")
        self.assertEqual(commands, [["xmutil", "listapps"], ["xmutil", "unloadapp"],
                                    ["xmutil", "loadapp", "demo"]] * 2)
        self.assertEqual((self.root / "demo" / "demo.bit.bin").read_bytes(), b"payload")

    def test_remote_validation_before_ssh(self):
        other = self.source / "other.json"
        other.write_text('{"different": true}')
        with mock.patch("onnx2fpga.kria_deploy.subprocess.run") as run:
            with self.assertRaisesRegex(BoardError, "compiled model"):
                program("kria.local", self.source, "kr260", other)
            with self.assertRaisesRegex(ValueError, "SSH host"):
                program("host;bad", self.source, "kr260", self.source / "manifest.json")
        run.assert_not_called()

    def test_remote_failure_reported(self):
        with mock.patch("onnx2fpga.kria_deploy.subprocess.run", return_value=
                        subprocess.CompletedProcess([], 1, "", "load failed")):
            with self.assertRaisesRegex(BoardError, "load failed"):
                program("kria.local", self.source, "kr260", self.source / "manifest.json")
