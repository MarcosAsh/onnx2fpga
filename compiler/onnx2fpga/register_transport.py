"""Deadline-bound connection to the native Kria register helper."""

import math
import os
import re
import selectors
import subprocess
import tempfile
import time

from .board import BoardError


def ssh_command(host, timeout):
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]*", host):
        raise ValueError("invalid SSH host")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    return ["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=%d" % max(1, math.ceil(timeout)), "--", host]


def connect(host, manifest, timeout=2.0):
    """Connects to an installed kria_regs helper using the user's SSH configuration."""
    command = ssh_command(host, timeout)
    layout = manifest["registers"]
    base = layout["base"]
    if (layout["span"] != 0x10000 or not isinstance(base, int)
            or base < 0 or base % 4096):
        raise BoardError("invalid Kria register window")
    return ProcessRegisters(command + ["kria_regs --base %x" % base], timeout)


class ProcessRegisters:
    """Owns one helper process; a failed exchange closes the connection."""

    def __init__(self, command, timeout=2.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self.process = None
        self.selector = selectors.DefaultSelector()
        self.errors = tempfile.TemporaryFile()
        self.pending = bytearray()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.errors,
                                            bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            if self._line(time.monotonic() + timeout) != b"OTFREG1":
                raise BoardError("unexpected Kria helper protocol")
        except Exception:
            self.close()
            raise

    def _line(self, deadline):
        while b"\n" not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise BoardError("Kria register transport timed out")
            chunk = os.read(self.process.stdout.fileno(), 256)
            if not chunk:
                raise BoardError("Kria register helper disconnected")
            self.pending.extend(chunk)
            if len(self.pending) > 256:
                raise BoardError("Kria register response is too long")
        line, _, rest = self.pending.partition(b"\n")
        self.pending = bytearray(rest)
        return bytes(line)

    def _exchange(self, command, reading):
        if self.process is None:
            raise BoardError("Kria register transport is closed")
        try:
            deadline = time.monotonic() + self.timeout
            # Commands fit in PIPE_BUF; a nonblocking write either completes or fails.
            if os.write(self.process.stdin.fileno(), command) != len(command):
                raise BoardError("incomplete Kria register command")
            response = self._line(deadline)
            if not reading and response == b"ok":
                return None
            if reading and response.startswith(b"ok "):
                word = response[3:]
                if 1 <= len(word) <= 8 and all(c in b"0123456789abcdefABCDEF" for c in word):
                    return int(word, 16)
            raise BoardError("invalid Kria register response")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _offset(offset):
        if not isinstance(offset, int) or not 0 <= offset < 0x10000 or offset % 4:
            raise ValueError("register offset must be aligned within the 64 KiB window")

    def read(self, offset):
        self._offset(offset)
        return self._exchange(("r %x\n" % offset).encode(), True)

    def write(self, offset, value):
        self._offset(offset)
        if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("register value must be an unsigned 32-bit integer")
        self._exchange(("w %x %x\n" % (offset, value)).encode(), False)

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            process.stdin.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            process.stdout.close()
        self.selector.close()
        self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
