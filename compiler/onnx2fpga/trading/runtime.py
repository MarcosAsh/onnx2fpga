"""Host control of the KR260 DDR trading replay engine."""

import io
import math
import time

from ..board import BoardError
from .board import OFFSETS
from .replay_records import read_records


class ReplaySession:
    """Owns buffer handoffs; register and buffer lifetimes remain with the caller."""

    def __init__(self, registers, buffer, manifest, timeout=30.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if manifest.get("kind") != "ddr_trading":
            raise BoardError("expected DDR trading manifest")
        entries = manifest["registers"]["map"]
        if any(entries[name]["offset"] != offset for name, offset in OFFSETS.items()):
            raise BoardError("unexpected DDR register layout")
        ddr = manifest["ddr"]
        if (ddr["record_bytes"] != 16 or ddr["base_alignment"] != 256
                or ddr["address_bits"] != 40 or ddr["mapped_base"] != 0
                or ddr["mapped_bytes"] != 0x80000000):
            raise BoardError("unexpected DDR address window")
        if (buffer.address < 0 or buffer.address % 256 or buffer.size < 16
                or buffer.address + buffer.size > ddr["mapped_bytes"]):
            raise BoardError("DMA allocation is outside the aligned DDR window")
        self.registers, self.buffer = registers, buffer
        self.timeout = timeout
        self.latency = entries["latency"]["value"]
        self.closed = True
        self.verified = False
        self.needs_abort = True

    def read(self, name):
        return self.registers.read(OFFSETS[name])

    def _wait(self, complete=False):
        deadline = time.monotonic() + self.timeout
        while True:
            status = self.read("status")
            if not status & 1 and (not complete or status & 2):
                return status
            if time.monotonic() >= deadline:
                raise BoardError("DDR replay timed out; abort before reusing the buffer")
            time.sleep(0.001)

    def open(self):
        if not self.closed:
            return self
        for name, expected in (("id", 0x4F544432), ("version", 1), ("latency", self.latency)):
            if self.read(name) != expected:
                raise BoardError("unexpected DDR %s register" % name)
        self.verified = True
        self.abort()
        self.closed = False
        return self

    def abort(self):
        if not self.verified:
            raise BoardError("open the DDR session before aborting")
        self.needs_abort = True
        self.registers.write(OFFSETS["control"], 2)
        self._wait()
        self.buffer.acquire()
        self.needs_abort = False

    def run(self, payload):
        if self.closed or self.needs_abort:
            raise BoardError("DDR session needs open or abort")
        if not isinstance(payload, bytes) or not payload or len(payload) % 16:
            raise ValueError("expected nonempty packed DDR records")
        if len(payload) > self.buffer.size or len(payload) // 16 > 0xFFFFFFFF:
            raise BoardError("replay exceeds DMA allocation or record count")
        count = sum(1 for _ in read_records(io.BytesIO(payload)))
        if self.read("status") & 1:
            self.needs_abort = True
            raise BoardError("DDR replay is already busy")
        self.needs_abort = True
        self.buffer.write(payload)
        self.buffer.submit()
        for name, value in (("base_low", self.buffer.address & 0xFFFFFFFF),
                            ("base_high", self.buffer.address >> 32), ("records", count)):
            self.registers.write(OFFSETS[name], value)
        self.registers.write(OFFSETS["control"], 1)
        status = self._wait(complete=True)
        self.buffer.acquire()
        if status & 12:
            raise BoardError("DDR replay failed with status 0x%x" % status)
        result = {name: self.read(name) for name in
                  ("delivered", "decisions", "orders", "cycles", "t_first", "t_last", "t_out")}
        if result["delivered"] != count:
            raise BoardError("DDR replay delivered %d of %d records" % (result["delivered"], count))
        elapsed = (result["t_out"] - result["t_last"]) & 0xFFFFFFFF
        if result["decisions"] and elapsed != self.latency:
            raise BoardError("DDR decision latency differs from the manifest")
        word = self.read("result")
        result["prediction"] = (word & 255) - (256 if word & 128 else 0)
        result["action"] = (word >> 8) & 3
        result["order"] = (b"" if not word & 1024 else b"".join(
            self.registers.read(OFFSETS["order"] + 4 * i).to_bytes(4, "little")
            for i in range(12))[:47])
        self.needs_abort = False
        return result

    def close(self):
        if self.verified and (not self.closed or self.needs_abort):
            self.abort()
            self.closed = True

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
