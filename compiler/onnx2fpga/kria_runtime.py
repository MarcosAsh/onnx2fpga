"""Prediction sessions over a Kria AXI4-Lite register transport."""

import math
import numbers
import time

from .board import BoardError, OFFSETS


class RegisterSession:
    """One inference at a time; reset is required after an interrupted transaction."""

    def __init__(self, registers, manifest, timeout=2.0, close_registers=False):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.registers = registers
        self.timeout = timeout
        self.close_registers = close_registers
        self.closed = True
        self.needs_reset = True
        layout = manifest["registers"]
        entries = layout["map"]
        for name in ("id", "latency", "feature_bits", "prediction_bits", "control",
                     "status", "cycles", "capabilities", "features", "prediction"):
            if entries[name]["offset"] != OFFSETS[name]:
                raise BoardError("unexpected register offset: %s" % name)
        self.latency = entries["latency"]["value"]
        self.widths = []
        for port, name, region in (("input", "feature_bits", "features"),
                                    ("output", "prediction_bits", "prediction")):
            stream = manifest[port]
            bits = stream["elem_bits"] * stream["elems_per_beat"]
            if (stream["beats"] != 1 or not 0 < bits <= 32768
                    or entries[name]["value"] != bits
                    or entries[region]["words"] != (bits + 31) // 32):
                raise BoardError("invalid fixed-mode %s register layout" % port)
            self.widths.append(bits)

    def read(self, name):
        return self.registers.read(OFFSETS[name])

    def open(self):
        for name, expected in (("id", 0x4F544631), ("latency", self.latency),
                               ("feature_bits", self.widths[0]),
                               ("prediction_bits", self.widths[1])):
            actual = self.read(name)
            if actual != expected:
                raise BoardError("%s reads %d, expected %d" % (name, actual, expected))
        if not self.read("capabilities") & 1:
            raise BoardError("bitstream lacks inference reset; rebuild the Kria shell")
        self.closed = False
        try:
            self.reset()
        except Exception:
            self.closed = True
            raise
        return self

    def _wait(self, ready):
        deadline = time.monotonic() + self.timeout
        while bool(self.read("status") & 1) != ready:
            if time.monotonic() >= deadline:
                raise BoardError("Kria %s timed out; reset before retrying" %
                                 ("prediction" if ready else "reset"))

    def reset(self):
        if self.closed:
            raise BoardError("Kria session is closed")
        self.needs_reset = True
        self.registers.write(OFFSETS["control"], 2)
        self._wait(False)
        self.needs_reset = False

    def run(self, frames):
        if self.closed or self.needs_reset:
            raise BoardError("Kria session needs open or reset")
        results = []
        for frame in frames:
            if (len(frame) != 1 or not isinstance(frame[0], numbers.Integral)
                    or not 0 <= frame[0] < (1 << self.widths[0])):
                raise BoardError("expected one packed feature vector")
            self.needs_reset = True
            value = int(frame[0])
            for index in range((self.widths[0] + 31) // 32):
                self.registers.write(OFFSETS["features"] + 4 * index,
                                     (value >> (32 * index)) & 0xFFFFFFFF)
            self.registers.write(OFFSETS["control"], 1)
            self._wait(True)
            value = sum(self.registers.read(OFFSETS["prediction"] + 4 * index)
                        << (32 * index) for index in range((self.widths[1] + 31) // 32))
            cycles = self.read("cycles")
            if cycles != self.latency:
                raise BoardError("prediction took %d cycles, expected %d" % (cycles, self.latency))
            results.append([value & ((1 << self.widths[1]) - 1)])
            self.needs_reset = False
        return results

    def close(self):
        self.closed = True
        self.needs_reset = True
        if self.close_registers:
            self.registers.close()
            self.close_registers = False

    def __enter__(self):
        return self.open() if self.closed else self

    def __exit__(self, *_):
        self.close()
