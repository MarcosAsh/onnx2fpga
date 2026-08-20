"""Fixed-point rescaling shared by every hardware node.

An integer accumulator is brought back to the output datatype by a per-channel
multiply and a single arithmetic right shift:

    out = clamp(((acc * mult[c] + 2**(shift-1)) >> shift) + zero_point, lo, hi)

The shift is shared across channels so the hardware needs one shifter, while
the multiplier stays per-channel so accuracy does not suffer. Ties round up on
both sides. The zero point is added after the shift and is zero for a symmetric
quantiser, which is the case every path took before asymmetric ONNX models
were accepted.

This function is the definition of correctness for the generated RTL: the
Verilator testbench compares against exactly this.
"""

import math

import numpy as np


class Requantizer:
    def __init__(self, multipliers, shift, out_dtype, lower_clamp=None,
                 zero_point=0, upper_clamp=None):
        self.multipliers = np.asarray(multipliers, dtype=np.int64)
        self.shift = int(shift)
        self.out_dtype = out_dtype
        self.zero_point = int(zero_point)
        self.lower_clamp = out_dtype.min if lower_clamp is None else lower_clamp
        self._upper_clamp = upper_clamp

    @property
    def channels(self):
        return int(self.multipliers.size)

    @property
    def upper_clamp(self):
        return self.out_dtype.max if self._upper_clamp is None else self._upper_clamp

    @classmethod
    def from_real_multipliers(cls, reals, out_dtype, mult_bits=18, relu=False,
                              zero_point=0):
        """A fused relu clamps at the integer standing for real zero, which is
        the zero point, not necessarily zero."""
        reals = np.atleast_1d(np.asarray(reals, dtype=np.float64))
        floor_value = int(zero_point) if relu else None
        limit = (1 << (mult_bits - 1)) - 1
        peak = float(np.max(np.abs(reals)))
        if peak == 0.0:
            return cls(np.zeros_like(reals, dtype=np.int64), 0, out_dtype,
                       floor_value, zero_point)
        shift = int(math.floor(math.log2(limit / peak)))
        shift = max(0, min(shift, 62))
        multipliers = np.rint(reals * (2.0 ** shift)).astype(np.int64)
        multipliers = np.clip(multipliers, -limit - 1, limit)
        return cls(multipliers, shift, out_dtype, floor_value, zero_point)

    def apply(self, accumulator):
        acc = np.asarray(accumulator, dtype=np.int64)
        scaled = acc * self.multipliers
        if self.shift > 0:
            scaled = (scaled + (1 << (self.shift - 1))) >> self.shift
        return np.clip(scaled + self.zero_point, self.lower_clamp, self.upper_clamp)

    def broadcast_to(self, channels):
        if self.channels == channels:
            return self
        return Requantizer(np.full(channels, self.multipliers.flat[0], dtype=np.int64),
                           self.shift, self.out_dtype, self.lower_clamp,
                           self.zero_point, self._upper_clamp)

    def __repr__(self):
        return "Requantizer(ch=%d, shift=%d, out=%s, lo=%d, zp=%d)" % (
            self.channels, self.shift, self.out_dtype, self.lower_clamp,
            self.zero_point)
