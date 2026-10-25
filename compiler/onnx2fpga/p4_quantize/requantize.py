"""Fixed-point rescaling, the arithmetic the generated RTL must match bit for bit."""

import math

import numpy as np


class Requantizer:
    def __init__(self, multipliers, shift, out_dtype, lower_clamp=None, zero_point=0):
        self.multipliers = np.asarray(multipliers, dtype=np.int64)
        self.shift = int(shift)
        self.out_dtype = out_dtype
        self.zero_point = int(zero_point)
        self.lower_clamp = out_dtype.min if lower_clamp is None else lower_clamp
        self.upper_clamp = out_dtype.max

    @classmethod
    def from_real_multipliers(cls, reals, out_dtype, mult_bits=18, relu=False,
                              zero_point=0):
        """A fused relu clamps at the zero point, which is not necessarily zero."""
        reals = np.atleast_1d(np.asarray(reals, dtype=np.float64))
        floor_value = int(zero_point) if relu else None
        limit = (1 << (mult_bits - 1)) - 1
        peak = float(np.max(np.abs(reals)))
        if peak == 0.0:
            return cls(np.zeros_like(reals, dtype=np.int64), 0, out_dtype,
                       floor_value, zero_point)
        shift = max(0, min(int(math.floor(math.log2(limit / peak))), 62))
        multipliers = np.clip(np.rint(reals * (2.0 ** shift)).astype(np.int64),
                              -limit - 1, limit)
        return cls(multipliers, shift, out_dtype, floor_value, zero_point)

    def apply(self, accumulator):
        scaled = np.asarray(accumulator, dtype=np.int64) * self.multipliers
        if self.shift > 0:
            scaled = (scaled + (1 << (self.shift - 1))) >> self.shift
        return np.clip(scaled + self.zero_point, self.lower_clamp, self.upper_clamp)

    def at_shift(self, shift):
        """The same rescale with multipliers for `shift`, rounded when it is smaller."""
        multipliers = np.rint(self.multipliers * 2.0 ** (shift - self.shift)).astype(np.int64)
        return Requantizer(multipliers, shift, self.out_dtype, self.lower_clamp,
                           self.zero_point)

    def broadcast_to(self, channels):
        if self.multipliers.size == channels:
            return self
        return Requantizer(np.full(channels, self.multipliers.flat[0], dtype=np.int64),
                           self.shift, self.out_dtype, self.lower_clamp, self.zero_point)
