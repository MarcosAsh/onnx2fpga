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
    #: Width the multipliers were generated within, when it was pinned rather
    #: than fitted. A fitted width follows the values and so follows the
    #: weights, which is the other half of what stops a netlist surviving a
    #: retrain; pinning the shift without pinning this pins nothing.
    width = None

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
                              zero_point=0, shift=None):
        """A fused relu clamps at the integer standing for real zero, which is
        the zero point, not necessarily zero.

        The shift is normally chosen to put the largest multiplier just under
        the width available, which is the most precision this representation
        can carry. Passing one instead pins it, and that is what makes a design
        survive a retrain: the shift is a module parameter baked into the RTL,
        so a model whose scales moved enough to change it needs the netlist
        rebuilt, while one whose shift is pinned needs only new memory images.
        The cost is precision, since the multipliers no longer fill the width.
        """
        reals = np.atleast_1d(np.asarray(reals, dtype=np.float64))
        floor_value = int(zero_point) if relu else None
        limit = (1 << (mult_bits - 1)) - 1
        peak = float(np.max(np.abs(reals)))
        if peak == 0.0:
            return cls(np.zeros_like(reals, dtype=np.int64), 0, out_dtype,
                       floor_value, zero_point)
        chosen = (int(math.floor(math.log2(limit / peak)))
                  if shift is None else int(shift))
        shift = max(0, min(chosen, 62))
        multipliers = np.rint(reals * (2.0 ** shift)).astype(np.int64)
        multipliers = np.clip(multipliers, -limit - 1, limit)
        built = cls(multipliers, shift, out_dtype, floor_value, zero_point)
        if shift is not None:
            built.width = int(mult_bits)
        return built

    def apply(self, accumulator):
        acc = np.asarray(accumulator, dtype=np.int64)
        scaled = acc * self.multipliers
        if self.shift > 0:
            scaled = (scaled + (1 << (self.shift - 1))) >> self.shift
        return np.clip(scaled + self.zero_point, self.lower_clamp, self.upper_clamp)

    def broadcast_to(self, channels):
        if self.channels == channels:
            return self
        wider = Requantizer(
            np.full(channels, self.multipliers.flat[0], dtype=np.int64),
            self.shift, self.out_dtype, self.lower_clamp,
            self.zero_point, self._upper_clamp)
        wider.width = self.width
        return wider

    def __repr__(self):
        return "Requantizer(ch=%d, shift=%d, out=%s, lo=%d, zp=%d)" % (
            self.channels, self.shift, self.out_dtype, self.lower_clamp,
            self.zero_point)
