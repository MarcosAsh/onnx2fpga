"""Register to register delay models, in ns, for choosing pipeline stages against a clock."""

import math


class DelayModel:
    """A path is a flip flop's clock to out and setup, plus a net and a cell for each step."""

    def __init__(self, register, net, lut, carry, carry_bit, mux, tree, fanout, multiply, dsp,
                 internal=None, margin=0.0, provenance="", lut_inputs=6):
        self.register = register
        self.net = net
        self.lut = lut
        self.carry = carry
        self.carry_bit = carry_bit
        self.mux = mux
        self.tree = tree
        self.fanout = fanout
        self.multiply = multiply
        self.dsp = dsp
        self.internal = internal
        # The share of the period the scheduler keeps free for the model's own error.
        self.margin = margin
        self.provenance = provenance
        self.lut_inputs = lut_inputs

    def path(self, *steps, fanout=1):
        """One register to register path: the source's fanout costs routing per doubling."""
        return self.register + self.fanout * math.log2(max(1, fanout)) + sum(steps)

    def adder(self, bits):
        """An adder, subtractor or comparison on a carry chain."""
        return self.net + self.carry + self.carry_bit * bits

    def adder_tree(self, levels, bits):
        """Binary levels of adds between two registers: one carry chain, then `tree` a level."""
        return self.adder(bits) + max(0, levels - 1) * self.tree if levels else self.net

    def logic(self, levels=1):
        return levels * (self.net + self.lut)

    def table(self, bits):
        """A constant table on a `bits` wide code: a level of LUTs and wide muxes per LUT's worth of address."""
        return self.logic(math.ceil(bits / self.lut_inputs))

    def select(self, levels):
        """A tree of 2:1 muxes with constant leaves, which LUTs absorb a level or two at a time."""
        return levels * self.mux

    def product(self, a_bits, b_bits, mreg=False):
        """A multiplier's pins to its result; a product wider than one multiplier adds the partial sums."""
        a, b = sorted((a_bits, b_bits), reverse=True)
        wide, narrow = sorted(self.dsp, reverse=True)
        pieces = math.ceil(a / wide) * math.ceil(b / narrow)
        delay = self.net + self.multiply[1 if mreg else 0]
        return delay + (self.adder(a_bits + b_bits) if pieces > 1 else 0.0)

    def registered_product(self, a_bits, b_bits, fanout=1):
        """The multiply with registers before and after it, inside the DSP slice where it has them."""
        if self.internal is None:
            return self.path(self.product(a_bits, b_bits, mreg=True), fanout=fanout)
        return max(self.internal, self.path(self.net, fanout=fanout))


# DS925 publishes DSP48E2 fmax only; the fabric figures are assumed, see docs/specs/002.
def ultrascale_plus(scale, all_registers, no_mreg, provenance):
    register = 0.15 * scale
    return DelayModel(
        register=register, net=0.40 * scale, lut=0.10 * scale, carry=0.10 * scale,
        carry_bit=0.015 * scale, mux=0.30 * scale, tree=0.80 * scale, fanout=0.10 * scale,
        multiply=(1000.0 / no_mreg - register, 1000.0 / all_registers - register),
        dsp=(27, 18), internal=1000.0 / all_registers, provenance=provenance)


ULTRASCALE_PLUS_2 = ultrascale_plus(
    1.0, 775.0, 544.0, "DSP48E2 from DS925 at 0.85 V, fabric assumed; unverified until a "
                       "Vivado run")
ULTRASCALE_PLUS_2LV = ultrascale_plus(
    1.24, 644.0, 440.0, "DSP48E2 from DS925 at 0.72 V (-2LE), fabric assumed and scaled; "
                        "unverified until a Vivado run")
