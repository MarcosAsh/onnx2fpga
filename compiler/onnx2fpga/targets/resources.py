"""Resource vectors and the arithmetic the folding allocator does on them."""


class Resources:
    """Device sites, one field per block RAM kind; a kind the budget lacks is never charged."""

    FIELDS = ("lut", "ff", "dsp", "bram36", "uram", "ebr")

    def __init__(self, lut=0, ff=0, dsp=0, bram36=0, uram=0, ebr=0):
        self.lut = lut
        self.ff = ff
        self.dsp = dsp
        self.bram36 = bram36
        self.uram = uram
        self.ebr = ebr

    def __add__(self, other):
        return Resources(*(getattr(self, f) + getattr(other, f) for f in self.FIELDS))

    def __mul__(self, factor):
        return Resources(*(getattr(self, f) * factor for f in self.FIELDS))

    def worst_utilisation(self, budget):
        return max(getattr(self, f) / getattr(budget, f)
                   for f in self.FIELDS if getattr(budget, f) > 0)

    def overruns(self, budget, limits):
        """Fields past their share of the budget as (field, used, allowed), worst first."""
        over = []
        for f in self.FIELDS:
            total = getattr(budget, f)
            if total <= 0:
                continue
            allowed = total * limits[f]
            if getattr(self, f) > allowed:
                over.append((getattr(self, f) / allowed, f, getattr(self, f), allowed))
        return [(f, used, allowed) for _, f, used, allowed in sorted(over, reverse=True)]

    def pressure(self, budget):
        """Total normalised spend across resource types, used to rank folding options by cost."""
        return sum(getattr(self, f) / getattr(budget, f)
                   for f in self.FIELDS if getattr(budget, f) > 0)

    def utilisation(self, budget):
        return {f: getattr(self, f) / getattr(budget, f)
                for f in self.FIELDS if getattr(budget, f) > 0}

    @classmethod
    def zero(cls):
        return cls()

    def __repr__(self):
        return "Resources(%s)" % ", ".join(
            "%s=%s" % (f, round(getattr(self, f), 1)) for f in self.FIELDS)


def memory_cost(depth, width):
    """AMD storage: LUT RAM if shallow, UltraRAM if huge and narrow, else block RAM."""
    if depth <= 64:
        return lut_memory_cost(depth, width)
    if depth * width >= 4 * 288 * 1024 and width <= 72:
        parallel = max(1, (width + 71) // 72)
        return Resources(uram=parallel * max(1, (depth + 4095) // 4096), ff=width)
    return block_ram_cost(depth, width)


def lut_memory_cost(depth, width):
    """AMD LUT RAM or ROM: a LUT6 holds 64 words of one bit."""
    return Resources(lut=max(1, (depth + 31) // 32) * width * 0.5, ff=width)


def block_ram_cost(depth, width):
    """AMD RAMB36: 512 words of 72 bits, or 1024 of 36."""
    parallel = max(1, (width + 71) // 72)
    slice_depth = 512 if width > 36 else 1024
    return Resources(bram36=parallel * max(1, (depth + slice_depth - 1) // slice_depth),
                     ff=width)
