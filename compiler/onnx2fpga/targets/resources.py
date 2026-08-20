"""Resource vectors and the arithmetic the folding allocator does on them."""


class Resources:
    FIELDS = ("lut", "ff", "dsp", "bram36", "uram")

    def __init__(self, lut=0, ff=0, dsp=0, bram36=0, uram=0):
        self.lut = lut
        self.ff = ff
        self.dsp = dsp
        self.bram36 = bram36
        self.uram = uram

    def __add__(self, other):
        return Resources(*(getattr(self, f) + getattr(other, f) for f in self.FIELDS))

    def __mul__(self, factor):
        return Resources(*(getattr(self, f) * factor for f in self.FIELDS))

    def fits_in(self, budget):
        return all(getattr(self, f) <= getattr(budget, f) for f in self.FIELDS)

    def worst_utilisation(self, budget):
        return max(getattr(self, f) / getattr(budget, f)
                   for f in self.FIELDS if getattr(budget, f) > 0)

    def pressure(self, budget):
        """Total normalised spend, summed across resource types. Used to rank
        folding options by cost; worst_utilisation decides feasibility."""
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


class MemoryStyle:
    """Chooses the storage primitive for a weight array of a given geometry."""

    LUTRAM_DEPTH_LIMIT = 64
    BRAM36_BITS = 36864
    URAM_BITS = 288 * 1024

    def __init__(self, name, cost_fn):
        self.name = name
        self._cost_fn = cost_fn

    def cost(self, depth, width):
        return self._cost_fn(depth, width)

    @classmethod
    def select(cls, depth, width, allow_uram=True):
        total_bits = depth * width
        if depth <= cls.LUTRAM_DEPTH_LIMIT:
            return cls.DISTRIBUTED
        if allow_uram and total_bits >= 4 * cls.URAM_BITS and width <= 72:
            return cls.ULTRA
        return cls.BLOCK


def _distributed_cost(depth, width):
    slices = max(1, (depth + 31) // 32)
    return Resources(lut=slices * width * 0.5, ff=width)


def _block_cost(depth, width):
    """Block RAMs are 36Kb but capped at 72 bits wide, so wide streams need
    several in parallel before depth is the binding constraint."""
    parallel = max(1, (width + 71) // 72)
    per_slice_depth = 512 if width > 36 else 1024
    deep = max(1, (depth + per_slice_depth - 1) // per_slice_depth)
    return Resources(bram36=parallel * deep, ff=width)


def _ultra_cost(depth, width):
    parallel = max(1, (width + 71) // 72)
    deep = max(1, (depth + 4095) // 4096)
    return Resources(uram=parallel * deep, ff=width)


MemoryStyle.DISTRIBUTED = MemoryStyle("distributed", _distributed_cost)
MemoryStyle.BLOCK = MemoryStyle("block", _block_cost)
MemoryStyle.ULTRA = MemoryStyle("ultra", _ultra_cost)
