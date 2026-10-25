"""Lattice ECP5, with a cost model fitted to yosys 0.52 and nextpnr-ecp5."""

import math

from .device import Device
from .resources import Resources


def chunks(bits):
    """18 bit pieces yosys cuts one multiplier operand into."""
    if bits <= 18:
        return 1
    return bits // 18 + (1 if bits % 18 >= 2 else 0)


def multipliers(a_bits, b_bits):
    """MULT18X18D for one signed a_bits x b_bits product."""
    return chunks(a_bits) * chunks(b_bits)


def constant_multipliers(value, var_bits):
    """MULT18X18D for a product with a constant; zero and powers of two become wiring."""
    value = abs(int(value))
    if value & (value - 1) == 0:
        return 0
    return multipliers(var_bits, value.bit_length() + 1)


def rom_luts(depth):
    """LUT4 per output bit of a combinational table of `depth` words."""
    if depth <= 1:
        return 0
    return 1.3 * (math.ceil(depth / 16) + max(0, math.ceil(depth / 64) - 1))


EBR_SHAPES = ((16384, 1), (8192, 2), (4096, 4), (2048, 9), (1024, 18), (512, 36))
LUTS_PER_DPR = 8


def ebr_blocks(depth, width):
    """DP16KD for one memory, in the cheapest of its shapes."""
    return min(math.ceil(depth / d) * math.ceil(width / w) for d, w in EBR_SHAPES)


def worth_a_block(depth, width):
    """Each DP16KD must save 128 LUTs, which leaves a 160 x 8 table in LUTs as yosys would."""
    return width * rom_luts(depth) >= 128 * ebr_blocks(depth, width)


def memory(depth, width):
    """A buffer with unregistered read: flip flops if tiny, DP16KD if deep and large, else LUT RAM."""
    if depth == 1 or depth * width <= 16:
        return Resources(lut=width, ff=depth * width)
    if depth > 32 and depth * width > 1024:
        return Resources(lut=width, ebr=ebr_blocks(depth, width))
    cells = math.ceil(width / 4) * math.ceil(depth / 16)
    read_mux = width * (math.ceil(depth / 16) - 1)
    return Resources(lut=cells * LUTS_PER_DPR + read_mux)


class Ecp5CostModel:
    """Per unit estimates in ECP5 sites, fitted to yosys and nextpnr."""

    def __init__(self, device):
        self.device = device

    def estimate(self, node):
        method = getattr(self, node.module, None)
        if method is None:
            raise NotImplementedError("no ECP5 cost model for %s" % node.module)
        return method(node)

    @staticmethod
    def _requant_multipliers(acc_bits, mult_bits, shift):
        """The rescale multiply; a shift past acc_bits + mult_bits builds no multiplier."""
        if shift >= acc_bits + mult_bits:
            return 0
        return multipliers(acc_bits, mult_bits)

    def otf_mvau(self, node):
        """Block RAM weights come out of a DP16KD register; the next stage holds the activations."""
        pe, simd = node.pe, node.simd
        acc = node.accumulator_dtype.bits
        act, wgt = node.act_dtype.bits, node.weight_dtype.bits
        sf, nf = node.synapse_fold, node.neuron_fold
        if sf * nf == 1:
            # Constant weights need no multiplier when zero or a power of two.
            dsp = sum(constant_multipliers(w, act) for w in node.weights.flat)
        else:
            dsp = pe * simd * multipliers(act, wgt)
        dsp += pe * self._requant_multipliers(acc, node.mult_bits, node.requant.shift)
        if nf > 2:
            dsp += constant_multipliers(sf, (nf - 1).bit_length())
        width = pe * simd * wgt
        if node.uses_block_ram(self.device):
            weights = Resources(ebr=ebr_blocks(sf * nf, width), ff=simd * act)
        else:
            weights = Resources(lut=width * rom_luts(sf * nf))
        tables = pe * (acc + node.mult_bits) * rom_luts(nf)
        datapath = 2.25 * pe * simd * acc + 150
        cost = weights + Resources(lut=tables + datapath,
                                   ff=pe * (acc + node.out_dtype.bits) + 35,
                                   dsp=dsp)
        if sf > 1:
            cost = cost + memory(sf, simd * act)
        elif nf > 1:
            cost = cost + Resources(ff=1.6 * pe * simd * act)
        return cost

    def otf_dense(self, node):
        acc, act = node.accumulator_dtype.bits, node.act_dtype.bits
        dsp = node.mh * self._requant_multipliers(acc, node.mult_bits, node.requant.shift)
        out = Resources(ff=node.mh * (acc + node.out_dtype.bits), dsp=dsp)
        if node.arithmetic == "constant":
            cost = node.constant_cost()
            return out + Resources(lut=2.25 * cost.lut, ff=cost.ff)
        product_bits = act + node.weight_dtype.bits
        if node.reloadable:
            dsp = node.multipliers * multipliers(act, node.weight_dtype.bits)
            out = out + node.bank_cost()
        else:
            dsp = sum(constant_multipliers(w, act) for w in node.weights.flat)
        return out + Resources(lut=2.25 * node.multipliers * acc,
                               ff=node.multipliers * product_bits, dsp=dsp)

    def _window(self, node):
        h_stride, w_stride = node.strides
        h_dil, w_dil = node.dilations
        dsp = 2 * constant_multipliers(node.ifm_dim[1], 32)
        dsp += sum(constant_multipliers(v, 32) for v in (h_stride, w_stride, h_dil, w_dil))
        depth = node.buffer_pixels * node.ifm_ch // node.sub_elems
        width = node.sub_elems * node.dtype.bits
        buffer = memory(depth, width) * node.parallel_columns
        control = Resources(lut=400 + 50 * node.parallel_columns,
                            ff=75 + node.simd * node.dtype.bits, dsp=dsp)
        return buffer + control

    otf_swu_line = _window
    otf_swu = _window

    def otf_act(self, node):
        pe = node.pe
        acc = node.in_dtype.bits + 2
        dsp = pe * self._requant_multipliers(acc, node.mult_bits, node.requant.shift)
        return Resources(lut=pe * (2 * acc + 60) + 20, ff=pe * node.out_dtype.bits + 10,
                         dsp=dsp)

    def otf_add(self, node):
        pe, bits = node.pe, node.in_dtype.bits
        dsp = pe * (constant_multipliers(node.left_mult, bits)
                    + constant_multipliers(node.right_mult, bits))
        return Resources(lut=pe * 130, ff=pe * node.out_dtype.bits, dsp=dsp)

    def otf_pool(self, node):
        pe, bits = node.pe, node.dtype.bits
        folds = node.channels // pe
        return memory(folds, pe * bits) + Resources(lut=12 + 3 * pe * bits,
                                                     ff=pe * bits + 4)

    def otf_dup(self, node):
        return Resources(lut=4 + 2 * len(node.outputs), ff=len(node.outputs))

    def otf_fifo(self, node):
        depth, width = node.depth, node.width
        control = Resources(lut=14 + 3 * max(1, (depth - 1).bit_length()),
                            ff=4 + 2 * max(1, (depth - 1).bit_length()))
        return memory(depth, width) + control

    def otf_dwc(self, node):
        wide = max(node.in_elems, node.out_elems) * node.elem_bits
        return Resources(lut=4 + 0.35 * wide, ff=wide + 4)

    def otf_trees(self, node):
        """A comparison against a constant becomes a CCU2C carry chain, two LUT4 sites per two bits."""
        return Resources(lut=(0.85 * node.comparators * node.feature_dtype.bits
                              + 0.26 * node.ensemble.splits * node.leaf_dtype.bits
                              + node.ensemble.trees * node.sum_dtype.bits),
                         ff=node.registers)


class Ecp5Device(Device):
    HARD_FIELDS = ("dsp", "ebr")

    def __init__(self, nextpnr_args, **kwargs):
        Device.__init__(self, block_ram="ebr", **kwargs)
        self.nextpnr_args = tuple(nextpnr_args)
        self.cost_model = Ecp5CostModel(self)

    def worth_a_block(self, depth, width):
        return worth_a_block(depth, width)

    def macs_per_dsp(self, act_bits, weight_bits):
        """One MULT18X18D per product, as yosys infers it; the 9x9 mode is never used."""
        return 1.0 / multipliers(act_bits, weight_bits)


Device.register(Ecp5Device(
    name="icepi25", part="LFE5U-25F-6CABGA256",
    # LUT4 and FF from nextpnr for the 25k die; EBR and multipliers from FPGA-DS-02012.
    budget=Resources(lut=24288, ff=24288, dsp=28, ebr=56),
    fmax_mhz=20.0, dsp_generation="sysDSP",
    nextpnr_args=("--25k", "--package", "CABGA256", "--speed", "6"),
    provenance="mixed",
    notes="Icepi Zero. 28 MULT18X18D (14 sysDSP), 56 EBR of 18 kbit and 194 kbit "
          "distributed RAM are Lattice's (FPGA-DS-02012 rev 3.4); 24288 LUT4 is "
          "nextpnr's count of what Lattice calls 24k. 20 MHz is the clock the board "
          "build runs at: the units' single cycle MAC and rescale path closes at "
          "about 23 to 25 MHz with yosys and nextpnr-ecp5.",
))
