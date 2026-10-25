"""Target devices; "estimated" figures are working numbers, not for sizing a real build."""

from .resources import Resources


class Device:
    _registry = {}

    # Block RAM and DSP sites are discrete, so a device may let them run to 100 percent.
    HARD_FIELDS = ()

    # nextpnr arguments that select the part, empty for parts Vivado builds.
    nextpnr_args = ()

    # A vendor specific estimator that prices each unit instead of its AMD model.
    cost_model = None

    def __init__(self, name, part, budget, fmax_mhz, dsp_generation,
                 provenance="estimated", notes="", block_ram="bram36"):
        self.name = name
        self.part = part
        self.budget = budget
        self.fmax_mhz = fmax_mhz
        self.dsp_generation = dsp_generation
        self.provenance = provenance
        self.notes = notes
        self.block_ram = block_ram

    @property
    def has_block_ram(self):
        return getattr(self.budget, self.block_ram) > 0

    def worth_a_block(self, depth, width):
        """Whether a weight table of `depth` words of `width` bits belongs in block RAM."""
        return depth > 64

    def limits(self, utilisation_limit):
        """Fraction of the budget each resource may use."""
        return {f: 1.0 if f in self.HARD_FIELDS else utilisation_limit
                for f in Resources.FIELDS}

    def macs_per_dsp(self, act_bits, weight_bits):
        """DSP packing factor: a DSP48E2 27x18 multiplier holds two INT8 or four INT4 products."""
        if self.dsp_generation not in ("DSP48E2", "DSP58"):
            return 1.0
        wide = max(act_bits, weight_bits)
        if wide <= 4:
            return 4.0
        if wide <= 8:
            return 2.0
        if act_bits <= 18 and weight_bits <= 27:
            return 1.0
        return 0.5

    def dsp_cost(self, mac_count, act_bits, weight_bits):
        return mac_count / self.macs_per_dsp(act_bits, weight_bits)

    @classmethod
    def register(cls, device):
        cls._registry[device.name] = device
        return device

    @classmethod
    def get(cls, name):
        try:
            return cls._registry[name]
        except KeyError:
            raise KeyError("unknown device %r, have %s"
                           % (name, sorted(cls._registry)))

    @classmethod
    def names(cls):
        return sorted(cls._registry)


Device.register(Device(
    name="vu9p", part="xcvu9p-flgb2104-2-i",
    budget=Resources(lut=1182240, ff=2364480, dsp=6840, bram36=2160, uram=960),
    fmax_mhz=300.0, dsp_generation="DSP48E2",
    provenance="vendor", notes="AWS EC2 F1 shell leaves roughly 20 percent to the platform"))

Device.register(Device(
    name="vu47p", part="xcvu47p-fsvh2892-2-e",
    budget=Resources(lut=1303680, ff=2607360, dsp=9024, bram36=2016, uram=960),
    fmax_mhz=300.0, dsp_generation="DSP48E2",
    provenance="mixed",
    notes="AWS EC2 F2. LUT/FF/DSP are vendor figures; bram36 and uram are estimated "
          "from the sibling VU37P and unverified. 16 GB HBM at about 460 GB/s."))

Device.register(Device(
    name="zu7ev", part="xczu7ev-ffvc1156-2-e",
    budget=Resources(lut=230400, ff=460800, dsp=1728, bram36=312, uram=96),
    fmax_mhz=250.0, dsp_generation="DSP48E2",
    provenance="vendor", notes="ZCU104 class board"))

Device.register(Device(
    name="k26", part="xck26-sfvc784-2LV-c",
    budget=Resources(lut=117120, ff=234240, dsp=1248, bram36=144, uram=64),
    fmax_mhz=322.265625, dsp_generation="DSP48E2",
    provenance="vendor",
    notes="Kria K26 SOM on the KV260 and KR260. Budgets from AMD DS987 v1.2 Table 5; "
          "-2LV speed grade (VCCINT 0.72 V). 322.265625 MHz is the 10GBASE-R "
          "clock we target, not a measured fmax."))

Device.register(Device(
    name="z7020", part="xc7z020-clg400-1",
    budget=Resources(lut=53200, ff=106400, dsp=220, bram36=140, uram=0),
    fmax_mhz=100.0, dsp_generation="DSP48E1",
    provenance="vendor", notes="PYNQ-Z1/Z2 class board"))

Device.register(Device(
    name="sim", part="verilator",
    budget=Resources(lut=10 ** 9, ff=10 ** 9, dsp=10 ** 9, bram36=10 ** 9, uram=10 ** 9),
    fmax_mhz=1000.0, dsp_generation="none",
    provenance="n/a", notes="unbounded pseudo-device for simulation-only builds"))


# Imported last: the ECP5 model subclasses Device and registers itself.
from . import ecp5  # noqa: E402,F401
