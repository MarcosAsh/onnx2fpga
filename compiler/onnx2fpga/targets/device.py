"""Target device models.

Resource counts marked provenance="vendor" are taken from published AMD device
tables. Anything marked "estimated" was not obtainable from a primary source
and is a working figure only; do not size a real build against it.
"""

from .resources import Resources


class Device:
    _registry = {}

    def __init__(self, name, part, budget, fmax_mhz, dsp_generation,
                 external_bandwidth_gbps=0.0, provenance="estimated", notes=""):
        self.name = name
        self.part = part
        self.budget = budget
        self.fmax_mhz = fmax_mhz
        self.dsp_generation = dsp_generation
        self.external_bandwidth_gbps = external_bandwidth_gbps
        self.provenance = provenance
        self.notes = notes

    def macs_per_dsp(self, act_bits, weight_bits):
        """DSP packing factor.

        A DSP48E2 holds a 27x18 multiplier. Two INT8 products that share one
        operand fit in it at once, and four INT4 products do, so the effective
        MAC density rises as precision falls. Anything wider than the native
        ports costs more than one slice.
        """
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

    def peak_macs_per_second(self, act_bits, weight_bits):
        return (self.budget.dsp * self.macs_per_dsp(act_bits, weight_bits)
                * self.fmax_mhz * 1e6)

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

    def __repr__(self):
        return "Device(%s, %s, %s)" % (self.name, self.part, self.budget)


Device.register(Device(
    name="vu9p", part="xcvu9p-flgb2104-2-i",
    budget=Resources(lut=1182240, ff=2364480, dsp=6840, bram36=2160, uram=960),
    fmax_mhz=300.0, dsp_generation="DSP48E2", external_bandwidth_gbps=64.0,
    provenance="vendor", notes="AWS EC2 F1 shell leaves roughly 20 percent to the platform"))

Device.register(Device(
    name="vu47p", part="xcvu47p-fsvh2892-2-e",
    budget=Resources(lut=1303680, ff=2607360, dsp=9024, bram36=2016, uram=960),
    fmax_mhz=300.0, dsp_generation="DSP48E2", external_bandwidth_gbps=460.0,
    provenance="mixed",
    notes="AWS EC2 F2. LUT/FF/DSP are vendor figures; bram36 and uram are estimated "
          "from the sibling VU37P and unverified. 16 GB HBM at about 460 GB/s."))

Device.register(Device(
    name="zu7ev", part="xczu7ev-ffvc1156-2-e",
    budget=Resources(lut=230400, ff=460800, dsp=1728, bram36=312, uram=96),
    fmax_mhz=250.0, dsp_generation="DSP48E2", external_bandwidth_gbps=19.2,
    provenance="vendor", notes="ZCU104 class board"))

Device.register(Device(
    name="zu5ev", part="xck26-sfvc784-2lv-c",
    budget=Resources(lut=117120, ff=234240, dsp=1248, bram36=144, uram=64),
    fmax_mhz=250.0, dsp_generation="DSP48E2", external_bandwidth_gbps=19.2,
    provenance="vendor",
    notes="Kria K26 SOM. A production module rather than a development board, and "
          "the cheapest DSP48E2 part inside the free Vivado tier, so the int8 DSP "
          "packing rule can be checked without a paid licence."))

Device.register(Device(
    name="z7020", part="xc7z020-clg400-1",
    budget=Resources(lut=53200, ff=106400, dsp=220, bram36=140, uram=0),
    fmax_mhz=100.0, dsp_generation="DSP48E1", external_bandwidth_gbps=4.2,
    provenance="vendor", notes="PYNQ-Z1/Z2 class board"))

Device.register(Device(
    name="sim", part="verilator",
    budget=Resources(lut=10 ** 9, ff=10 ** 9, dsp=10 ** 9, bram36=10 ** 9, uram=10 ** 9),
    fmax_mhz=1000.0, dsp_generation="none",
    provenance="n/a", notes="unbounded pseudo-device for simulation-only builds"))
