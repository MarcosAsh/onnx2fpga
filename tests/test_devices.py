"""The target device registry, and the DSP packing rule it exists to state."""

import unittest

from onnx2fpga.targets.device import Device


class RegistryTest(unittest.TestCase):
    def test_every_device_budgets_something(self):
        for name in Device.names():
            device = Device.get(name)
            for field in device.budget.FIELDS:
                self.assertGreaterEqual(getattr(device.budget, field), 0,
                                        "%s.%s is negative" % (name, field))
            self.assertGreater(device.budget.lut, 0, name)
            self.assertGreater(device.fmax_mhz, 0, name)

    def test_unknown_device_names_what_it_has(self):
        with self.assertRaises(KeyError) as caught:
            Device.get("no-such-part")
        self.assertIn("z7020", str(caught.exception))


class DspPackingTest(unittest.TestCase):
    """int8 packing is a property of the DSP generation, not of the board. It
    is the reason a part is worth adding: z7020 cannot exercise the rule, and
    until zu5ev arrived nothing in the free Vivado tier could."""

    def test_dsp48e2_packs_two_int8_multiplies(self):
        self.assertEqual(Device.get("zu5ev").macs_per_dsp(8, 8), 2.0)
        self.assertEqual(Device.get("vu9p").macs_per_dsp(8, 8), 2.0)

    def test_dsp48e1_packs_none(self):
        self.assertEqual(Device.get("z7020").macs_per_dsp(8, 8), 1.0)

    def test_packing_rises_as_precision_falls(self):
        device = Device.get("zu5ev")
        self.assertEqual(device.macs_per_dsp(4, 4), 4.0)
        self.assertEqual(device.macs_per_dsp(8, 8), 2.0)
        self.assertEqual(device.macs_per_dsp(16, 16), 1.0)
        self.assertEqual(device.macs_per_dsp(32, 32), 0.5)

    def test_int8_costs_half_the_slices_of_int16(self):
        device = Device.get("zu5ev")
        self.assertEqual(device.dsp_cost(1024, 8, 8),
                         device.dsp_cost(1024, 16, 16) / 2)


if __name__ == "__main__":
    unittest.main()
