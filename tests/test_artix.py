"""Artix budgets and memory allocation on devices without UltraRAM."""

import unittest

from onnx2fpga.p3_ops.hardware_ops import StreamFifo
from onnx2fpga.targets.device import Device
from onnx2fpga.targets.resources import memory_cost


class ArtixTest(unittest.TestCase):
    def test_parts_and_budgets(self):
        for name, part, lut, dsp, bram in (
            ('a7_35t', 'xc7a35ticsg324-1L', 20800, 90, 50),
            ('a7_100t', 'xc7a100tcsg324-1', 63400, 240, 135),
        ):
            with self.subTest(device=name):
                device = Device.get(name)
                self.assertEqual(device.part, part)
                self.assertEqual((device.budget.lut, device.budget.ff,
                                  device.budget.dsp, device.budget.bram36),
                                 (lut, lut * 2, dsp, bram))
                self.assertEqual(device.budget.uram, 0)
                self.assertEqual(device.macs_per_dsp(8, 8), 1)

    def test_fifo_needs_distributed_memory(self):
        fifo = StreamFifo('large', ['x'], ['y'], depth=32768, width=64)
        for name in ('a7_35t', 'a7_100t', 'z7020', 'k26'):
            with self.subTest(device=name):
                device = Device.get(name)
                cost = fifo.resources(device)
                self.assertEqual(cost.uram, 0)
                self.assertEqual(cost.bram36, 0)
                if name == 'a7_35t':
                    self.assertIn('lut', [field for field, _, _ in
                                            cost.overruns(device.budget, device.limits(.8))])

    def test_fifo_synthesis_bounds(self):
        for depth, lut, ff in ((1024, 273, 31), (4096, 1022, 86)):
            with self.subTest(depth=depth):
                cost = StreamFifo('fifo', ['x'], ['y'], depth, 8).resources(
                    Device.get('a7_35t'))
                self.assertGreaterEqual(cost.lut, lut)
                self.assertLess(cost.lut, 1.2 * lut)
                self.assertGreaterEqual(cost.ff, ff)

    def test_small_memory_unchanged(self):
        for depth, width in ((16, 8), (64, 32), (1024, 72)):
            with self.subTest(depth=depth, width=width):
                self.assertEqual(vars(memory_cost(depth, width)),
                                 vars(memory_cost(depth, width, allow_uram=False)))


if __name__ == '__main__':
    unittest.main()
