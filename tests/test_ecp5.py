"""The ECP5 model: Lattice budget, yosys multipliers, folding within 28 of them."""

import unittest

import numpy as np

from support import Fixtures, Hdl

from onnx2fpga.compile import Compiler
from onnx2fpga.measure import YosysRunner
from onnx2fpga.targets.device import Device
from onnx2fpga.targets.ecp5 import ebr_blocks, memory, multipliers, worth_a_block
from onnx2fpga.targets.resources import Resources
from onnx2fpga.trading.replay import tiny_mlp


def compile_mlp(target, device="icepi25", hdl="sv", out="ecp5_mlp"):
    return Compiler(device=device, target_cycles=target, hdl=hdl).compile(
        Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
        Fixtures.build_dir(out))


class Ecp5ModelTest(unittest.TestCase):
    def test_budget_is_lfe5u_25(self):
        device = Device.get("icepi25")
        self.assertEqual(device.part, "LFE5U-25F-6CABGA256")
        budget = device.budget
        self.assertEqual((budget.lut, budget.dsp, budget.ebr), (24288, 28, 56))
        self.assertEqual((budget.bram36, budget.uram), (0, 0))

    def test_int8_mac_takes_a_whole_multiplier(self):
        """yosys synth_ecp5 never infers the 9x9 mode."""
        self.assertEqual(Device.get("icepi25").macs_per_dsp(8, 8), 1.0)
        self.assertEqual(Device.get("vu9p").macs_per_dsp(8, 8), 2.0)

    def test_operands_split_like_yosys(self):
        """Measured with synth_ecp5; a 1 bit remainder goes to soft logic."""
        measured = {(8, 8): 1, (18, 18): 1, (19, 18): 1, (20, 18): 2, (36, 18): 2,
                    (37, 18): 2, (20, 2): 2, (35, 35): 4}
        for (a, b), count in measured.items():
            self.assertEqual(multipliers(a, b), count, (a, b))

    def test_buffers_land_where_yosys_puts_them(self):
        self.assertEqual(memory(50, 32).ebr, 1)
        self.assertEqual(memory(266, 8).ebr, 1)
        self.assertEqual(memory(66, 72).ebr, 2)
        self.assertEqual(memory(128, 8).ebr, 0)
        self.assertEqual(memory(25, 72).ebr, 0)
        self.assertEqual(memory(2, 8).ff, 16)

    def test_weights_go_in_ebr_when_they_save_luts(self):
        """Block counts measured with synth_ecp5 on ROMs marked ram_style block."""
        shapes = ((160, 8), (32, 36), (4, 72), (64, 36), (128, 128))
        self.assertEqual([worth_a_block(*s) for s in shapes],
                         [False, False, False, True, True])
        self.assertEqual([ebr_blocks(*s) for s in shapes], [1, 1, 2, 1, 4])
        self.assertEqual(ebr_blocks(64, 256), 8)

    def test_synth_commands(self):
        synth, place = YosysRunner(Device.get("icepi25")).commands()
        self.assertIn("synth_ecp5 -top otf_top", synth[-1])
        self.assertEqual(place[1:7], ["--25k", "--package", "CABGA256", "--speed", "6",
                                      "--json"])
        self.assertIn("--pack-only", place)
        self.assertNotIn("--pack-only", YosysRunner(Device.get("icepi25"),
                                                    implement=True).commands()[1])

    def test_rescale_past_datapath_costs_no_multiplier(self):
        result = compile_mlp(512)
        unit = next(n for n in result.hardware_graph.topological_order()
                    if n.module == "otf_mvau")
        model = Device.get("icepi25").cost_model
        before = model.estimate(unit).dsp
        unit.requant.shift = unit.accumulator_dtype.bits + unit.mult_bits
        after = model.estimate(unit).dsp
        self.assertEqual(before - after, unit.pe * multipliers(
            unit.accumulator_dtype.bits, unit.mult_bits))


class Ecp5FoldingTest(unittest.TestCase):
    def test_folding_stops_at_multiplier_budget(self):
        result = compile_mlp(64)
        total = sum((n.resources(Device.get("icepi25"))
                     for n in result.hardware_graph.nodes), Resources.zero())
        self.assertLessEqual(total.dsp, 28)
        notes = [line for line in result.context.log if line.startswith("fold:")]
        self.assertTrue(any("does not fit the dsp budget of icepi25" in line
                            for line in notes), notes)
        # Every stage at 256 cycles is the best 28 multipliers allow, not fc1 at 64 and fc0 at 512.
        self.assertEqual(result.folding.bottleneck_cycles, 256)


    def test_amd_devices_fold_as_before(self):
        result = compile_mlp(64, device="vu9p", out="ecp5_mlp_vu9p")
        foldings = {name: folding for name, _, folding, _, _ in result.folding.rows}
        self.assertEqual(foldings["fc0"], {"pe": 2, "simd": 16})
        self.assertEqual(result.folding.total.dsp, 30)
        self.assertEqual(result.folding.total.ebr, 0)


@unittest.skipIf(not YosysRunner(Device.get("icepi25")).available,
                 "yosys and nextpnr-ecp5 not installed")
@unittest.skipIf(Hdl.skip_reason("hardcaml"), Hdl.skip_reason("hardcaml"))
class Ecp5SynthesisTest(unittest.TestCase):
    def test_estimate_tracks_yosys_and_nextpnr(self):
        device = Device.get("icepi25")
        result = compile_mlp(512, hdl="hardcaml", out="ecp5_mlp_synth")
        measured, _ = YosysRunner(device).run(result.build.root)
        predicted = sum((n.resources(device) for n in result.hardware_graph.nodes),
                        Resources.zero())
        self.assertEqual(predicted.dsp, measured.dsp)
        self.assertGreater(measured.ebr, 0)
        self.assertEqual(predicted.ebr, measured.ebr)
        self.assertLess(abs(predicted.lut / measured.lut - 1), 0.25)
        self.assertLess(abs(predicted.ff / measured.ff - 1), 0.25)

    def test_clock(self):
        """Pipelined for 120 MHz, the 4-8-1 MLP meets it after place and route, near the model's fmax."""
        result = Compiler(device="icepi25", mode="fixed", hdl="hardcaml",
                          clock_mhz=120).compile(
            tiny_mlp(), [{"x": x} for x in np.random.default_rng(0).normal(0, 1, (64, 1, 4))],
            Fixtures.build_dir("ecp5_clock"))
        _, fmax = YosysRunner(result.context.device, implement=True,
                              freq_mhz=120).run(result.build.root)
        self.assertGreaterEqual(fmax, 120)
        self.assertLess(abs(result.fmax_mhz / fmax - 1), 0.1)


if __name__ == "__main__":
    unittest.main()
