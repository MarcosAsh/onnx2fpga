"""Can a retrained model reuse a netlist, or does it need a new one?

This is the compiler half of weight hot-swap. If the generated RTL depends on
the weights, then every retrain is a place-and-route and the tool is unusable
for anything that retrains daily, whatever the silicon does. If it does not,
the swap is a memory reload.
"""

import hashlib
import pathlib
import unittest

import numpy as np

from support import Fixtures

from onnx2fpga.compile import Compiler

SEEDS = (0, 999, 4242, 77)


def digest(build, subdir):
    return [path.name + ":" + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(pathlib.Path(build, subdir).glob("*"))]


def retrained(seed):
    """The same architecture with entirely different weights, which is what a
    retrain looks like to this compiler."""
    factory = Fixtures.factory()
    factory.rng = np.random.default_rng(seed)
    return factory.mlp()


def built(seed, tag, **options):
    out = Fixtures.build_dir("reweight_%s_%d" % (tag, seed))
    result = Compiler(device="vu9p", target_cycles=64, **options).compile(
        retrained(seed), Fixtures.samples(Fixtures.MLP_SHAPE), out)
    return out, result


class WithoutPinningTest(unittest.TestCase):
    """The state of things, recorded so the change below has a baseline.

    Note what the claim is and is not. An unpinned retrain does not always move
    the RTL: the shift only changes when the scales cross a power of two, so
    two models can differ throughout and still land on the same netlist. That
    is worse than a guaranteed change, not better, because it means the
    property holds by luck and nobody finds out which retrain is the one that
    needs a rebuild until it is built."""

    def test_a_retrain_is_not_guaranteed_to_leave_the_rtl_alone(self):
        digests = [digest(built(seed, "loose")[0], "rtl") for seed in SEEDS]
        self.assertNotEqual(len({tuple(d) for d in digests}), 1,
                            "no retrain moved the RTL, so this baseline says "
                            "nothing; pick seeds whose scales differ more")


class PinnedRepresentationTest(unittest.TestCase):
    """Pinning the requantiser's shift and multiplier width makes the netlist
    independent of the weights. Both, not either: the shift is a module
    parameter and so is the width, and the width is fitted to the multiplier
    values, so pinning only the shift pins nothing."""

    SHIFT = 24

    def test_every_retrain_produces_the_same_rtl(self):
        digests = [digest(built(seed, "pinned", fixed_shift=self.SHIFT)[0], "rtl")
                   for seed in SEEDS]
        for other in digests[1:]:
            self.assertEqual(digests[0], other)

    def test_the_memories_still_differ_because_the_weights_did(self):
        """If these matched too, the compiler would be ignoring the retrain."""
        first = built(SEEDS[0], "pinned", fixed_shift=self.SHIFT)[0]
        other = built(SEEDS[1], "pinned", fixed_shift=self.SHIFT)[0]
        self.assertNotEqual(digest(first, "mem"), digest(other, "mem"))

    def test_pinning_below_the_fitted_shift_costs_no_accuracy(self):
        """Down is the safe direction: the multipliers give up some precision
        they were not using. Up is not, see below."""
        _, loose = built(SEEDS[0], "acc_loose")
        _, pinned = built(SEEDS[0], "acc_pinned", fixed_shift=self.SHIFT)
        self.assertLessEqual(pinned.accuracy.worst.relative,
                             loose.accuracy.worst.relative * 1.1)

    def test_pinning_too_high_clips_the_multipliers_and_says_so(self):
        """A shift above the fitted one pushes multipliers past the width they
        have and they saturate. The accuracy report is what catches it, which
        is the argument for having built that first."""
        _, loose = built(SEEDS[0], "acc_ref")
        _, high = built(SEEDS[0], "acc_high", fixed_shift=30)
        self.assertGreater(high.accuracy.worst.relative,
                           loose.accuracy.worst.relative * 5)

    def test_the_pinned_width_is_the_one_asked_for(self):
        from onnx2fpga.p3_ops.hardware_ops import MatVecUnit
        _, result = built(SEEDS[0], "width", fixed_shift=self.SHIFT, mult_bits=18)
        units = [n for n in result.hardware_graph.nodes
                 if isinstance(n, MatVecUnit)]
        self.assertTrue(units)
        for unit in units:
            self.assertEqual(unit.mult_bits, 18, unit.name)


if __name__ == "__main__":
    unittest.main()
