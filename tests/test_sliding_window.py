"""The sliding window generators, checked against numpy across a config sweep.

Window geometry is where an accelerator quietly goes wrong: an off by one in
padding or dilation still produces plausible output. Both strategies are swept
over the same configurations so they also have to agree with each other.
"""

import shutil
import unittest
import zlib

import numpy as np

from support import Fixtures, SingleUnitProject, run

from onnx2fpga.p2_graph.datatype import INT8
from onnx2fpga.p3_ops.hardware_ops import SlidingWindowUnit

# ifm, channels, kernel, stride, pad, dilation, simd
CONFIGS = [
    ("basic",        (8, 8),   4, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 1),
    ("folded",       (8, 8),   4, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 4),
    ("single_chan",  (12, 12), 1, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 1),
    ("same_pad",     (7, 7),   4, (5, 5), (1, 1), (2, 2, 2, 2), (1, 1), 4),
    ("stride_pad",   (9, 9),   2, (3, 3), (2, 2), (1, 1, 1, 1), (1, 1), 2),
    ("pool_window",  (10, 10), 4, (2, 2), (2, 2), (0, 0, 0, 0), (1, 1), 2),
    ("dilated",      (8, 8),   2, (3, 3), (1, 1), (0, 0, 0, 0), (2, 2), 2),
    ("pointwise",    (6, 6),   8, (1, 1), (1, 1), (0, 0, 0, 0), (1, 1), 8),
    ("kernel_eq_ifm", (5, 5),  1, (5, 5), (1, 1), (0, 0, 0, 0), (1, 1), 1),
    ("non_square",   (6, 10),  2, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 2),
    # simd above the channel count emits several kernel columns per beat
    ("par_gray",     (12, 12), 1, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 3),
    ("par_padded",   (9, 9),   1, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 3),
    ("par_nonsquare", (6, 10), 2, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 6),
    ("par_stride",   (11, 11), 2, (3, 3), (2, 2), (1, 1, 1, 1), (1, 1), 6),
    ("par_dilated",  (10, 10), 1, (3, 3), (1, 1), (0, 0, 0, 0), (2, 2), 3),
    ("par_pool",     (10, 10), 4, (2, 2), (2, 2), (0, 0, 0, 0), (1, 1), 8),
    ("par_k5",       (9, 9),   1, (5, 5), (1, 1), (2, 2, 2, 2), (1, 1), 5),
    ("par_partial",  (8, 8),   2, (4, 4), (1, 1), (0, 0, 0, 0), (1, 1), 4),
]

FRAME_REGRESSION = {"basic", "same_pad", "pool_window"}


def uses_parallel_columns(config):
    return config[7] > config[2]


def make_unit(config, strategy):
    _, ifm, channels, kernel, stride, pads, dilation, simd = config
    unit = SlidingWindowUnit("swu", ["uin"], ["uout"], ifm_dim=ifm, ifm_ch=channels,
                             kernel=kernel, strides=stride, pads=pads,
                             dilations=dilation, dtype=INT8, strategy=strategy)
    unit.apply_folding({"simd": simd})
    return unit


@unittest.skipIf(shutil.which("verilator") is None, "verilator not installed")
class SlidingWindowRtlTest(unittest.TestCase):
    def _simulate(self, config, strategy, duties=((100, 100),)):
        name, ifm, channels = config[0], config[1], config[2]
        unit = make_unit(config, strategy)
        # crc32 rather than hash(): string hashing is randomised per process,
        # so hash() would pick a different image every run and a failure could
        # not be reproduced.
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        image = rng.integers(-128, 128, (1, ifm[0], ifm[1], channels), dtype=np.int64)

        build = Fixtures.build_dir("swu_%s_%s" % (strategy, name))
        project = SingleUnitProject(unit, (1, ifm[0], ifm[1], channels), INT8)
        project.write(build, image)

        built = run(["make", "-s"], cwd=build)
        self.assertEqual(built.returncode, 0, built.stderr[-3000:])
        for input_duty, output_duty in duties:
            result = run(["./obj_dir/Votf_top", "--input", "golden/input.hex",
                          "--expected", "golden/expected.hex",
                          "--input-duty", str(input_duty),
                          "--output-duty", str(output_duty), "--quiet"], cwd=build)
            self.assertEqual(result.returncode, 0,
                             "%s %s duty=%s\n%s%s" % (strategy, name,
                                                      (input_duty, output_duty),
                                                      result.stdout, result.stderr))
        return unit

    def test_line_buffer_matches_numpy(self):
        for config in CONFIGS:
            with self.subTest(config=config[0]):
                self._simulate(config, "line",
                               duties=((100, 100), (40, 40), (100, 25), (25, 100)))

    def test_frame_buffer_still_matches_numpy(self):
        for config in CONFIGS:
            if config[0] not in FRAME_REGRESSION:
                continue
            with self.subTest(config=config[0]):
                self._simulate(config, "frame", duties=((100, 100), (40, 40)))


class SlidingWindowModelTest(unittest.TestCase):
    def test_line_buffer_never_costs_more_cycles_than_frame(self):
        for config in CONFIGS:
            if uses_parallel_columns(config):
                continue
            with self.subTest(config=config[0]):
                line = make_unit(config, "line")
                frame = make_unit(config, "frame")
                self.assertLessEqual(line.cycles, frame.cycles)

    def test_parallel_columns_are_offered_only_by_the_line_buffer(self):
        config = ("c", (12, 12), 1, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 1)
        line = make_unit(config, "line")
        frame = make_unit(config, "frame")
        self.assertEqual([o["simd"] for o in frame.folding_options()], [1])
        self.assertEqual([o["simd"] for o in line.folding_options()], [1, 3])

    def test_parallel_columns_fold_a_single_channel_layer(self):
        """One input channel leaves nothing to fold on channels, which is what
        pinned the example CNN."""
        config = ("c", (12, 12), 1, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 3)
        wide = make_unit(config, "line")
        narrow = make_unit(("c",) + config[1:7] + (1,), "line")
        self.assertEqual(wide.parallel_columns, 3)
        self.assertEqual(narrow.parallel_columns, 1)
        self.assertEqual(wide.cycles * 3, narrow.cycles)
        self.assertEqual(wide.sub_elems, narrow.sub_elems)

    def test_line_buffer_is_sized_for_liveness_without_waste(self):
        """The buffer must cover the window span plus the backward reach at an
        output row wrap, plus one so the writer can stay ahead. Rounding to a
        power of two may never more than double that."""
        for config in CONFIGS:
            with self.subTest(config=config[0]):
                unit = make_unit(config, "line")
                required = unit.span_pixels + unit.backward_reach + 1
                self.assertGreaterEqual(unit.buffer_pixels, required)
                self.assertLess(unit.buffer_pixels, 2 * required)

    def test_line_buffer_is_far_smaller_than_a_frame_on_a_tall_image(self):
        config = ("c", (224, 224), 8, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 8)
        line = make_unit(config, "line")
        frame = make_unit(config, "frame")
        self.assertLess(line.buffer_pixels * 8, frame.buffer_pixels)

    def test_line_buffer_cost_is_the_replay_term(self):
        config = ("c", (28, 28), 16, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 4)
        unit = make_unit(config, "line")
        self.assertEqual(unit.cycles, unit.replay_cycles)


if __name__ == "__main__":
    unittest.main()
