"""What the compiler says quantization cost, before anyone builds anything."""

import unittest

import numpy as np

from support import Fixtures

from onnx2fpga.accuracy import AccuracyReport
from onnx2fpga.compile import Compiler


class AccuracyReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = Fixtures.samples(Fixtures.MLP_SHAPE)
        cls.result = Compiler(device="vu9p", target_cycles=64).compile(
            Fixtures.factory().mlp(), cls.samples,
            Fixtures.build_dir("accuracy_mlp"))

    def test_compiling_produces_a_report(self):
        self.assertIsNotNone(self.result.accuracy)
        self.assertEqual(len(self.result.accuracy.rows),
                         len(self.result.hardware_graph.outputs))

    def test_it_names_every_output(self):
        names = [row.name for row in self.result.accuracy.rows]
        self.assertEqual(names, list(self.result.hardware_graph.outputs))

    def test_the_error_is_small_but_not_zero(self):
        """Zero would mean it is comparing something against itself."""
        row = self.result.accuracy.rows[0]
        self.assertGreater(row.worst, 0.0)
        self.assertLess(row.relative, 0.2)

    def test_the_mean_is_no_worse_than_the_worst(self):
        for row in self.result.accuracy.rows:
            self.assertLessEqual(row.mean, row.worst)

    def test_a_model_with_no_samples_reports_nothing_rather_than_guessing(self):
        report = AccuracyReport.measure(
            self.result.float_graph, self.result.hardware_graph,
            self.result.plan, [])
        self.assertIsNone(report)

    def test_the_render_states_the_sample_count(self):
        """How many samples it saw, because an error measured over three of
        them is a different claim from one measured over thirty."""
        text = self.result.accuracy.render()
        self.assertIn("accuracy against the float model", text)
        self.assertEqual(self.result.accuracy.samples, len(self.samples))
        self.assertIn("%d samples" % len(self.samples), text)


class WhatWideningActuallyMovesTest(unittest.TestCase):
    """The distinction the report exists to get right.

    Widening the datapath shrinks the absolute error and the relative error,
    and leaves the error measured in quantization steps almost unchanged,
    because the step shrinks with it. A report that told someone to widen the
    datapath to improve a step count would be advice that does not work, and
    the first draft of this one did exactly that."""

    def report_at(self, bits, name):
        return Compiler(device="vu9p", target_cycles=64, act_bits=bits,
                        weight_bits=bits).compile(
            Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
            Fixtures.build_dir(name)).accuracy

    def test_widening_shrinks_the_absolute_and_relative_error(self):
        narrow = self.report_at(8, "acc_narrow").rows[0]
        wide = self.report_at(16, "acc_wide").rows[0]
        self.assertLess(wide.worst, narrow.worst / 50)
        self.assertLess(wide.relative, narrow.relative / 50)

    def test_widening_barely_moves_the_step_count(self):
        narrow = self.report_at(8, "acc_narrow_steps").rows[0]
        wide = self.report_at(16, "acc_wide_steps").rows[0]
        self.assertLess(abs(wide.worst_steps - narrow.worst_steps),
                        0.5 * narrow.worst_steps)

    def test_the_advice_is_attached_to_the_number_it_moves(self):
        """int8 on this model is loose enough to suggest widening; int16 is
        not, and must not repeat the suggestion."""
        narrow = self.report_at(8, "acc_advice_narrow")
        wide = self.report_at(16, "acc_advice_wide")
        self.assertIn("--act-bits 16", narrow.render())
        self.assertNotIn("--act-bits 16", wide.render())


if __name__ == "__main__":
    unittest.main()
