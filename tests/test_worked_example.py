"""The README's worked example has to keep working.

A broken example is worse than none: it is the first thing anyone runs, and it
is the one file where a change three passes away shows up as "this project
does not work". So the example is tested like anything else.
"""

import unittest

from support import Fixtures

import worked_example


class WorkedExampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = worked_example.main(
            ["--quiet", "--out", str(Fixtures.build_dir("worked_example"))])

    def test_the_float_model_actually_learns_the_task(self):
        """If it did not, the comparison underneath it would be meaningless:
        two bad models agreeing says nothing about quantization."""
        self.assertGreater(self.summary["float_accuracy"], 0.9)

    def test_quantization_costs_almost_nothing_on_this_task(self):
        self.assertGreaterEqual(self.summary["integer_accuracy"],
                                self.summary["float_accuracy"] - 0.02)

    def test_the_compiled_design_agrees_with_the_float_model(self):
        self.assertGreater(self.summary["agreement"], 0.95)

    def test_it_leaves_a_buildable_design_behind(self):
        build = Fixtures.build_dir("worked_example")
        for expected in ("Makefile", "manifest.json", "worked.onnx"):
            self.assertTrue((build / expected).exists(), expected)
        self.assertTrue((build / "rtl").is_dir())
        self.assertTrue((build / "golden" / "input.hex").exists())

    def test_it_reports_what_quantization_cost(self):
        self.assertIsNotNone(self.summary["result"].accuracy)


if __name__ == "__main__":
    unittest.main()
