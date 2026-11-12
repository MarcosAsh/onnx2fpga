"""Streaming replay comparison must reject missing, extra and incorrect outputs."""

from types import SimpleNamespace
import unittest

from onnx2fpga.trading.reference import Config
from onnx2fpga.trading.replay import ReplayError, compare


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.config = Config(levels=1)
        self.expected = [SimpleNamespace(snapshot=(0, ((100, 5),), ((101, 8),), 0),
                                         features=(1, -2, 3, -4), action=0, prediction=7,
                                         order=b"", message=SimpleNamespace(
                                             data=b"test", first_beat=5, last_beat=10))]
        self.lines = ["beat0 0", "B 0 500000064 800000065 0", "F fc03fe01",
                      "D 32 0 7 0 0 5 10 32", "cycles 40"]

    def test_one_pass_iterator(self):
        self.assertEqual(compare(iter(self.lines), self.expected, self.config, 22).tolist(), [27])

    def test_missing_and_extra_records(self):
        for index in range(4):
            with self.subTest(missing=index), self.assertRaises(ReplayError):
                compare(iter(self.lines[:index] + self.lines[index + 1:]), self.expected,
                        self.config, 22)
        for index in (1, 2, 3):
            with self.subTest(extra=index), self.assertRaises(ReplayError):
                compare(iter(self.lines + [self.lines[index]]), self.expected, self.config, 22)

    def test_bad_value_and_latency(self):
        for index, replacement in ((2, "F fc03fe02"), (3, "D 32 0 7 0 0 5 10 33")):
            lines = self.lines.copy()
            lines[index] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ReplayError):
                compare(iter(lines), self.expected, self.config, 22)
