"""DDR record format against the existing MoldUDP64 beat representation."""

import io
import unittest

from onnx2fpga.trading import itch, replay_records as replay


class Records(unittest.TestCase):
    def test_packet_boundaries_and_byte_order(self):
        packets = [bytes(range(20)), bytes(range(21)), bytes(range(1, 33))]
        beats = itch.beats(packets)
        encoded = b"".join(replay.records(beats))
        self.assertEqual(encoded[:16], bytes(range(8)) + b"\xff\x00" + bytes(6))
        self.assertEqual(list(replay.read_records(io.BytesIO(encoded))), beats)
        self.assertEqual(len(encoded), len(beats) * 16)

    def test_bad_records(self):
        valid = b"".join(replay.records([(1, 255, True)]))
        for broken in (valid[:-1], valid[:8] + b"\x05" + valid[9:],
                       valid[:9] + b"\x02" + valid[10:], valid[:-1] + b"\x01",
                       valid[:9] + b"\x00" + valid[10:]):
            with self.subTest(record=broken), self.assertRaises(ValueError):
                list(replay.read_records(io.BytesIO(broken)))

    def test_bad_beats(self):
        for beat in ((-1, 255, True), (1 << 64, 255, True), (0, 0, True),
                     (0, 5, True), (0, 7, False), (0, 255, 2), (0, 255, False)):
            with self.subTest(beat=beat), self.assertRaises(ValueError):
                list(replay.records([beat]))
