"""BinaryFILE framing and symbol selection preserve the source payloads."""

import io
import unittest

from onnx2fpga.trading.recorded import messages, select
from onnx2fpga.trading.itch import packets, HEADER, MTU_PAYLOAD


def binary(payloads, end=True):
    return io.BytesIO(b"".join(len(p).to_bytes(2, "big") + p for p in payloads)
                      + (b"\0\0" if end else b""))


def directory(symbol, locate):
    return b"R" + locate.to_bytes(2, "big") + bytes(8) + symbol.ljust(8, b" ") + bytes(20)


class RecordedTest(unittest.TestCase):
    def test_oversize_message_rejected(self):
        for payload in (b"", bytes(MTU_PAYLOAD - HEADER - 1)):
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                packets([payload])
        self.assertTrue(packets([bytes(MTU_PAYLOAD - HEADER - 2)]))

    def test_framing(self):
        self.assertEqual(list(messages(binary([b"abc", b"defg"]))), [b"abc", b"defg"])
        self.assertEqual(list(messages(binary([]))), [])

    def test_incomplete_and_trailing_bytes(self):
        for data in (b"", b"\0", b"\0\x03ab", b"\0\x01a", b"\0\0x"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                list(messages(io.BytesIO(data)))

    def test_eof_capture(self):
        wanted = directory(b"AAPL", 123)
        selected, record = select(binary([wanted], end=False), "AAPL", 10, allow_eof=True)
        self.assertEqual(selected, [wanted])
        self.assertTrue(record["source_file_exhausted"])
        self.assertFalse(record["source_session_complete"])
        _, terminated = select(binary([wanted]), "AAPL", 10, allow_eof=True)
        self.assertTrue(terminated["source_session_complete"])
        _, prefix = select(binary([wanted]), "AAPL", 1, allow_eof=True)
        self.assertFalse(prefix["source_file_exhausted"])
        self.assertFalse(prefix["source_session_complete"])
        for data in (b"\0", b"\0\x03ab", b"\0\0x"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                list(messages(io.BytesIO(data), allow_eof=True))

    def test_symbol_filter_and_prefix(self):
        wanted = directory(b"AAPL", 123)
        other = directory(b"MSFT", 456)
        event = b"D\0{" + bytes(16)
        selected, record = select(binary([other, wanted, event]), "AAPL", 2)
        self.assertEqual(selected, [wanted, event])
        self.assertEqual(record["locate"], 123)
        self.assertEqual(record["source_messages_scanned"], 3)
        self.assertFalse(record["source_session_complete"])
        selected, record = select(binary([other, wanted, event]), "AAPL", 10)
        self.assertTrue(record["source_session_complete"])
        self.assertEqual(selected, [wanted, event])

    def test_missing_and_changed_directory(self):
        with self.assertRaisesRegex(ValueError, "no symbol"):
            select(binary([directory(b"MSFT", 1)]), "AAPL", 10)
        with self.assertRaisesRegex(ValueError, "changed stock locate"):
            select(binary([directory(b"AAPL", 1), directory(b"AAPL", 2)]), "AAPL", 10)
        with self.assertRaisesRegex(ValueError, "reassigned"):
            select(binary([directory(b"AAPL", 1), directory(b"MSFT", 1)]), "AAPL", 10)
