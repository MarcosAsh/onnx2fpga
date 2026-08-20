"""Stage 1: the protobuf codec and the ONNX reader."""

import unittest

import numpy as np

from support import ROOT, Fixtures

from onnx2fpga.p1_ingest import ModelBuilder, OnnxModel
from onnx2fpga.p1_ingest.protobuf_codec import Decoder, Encoder, LEN, VARINT


class ProtobufCodecTest(unittest.TestCase):
    def test_varint_round_trip(self):
        for value in (0, 1, 127, 128, 300, 2 ** 31, 2 ** 63 - 1):
            encoder = Encoder()
            encoder.varint_field(1, value)
            decoder = Decoder(encoder.to_bytes())
            number, wire = next(iter(decoder.fields()))
            self.assertEqual((number, wire), (1, VARINT))
            self.assertEqual(decoder.varint(), value)

    def test_skip_consumes_exactly_one_field(self):
        encoder = Encoder()
        encoder.string_field(2, "abcd")
        encoder.varint_field(3, 99)
        decoder = Decoder(encoder.to_bytes())
        fields = decoder.fields()
        number, wire = next(fields)
        self.assertEqual((number, wire), (2, LEN))
        decoder.skip(wire)
        number, wire = next(fields)
        self.assertEqual(number, 3)
        self.assertEqual(decoder.varint(), 99)

    def test_packed_and_unpacked_repeated_fields(self):
        encoder = Encoder()
        encoder.packed_varints(1, [1, 2, 300])
        decoder = Decoder(encoder.to_bytes())
        number, wire = next(iter(decoder.fields()))
        self.assertEqual(wire, LEN)


class OnnxReaderTest(unittest.TestCase):
    def test_round_trip_preserves_everything(self):
        weights = np.arange(12, dtype=np.float32).reshape(3, 4)
        model = (ModelBuilder("t", opset=13)
                 .add_input("x", (1, 3)).add_output("y", (1, 4))
                 .add_initializer("W", weights)
                 .add_node("Gemm", ["x", "W"], ["y"], alpha=1.0, transB=0)
                 .build())
        path = Fixtures.build_dir("ingest") / "round_trip.onnx"
        model.save(path)
        reloaded = OnnxModel.load(path)

        self.assertEqual(reloaded.opset, 13)
        self.assertEqual(reloaded.graph_inputs(), ["x"])
        self.assertEqual(reloaded.graph_outputs(), ["y"])
        self.assertEqual(reloaded.value_shape("x"), (1, 3))
        np.testing.assert_array_equal(reloaded.initializer("W"), weights)
        self.assertEqual(reloaded.nodes()[0].attributes()["transB"], 0)

    def test_third_party_models_parse(self):
        """Any .onnx dropped into tests/data must parse. Nothing is vendored,
        so this is a no-op until someone adds a file."""
        data = ROOT / "tests" / "data"
        models = sorted(data.glob("*.onnx")) if data.exists() else []
        if not models:
            self.skipTest("no models in tests/data")
        for path in models:
            with self.subTest(model=path.name):
                model = OnnxModel.load(path)
                self.assertTrue(model.nodes())
                self.assertTrue(model.graph_inputs())


if __name__ == "__main__":
    unittest.main()
