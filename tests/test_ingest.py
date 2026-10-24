"""The protobuf codec and the ONNX reader."""

import unittest

import numpy as np

from support import ROOT, Fixtures

from onnx2fpga.p1_ingest import ModelBuilder, OnnxModel
from onnx2fpga.p1_ingest.onnx_schema import TensorProto
from onnx2fpga.p1_ingest.protobuf_codec import Decoder, Encoder, LEN, VARINT


class CodecTest(unittest.TestCase):
    def test_varint_round_trip(self):
        for value in (0, 1, 127, 128, 300, 2 ** 31, 2 ** 63 - 1):
            encoder = Encoder()
            encoder.varint_field(1, value)
            decoder = Decoder(encoder.to_bytes())
            self.assertEqual(next(decoder.fields()), (1, VARINT))
            self.assertEqual(decoder.varint(), value)

    def test_skip_consumes_one_field(self):
        encoder = Encoder()
        encoder.string_field(2, "abcd")
        encoder.varint_field(3, 99)
        decoder = Decoder(encoder.to_bytes())
        fields = decoder.fields()
        number, wire = next(fields)
        self.assertEqual((number, wire), (2, LEN))
        decoder.skip(wire)
        self.assertEqual(next(fields)[0], 3)
        self.assertEqual(decoder.varint(), 99)

    def test_packed_and_unpacked_repeated_fields(self):
        packed = TensorProto(dims=[1, 2, 300], float_data=[0.5, -2.0]).encode()
        decoded = TensorProto.decode(packed)
        self.assertEqual(decoded.dims, [1, 2, 300])
        self.assertEqual(decoded.float_data, [0.5, -2.0])
        unpacked = Encoder()
        for value in (4, 5):
            unpacked.varint_field(1, value)
        self.assertEqual(TensorProto.decode(unpacked.to_bytes()).dims, [4, 5])


class ReaderTest(unittest.TestCase):
    def test_round_trip(self):
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
        self.assertEqual(reloaded.nodes()[0].attributes(), {"alpha": 1.0, "transB": 0})

    def test_models_in_tests_data_parse(self):
        """A no-op until someone drops an .onnx file into tests/data."""
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
