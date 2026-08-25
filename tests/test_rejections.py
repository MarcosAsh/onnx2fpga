"""Refusing a model is an answer, and it has to be a useful one.

Every one of these paths is reachable by someone with a real model and no
knowledge of this compiler's internals. "unsupported op Softmax" tells them
their afternoon is over; naming the node and the rewrite tells them what to
change. These tests assert the second kind of message, because the first kind
is what you get by default and it comes back the moment nobody is looking.
"""

import unittest

import numpy as np

from support import Fixtures  # noqa: F401

from onnx2fpga.compile import Compiler
from onnx2fpga.p1_ingest.onnx_model import ModelBuilder
from onnx2fpga.p2_graph.onnx_importer import ImportError_, OnnxImporter


def imported(builder):
    return OnnxImporter(builder.build()).run()


class UnsupportedOperatorTest(unittest.TestCase):
    def model_with(self, op_type, **attrs):
        builder = ModelBuilder("reject").add_input("x", (1, 4))
        builder.add_node(op_type, ["x"], ["y"], name="the_node", **attrs)
        return builder.add_output("y", (1, 4))

    def test_it_names_the_operator_and_the_node(self):
        with self.assertRaises(ImportError_) as caught:
            imported(self.model_with("Softmax"))
        message = str(caught.exception)
        self.assertIn("Softmax", message)
        self.assertIn("the_node", message)

    def test_it_offers_the_rewrite_when_there_is_one(self):
        with self.assertRaises(ImportError_) as caught:
            imported(self.model_with("Softmax"))
        self.assertIn("argmax", str(caught.exception))

    def test_it_lists_what_is_supported(self):
        """So the reader can see whether their model is one edit away or a
        different shape entirely."""
        with self.assertRaises(ImportError_) as caught:
            imported(self.model_with("Softmax"))
        message = str(caught.exception)
        for supported in ("Conv", "Gemm", "MaxPool", "Relu"):
            self.assertIn(supported, message)

    def test_an_operator_with_no_advice_still_names_itself(self):
        with self.assertRaises(ImportError_) as caught:
            imported(self.model_with("Erf"))
        message = str(caught.exception)
        self.assertIn("Erf", message)
        self.assertIn("the_node", message)

    def test_every_rewrite_is_a_sentence_not_a_shrug(self):
        for op, advice in OnnxImporter.REWRITES.items():
            with self.subTest(op=op):
                self.assertGreater(len(advice), 30, op)
                self.assertNotIn("unsupported", advice.lower())

    def test_no_rewrite_describes_an_operator_that_is_supported(self):
        """A rewrite for a supported operator is dead advice that contradicts
        the list printed beside it. Squeeze was exactly that until this
        test."""
        supported = set(OnnxImporter(
            self.model_with("Relu").build())._dispatch)
        overlap = supported & set(OnnxImporter.REWRITES)
        self.assertEqual(overlap, set(),
                         "these are supported and should not have rewrites: %s"
                         % sorted(overlap))


class UnsupportedAttributeTest(unittest.TestCase):
    def conv(self, **attrs):
        builder = ModelBuilder("rejectconv").add_input("x", (1, 2, 8, 8))
        builder.add_initializer("w", np.zeros((4, 1, 3, 3), dtype=np.float32))
        builder.add_node("Conv", ["x", "w"], ["y"], name="the_conv",
                         kernel_shape=[3, 3], **attrs)
        return builder.add_output("y", (1, 4, 6, 6))

    def test_grouped_convolution_says_how_to_split_it(self):
        with self.assertRaises(ImportError_) as caught:
            imported(self.conv(group=2))
        message = str(caught.exception)
        self.assertIn("the_conv", message)
        self.assertIn("groups=2", message)
        self.assertIn("separate Conv", message)

    def gemm(self, **attrs):
        builder = ModelBuilder("rejectgemm").add_input("x", (1, 4))
        builder.add_initializer("w", np.zeros((4, 3), dtype=np.float32))
        builder.add_node("Gemm", ["x", "w"], ["y"], name="the_gemm", **attrs)
        return builder.add_output("y", (1, 3))

    def test_alpha_scaling_says_to_fold_it_into_the_weights(self):
        with self.assertRaises(ImportError_) as caught:
            imported(self.gemm(alpha=2.0))
        message = str(caught.exception)
        self.assertIn("the_gemm", message)
        self.assertIn("weights", message)

    def test_transa_says_what_to_export_instead(self):
        with self.assertRaises(ImportError_) as caught:
            imported(self.gemm(transA=1))
        message = str(caught.exception)
        self.assertIn("the_gemm", message)
        self.assertIn("transA=0", message)


class CalibrationTest(unittest.TestCase):
    def test_a_float_model_with_no_samples_says_which_flag_to_pass(self):
        with self.assertRaises(ValueError) as caught:
            Compiler(device="vu9p").compile(
                Fixtures.factory().mlp(), None,
                Fixtures.build_dir("reject_nosamples"))
        message = str(caught.exception)
        self.assertIn("--calibration", message)
        self.assertIn("--samples", message)

    def test_it_says_random_samples_are_not_an_accuracy_claim(self):
        """Someone reaching for --samples to get past the error should be told
        what they are getting."""
        with self.assertRaises(ValueError) as caught:
            Compiler(device="vu9p").compile(
                Fixtures.factory().mlp(), None,
                Fixtures.build_dir("reject_nosamples2"))
        self.assertIn("accuracy", str(caught.exception))


class UnrollRefusalTest(unittest.TestCase):
    def test_it_names_the_device_the_resource_and_the_alternatives(self):
        from onnx2fpga.p2_graph.graph import GraphError
        with self.assertRaises(GraphError) as caught:
            Compiler(device="z7020", unroll=True).compile(
                Fixtures.factory().mlp(), Fixtures.samples(Fixtures.MLP_SHAPE),
                Fixtures.build_dir("reject_unroll"))
        message = str(caught.exception)
        self.assertIn("z7020", message)
        self.assertIn("dsp", message)
        self.assertIn("larger device", message)


if __name__ == "__main__":
    unittest.main()
