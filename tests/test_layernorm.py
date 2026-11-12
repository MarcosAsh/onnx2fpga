"""Layer normalization import, integer arithmetic and fixed-latency RTL."""

import shutil
import unittest

import numpy as np

import onnx2fpga as otf
from onnx2fpga.errors import ConfigError, UnsupportedOpError
from onnx2fpga.p1_ingest import ModelBuilder
from onnx2fpga.p2_graph.onnx_importer import OnnxImporter
from onnx2fpga.p3_ops.reductions import SoftmaxUnit
from onnx2fpga.p4_quantize.calibrate import Calibrator
from onnx2fpga.p4_quantize.lower import LowerToHardware
from onnx2fpga.pipeline import CompilerContext
from onnx2fpga.targets.device import Device
from support import Fixtures, Hdl


def model(lanes=8, bias=True, outputs=None, **attrs):
    builder = ModelBuilder('layernorm', opset=17).add_input('x', (1, lanes))
    builder.add_initializer('scale', np.linspace(-1.5, 1.5, lanes).astype(np.float32))
    inputs = ['x', 'scale']
    if bias:
        builder.add_initializer('bias', np.linspace(-.3, .7, lanes).astype(np.float32))
        inputs.append('bias')
    builder.add_node('LayerNormalization', inputs, outputs or ['y'], name='norm', **attrs)
    return builder.add_output('y', (1, lanes)).build()


def reference(x, epsilon):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + epsilon)


class Arithmetic(unittest.TestCase):
    def test_centered_normalization(self):
        rng = np.random.default_rng(17)
        for lanes in (1, 3, 8, 16):
            for epsilon in (1e-5, .1):
                with self.subTest(lanes=lanes, epsilon=epsilon):
                    graph = OnnxImporter(model(lanes, epsilon=epsilon)).run()
                    samples = [{'x': row[None]} for row in rng.normal(size=(32, lanes))]
                    plan = Calibrator(graph).plan(samples)
                    graph = LowerToHardware(plan, fixed=True).run(
                        graph, CompilerContext(Device.get('sim')))
                    unit = next(n for n in graph.nodes if isinstance(n, SoftmaxUnit))
                    values = rng.integers(-128, 128, size=(512, lanes))
                    values[:256] = np.arange(-128, 128)[:, None]
                    if lanes > 1:
                        values[256:384] = 64
                        values[256:384, 0] += 1
                    scale = plan.scale(unit.outputs[0])
                    expected = reference(values * plan.scale('x'), epsilon)
                    got = np.concatenate([unit.execute(graph, {'x': row[None]})[0]
                                          for row in values])
                    self.assertLess(np.max(np.abs(got * scale - expected)), .6 * scale)
                    np.testing.assert_array_equal(got[:256], 0)

    def test_import_contract(self):
        for attrs, outputs in (({'axis': 2}, None), ({'epsilon': 0}, None),
                               ({'epsilon': float('nan')}, None), ({'stash_type': 10}, None),
                               ({}, ['y', 'mean'])):
            with self.subTest(attrs=attrs, outputs=outputs):
                with self.assertRaises(UnsupportedOpError):
                    OnnxImporter(model(outputs=outputs, **attrs)).run()
        with self.assertRaises(ConfigError):
            otf.load(model()).compile(mode='stream', calibration=np.ones((4, 8)))


@unittest.skipIf(Hdl.skip_reason('hardcaml'), 'Hardcaml unavailable')
class Hardware(unittest.TestCase):
    def test_reload_keeps_normalization(self):
        samples = np.random.default_rng(19).normal(size=(32, 8))
        acc = otf.load(model())
        acc.compile(device='k26', mode='fixed', weights='reloadable', calibration=samples,
                    out_dir=Fixtures.build_dir('layernorm_reload'))
        acc._retarget(model())
        with self.assertRaisesRegex(ConfigError, 'frozen'):
            acc._retarget(model(epsilon=.1))

    @unittest.skipUnless(shutil.which('verilator'), 'Verilator unavailable')
    def test_affine_rtl(self):
        cases = (('k26', 322.265625, 1, True, 1e-5),
                 ('k26', 322.265625, 3, False, .1),
                 ('k26', 322.265625, 8, True, 1e-5),
                 ('icepi25', 80, 3, True, .1))
        for device, clock, lanes, bias, epsilon in cases:
            with self.subTest(device=device, lanes=lanes, bias=bias):
                rng = np.random.default_rng(71)
                x = rng.normal(size=(256, lanes))
                x[:32] = np.arange(-16, 16)[:, None] / 16
                if lanes > 1:
                    x[32:64] = 1
                    x[32:64, 0] += .05
                acc = otf.load(model(lanes, bias=bias, epsilon=epsilon))
                acc.compile(device=device, mode='fixed', clock_mhz=clock,
                            out_dir=Fixtures.build_dir('layernorm_%s_%d' % (device, lanes)),
                            calibration=x)
                report = acc.simulate(x=x)
                self.assertTrue(report.ok, report.first.output[-2000:])
                self.assertEqual(report.frames, len(x))
                self.assertEqual(report.latency, acc.latency)
                expected = reference(x, epsilon) * np.linspace(-1.5, 1.5, lanes)
                if bias:
                    expected += np.linspace(-.3, .7, lanes)
                np.testing.assert_allclose(acc.predict(x), expected, atol=.1, rtol=.03)


if __name__ == '__main__':
    unittest.main()
