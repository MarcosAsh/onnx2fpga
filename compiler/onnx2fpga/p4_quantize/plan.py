"""The quantization plan: what integer grid every tensor lives on.

The plan can be filled two ways. Calibration measures activation ranges on
float data and derives symmetric scales. Annotation reads the scales an
external quantizer already chose and recorded in the model as QuantizeLinear
and DequantizeLinear pairs. Everything downstream reads the plan and does not
care which produced it.

Weights are per output channel, activations per tensor. Activation zero points
may be non zero; weight zero points may not, because a weight zero point
introduces a term proportional to the sum of the input, which is not a compile
time constant and would cost real hardware.
"""

import numpy as np

from ..p2_graph.datatype import INT8, QuantSpec

TINY = 1e-12


class QuantizationPlan:
    def __init__(self, act_dtype=INT8, weight_dtype=INT8, mult_bits=18):
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.mult_bits = mult_bits
        self.specs = {}
        self.from_model = False

    @classmethod
    def from_peaks(cls, peaks, act_dtype=INT8, weight_dtype=INT8, mult_bits=18):
        plan = cls(act_dtype, weight_dtype, mult_bits)
        for name, peak in peaks.items():
            plan.annotate(name, QuantSpec(max(peak, TINY) / act_dtype.max, 0))
        return plan

    @classmethod
    def from_annotations(cls, annotations, act_dtype=INT8, weight_dtype=INT8,
                         mult_bits=18):
        plan = cls(act_dtype, weight_dtype, mult_bits)
        for name, spec in annotations.items():
            plan.annotate(name, spec)
        plan.from_model = True
        return plan

    def annotate(self, name, spec):
        self.specs[name] = spec.to_signed()
        return self.specs[name]

    def has(self, name):
        return name in self.specs

    def spec(self, name):
        try:
            return self.specs[name]
        except KeyError:
            raise KeyError("no quantization recorded for tensor %r; calibrate "
                           "the model or quantize it before compiling" % name)

    def propagate(self, source, target):
        """Scale preserving operations pass their grid through unchanged."""
        if source in self.specs:
            self.specs[target] = self.specs[source]
        return self.specs.get(target)

    def scale(self, name):
        return float(self.spec(name).scale.flat[0])

    def zero_point(self, name):
        return int(self.spec(name).zero_point.flat[0])

    def offset(self, name):
        return self.spec(name).offset

    def weight_scales(self, weights, spec=None):
        """Per output channel, with the output channel axis last."""
        if spec is not None:
            if not spec.is_symmetric:
                raise NotImplementedError(
                    "weight zero points must be zero: a non zero one adds a term "
                    "proportional to the running input sum, which is not constant")
            scales = np.asarray(spec.scale, dtype=np.float64).reshape(-1)
            if scales.size == 1:
                return np.full(weights.shape[-1], float(scales[0]))
            return scales
        peaks = np.max(np.abs(weights), axis=tuple(range(weights.ndim - 1)))
        return np.maximum(peaks, TINY) / self.weight_dtype.max

    def quantize_weights(self, weights, scales):
        return self.weight_dtype.clamp(np.rint(weights / scales))

    def __repr__(self):
        asymmetric = sum(1 for s in self.specs.values() if not s.is_symmetric)
        return "QuantizationPlan(%d tensors, %d asymmetric, act=%s, weight=%s)" % (
            len(self.specs), asymmetric, self.act_dtype, self.weight_dtype)
