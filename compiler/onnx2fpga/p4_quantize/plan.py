"""The integer grid of every tensor, from calibration or from the model's QDQ nodes."""

import numpy as np

from ..errors import ConfigError
from ..p2_graph.datatype import INT8, QuantSpec

TINY = 1e-12


class QuantizationPlan:
    def __init__(self, mult_bits=18):
        self.act_dtype = INT8
        self.weight_dtype = INT8
        self.mult_bits = mult_bits
        self.specs = {}
        self.from_model = False
        self.source = "calibration"
        self.layer_weight_dtypes = {}

    @classmethod
    def from_peaks(cls, peaks, mult_bits=18):
        plan = cls(mult_bits)
        for name, peak in peaks.items():
            plan.annotate(name, QuantSpec(max(peak, TINY) / plan.act_dtype.max, 0))
        return plan

    @classmethod
    def from_annotations(cls, annotations, mult_bits=18):
        plan = cls(mult_bits)
        for name, spec in annotations.items():
            plan.annotate(name, spec)
        plan.from_model = True
        plan.source = "model"
        return plan

    def annotate(self, name, spec):
        self.specs[name] = spec.to_signed()

    def has(self, name):
        return name in self.specs

    def spec(self, name):
        try:
            return self.specs[name]
        except KeyError:
            raise KeyError("no quantization for tensor %r" % name)

    def propagate(self, source, target):
        """Scale preserving operations pass their grid through unchanged."""
        if source in self.specs:
            self.specs[target] = self.specs[source]

    def quantize(self, name, real):
        """Codes for real values of tensor `name`, per element when its grid is per feature."""
        spec = self.spec(name)
        return self.act_dtype.clamp(np.rint(real / spec.scale) + spec.zero_point)

    def scale(self, name):
        return float(self.spec(name).scale.flat[0])

    def zero_point(self, name):
        return int(self.spec(name).zero_point.flat[0])

    def weight_dtype_for(self, layer):
        return self.layer_weight_dtypes.get(layer, self.weight_dtype)

    def weight_scales(self, weights, spec, layer):
        """Per output channel, with the output channel axis last."""
        if spec is None:
            peaks = np.max(np.abs(weights), axis=tuple(range(weights.ndim - 1)))
            return np.maximum(peaks, TINY) / self.weight_dtype_for(layer).max
        if layer in self.layer_weight_dtypes:
            raise ConfigError("%s has weights quantized by the model itself, so their "
                              "precision cannot be overridden" % layer)
        if not spec.is_symmetric:
            raise NotImplementedError("weight zero points must be zero")
        scales = np.asarray(spec.scale, dtype=np.float64).reshape(-1)
        if scales.size == 1:
            return np.full(weights.shape[-1], float(scales[0]))
        return scales

    def quantize_weights(self, weights, scales, layer):
        return self.weight_dtype_for(layer).clamp(np.rint(weights / scales))
