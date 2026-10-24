"""The quantization plan: the integer grid of every tensor, from calibration or QDQ annotation."""

import numpy as np

from ..errors import ConfigError
from ..p2_graph.datatype import INT8, QuantSpec

TINY = 1e-12


class QuantizationPlan:
    def __init__(self, act_dtype=INT8, weight_dtype=INT8, mult_bits=18):
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.mult_bits = mult_bits
        self.specs = {}
        self.from_model = False
        self.layer_weight_dtypes = {}

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

    def weight_dtype_for(self, layer):
        """The weight precision of a layer if set, else the plan; activations keep one width."""
        return self.layer_weight_dtypes.get(layer, self.weight_dtype)

    def weight_scales(self, weights, spec=None, dtype=None, layer=None):
        """Per output channel, with the output channel axis last."""
        if spec is not None and layer in self.layer_weight_dtypes:
            raise ConfigError(
                "%s's weights were quantized by the model itself, so its precision "
                "cannot be overridden. Quantize the model with the precision you want "
                "instead, or start from the float model" % layer)
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
        return np.maximum(peaks, TINY) / (dtype or self.weight_dtype).max

    def quantize_weights(self, weights, scales, dtype=None):
        return (dtype or self.weight_dtype).clamp(np.rint(weights / scales))

    def __repr__(self):
        asymmetric = sum(1 for s in self.specs.values() if not s.is_symmetric)
        return "QuantizationPlan(%d tensors, %d asymmetric, act=%s, weight=%s)" % (
            len(self.specs), asymmetric, self.act_dtype, self.weight_dtype)
