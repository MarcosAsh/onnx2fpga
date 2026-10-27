# Quantization

Status: partial (per layer activation precision is not built)

The hardware only does integer arithmetic, so every tensor needs a scale. Those
scales come from one of two places.

If the model was already quantized and saved in ONNX's QDQ form, with
QuantizeLinear and DequantizeLinear pairs, it carries its own scales. We read
them and you need no data at all.

A float model has to be calibrated. Pass a batch of real inputs as
`calibration=` to `compile`. Leave it out and we calibrate on random normal
noise and print a warning. That is fine for checking that a model compiles. It
is useless for accuracy.

Activations get one scale per tensor and can be asymmetric, signed or
unsigned. The one exception is an input the model scales itself, through a
`Scaler` or a `BatchNormalization`: it gets a scale and zero point per feature,
fitted to the calibration data, which the first layer's weights and bias take
in ([009](../specs/009-operator-coverage.md)). Weights get one scale per output channel and are symmetric. Two
tricks keep this cheap in hardware. An input zero point folds into the bias at
compile time and costs nothing. And an unsigned activation rides on the signed
grid with a recorded offset, so there is one datapath, not two.

To see what quantization cost you:

```python
acc.evaluate(x_test, y_test)
# {'accuracy': ..., 'float_accuracy': ..., 'agreement': ...}
```

`accuracy` is the integer model, which the hardware matches bit for bit.
`agreement` is how often it picks the same class as the float model. You get
that one even without labels.

Narrower weights for a single layer:

```python
acc.layers["fc1"].quantize(weights="int4")
```

Anything from int2 to int8 works. On AMD parts narrower weights also pack more
multiplies into each DSP. A QDQ model refuses the override, since its weights
are already fixed.

A tree ensemble takes its leaves as weights and its features as activations,
each int2 to int16, 16 bits by default:

```python
acc.layers["TreeEnsembleClassifier"].quantize(weights="int8", activations="int12")
```

On the corpus that takes 45 percent of the LUTs off every tree model and keeps
the labels within 2 percent of ONNX Runtime's, see
[013](../specs/013-tree-widths.md).

## Not built yet

`quantize(activations=...)` raises `ConfigError` on any layer but a tree
ensemble. Every stream in the
pipeline is one width, and giving a layer its own would need converters on
both sides that the compiler does not insert yet.

A QONNX model, as Brevitas and the hls4ml tools export, states its own grids
with `Quant` nodes, and they are read like QDQ ones: weights keep their 2 to 8
bit width, and activations of 2 to 8 bits are clamped to their range on the
8 bit streams, see [014](../specs/014-qonnx.md).
