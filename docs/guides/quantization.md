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

## Not built yet

`quantize(activations=...)` raises `ConfigError` today. Every stream in the
pipeline is one width, and giving a layer its own would need converters on
both sides that the compiler does not insert yet. Importing models trained with
quantization in QONNX, Brevitas or HGQ is not planned right now.
