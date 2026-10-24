# The workflow

Status: implemented

If you have used Keras, this will feel familiar. There is one object, a handful
of verbs, and the defaults work. You add control only when you want it.

```python
acc = otf.load("model.onnx")
acc.compile(device="icepi25", calibration=x_train[:64])
acc.summary()
acc.evaluate(x_test, y_test)
acc.simulate(x_test[:4])
acc.build(board="icepi_zero")
acc.deploy()
acc.predict(x_test)
```

`load` reads and imports the ONNX file right away. That matters, because an
operator we do not support fails here, by name, instead of halfway through a
compile. `compile` quantizes the model, folds every layer to fit the device,
and writes a build directory. `summary` prints what it chose.

The next two steps are about trust. `evaluate` runs the integer model that the
hardware reproduces and compares it with the float model. `simulate` builds the
generated RTL in Verilator and checks every output beat against that same
integer model. If both look right, `build` turns the design into a bitstream
and `deploy` loads it and opens the serial port. `predict` runs on the integer
model until you deploy, and on the board after.

Everything can be repeated. Call `compile()` with no arguments and it rebuilds
with the settings from last time, which makes tuning a layer a short loop:

```python
acc.layers["conv0"].fold(pe=4)
acc.compile()
acc.summary()
```

## More control

Each layer can be pinned before a compile. Pins are checked when you compile,
and a bad one tells you the values that would work.

```python
acc.layers["conv0"].fold(simd=4, pe=2)       # pin both factors
acc.layers["conv0"].fold(pe=4)               # pin one, the compiler picks the other
acc.layers["conv0"].fold()                   # hand it back to the compiler
acc.layers["fc1"].quantize(weights="int4")   # narrower weights for this layer
```

The compile itself takes a speed target, a backend and a strictness flag:

```python
acc.compile(device="icepi25", target_cycles=400)   # stop at this speed
acc.compile(device="icepi25", target_fps=10_000)
acc.compile(hdl="sv")
acc.compile(strict=False)   # keep the nearest design if the target cannot fit
```

And if a model needs an operator we lack, you can write the unit yourself. See
[Custom hardware layers](custom-layers.md).

## Fixed latency

A trading path cannot wait on a FIFO. `mode="fixed"` builds the same model as a
pipeline that takes a new input on every cycle and answers a fixed number of
cycles later, with no ready signal anywhere:

```python
acc = otf.load("examples/models/mlp.onnx")
acc.compile(device="vu9p", mode="fixed", calibration=x_train[:64])
acc.latency          # 9
acc.summary()
acc.simulate()       # 10000 random vectors, one per cycle
```

Each dense layer costs three cycles: one for the products, one for the sum and
one for the rescale. The example MLP has three layers, so it answers 9 cycles
after its input, 30 ns at the 300 MHz the vu9p model assumes. `simulate`
checks every one of the 10000 answers lands on exactly that cycle and matches
the integer model. The summary lists each layer's arithmetic, multipliers and
latency:

```
 Layer (kind)                    In -> out  Arithmetic Multipliers  Latency      LUT    DSP
===========================================================================================
 fc0 (Gemm+Relu)                  32 -> 64    constant           0        3    59364     64
 fc1 (Gemm+Relu)                  64 -> 16    constant           0        3    37190     16
 fc2 (Gemm)                       16 -> 10    constant           0        3     6064     10
===========================================================================================
 Total                                                           0        9   102618     90
```

The weights are fixed when the hardware is generated, so by default each
product becomes logic for its own weight instead of a multiplier: a zero weight
costs nothing, a power of two is a shift, and anything else is a few shifts and
adds. The answers and the 9 cycles do not change. The 90 DSPs left are the
rescale, one per output channel. A layer can be told which to use:

```python
acc.layers["fc1"].arithmetic("multiplier")   # one multiplier per weight
acc.layers["fc1"].arithmetic("constant")     # shifts and adds
acc.layers["fc1"].arithmetic("auto")         # the default
```

See [004](../specs/004-weight-specialised-arithmetic.md) for what this costs
in LUTs and how it is checked.

Fixed mode builds Gemm and MatMul layers with a fused Relu, and treats Flatten
and Reshape as wiring. Anything else, a `Conv` for instance, raises
`ConfigError` for now. It needs the Hardcaml backend, and the bitstream flow
for the Icepi Zero does not take it yet. See
[001](../specs/001-fixed-latency.md) for the design.

## What predict gives back

`predict(x)` takes a batch in the model's own layout and returns floats, so it
drops into code that was written for the float model. Pass `quantized=True` to
get the raw integer codes, which is exactly what came out of the hardware.
