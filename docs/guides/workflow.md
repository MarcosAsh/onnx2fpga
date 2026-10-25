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
 Layer (kind)                    In -> out  Arithmetic Multipliers  Latency   Stages   Fmax      LUT    DSP
===========================================================================================================
 fc0 (Gemm+Relu)                  32 -> 64    constant           0        3    0/0/0    180    59364     64
 fc1 (Gemm+Relu)                  64 -> 16    constant           0        3    0/0/0    162    37190     16
 fc2 (Gemm)                       16 -> 10    constant           0        3    0/0/0    205     6064     10
===========================================================================================================
 Total                                                           0        9             162   102618     90
```

The Fmax column is what the device's delay model expects each layer to reach.
Three stages a layer put a multiply, an adder tree or a whole rescale in one
cycle, which the vu9p model puts at 162 MHz. To run faster, ask for a clock:

```python
acc.compile(device="k26", mode="fixed", clock_mhz=322.265625, utilisation=0.9)
acc.latency          # 22
```

The compiler adds the fewest register stages each layer needs: after the
multiplies, inside the adder tree and inside the rescale, shown as the Stages
column in that order. On the `k26` the example MLP takes 22 cycles, 68.3 ns at
322.265625 MHz, and the model predicts 343 MHz. The `k26` model takes its DSP
figures from AMD's datasheet and assumes the fabric ones, so it waits for a
Vivado run; the `icepi25` model is fitted to nextpnr. A clock no pipelining
reaches raises `BudgetError` naming the slowest path. See
[002](../specs/002-pipelined-matvec.md).

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

## Reloading weights

A retrained model should not need a Vivado rebuild. With
`weights="reloadable"` every dense layer keeps its weights, biases and rescale
multipliers in two banks of registers. A host writes the idle bank while the
design keeps answering, then commits, and every inference whose features
arrive after the commit uses the new bank from start to finish:

```python
acc.compile(device="vu9p", mode="fixed", weights="reloadable",
            calibration=x_train[:64])
acc.simulate(reload="retrained.onnx", commits=[5000])

from onnx2fpga.simulate import SimulatedRegisters
with SimulatedRegisters(acc.out_dir) as registers:
    acc.load_weights("retrained.onnx", registers=registers)
acc.predict(x)       # now the retrained model
```

The latency stays 9 cycles. The price is that every weight gets a general
multiplier, zeros included, since a reload can change any of them, so constant
arithmetic does not apply: the example MLP needs 3322 DSP48E2 in yosys against
90 with frozen constant arithmetic. `simulate` writes the new model into the
idle bank while 10000 inputs stream past and checks that every answer carries
the bank that was active when its features arrived and matches that bank's
integer model.

`load_weights` refuses a model with another architecture or one that does not
fit the compiled grid, naming the first difference. It writes the bank through
the AXI4-Lite registers of the Kria shell, checks a CRC-32 before and after the
commit, and checks the version counter. `SimulatedRegisters` is that register
file under Verilator; on a board it waits for the Kria runtime. See
[003](../specs/003-reloadable-weights.md) for the register map and what stays
fixed between reloads.

## Tree ensembles

Gradient boosted trees exported by onnxmltools or skl2onnx compile the same
way, in fixed mode only. There is nothing to calibrate, because each feature's
grid comes from the thresholds the trees test it against:

```python
acc = otf.load("tests/fixtures/xgb_regressor.onnx")   # 100 trees of depth 4
acc.compile(device="vu9p", mode="fixed")
acc.latency          # 6
acc.evaluate(x)      # {'mean_abs_error': ...} for a regressor, label agreement for a classifier
acc.simulate()
```

Every comparison happens in the first cycle, each tree picks its leaf in the
second, and an adder tree of fan in 4 needs four more for 100 trees and the
base value. A binary classifier outputs one bit, the label, decided on the
margin. Features enter as 16 bit codes on per feature grids, which
`manifest.json` lists as a scale and zero point per feature. See
[005](../specs/005-tree-ensembles.md) for the design and what is not supported.

## What predict gives back

`predict(x)` takes a batch in the model's own layout and returns floats, so it
drops into code that was written for the float model. Pass `quantized=True` to
get the raw integer codes, which is exactly what came out of the hardware.
