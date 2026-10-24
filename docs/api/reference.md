# Python API

Status: implemented unless marked otherwise

```python
import onnx2fpga as otf
```

## Loading

### `otf.load(model, custom_objects=None, name=None)`

Reads an ONNX file (a path or an `OnnxModel`) and returns an `Accelerator`. The
model is imported straight away, so an unsupported operator raises
`UnsupportedOpError` here, naming the operator, the node and the supported list.

`custom_objects` maps an ONNX op type to a `HardwareLayer` subclass.

### `otf.from_keras(model, path=None, custom_objects=None)`

Exports a Keras 3 model to ONNX and loads it. Keras is imported only when this
is called. Status: partial and no longer a priority; it exists but is untested.

## `Accelerator`

### `compile(device=None, target=None, target_fps=None, target_cycles=None, hdl=None, calibration=None, utilisation=None, out_dir=None, strict=True, verbose=False, mult_bits=None, top=None, mode=None)`

Quantizes, folds and emits the design.

| Argument | Default | Meaning |
|---|---|---|
| `device` | `vu9p` | Device model to budget against. `onnx2fpga devices` lists them. See Open questions. |
| `mode` | `"stream"` | `"stream"` folds layers onto handshaked streams. `"fixed"` unrolls every layer so the design takes a vector every cycle and answers a fixed number of cycles later, see [001](../specs/001-fixed-latency.md). |
| `target` | `"fit"` | `"fit"` is the fastest design that fits. |
| `target_fps`, `target_cycles` | none | A throughput to balance to and stop at. |
| `hdl` | see [Backends](../guides/backends.md) | `"hardcaml"` or `"sv"`. |
| `calibration` | random, with a warning | A batch of representative inputs for a float model. |
| `utilisation` | 0.8 | LUT ceiling as a fraction of the device. |
| `out_dir` | `build/<model>` | Where the build directory goes. |
| `strict` | `True` | Raise `BudgetError` when the target cannot fit, instead of keeping the nearest design. |

Arguments left out keep their value from the previous call.

With `mode="fixed"` there is nothing to fold, so `target`, `target_fps`,
`target_cycles` and any `Layer.fold` pin raise `ConfigError`. The backend is
Hardcaml, and `hdl="sv"` raises `ConfigError` because the fixed latency units
have no SystemVerilog version. A model with a layer fixed mode cannot build yet,
such as a `Conv`, raises `ConfigError` naming it, and one too large to unroll on
the device raises `BudgetError` naming its largest layer. The default build
directory is `build/<model>_fixed`.

### `summary(print_fn=None)`

Before compile: layers, output shapes, parameter counts and any pins. After:
one row per unit with folding, cycles, LUT, DSP and block RAM, the bottleneck
marked, totals against the budget, throughput, quantization and build path.
Shapes are channel last. A fixed latency design prints one row per layer with
its arithmetic, multipliers and latency in cycles instead, then the total
latency in cycles and in nanoseconds at the device's clock, and how many
multipliers constant arithmetic saved.

### `evaluate(x, y=None)`

Returns a dict. Always `agreement`; with labels also `accuracy` (integer model)
and `float_accuracy`.

### `predict(x, on=None, quantized=False)`

Runs a batch. `on=None` means the board after `deploy`, otherwise the integer
reference. `on="reference"` or `on="board"` forces one. Returns dequantized
floats, or integer codes with `quantized=True`.

### `simulate(x=None, sweep=False)`

Builds the RTL with Verilator and checks it against the integer model. With `x`,
those inputs are used; without, the build's own golden vectors. `sweep=True`
repeats under randomised backpressure. Returns a `SimulationReport` with `ok`,
`cycles`, `latency` and `sweep`.

In fixed mode the harness drives one vector on every cycle, 10000 random ones
when `x` is left out, and fails unless every answer arrives exactly
`acc.latency` cycles after its input and matches the integer model.
`sweep=True` repeats with `features_valid` high on 50 and 10 percent of cycles.

### `build(board="icepi_zero", verbose=False, **options)`

Builds a bitstream and returns its path. Compiles for the board's device first
if nothing is compiled. Options are board specific, see
[Boards](../guides/boards.md). A fixed latency design raises `ConfigError`,
since the board top streams frames over a serial port.

### `deploy(port=None, program=True, timeout=2.0)`

Loads the bitstream into SRAM and opens the serial port. `program=False` only
opens the port, for a board already loaded. `port` defaults to the board's USB
serial device.

### `disconnect()`

Closes the serial port. `predict` goes back to the reference.

### Properties

`device`, `hdl`, `mode`, `out_dir`, `deployed`, `input_name`, `input_shape`,
`layers`.

`latency` is the number of cycles from `features_valid` to `prediction_valid`
in a fixed latency design, 9 for the example MLP. It is `None` before a compile
and for a streaming design, whose latency depends on what came before.

## Layers

`acc.layers` is a mapping from layer name to `Layer`.

### `Layer.fold(**factors)`

Pins folding factors, for example `fold(simd=4, pe=2)`. `fold()` with none
removes the pin. An invalid value raises `ConfigError` at the next `compile`,
listing the valid ones.

### `Layer.quantize(weights=None, activations=None)`

`weights="int2"` to `"int8"`. `activations` is planned and raises
`ConfigError` today.

### `Layer.arithmetic(kind="auto")`

How a fixed latency design multiplies by this layer's weights.
`"multiplier"` gives every weight its own multiplier. `"constant"` turns each
product into logic for that weight: nothing for a zero, an add or subtract for
plus or minus one, a shift for a power of two and a few shifts and adds for
anything else. The outputs and the latency in cycles are the same either way,
see [004](../specs/004-weight-specialised-arithmetic.md). `"auto"`, the
default, takes constant arithmetic unless the layer's estimate on the device
then fills its busiest resource further than with multipliers. A layer without
weights, or a value other than these three, raises `ConfigError`, and so does
compiling with `mode="stream"` while a layer is set to `"constant"`.

### `Layer.folding`, `Layer.cycles`, `Layer.latency`, `Layer.units`

What the last compile chose for this layer. `latency` is the cycles the layer
adds in a fixed latency design and `None` otherwise.

## `otf.HardwareLayer`

See [Custom hardware layers](../guides/custom-layers.md).

## Environment variables

| Variable | Effect |
|---|---|
| `ONNX2FPGA_HDL` | Default backend, `hardcaml` or `sv`. |
| `OTF_HARDCAML` | Path to a checkout's `hardware/hardcaml`, for an installed copy. |

## Open questions

- The default device is `vu9p`, a datacentre part almost nobody has on a desk.
  Should it be `icepi25`, or should `compile` require `device=`?
- `predict` needs `compile` first even on the reference model. Should `load`
  alone allow float `predict`, like an uncompiled Keras model?
