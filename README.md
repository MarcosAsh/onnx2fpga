# onnx2fpga

Compiles small ONNX models (linear models, MLPs, CNNs, gradient boosted trees)
into fixed-latency FPGA pipelines, in Hardcaml or SystemVerilog.

Documentation: https://marcosash.github.io/onnx2fpga/

```python
import onnx2fpga as otf

acc = otf.load("model.onnx")
acc.compile(device="k26", mode="fixed", clock_mhz=322, calibration=x_train[:64])
acc.summary()
acc.evaluate(x_test, y_test)
acc.simulate(x_test[:4])
acc.build(board="kr260")
```

Some numbers from simulation:

| design | latency |
|---|---|
| MLP 32-64-16-10, int8, fixed mode | 9 cycles, one input per cycle |
| 100-tree XGBoost ensemble | 6 cycles, 18.6 ns |
| K26 ITCH message to OUCH order, tick-to-trade stage A | 22 cycles |

The K26 tick-to-trade pipeline has a 279.72 MHz out-of-context routed estimate
in Vivado 2026.1. It does not yet meet the 322.265625 MHz target; see the
[timing measurements](docs/guides/tick-to-trade.md).

The example MLP also runs on an Icepi Zero (Lattice ECP5) built with yosys and
nextpnr, bit exact against the integer reference. K26 model-only
[Vivado measurements](docs/guides/k26-measurements.md) cover the 22-model
corpus. The full Kria board builds have not been validated yet.

## Running it

Needs Python 3.10+, numpy, Verilator 5 and a C++17 compiler. Hardcaml needs the
OxCaml switch described in `hardware/hardcaml`.

```
./onnx2fpga doctor
./onnx2fpga simulate examples/models/cnn.onnx --sweep
make test
```

License: Apache-2.0.
