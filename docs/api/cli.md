# Command line

Status: implemented

The command line is a thin layer over the Python API.

| Command | Does |
|---|---|
| `onnx2fpga doctor` | Reports what this machine can compile, simulate, build and load. |
| `onnx2fpga inspect MODEL` | Prints what the compiler sees in an ONNX file. |
| `onnx2fpga compile MODEL --out DIR --device D [--target-cycles N] [--hdl H]` | `load` then `compile`. |
| `onnx2fpga simulate MODEL [--sweep] [--hdl H]` | Compiles and checks the RTL in Verilator. |
| `onnx2fpga synth MODEL --device D [--period NS] [--impl] [--dry-run]` | Measures real cost: Vivado for AMD parts such as `k26`, yosys and nextpnr for ECP5. `--period` defaults to the device's clock. |
| `onnx2fpga bitstream MODEL --board icepi_zero` | Compiles and builds a bitstream. |
| `onnx2fpga devices` | Lists device models and their budgets. |

Common options: `--samples` and `--seed` for random calibration, `--quiet`.

Exit status: 0 success, 1 a check failed, 2 a missing tool or bad usage.
