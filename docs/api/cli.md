# Command line

Status: implemented

The command line is a thin layer over the Python API.

| Command | Does |
|---|---|
| `onnx2fpga doctor` | Reports what this machine can compile, simulate, build and load. |
| `onnx2fpga inspect MODEL` | Prints what the compiler sees in an ONNX file. |
| `onnx2fpga check MODEL [--device D] [--mode stream\|fixed] [--clock-mhz MHz] [--json]` | Reports import compatibility without generating hardware. |
| `onnx2fpga compile MODEL --out DIR --device D [--target-cycles N] [--hdl H]` | `load` then `compile`. |
| `onnx2fpga simulate MODEL [--sweep] [--hdl H]` | Compiles and checks the RTL in Verilator. |
| `onnx2fpga synth MODEL --device D [--period NS] [--impl] [--dry-run]` | Measures real cost: Vivado for AMD parts such as `k26`, yosys and nextpnr for ECP5. `--period` defaults to the device's clock. |
| `onnx2fpga bitstream MODEL --board icepi_zero` | Compiles and builds a bitstream. |
| `onnx2fpga devices` | Lists device models and their budgets. |

Common options: `--samples` and `--seed` for random calibration, `--quiet`.

Exit status: 0 success, 1 a check failed, 2 a missing tool or bad usage.

`check` collects independent import failures and marks nodes downstream of a
failure as unassessed. Exit status 0 means ready to attempt compilation;
errors or unknown findings return 1. Warnings alone return 0. The JSON output
contains `ok`, `device`, `mode`, `findings` and `unchecked`. Findings identify
the original node index and name, operator domain and declared opset when
available. Indices start at zero.

This check uses the built-in importer. It does not run full ONNX schema/opset
validation, calibration, hardware lowering, synthesis or physical board tests.
Resource fit, numerical accuracy and routed timing remain separate checks.
