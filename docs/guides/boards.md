# Boards

## Icepi Zero

Status: implemented

The Icepi Zero carries a Lattice ECP5 LFE5U-25F, and we build for it with a
fully open toolchain: yosys, nextpnr-ecp5, ecppack and openFPGALoader. The
device is called `icepi25` and the board `icepi_zero`.

| | |
|---|---|
| LUTs | 24288 |
| Multipliers | 28, each 18x18 |
| Block RAM | 56 blocks of 18 kbit |
| Clock | 20 MHz, from the 50 MHz oscillator through the PLL |
| Host link | UART over the board's own USB port, 1 Mbaud |

The FT231X that programs the FPGA has its serial lines wired to it as well. So
one cable programs the chip and carries the data, and there is nothing to wire
up. If you want the serial pair on the header instead, for a USB to serial
adapter or a Pico, build with `uart="header"` and use pins 8 (TX) and 10 (RX).

`build()` also takes `rev` for the revision printed on the silkscreen (v1.4 by
default, and v1.3 has the same pins), `mhz`, `baud` and `seeds`.

Button 0 resets the design. LED 0 blinks as a heartbeat. LED 1 lights if a
serial frame arrived broken or a buffer filled up.

## Arty A7

The `arty_a7_35t` and `arty_a7_100t` flows use Vivado, the 100 MHz board
oscillator and USB UART at 1 Mbaud. Use devices `a7_35t` and `a7_100t`,
respectively. Frozen-weight fixed and streaming models are supported.
Physical board operation remains unverified.

```python
acc.compile(device="a7_35t", mode="fixed", calibration=x_train[:64])
acc.build(board="arty_a7_35t")
acc.deploy(port="/dev/ttyUSB1")  # use your board's UART port
```

Builds check setup, hold, pulse-width and internal constraints before writing
a bitstream. The serial bridge buffers one fixed-mode inference at a time;
its throughput differs from the accelerator's native throughput. Button 0
resets the design, and a serial break resets framing. See the
[Arty board guide](https://github.com/MarcosAsh/onnx2fpga/blob/main/hardware/boards/arty_a7/README.md)
for build requirements and the serial protocol.

## KV260 and KR260

Status: partial, see [006](../specs/006-kv260.md). A small KR260 inference
design passes full implementation in Vivado 2026.1 and produces a bitstream.
Physical board operation, host access and I/O remain untested.

Both kits carry the Kria K26 SOM, a Zynq UltraScale+ part, and share the device
`k26`. The KR260 has one SFP+ cage, which the 10G demo of
[007](../specs/007-tick-to-trade.md) needs. The KV260 has no SFP+ and no 10G
port at all.

| | |
|---|---|
| Part | `xck26-sfvc784-2LV-c` |
| LUTs | 117120 |
| DSP48E2 | 1248, two int8 products each |
| Block RAM | 144 of 36 kbit, plus 64 UltraRAMs |
| Clock | 322.265625 MHz from an MMCM on the 100 MHz `pl_clk0` |
| Host link | AXI4-Lite from the ARM cores, 64 KiB at 0xA0000000 |

These boards take fixed latency designs only:

```python
acc.compile(device="k26", mode="fixed", calibration=x_train[:64])
acc.build(board="kr260")          # or "kv260"
acc.summary()
```

`build` writes a Vivado project to `build/<model>_fixed/kr260/` and runs
`vivado -mode batch -source build.tcl` there if `vivado` is on PATH. Without it,
you get a `ToolchainError` naming the directory, which you can copy to a machine
with Vivado and run the same command in. The run leaves `kr260.bit`, `kr260.bin`,
`kr260.xsa`, `timing_summary.rpt` and `utilization.rpt`, and from then on
`summary()` prints Vivado's fmax beside the 322 MHz estimate. `build` takes
`jobs` for the number of parallel Vivado jobs (8 by default).

The register map is in `manifest.json` and in the spec. On the board,
`runtime/host/kria_run.cpp` runs the golden vectors through it as root and
prints the cycles the hardware counted between start and answer.

A design compiled with `weights="reloadable"` puts its weight banks behind the
same registers, at 0x8000, with a commit register, a version counter and a
CRC of each bank at 0x0020 to 0x0034 (see
[003](../specs/003-reloadable-weights.md)). `load_weights` speaks that
protocol to anything with `read` and `write`, and the tests drive it against
the generated `kria_regs` under Verilator. `deploy(registers=regs, program=False)`
opens a prediction session over that transport and checks the shell identity,
widths and reset support. `reset()` discards pending inference while preserving
weight banks. `deploy(host="user@kria.local", program=False)` uses the native
`kria_regs` helper over SSH; see the [API reference](../api/reference.md).
The helper and physical MMIO still need validation on a board. The example MLP does not
fit reloadable on the K26, since every weight then needs a multiplier and the
estimate is 1706 DSPs; the 16-32-8 MLP of the tests needs 424 DSPs and 24096
LUTs, 34 and 21 percent.

On the KR260 the project also carries `sfp.xdc`, disabled, with the SFP+ pins
and the 156.25 MHz reference clock reserved for the 10G MAC.

The example MLP fits the K26 unrolled only with one layer on multipliers. On
multipliers alone it needs 1706 DSPs on the estimate, and with constant
arithmetic everywhere 102618 LUTs, both past the 80 percent limit. With
`acc.layers["fc1"].arithmetic("multiplier")` it needs 86932 LUTs and 602 DSPs.

## Reproduce the KR260 build

From a checkout with Hardcaml, Verilator and Vivado available:

```sh
python3 tools/build_kr260.py
```

This creates a seeded 4-8-1 MLP, compiles it for 322.265625 MHz, checks 16
vectors in Verilator, and runs the complete PS, SmartConnect and inference
shell through Vivado. `--generate-only` performs simulation and writes the
project without starting Vivado. `--out` selects the build directory.
The default is `build/kr260_smoke`.

To build your own model, pass `--model model.onnx --samples samples.npy`.
The sample array contains a batch of model inputs. The compiler uses it for
calibration and checks up to 16 samples in simulation before starting
Vivado. Supply both options together and use `--out` to keep separate builds.

The full build on `xck26-sfvc784-2LV-c` uses 1530 LUTs, 1760 flip-flops and
9 DSPs, with no block RAM. Its routed clock is 322.266 MHz, with 0.561 ns
setup slack, 0.013 ns hold slack and 0.052 ns pulse-width slack. These are
Vivado results for the small inference example, not the tick-to-trade design
or measurements from a board. The [measurement record](../measurements/kr260-mlp-2026-11-11.json)
includes the model seed and hashes of the reports and build artifacts.

The 100-tree, depth-4 XGBoost regressor also passes the full build. Its
6-cycle pipeline uses 6489 LUTs and 4609 flip-flops, with no DSPs or block
RAM. At the same clock, setup slack is 0.473 ns, hold slack is 0.024 ns and
pulse-width slack is 0.052 ns. See the
[tree measurement record](../measurements/kr260-trees-2026-11-11.json).
To reproduce it:

```sh
python3 - <<'PY'
import numpy as np
with np.load('tests/fixtures/trees_expected.npz') as data:
    np.save('/tmp/onnx2fpga-tree-samples.npy', data['x'])
PY
python3 tools/build_kr260.py --model tests/fixtures/xgb_regressor.onnx \
  --samples /tmp/onnx2fpga-tree-samples.npy --out build/kr260_tree_validated
```

The PS uses IOPLL for nominal 100 MHz `pl_clk0`. Vivado derives `otf_clk`
through the MMCM, preserving the clock relationship and jitter. The build
rejects an independent clock, the wrong period, unconstrained internal paths,
critical methodology violations or failing setup, hold or pulse-width timing.
A failed rebuild removes the previous exported bitstream and handoff.

The output directory contains `kr260.bit`, `kr260.bin`, `kr260.xsa`,
`timing_summary.rpt`, `utilization.rpt` and `methodology.rpt`. The generated
`model.onnx`, simulation vectors and manifest remain in the parent directory.
The host runtime and Linux overlay still need validation on a board; the
bitstream alone does not establish that Linux can access the accelerator.

### Prepare a Linux firmware package

After a successful build, install `device-tree-compiler` on the build machine and run:

```sh
python3 tools/package_kria.py build/kr260_runtime_reset build/kr260_firmware
```

Use the output directory of your own build as the first argument. `--board kv260`
selects a KV260 build; `--name my_model` changes the firmware application name.
The destination must not already exist.

The package contains `onnx2fpga.bit.bin`, `onnx2fpga.dtbo`, `shell.json`, the
overlay source, the model manifest, the successful build record and a hash record.
`board.run()` writes `build.json` only after Vivado succeeds, tying the generated
sources and manifest to the bitstream, XSA and reports. A failed rebuild removes
that record, so its older output files cannot be packaged. Source changes during
the build and artifact changes before packaging are rejected. Builds predating
this record need another `board.run()` before packaging.

Packaging also checks that the XSA
contains the exported bitstream and that the binary is its payload. Register
addresses, PS clock rate and AXI interface widths come from the XSA. The overlay
enables PL0 through `xlnx,fclk` and configures the PS interfaces through
`xlnx,afi-fpga`.

This targets AMD Linux images with FPGA-manager overlays, `xmutil`/`dfx-mgr`,
both drivers above, and device-tree symbols `fpga_full`, `amba`, `zynqmp_clk` and
`zynqmp_reset`. Check those requirements on the intended image before loading.
The package follows AMD's [Vivado deployment flow](https://xilinx.github.io/kria-apps-docs/creating_applications/2022.1/build/html/docs/vivado_accel_example.html)
and derives interface settings using the mappings in AMD's
[device-tree generator](https://github.com/Xilinx/system-device-tree-xlnx/tree/xlnx_rel_v2025.2/device_tree/data).

Copy the package and `tools/install_kria.py` to a matching board image. On the
board, install with:

```sh
sudo python3 install_kria.py kr260_firmware --board kr260
```

The installer checks file hashes and the build record, then installs the
application under `/lib/firmware/xilinx/<name>`. It refuses an existing
application directory. The selected board must match the package; the tool
does not detect the carrier automatically.

Add `--program` to install and load in one command. Before installation, this
checks the four device-tree symbols and runs `xmutil listapps`. After installation,
it runs `xmutil unloadapp` followed by `xmutil loadapp <name>`, stopping on failure.
Unloading stops the current PL application; stop all accelerator clients first.
If loading fails, the installed package remains available for diagnosis and a
manual `xmutil loadapp <name>` retry. Preflight does not establish driver or
firmware compatibility.

Install the native `kria_regs` helper before connecting with
`deploy(host=..., program=False)`. The `deploy()` API does not install or program
Kria firmware. The trading demo's
[Linux setup](../specs/007-tick-to-trade.md#provision-linux-for-replay) includes
the helper installation and persistent DMA-buffer provisioning commands.

Package tests compile and apply the overlay to a test device tree and reject
mismatched artifacts. Installer tests use temporary directories and a mocked
`xmutil`. They do not validate the board's Linux image, firmware, clock drivers
or physical MMIO.

## Adding a board

A board is a class in `compiler/onnx2fpga/board.py` and a directory under
`hardware/boards/` with its constraints and build scripts. The
board top wraps the generated accelerator with a clock, a reset and a link to
the host. The Icepi Zero support is about a thousand lines including its UART
and tests, and it is the one to copy.
