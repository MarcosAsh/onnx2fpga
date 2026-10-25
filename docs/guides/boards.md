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

## KV260 and KR260

Status: partial, see [006](../specs/006-kv260.md). The projects are generated
and checked against a Vivado stub, but have not been through real Vivado or
run on a board.

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
the generated `kria_regs` under Verilator. `deploy(host=...)` still waits for a
Kria runtime that can hand it the board's registers. The example MLP does not
fit reloadable on the K26, since every weight then needs a multiplier and the
estimate is 1706 DSPs; the 16-32-8 MLP of the tests needs 424 DSPs and 24096
LUTs, 34 and 21 percent.

On the KR260 the project also carries `sfp.xdc`, disabled, with the SFP+ pins
and the 156.25 MHz reference clock reserved for the 10G MAC.

The example MLP fits the K26 unrolled only with one layer on multipliers. On
multipliers alone it needs 1706 DSPs on the estimate, and with constant
arithmetic everywhere 102618 LUTs, both past the 80 percent limit. With
`acc.layers["fc1"].arithmetic("multiplier")` it needs 86932 LUTs and 602 DSPs.

## Adding a board

A board is a class in `compiler/onnx2fpga/board.py` and a directory under
`hardware/boards/` with its constraints and build scripts. The
board top wraps the generated accelerator with a clock, a reset and a link to
the host. The Icepi Zero support is about a thousand lines including its UART
and tests, and it is the one to copy.
