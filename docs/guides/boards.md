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

## KV260

Status: planned, see [006](../specs/006-kv260.md).

## Adding a board

A board is a class in `compiler/onnx2fpga/board.py` and a directory under
`hardware/boards/` with its constraints, a Makefile and a host script. The
board top wraps the generated accelerator with a clock, a reset and a link to
the host. The Icepi Zero support is about a thousand lines including its UART
and tests, and it is the one to copy.
