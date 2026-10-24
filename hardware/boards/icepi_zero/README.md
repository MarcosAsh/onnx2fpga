# Icepi Zero

A compiled model on an [Icepi Zero](https://github.com/cheyao/icepi-zero) (Lattice ECP5
LFE5U-25F-6CABGA256, 50 MHz oscillator), driven from a PC over the board's own USB port.
The flow is open source: the Hardcaml generator, yosys, nextpnr-ecp5, ecppack and
openFPGALoader (all in the OSS CAD Suite).

```sh
./onnx2fpga bitstream examples/models/mlp.onnx --board icepi_zero --target-cycles 256
make -C build/mlp_icepi_zero/icepi_zero prog      # load into SRAM over USB
python3 hardware/boards/icepi_zero/host.py build/mlp_icepi_zero
```

`bitstream` compiles with `--hdl hardcaml`, puts the accelerator behind a UART
(`hardware/hardcaml/src/icepi_top.ml`), and runs this directory's `Makefile`, which
sweeps placement seeds until one meets timing and prints fmax and utilisation. Folding
budgets against the `icepi25` device model, so a target the part cannot meet is relaxed
to the fastest one that fits, and the compiler says which resource ran out. `host.py` sends
`golden/input.hex` a thousand times, checks every answer against
`golden/expected.hex` and prints frames per second. It needs pyserial.

`--rev` takes the revision printed on the board (v1.4 by default), `--mhz` the PLL
clock (20 by default, any divisor of 480) and `--baud` the serial rate (1000000).

## Pins

No wiring: the FT231X that openFPGALoader programs through is also a serial port, and
its TXD and RXD go to the FPGA, so the accelerator appears as `/dev/ttyUSB0`
(`/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_*`). Programming borrows the same chip,
so the port vanishes for a moment during `make prog` and comes back.

| Signal   | Ball                     | What                                   |
|----------|--------------------------|----------------------------------------|
| rxd      | K16                      | from the FT231X's TXD                  |
| txd      | K15                      | to the FT231X's RXD                    |
| clk      | M1 (v1.3+), M2 (v1.0-1.2)| 50 MHz oscillator, into the PLL        |
| button_n | C4                       | button 0: hold to reset                |
| led[0]   | E13 (v1.2+)              | blinks while the clock runs            |
| led[1]   | D14 (v1.2+)              | a framing error or a full input buffer |

`--uart header` moves the pair to the 40-pin header where a Raspberry Pi has its UART,
for a 3.3 V USB serial adapter or a Pico instead:

| Header pin | Ball | FPGA | Adapter | Pico (UART0) |
|-----------:|------|------|---------|--------------|
| 8          | P1   | txd  | RX      | GP1 RX       |
| 10         | N1   | rxd  | TX      | GP0 TX       |
| 6          | GND  |      | GND     | GND          |

## The link

8N1 at the chosen baud. A break (the line held low for at least twenty bit times)
clears the bridge and the accelerator, so `host.py` sends one before anything else and
the next byte starts a frame. Each beat then travels as `ceil(width / 8)` bytes, least
significant first, in both directions; `manifest.json` gives the widths and the beats
per frame. The board has 2 KiB of buffer each way and cannot hold the line, so the host
keeps no more frames in flight than both buffers hold.

The serial line is the bottleneck: the example MLP takes 32 bytes a frame, so 1 Mbaud
allows about 3100 frames a second, while the accelerator itself manages one every 256
cycles. The clock is 20 MHz because the matrix vector units reach about 25 MHz on this
part (weight table, multiplier and requantiser in one cycle); raising it buys nothing
while the line is slower.

## Without the board

`make sim` here runs the generated `icepi_top.v` in Verilator, with `sim/ehxplll.v`
standing in for the PLL and `sim/uart_bench.cpp` for the host. `tests/test_icepi_board.py`
(part of `make test`) uses it to check the MLP and the CNN bit for bit, with the host's
clock a couple of percent off and gaps between bytes.
