# Arty A7

The `arty_a7_35t` and `arty_a7_100t` boards use the 100 MHz oscillator and
USB UART at 1 Mbaud. They accept frozen-weight fixed or streaming models.
The fixed-mode serial bridge holds one prediction until the UART accepts it.
Serial throughput is limited by the input and output byte counts, separately
from the accelerator's native initiation interval.

```python
import numpy as np
import onnx2fpga as otf

with np.load("tests/fixtures/corpus/mlp_regressor.npz") as data:
    calibration = data["calibration"]
acc = otf.load("tests/fixtures/corpus/mlp_regressor.onnx")
acc.compile(device="a7_35t", mode="fixed", calibration=calibration,
            out_dir="build/arty_mlp")
acc.build(board="arty_a7_35t")
```

Use device `a7_100t` with board `arty_a7_100t` for the larger part. The CLI
also accepts these board names through `onnx2fpga bitstream --board`; its
calibration inputs are random, so use the Python API with representative data
when numerical accuracy matters. Fixed mode is the Arty default; streaming
models need `--mode stream`. Vivado and the Hardcaml toolchain must be on PATH.

The build rejects failed setup, hold, pulse-width, internal constraint and
critical methodology checks before writing its bitstream. Reports and the
routed checkpoint stay in the board's build directory. Physical programming,
serial communication and operation on the board remain unverified.

After connecting the board, load SRAM and open its serial port with
`acc.deploy(port="/dev/ttyUSB1")`; `acc.predict(x)` then uses the board.
Programming requires openFPGALoader, and serial access requires pyserial.
Choose the UART device for your board rather than assuming that example port.
For a board already programmed, use `program=False`.

Button 0 resets the link and accelerator. LED 0 is a heartbeat; LED 1 latches
a serial framing error or input overflow until reset or a serial break.
The host sends a break before each session. Framing matches the
[Icepi link](../icepi_zero/README.md#the-link), with 2 KiB buffers in each direction.

Pin assignments follow Digilent's
[35T](https://github.com/Digilent/digilent-xdc/blob/master/Arty-A7-35-Master.xdc)
and [100T](https://github.com/Digilent/digilent-xdc/blob/master/Arty-A7-100-Master.xdc)
constraints. The selected parts match revision E.0 board definitions.
The asynchronous button and UART input paths end at synchronizers; UART and
LED outputs have no external clock. Those I/O paths are excluded from synchronous
timing analysis. Board I/O validation still requires physical testing.
