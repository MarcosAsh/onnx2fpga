# Getting started

Status: implemented

Install from a checkout and ask the tool what your machine can do:

```
pip install .
onnx2fpga doctor
```

Compiling needs only Python 3.9 and numpy. Simulating needs Verilator 5 and a
C++17 compiler. Building for the Icepi Zero needs yosys, nextpnr-ecp5, ecppack
and openFPGALoader, which all come in the OSS CAD Suite, plus the OCaml switch
with Hardcaml, because yosys cannot read our SystemVerilog library. `doctor`
tells you which of these it found.

## A first accelerator

We ship a small MLP, 32 inputs to 10 classes, in `examples/models`.

```python
import numpy as np
import onnx2fpga as otf

acc = otf.load("examples/models/mlp.onnx")
acc.compile(device="icepi25")
acc.summary()
```

The summary has one row per hardware unit. Each row shows how the unit was
folded, how many cycles it spends per frame, and what it costs. On the Icepi
Zero the second layer comes out at 256 cycles and is marked as the bottleneck,
because making it faster would need 30 multipliers and the chip has 28.

Before any hardware gets involved, check what int8 did to the model:

```python
x = np.random.default_rng(0).standard_normal((300, 32)).astype(np.float32)
acc.evaluate(x)          # {'agreement': 0.977}
```

So the integer model picks the same class as the float one 97.7 percent of the
time on random inputs. With real data and labels you also get accuracy.

Then run the generated hardware in Verilator. Every output is compared bit for
bit against the integer model:

```python
acc.simulate(x[:4]).ok   # True
```

## On the board

Plug the Icepi Zero in. One USB cable is enough, because the chip that programs
the FPGA also carries the serial link.

```python
acc.build(board="icepi_zero")   # yosys, nextpnr-ecp5, ecppack; about 30 s
acc.deploy()                    # loads it into SRAM, opens the serial port
y = acc.predict(x)              # now this runs on the board
acc.disconnect()
```

`deploy()` only writes SRAM. Power cycle the board and whatever was in its
flash comes back.

From the shell it is three commands:

```
onnx2fpga bitstream examples/models/mlp.onnx --board icepi_zero
make -C build/mlp_icepi_zero/icepi_zero prog
python3 hardware/boards/icepi_zero/host.py build/mlp_icepi_zero
```

## What to expect

We measured this MLP on our board on 24 October 2026. It ran at 3112 frames
a second, and 300 random inputs came back identical to the integer model. The
accelerator itself could do 78125 frames a second at its 20 MHz clock, so the 1
Mbaud serial link is the bottleneck by a factor of 25. The design used about
5300 of the chip's 24288 LUTs and 26 of its 28 multipliers.

Those numbers are the starting line. The serial link only matters for this
demo. A trading design takes its inputs straight from the network, and what
counts there is cycles from input to answer, which is what specs
[001](specs/001-fixed-latency.md) and [002](specs/002-pipelined-matvec.md) are
about.
