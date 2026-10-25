# onnx2fpga

onnx2fpga compiles a machine learning model into FPGA hardware. You hand it an
ONNX file and it gives back a design where every layer is its own hardware unit,
checked bit for bit against an integer model of the same network, and a
bitstream that runs it on a real board.

We are building it toward trading. The goal is models that answer in a fixed
number of cycles, weights a host can swap without a rebuild, gradient boosted
trees, and a market data demo on 10G Ethernet. The hardware is written in
Hardcaml, Jane Street's OCaml hardware language, and kept cycle for cycle
identical to a SystemVerilog twin.

What runs today: the Python API below, both backends, a fixed latency mode that
answers the example MLP exactly 9 cycles after its input and a 100 tree XGBoost
model 6 cycles after its input, a small MLP on an Icepi Zero that came back bit
exact on 300 random inputs, and, in simulation, a tick to trade path from ITCH
packets to orders that decides 14 cycles after each message's last byte. Pages
marked "planned" are specs for work not built yet. We write the page first and
then make the code match it.

```python
import onnx2fpga as otf

acc = otf.load("examples/models/mlp.onnx")
acc.compile(device="icepi25")
acc.summary()
acc.build(board="icepi_zero")
acc.deploy()
acc.predict(x)
```

```{toctree}
:caption: Install
:maxdepth: 1

getting-started
```

```{toctree}
:caption: Usage
:maxdepth: 1

guides/workflow
guides/folding
guides/quantization
guides/boards
guides/backends
guides/custom-layers
guides/tick-to-trade
```

```{toctree}
:caption: Python API Reference
:maxdepth: 1

api/python
api/reference
api/errors
api/cli
```

```{toctree}
:caption: Hardware
:maxdepth: 1

hardware/streams
hardware/requant
hardware/sliding-window
```

```{toctree}
:caption: Specs
:maxdepth: 1

specs/001-fixed-latency
specs/002-pipelined-matvec
specs/003-reloadable-weights
specs/004-weight-specialised-arithmetic
specs/005-tree-ensembles
specs/006-kv260
specs/007-tick-to-trade
specs/008-weights-in-block-ram
specs/009-operator-coverage
specs/010-lookup-tables-and-elementwise
specs/011-reductions-and-softmax
```

```{toctree}
:caption: Further Reading
:maxdepth: 1

contributing
```
