# onnx2fpga

Compiles an ONNX model into a streaming dataflow accelerator: one hardware unit
per layer, connected by handshaked streams, with no shared instruction engine
and no round trip to external memory between layers.

Runs on Python, numpy and Verilator. Nothing else is required to build, test or
simulate. The compiler reads and writes `.onnx` files with its own protobuf
decoder rather than depending on the reference `onnx` package.

## What works today

Both example models compile from `.onnx` to SystemVerilog and reproduce the
integer reference model exactly in simulation, under randomised backpressure on
both ends of the pipeline.

| model | shape | bottleneck | measured frame | DSP | LUT | verified |
|---|---|---|---|---|---|---|
| MLP 32-64-16-10 | int8, target 64 cycles | 64 cycles | 169 cycles | 30 | 1240 | 21 backpressure runs, bit exact |
| CNN 12x12 conv3x3x4 + maxpool + fc | int8, target 400 cycles | 400 cycles | 700 cycles | 8 | 961 | 21 backpressure runs, bit exact |
| the same CNN, pushed | int8, target 25 cycles | 300 cycles | 383 cycles | 28 | 1715 | bit exact |

The sliding window generators are swept separately over eighteen geometries
(padding, stride, dilation, pointwise, kernel equal to image, non square, and
each of those again with several kernel columns per beat) at four backpressure
settings each, against the numpy reference.

Cycle counts are measured in simulation. Resource figures are model estimates,
not synthesis results, and the 300 MHz used to derive frames per second is an
assumption. Nothing here has been through synthesis yet; see *Measuring
against synthesis* below for the path to real numbers.

Supported operators: `Conv`, `Gemm`, `MatMul`, `Relu`, `MaxPool`, `Add`,
`Flatten`, `Reshape`, `Identity`, `QuantizeLinear`, `DequantizeLinear`.
Skip connections work: a stream that branches gets a fork unit, and the
buffers on the two paths are sized so the join cannot deadlock.

Quantization comes from one of two places. A model already quantized in the
ONNX QDQ form states its own scales, and the compiler reads them. A float model
is calibrated on samples instead. Activations are per tensor and may be
asymmetric, signed or unsigned; weights are per output channel and symmetric.

On the example MLP the integer pipeline reproduces the QDQ specification, that
is the model evaluated in float between each quantize and dequantize pair,
**exactly** for signed activations and within **one output step** for unsigned
ones over fifty random inputs.

## Quickstart

It is a command line tool. From a checkout, with nothing installed:

```
./onnx2fpga doctor                       what this machine can do
./onnx2fpga inspect examples/models/cnn.onnx
./onnx2fpga simulate examples/models/cnn.onnx --sweep
```

`simulate` compiles the model, builds the generated RTL with Verilator, and
checks every output beat against vectors produced from the compiler's own numpy
model. `--sweep` repeats it under randomised backpressure, which is where
stream designs actually fail.

### The whole path, from nothing

Everything above starts from an `.onnx` that already exists. This starts from
no model at all, trains one, and ends by saying what quantization cost:

```
python3 examples/worked_example.py
```

It makes a small classification problem, fits an 8-16-3 network to it in numpy,
exports that to ONNX, compiles it, and compares the compiled design's answers
against the float model's on held-out data:

```
accuracy on 200 held out samples
  float model     99.5%
  compiled design 99.5%
  same prediction on 100.0% of them
```

That last line is the one worth reading. The error report above it says the
worst output moved by about one percent of the model's own range, and the
predictions did not change at all, which is the usual outcome when the decision
is an argmax: int8 is far more precision than a ranking needs.

It leaves a buildable design behind, so building and simulating it is one more
command:

```
cd build/worked && make run
```

There is no framework involved. Training is thirty lines of numpy, the same
dependency the compiler already has.

Installed, the same commands are on PATH:

```
pip install .
onnx2fpga compile model.onnx --out build/model --device vu9p --target-cycles 400
```

The wheel carries the RTL library, the C++ runtime and the synthesis script, so
an installed copy can build and simulate what it emits without the checkout.

An already quantized model needs no calibration data at all:

```
onnx2fpga compile examples/models/qdq_mlp_uint8.onnx --out build/qdq
```

Commands: `doctor`, `inspect`, `compile`, `simulate`, `synth`, `devices`.

For working on the compiler itself:

```
make test                   # unit, C++ contract, Verilator, packaging
make test-flaky             # the suite again, shuffled, to catch flakiness
make wheel                  # stage assets and build an installable wheel
```

The suite is deterministic: compiling the same model twice produces byte
identical artifacts, and `make test-flaky` reruns everything under shuffled
orderings and varied hash seeds.

## Layout

The compiler is a pipeline and the directory names are the pipeline.

```
compiler/onnx2fpga/         Python. Graph rewriting. Runs once, in milliseconds.
  p1_ingest/                .onnx bytes         -> ONNX protos
    protobuf_codec.py         the wire format
    onnx_schema.py            the subset of onnx.proto3 used here
    onnx_model.py             numpy view over the protos, plus a builder
  p2_graph/                 ONNX protos         -> compiler IR, channel-last
    onnx_importer.py          the translation, one handler class per ONNX op
    graph.py node.py tensor.py datatype.py layout.py
  p3_ops/                   the operator library
    onnx_ops.py               float ops, for calibration and shape inference
    hardware_ops.py           streaming units, one per module in hardware/rtl
  p4_quantize/              float IR            -> integer hardware IR
    calibrate.py plan.py      measure ranges, choose scales
    requantize.py             the fixed point rescale contract
    lower.py                  float ops -> hardware ops
  p5_schedule/              hardware IR         -> a balanced pipeline
    folding.py                how parallel each unit gets
    streams.py                width converters and FIFOs
  p6_emit/                  hardware IR         -> a build directory
    systemverilog.py memories.py stream_io.py project.py
  targets/                  device models: resource budgets, DSP packing rules
  reference/                numpy execution, the golden model

hardware/rtl/               SystemVerilog. Hand written, parameterised, linted.
runtime/                    C++. Everything that runs per inference.
  include/otf/                kernels, manifest, accelerator interface
  sim/harness.cpp             the Verilator driver loop
  kernels/                    bit exact integer reference and its checker
  host/otf_run.cpp            host driver over a compiled design
```

`hardware/` and `runtime/` are data the tool needs at run time, not just
sources. From a checkout they are found beside the package; in a wheel they are
staged inside it. `compiler/onnx2fpga/assets.py` is the only thing that knows
the difference.

### Why Python here and C++ there

The compiler is a graph rewriter. It runs once and finishes in a few
milliseconds, so its speed is irrelevant and its clarity is not. C++ lives
where work happens per inference or per clock edge: the Verilator harness runs
its loop once per cycle for the whole test, the integer kernels are meant to
sweep datasets that numpy is too slow for, and the host driver is what will
talk to an actual card.

The same arithmetic therefore exists three times: numpy in `p4_quantize`, C++
in `runtime/include/otf`, and SystemVerilog in `hardware/rtl`. That is
deliberate. Agreement between three independent implementations of one written
contract is what makes a mismatch a real finding instead of a shared
misreading, and `make test-cpp` and `make test-sim` check exactly that.

## How it works

A folded matrix vector engine is the workhorse. `SIMD` sets how many input
elements it consumes per cycle and `PE` how many output channels it produces in
parallel, so one vector costs `(MW/SIMD) * (MH/PE)` cycles. Convolution lowers
to a sliding window generator feeding that same engine.

The window generator holds only the span a window can reach, roughly `KH` rows
rather than a whole feature map, and captures the next pixels while replaying
the current ones. On a 224x224 layer that is 49 times less memory than buffering
the frame. It can also emit several kernel columns per beat, which is the only
way a first layer with one input channel folds at all: there are no channels to
spread across lanes, so the parallelism has to come from the window. The older
frame buffering strategy is still selectable with `strategy="frame"` and is kept
under test, because having two implementations that must agree is how the window
arithmetic gets checked.

Asymmetric quantization costs almost nothing in hardware. Expanding
`sum_k (x_k - zx) * w_kc` leaves `-zx * sum_k w_kc`, a per channel constant,
so the input zero point folds into the bias at compile time. Unsigned
activations are rebased onto the signed grid, since a uint8 value `u` with zero
point `z` is exactly the int8 value `u-128` with zero point `z-128`. That keeps
the whole datapath signed and costs one recorded offset instead of a second set
of RTL.

Buffer depths are computed, not guessed. `TokenAnalysis` works out how many
beats can be in flight on each edge under an as-soon-as-possible schedule. On a
chain that is a handful; across a fork that rejoins it is the whole skew
between the branches, and undersizing there deadlocks rather than slows. On the
example residual the skip branch needs sixteen beats, the analysis says
sixteen, and capping it at two stops the design dead. A structural check
refuses any graph where a stream still feeds two consumers, because Verilator
accepts that netlist silently.

Folding is chosen per unit, which can leave two neighbours meeting at widths
like 2 and 5. That converts only by stepping down to their common divisor and
back up, throttling the edge to one element per cycle. An alignment pass
retunes one side to an integer ratio first, and the two step chain remains as a
correct fallback.

Throughput of a streaming pipeline is set by its slowest stage, so resources
spent anywhere else are wasted. The folding allocator repeatedly finds the
current bottleneck and buys it the cheapest folding step that actually makes it
faster, stopping when the budget runs out or the target is met. When a stage
cannot fold any further it becomes the floor, and nothing else is bought speed
beyond it:

```
fold: conv0_swu cannot fold further and set the floor at 1044 cycles
```

Three details do most of the work on the hardware side. Accumulators are
narrowed to the width the weights actually need, so a 32x16 int8 layer uses a
19 bit accumulator instead of 32. The rescale is a per channel multiply and one
shared arithmetic shift, so the hardware needs one shifter and accuracy still
tracks per channel. Compute lives entirely in hand written RTL, which for this
class of unit is reported to produce circuits up to 80 percent faster with half
the block RAM and an order of magnitude fewer flip flops than the equivalent
generated from high level synthesis, at roughly a tenth of the synthesis time
([arXiv:2201.11409](https://arxiv.org/abs/2201.11409)).

The device model knows that a DSP48E2 fits two int8 multiplies that share an
operand, and four int4 ones, so falling precision raises MAC density rather
than just shrinking memory
([DSP-Packing, arXiv:2203.11028](https://arxiv.org/pdf/2203.11028)).

## Measuring against synthesis

Every resource figure above is a model estimate. `onnx2fpga synth`
compiles a model, synthesises each of its units out of context, and prints
measured cost beside predicted cost:

```
onnx2fpga doctor                          is Vivado usable here
onnx2fpga synth model.onnx --dry-run      print the commands, run nothing
onnx2fpga synth model.onnx --device z7020 synthesise and compare
onnx2fpga synth model.onnx --impl         place and route too
```

The free Vivado ML Standard edition covers `z7020`, which is enough to
calibrate the estimator. `vu9p` and `vu47p` are Virtex UltraScale+ and need the
paid edition or an AWS F1 or F2 instance.

## Requirements

- Python 3.9 or newer, numpy
- Verilator 5 for simulation
- A C++17 compiler for the runtime
- Vivado only to measure resources and timing, or to build a bitstream, which
  this project does not produce yet

## Status

This is an early prototype. The compiler is correct on what it supports, and
the operator list above is exhaustive. Everything about resources and timing is
still a model estimate rather than a synthesis result, and no bitstream has
been built.

License: Apache-2.0.
