# Model and device support

Support depends on an operator's attributes, input shapes, data types and
execution mode. Start with `onnx2fpga check model.onnx --device k26 --mode fixed`.
This checks the built-in importer and known mode restrictions. It reports
unassessed properties separately; compilation and timing may still fail.

## Operator families

| Family | Execution modes | Restrictions and details |
|---|---|---|
| Gemm, MatMul | Stream and fixed | Constant weights; Gemm `transA` is unsupported. |
| Conv, MaxPool | Stream | MaxPool ceil mode and indices output are unsupported. Grouped Conv requires per-tensor activation grids. Fixed mode rejects these units. |
| GlobalAveragePool | Stream | Static batch-one 2D images, per-tensor activation grids, sums up to 32 bits. |
| AveragePool | Stream | Static batch-one 2D images, floor output sizes, either padding-count mode. |
| Relu, Add | Stream and fixed | Shapes and quantization must satisfy the selected hardware unit. |
| Other live elementwise operations | Fixed | See [elementwise functions and tables](../specs/010-lookup-tables-and-elementwise.md) for operations and broadcasting limits. |
| Reductions, Softmax, Normalizer, LayerNormalization | Fixed | See [axes, shapes and normalization](../specs/011-reductions-and-softmax.md). |
| Tree ensembles | Fixed | See [tree ensembles](../specs/005-tree-ensembles.md) and [multiclass trees](../specs/012-multiclass-trees.md) for branch modes and output semantics. |
| Layout operations and constant arithmetic | Depends on the imported graph | Many fold away. Live views can become dense layers; live Concat can require fixed mode. See [import rules](../specs/009-operator-coverage.md). |
| BatchNormalization and ONNX-ML linear operators | Depends on the imported graph | Inference BatchNormalization folds into arithmetic; training mode is rejected. LinearRegressor requires `post_transform=NONE`. |
| QDQ, QOperator and QONNX imports | Depends on the lowered operations | Static grids are required. See [quantized imports](../specs/009-operator-coverage.md) and [QONNX](../specs/014-qonnx.md). |

These families do not imply support for every ONNX opset or attribute value.
MaxPool uses stride one when omitted and ignores padded positions when taking
the maximum, including windows containing only negative input values.
The importer converts declared input shapes to its internal layout and uses
batch size 1 for symbolic batch dimensions. The checker marks unknown feature
dimensions as unassessed. Quantized hardware introduces numerical error even
when the generated RTL exactly matches the integer reference model.

## Device and board evidence

| Target | Compiler/model evidence | Full board flow | Physical board evidence |
|---|---|---|---|
| K26 / KR260 | [22-model routed corpus](k26-measurements.md) | MLP, 100-tree and [32K DDR trading](tick-to-trade.md#k26-clock-target) builds pass Vivado timing | Pending |
| K26 / KV260 | Shares the K26 device model | Project generator and stub tests; a real full build remains unverified | Pending |
| ECP5-25K / Icepi Zero | Compiler, simulation and open-tool builds | Implemented | [Recorded MLP test](../getting-started.md#what-to-expect) |
| VU9P, VU47P, ZU7EV, Zynq-7020 | Device profiles; evidence varies by model | No corresponding board integration in the board registry | Unverified here |
| Artix-7 35T / 100T | [22 models routed on each part at 100 MHz](artix-measurements.md) | Arty A7 MLP and tree builds pass at 100 MHz | Pending |

A device profile supplies budgets and estimation parameters. It does not
establish that a board can be programmed or that a model meets timing.

Artix budgets come from [AMD DS180](https://docs.amd.com/api/khub/documents/2LByHkO~nSZXcei2D55fTg/content).
The selected parts match Digilent's E.0 board definitions for the
[Arty A7-35](https://github.com/Digilent/vivado-boards/blob/master/new/board_files/arty-a7-35/E.0/1.1/board.xml)
and [Arty A7-100](https://github.com/Digilent/vivado-boards/blob/master/new/board_files/arty-a7-100/E.0/1.1/board.xml).

## Reproduce the corpus report

### CNN baseline

`tests/fixtures/cnn` contains six untrained, batch-one image models exported
from PyTorch. Their sidecar JSON files record versions, seeds and file hashes.
Each has 512 calibration images and 256 separate evaluation images with ONNX
Runtime outputs. The exporter also checks those outputs against PyTorch.
These fixtures measure numerical error, not classifier task accuracy.

```sh
python3 tests/fixtures/make_cnn.py
python3 tools/report_corpus.py --corpus-dir tests/fixtures/cnn --mode stream \
    --inspect-only --out build/cnn_inspection
python3 tools/report_corpus.py maxpool residual --corpus-dir tests/fixtures/cnn \
    --mode stream --frames 256 --sweep --out build/cnn_baseline
```

Exporting needs PyTorch, ONNX and ONNX Runtime. Existing fixtures can be checked
without those packages. All seven CNN fixtures pass inspection. They remain separate from the vector corpus.

The trained `mnist` fixture uses the standard 60,000/10,000 split, seed 81 and
10 CPU training epochs. Full-test accuracy is 97.53% in ONNX Runtime and 97.45%
with 8-bit weights and activations. The 256-image numerical fixture has
normalized mean absolute error 0.02435. Frozen gates are 95% float accuracy,
94.5% quantized accuracy, at most 0.5 percentage points lost, and error below
0.025. Calibration uses 512 training images; RTL inputs come from the test split.

MNIST is by Yann LeCun and Corinna Cortes, distributed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), according to
[the dataset documentation](https://keras.io/api/datasets/mnist/). The included
subset is normalized to float32 pixels divided by 255 and includes its split
indices. The sidecar records dataset and model hashes, training settings and
accuracy.

```sh
python3 tests/fixtures/make_cnn.py mnist --mnist-data build/datasets/mnist.npz
```

The [baseline report](../measurements/cnn-baseline-2026-11-13.json) records
the max-pool CNN and residual CNN passing numerical comparison and bit-exact
Hardcaml simulation on 256 images, including 15 stall runs per model. Their
normalized mean absolute errors are 0.00754 and 0.01014, below the existing
0.025 threshold. This establishes the small residual fixture's stream behavior;
other branch shapes still need their own checks. CNN routing and physical
measurements remain unverified.

Stream reports include total cycles, time to the first output beat, per-frame
latency from first accepted input to final accepted output, and the mean
interval between completed frames. Every stall run retains these measurements.
A separate run resets halfway through the first input frame under backpressure,
then checks all replayed frames. Its cycle counts begin after reset. Compiler
cycles per frame remain schedule estimates; physical throughput is unmeasured.

### Grouped convolution

`Conv` supports groups and depthwise channel multipliers in stream mode.
Input channels must equal the weight tensor's input channels times `group`;
output channels must divide evenly into groups, following
[ONNX](https://onnx.ai/onnx/operators/onnx__Conv.html).
Weights use per-output-channel scales. Grouped convolution requires per-tensor
activation grids; unsupported grids fail before hardware generation.

The window generator emits each group's kernel before advancing to the next
group. The matrix engine reuses those inputs for that group's output channels.
Tests cover bias, asymmetric padding, stride, dilation, depthwise multipliers
one and two. Unit comparisons include reset under stalls. Routed CNN timing
remains unqualified.

### Average pooling

Both average-pool operators require stream mode, static batch-one 2D images,
per-tensor activation grids and sums that fit in 32 bits.

[`AveragePool`](https://onnx.ai/onnx/operators/onnx__AveragePool.html) accepts
positive strides and dilations, nonnegative asymmetric padding, and
`SAME_UPPER`, `SAME_LOWER` or `VALID` auto padding. Explicit and automatic
padding cannot be combined. Strides default to one; only `ceil_mode=0` is
supported. `count_include_pad=1` divides by the kernel area; the default
excludes padded positions from the divisor.

[`GlobalAveragePool`](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html)
reduces height and width to one value per channel, retaining both spatial axes.
It accepts no attributes.

Both use signed sums followed by integer rescaling with zero-point correction,
rounding and clamping. Output stalls preserve pending results; reset discards
the partial frame. Tests cover partial-frame reset, signed extremes, odd and
one-pixel images. ONNX Runtime comparisons cover asymmetric edges, dilation
and auto padding; both HDL backends pass padded negative and constant images.

The [pooling report](../measurements/average-pool-2026-11-13.json) records both
CNNs passing the 0.025 numerical limit and 256-frame stall sweeps.
The [Artix-7 35T global-average build](../measurements/global-average-a7-2026-11-13.json)
fails 100 MHz timing with -10.279 ns setup slack through the convolution
arithmetic and rescale. Its numerical and simulation stages pass. Physical
operation remains unverified.

### Vector corpus

From a checkout, using Python, NumPy, Hardcaml and Verilator:

```sh
python3 tools/report_corpus.py --clock-mhz 322.265625
```

The command compiles each corpus model for K26 in fixed mode, compares its
integer predictions with stored independent outputs on 256 evaluation inputs,
and simulates 16 inputs. Use `--frames` to change the simulation count, up to
the fixture's available inputs. Model names select a subset:

```sh
python3 tools/report_corpus.py linear_regression xgb_small --clock-mhz 322.265625
python3 tools/report_corpus.py --inspect-only --mode stream --out build/stream_coverage
```

`build/corpus_report/report.json` contains model and fixture hashes, exporter
metadata, tool versions, the source revision and working-tree status. Each
model records inspection, compilation, numerical comparison and simulation
separately. The report includes compiler resource estimates, cycle counts
and throughput at the assumed clock. By default, implementation and physical
board testing remain `not_run`.

Add `--implement` to place and route a model:

```sh
python3 tools/report_corpus.py linear_regression --device a7_100t --frames 256 \
    --implement --out build/artix_report
```

The Vivado flow records measured resources, setup, hold and pulse-width slack, internal
constraint checks, and hashes of the reports and routed checkpoint. Any failed
timing check fails the model's report. This build is out of context: board I/O
timing and physical operation remain unassessed. Artix profiles use a 100 MHz
default target and a [calibrated delay model](artix-measurements.md#pipeline-delay-calibration).
Use `--clock-mhz` to select the pipeline target. Routed timing remains required.

For ECP5 targets, the same option runs Yosys and nextpnr-ecp5:

```sh
python3 tools/report_corpus.py torch_layernorm_mlp --device icepi25 --clock-mhz 80 \
    --frames 256 --implement --out build/layernorm_ecp5
```

The report records resource use, the requested and achieved clock frequency,
tool versions and hashes of the synthesis netlist, commands and routing report.
An absent clock, a different constraint or a frequency below the requested
target fails implementation. nextpnr setup estimates do not establish hold,
pulse-width or board I/O timing.

Stream mode shares arithmetic across cycles. To measure a model that exceeds
the fixed pipeline's resource budget, request that mode explicitly:

```sh
python3 tools/report_corpus.py brevitas_mlp qonnx_mlp --device icepi25 \
    --mode stream --frames 256 --implement --out build/quantized_ecp5_stream
```

Stream reports use the device's default clock assumption, 20 MHz for Icepi
Zero, and record cycles per frame. `--clock-mhz` selects fixed-mode pipelining
and cannot be used with stream mode. Throughput estimates assume that clock
and continuous traffic without downstream stalls; physical throughput remains
unmeasured.

Both models pass [stream-mode routing](../measurements/quantized-ecp5-stream-2026-11-12.json)
at 20 MHz and match 256 RTL vectors each. Brevitas uses 27 DSPs and reaches
28.02 MHz; QONNX uses 28 DSPs and reaches 27.06 MHz. Both schedule 32 cycles
per frame, or 625,000 frames/second at 20 MHz under the assumptions above.

The [12 November 2026 report](../measurements/corpus-k26-2026-11-12.json),
from clean revision `bef6835`, passes all 22 models. It used the command above
with `--frames 256`, for 5,632 bit-exact simulation vectors. Fixed pipeline
latencies range from 4 to 22 cycles at the requested 322.265625 MHz clock.
This run did not measure achievable clock frequency.

Regression error must be below 2.5 percent of the reference output's standard
deviation. A constant reference requires exact agreement. Classifier labels
must agree on at least 98 percent of inputs. These are corpus regression
thresholds; applications need their own accuracy requirements. Reference
outputs come from ONNX Runtime, or QONNX for the QONNX/Brevitas fixtures;
this command does not regenerate or independently revalidate those outputs.

The three `ort_*` fixtures use ONNX Runtime's `reduce_range=True` to keep signed
weights within seven bits. This avoids the documented U8S8 pairwise saturation
on [AVX2 CPUs without VNNI](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
CI checks the stored outputs against ONNX Runtime with strict tolerances and
checks the weight range. Quantizer version and range settings are stored in the
models' metadata. Older measurement reports identify the hashes of their
original eight-bit-weight fixtures.

The corpus now includes the PyTorch LayerNorm MLP. A regression run at
`6d727c4` checked numerical accuracy and 256 RTL vectors per accepted model:

| Target and scheduling clock | Passed | Refused before simulation | Report |
|---|---|---|---|
| K26, 322.265625 MHz | 23 | 0 | [K26](../measurements/corpus-regression-k26-2026-11-12.json) |
| Artix-7 35T, 100 MHz | 23 | 0 | [Artix](../measurements/corpus-regression-a7_35t-2026-11-12.json) |
| ECP5-25K, 80 MHz | 21 | 2 | [ECP5](../measurements/corpus-regression-ecp5-2026-11-12.json) |

Both ECP5 refusals are fixed-mode DSP limits: `brevitas_mlp` and `qonnx_mlp`
each need 49 DSPs against a budget of 28. These runs do not establish routed
timing. Their metadata records four uncommitted trading files; the inference
compiler and corpus match the recorded revision. Reproduce each row with
`--device`, `--clock-mhz` and `--frames 256`, omitting `--implement`.

The JSON file is replaced after each model. `complete=false` means the run
was interrupted, and any failed model causes a nonzero exit status. An
inspection-only run records `inspected`, with later stages left `not_run`.
Use the [K26 measurement flow](k26-measurements.md#reproduce) and
[KR260 board builds](boards.md#reproduce-the-kr260-build) for routed evidence.
