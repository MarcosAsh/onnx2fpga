# Model and device support

Support depends on an operator's attributes, input shapes, data types and
execution mode. Start with `onnx2fpga check model.onnx --device k26 --mode fixed`.
This checks the built-in importer and known mode restrictions. It reports
unassessed properties separately; compilation and timing may still fail.

## Operator families

| Family | Execution modes | Restrictions and details |
|---|---|---|
| Gemm, MatMul | Stream and fixed | Constant weights; Gemm `transA` is unsupported. |
| Conv, MaxPool | Stream | Spatial operators; grouped convolution is unsupported. Fixed mode rejects these units. |
| Relu, Add | Stream and fixed | Shapes and quantization must satisfy the selected hardware unit. |
| Other live elementwise operations | Fixed | See [elementwise functions and tables](../specs/010-lookup-tables-and-elementwise.md) for operations and broadcasting limits. |
| Reductions, Softmax, Normalizer, LayerNormalization | Fixed | See [axes, shapes and normalization](../specs/011-reductions-and-softmax.md). |
| Tree ensembles | Fixed | See [tree ensembles](../specs/005-tree-ensembles.md) and [multiclass trees](../specs/012-multiclass-trees.md) for branch modes and output semantics. |
| Layout operations and constant arithmetic | Depends on the imported graph | Many fold away. Live views can become dense layers; live Concat can require fixed mode. See [import rules](../specs/009-operator-coverage.md). |
| BatchNormalization and ONNX-ML linear operators | Depends on the imported graph | Inference BatchNormalization folds into arithmetic; training mode is rejected. LinearRegressor requires `post_transform=NONE`. |
| QDQ, QOperator and QONNX imports | Depends on the lowered operations | Static grids are required. See [quantized imports](../specs/009-operator-coverage.md) and [QONNX](../specs/014-qonnx.md). |

These families do not imply support for every ONNX opset or attribute value.
The importer converts declared input shapes to its internal layout and uses
batch size 1 for symbolic batch dimensions. The checker marks unknown feature
dimensions as unassessed. Quantized hardware introduces numerical error even
when the generated RTL exactly matches the integer reference model.

## Device and board evidence

| Target | Compiler/model evidence | Full board flow | Physical board evidence |
|---|---|---|---|
| K26 / KR260 | [22-model routed corpus](k26-measurements.md) | MLP and 100-tree builds pass Vivado timing | Pending |
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

Add `--implement` to run Vivado place-and-route for a fixed-mode model:

```sh
python3 tools/report_corpus.py linear_regression --device a7_100t --frames 256 \
    --implement --out build/artix_report
```

This records measured resources, setup, hold and pulse-width slack, internal
constraint checks, and hashes of the reports and routed checkpoint. Any failed
timing check fails the model's report. This build is out of context: board I/O
timing and physical operation remain unassessed. Artix profiles use a 100 MHz
default target and a [calibrated delay model](artix-measurements.md#pipeline-delay-calibration).
Use `--clock-mhz` to select the pipeline target. Routed timing remains required.

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

The JSON file is replaced after each model. `complete=false` means the run
was interrupted, and any failed model causes a nonzero exit status. An
inspection-only run records `inspected`, with later stages left `not_run`.
Use the [K26 measurement flow](k26-measurements.md#reproduce) and
[KR260 board builds](boards.md#reproduce-the-kr260-build) for routed evidence.
