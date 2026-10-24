# onnx2fpga documentation

We write these pages before the code. A page says how a feature should behave,
and then we make the code do that. So some pages describe things that do not
exist yet, and each one says so in its first line: implemented, partial, or
planned. A planned page is a spec. When the code and a page disagree, one of
them is wrong, and the commit that fixes it should say which.

If you are new, read [Getting started](getting-started.md). It takes a small
MLP from an ONNX file to answers coming back from an Icepi Zero, in about ten
lines of Python.

## Guides

[The workflow](guides/workflow.md) walks through the verbs: load, compile,
summary, evaluate, simulate, build, deploy, predict.
[Folding and budgets](guides/folding.md) explains how fast a design runs and
what stops it going faster. [Quantization](guides/quantization.md) covers where
the integer scales come from. [Boards](guides/boards.md) is the Icepi Zero
today and the KV260 next. [Custom hardware layers](guides/custom-layers.md) is
for when a model uses an operator we do not have, and
[Backends](guides/backends.md) is about Hardcaml versus SystemVerilog.

## Reference

The [Python API](api/reference.md), the [command line](api/cli.md) and the
[errors](api/errors.md) it raises. Two pages pin down the hardware: the
[stream contract](hardware/streams.md), which is the handshake every unit obeys,
and the [rescale contract](hardware/requant.md), the integer arithmetic they
all share. [Sliding window unit](hardware/sliding-window.md) explains the one
unit whose buffer sizing is easy to get wrong.

## Specs

| | |
|---|---|
| [001](specs/001-fixed-latency.md) | Fixed latency mode |
| [002](specs/002-pipelined-matvec.md) | Pipelining and clock targets |
| [003](specs/003-reloadable-weights.md) | Reloadable weights |
| [004](specs/004-weight-specialised-arithmetic.md) | Weight specialised arithmetic |
| [005](specs/005-tree-ensembles.md) | Tree ensembles |
| [006](specs/006-kv260.md) | KV260 board |
| [007](specs/007-tick-to-trade.md) | Tick to trade demo |
| [008](specs/008-weights-in-block-ram.md) | Weights in block RAM |

We are building toward trading: models that answer in a fixed number of cycles,
weights a host can swap without a rebuild, gradient boosted trees, and a 10G
market data demo on the Kria KR260. Keras import and a faster UART were dropped on 24
October 2026 when we made that call; git has them if they come back.

## How we write these

Say what the user types and what happens. Every code example has to run as
written. Use real numbers from real runs. Plain ASCII, short and long sentences
mixed, no marketing, no hedging. A spec states the behaviour, the API, the
errors, and how we will know it is done. Anything undecided goes under "Open
questions" so somebody answers it.
