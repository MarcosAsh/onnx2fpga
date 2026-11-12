# Artix-7 measurements

Three fixed-mode corpus models pass simulation and Vivado 2026.1 routing at
100 MHz on the Arty A7-35 and A7-100 parts. The MLP and tree models also pass
full board builds with the UART, buffers and reset logic. These results do
not establish physical programming, board I/O or sustained serial throughput.

## Model-only builds

Each model passed its numerical threshold and 256 bit-exact simulation inputs
on each target. These builds exclude the board shell. The
[35T report](../measurements/corpus-a7-35t-2026-11-12.json) comes from clean
revision `d0206b3`; the [100T report](../measurements/corpus-a7-100t-2026-11-12.json)
comes from clean revision `a5dda1b`. Both retain model, fixture, generated RTL
and routed artifact hashes, tool versions and all timing checks.

| Model | Cycles | Latency at 100 MHz | 35T LUTs | 100T LUTs | DSPs | 35T setup slack | 100T setup slack |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear_regression` | 3 | 30 ns | 264 | 264 | 1 | 1.104 ns | 1.077 ns |
| `mlp_regressor` | 6 | 60 ns | 3120 | 3114 | 17 | 0.022 ns | 0.227 ns |
| `xgb_small` | 5 | 50 ns | 813 | 812 | 0 | 5.243 ns | 5.371 ns |

All six runs have nonnegative setup, hold and pulse-width slack and no failing
internal constraint checks. Their native initiation interval is one cycle,
which permits 100 million vectors/s at the declared clock if the surrounding
system can supply and receive them. That is an RTL capacity, not a measured
board rate. The UART is much slower.

The compiler's LUT estimates exceed these measurements by about 14–55 percent.
DSP estimates match these examples. The Artix profiles still have no calibrated
delay model, so these builds do not use `--clock-mhz` to insert pipeline stages.
A successful build at 100 MHz does not guarantee timing for another model.

```sh
python3 tools/report_corpus.py linear_regression mlp_regressor xgb_small \
    --device a7_35t --frames 256 --implement --out build/artix35
python3 tools/report_corpus.py linear_regression mlp_regressor xgb_small \
    --device a7_100t --frames 256 --implement --out build/artix100
```

## Full board builds

The [board records](../measurements/arty-2026-11-12.json) include bitstream,
checkpoint, source and report hashes. The sources match revision `0f5108e`;
the builds ran from its worktree before the commit. Regenerating each shell
with that revision produces the same Verilog hash.

| Board | Model | LUTs | Registers | DSPs | RAMB36 equivalents | Setup slack | Hold slack |
|---|---|---:|---:|---:|---:|---:|---:|
| 35T | MLP | 3365 | 1912 | 17 | 1 | 0.194 ns | 0.006 ns |
| 35T | Trees | 1066 | 1106 | 0 | 1 | 2.391 ns | 0.106 ns |
| 100T | MLP | 3354 | 1785 | 17 | 1 | 0.119 ns | 0.054 ns |
| 100T | Trees | 1068 | 1106 | 0 | 1 | 2.352 ns | 0.119 ns |

All four bitstreams pass setup, hold, pulse-width and internal constraint checks
at 100 MHz. Pulse-width slack is 4.5 ns, with no critical methodology violations.
Vivado reports an advisory SYNTH-6 warning for the UART RAM output registers.
Asynchronous button/UART input paths and UART/LED outputs are excluded from
synchronous timing analysis; their physical behavior still needs board tests.

Generated MLP and tree board RTL each passed eight serial frames against the
integer model with clock skew, byte gaps and reset. Separate Hardcaml tests
cover stalled predictions, reset during pending work and serial break recovery.
The Python deployment path also passed a pseudo-terminal test. None of these
checks substitutes for a physical UART test.

To reproduce a full build after the corpus commands above:

```python
from pathlib import Path
from onnx2fpga.board import ArtyA735, ArtyA7100

for board, root in ((ArtyA735, "build/artix35"), (ArtyA7100, "build/artix100")):
    for model in ("mlp_regressor", "xgb_small"):
        flow = board(Path(root) / model)
        flow.run()
        print(flow.bitstream, flow.render_report())
```

The [board guide](boards.md#arty-a7) describes API deployment and serial limits.
