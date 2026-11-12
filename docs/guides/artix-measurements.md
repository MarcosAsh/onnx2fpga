# Artix-7 measurements

All 22 fixed-mode corpus models pass simulation and Vivado 2026.1 routing at
100 MHz on the Arty A7-35 and A7-100 parts. The MLP and tree models also pass
full board builds with the UART, buffers and reset logic. These results do
not establish physical programming, board I/O or sustained serial throughput.

## Initial model-only builds

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

The compiler's LUT estimates exceed these measurements by about 14 to 55 percent.
DSP estimates match these examples. These initial builds predate the calibrated
delay model below and do not insert clock-driven pipeline stages.

```sh
python3 tools/report_corpus.py linear_regression mlp_regressor xgb_small \
    --device a7_35t --frames 256 --implement --out build/artix35
python3 tools/report_corpus.py linear_regression mlp_regressor xgb_small \
    --device a7_100t --frames 256 --implement --out build/artix100
```

## Pipeline delay calibration

The Artix delay model uses DSP48E1 limits from
[DS181 v1.22, table 31](https://www.farnell.com/datasheets/2301213.pdf), with fabric
estimates fitted to routed corpus results and a 15 percent scheduling margin.
It applies to the supported 35T and 100T parts at their declared speed grades.
The default target remains 100 MHz. Estimates guide pipeline insertion; every
new design still needs a routed timing check.

The [35T 100 MHz sweep](../measurements/artix-calibration-35t-100-2026-11-12.json)
passes all 22 models, their numerical thresholds, and 256 bit-exact inputs per
model. Every routed build passes setup, hold, pulse-width and internal constraint
checks. The smallest setup slack is 0.999 ns.

The [100T 100 MHz sweep](../measurements/artix-calibration-100t-100-2026-11-12.json)
also passes all 22 models and the same checks, with a minimum setup slack of
0.967 ns. Fresh compilations with the production profile match every generated
design hash in both 100 MHz sweeps.

At [200 MHz](../measurements/artix-calibration-35t-200-2026-11-12.json), all 17
models accepted by the compiler pass numerical, simulation and routing checks;
the smallest setup slack is 0.151 ns. The compiler rejects `lgbm_multiclass`,
`mlp_sigmoid`, `normalizer_logistic`, `ort_qoperator_gemm` and `xgb_multiclass`
because its comparator, lookup-table or normalization estimates exceed the
available cycle time. Those five were not routed at 200 MHz with this profile.
These refusals are model estimates, not measured limits of the devices.

The records retain the experimental override used for calibration. Its numerical
parameters are the same as the production `ARTIX_7_1` profile. To reproduce:

```sh
python3 tools/report_corpus.py --device a7_35t --clock-mhz 100 \
    --frames 256 --implement --out build/artix35_calibrated
python3 tools/report_corpus.py --device a7_35t --clock-mhz 200 \
    --frames 256 --implement --out build/artix35_200
python3 tools/report_corpus.py --device a7_100t --clock-mhz 100 \
    --frames 256 --implement --out build/artix100_calibrated
```

The 200 MHz command exits with failure because of the five compilation refusals;
its complete JSON report distinguishes them from routed timing failures.

## FIFO resources

Stream FIFOs have an asynchronous read port. Vivado maps their storage to
distributed RAM; increasing their depth does not move them into block RAM or
UltraRAM. The compiler charges LUTs for this storage and its selection logic,
plus registers for the pointers, occupancy count and replicated controls.
Depth-one buffers use registers.

The [FIFO measurements](../measurements/fifo-resources-2026-11-12.json) use
Vivado 2026.1 synthesis on the 35T part, with a 100 MHz constraint:

| Depth x width | Backend | Estimated LUTs | Synthesized LUTs | Estimated FFs | Synthesized FFs |
|---|---|---:|---:|---:|---:|
| 1024 x 8 | SystemVerilog | 304 | 273 | 61 | 31 |
| 1024 x 8 | Hardcaml | 304 | 273 | 61 | 31 |
| 4096 x 8 | SystemVerilog | 1080 | 1022 | 163 | 45 |
| 4096 x 8 | Hardcaml | 1080 | 1003 | 163 | 86 |

All four use zero block RAMs, UltraRAMs and DSPs. These are synthesis resource
checks, not routed timing results. The estimates are conservative for these
configurations; other sizes still need implementation checks, including the
device's available memory-capable slices. The fixed-mode corpus results above
do not use these streaming FIFOs.

```sh
python3 tools/measure_fifos.py --device a7_35t --depths 1024 4096 --width 8 \
    --out build/fifo_resources
```

## Full board builds

With the calibrated profile, the
[four board builds](../measurements/arty-calibrated-2026-11-12.json) pass the same
100 MHz timing and methodology checks:

| Board | Model | LUTs | Registers | DSPs | RAMB36 equivalents | Setup slack | Hold slack |
|---|---|---:|---:|---:|---:|---:|---:|
| 35T | MLP | 3310 | 2566 | 17 | 1 | 2.083 ns | 0.038 ns |
| 35T | Trees | 1066 | 1106 | 0 | 1 | 2.391 ns | 0.106 ns |
| 100T | MLP | 3307 | 2566 | 17 | 1 | 2.037 ns | 0.064 ns |
| 100T | Trees | 1068 | 1106 | 0 | 1 | 2.352 ns | 0.119 ns |

The MLP's extra registers provide more setup margin, at a latency cost: its
inference takes 10 cycles (100 ns), compared with 6 cycles (60 ns) in the initial
builds. Trees remain at 5 cycles (50 ns). Both retain an initiation interval of
one cycle. All four builds retain 4.5 ns pulse-width slack and zero critical
methodology violations. The records
include profile-source and artifact hashes. Physical operation remains untested.

### Initial builds without delay calibration

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
