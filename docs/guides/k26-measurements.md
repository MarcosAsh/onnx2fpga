# K26 measurements

Status: measured out of context in Vivado 2026.1 on 11 November 2026.

We placed and routed all 22 corpus models for `xck26-sfvc784-2LV-c`, with
constant weights, fixed latency and a 3.103 ns clock constraint (about
322.27 MHz). The compiler target was 322.265625 MHz. Twenty-one met the
clock constraint. `ort_qoperator_gemm` missed setup timing by 0.106 ns;
the table mapping fix below closes that path in a separate routed run.
These are model-only runs: they do not include the Kria register shell,
PS, MMCM, board I/O or the tick-to-trade pipeline. No result here is a
measurement from a physical board.

## Baseline

The RTL came from revision `dc152d0`. Counts below come from Vivado's
utilization reports; the first run exposed a DSP counting bug that was
corrected against those reports. The [JSON results](../measurements/k26-2026-11-11.json)
also retain the compiler's resource and clock estimates. Positive setup
slack means the constrained register paths met timing. Latency is the
compiler's cycle count.

| Model | Cycles | LUTs | Registers | DSPs | Setup slack (ns) |
|---|---:|---:|---:|---:|---:|
| `brevitas_mlp` | 18 | 6526 | 3142 | 47 | 0.111 |
| `forest_binary` | 4 | 740 | 353 | 0 | 1.110 |
| `forest_multiclass` | 6 | 1249 | 847 | 0 | 0.681 |
| `lgbm_multiclass` | 7 | 1888 | 1760 | 0 | 0.922 |
| `lgbm_small` | 5 | 751 | 690 | 0 | 1.058 |
| `linear_regression` | 6 | 253 | 164 | 1 | 0.741 |
| `logistic_binary` | 5 | 235 | 189 | 0 | 1.347 |
| `logistic_multiclass` | 7 | 569 | 370 | 3 | 0.466 |
| `mlp_regressor` | 13 | 3084 | 2355 | 17 | 0.378 |
| `mlp_sigmoid` | 14 | 2975 | 2166 | 17 | 0.097 |
| `normalizer_logistic` | 16 | 1021 | 635 | 11 | 0.365 |
| `ort_qdq_mlp` | 13 | 3639 | 2174 | 17 | 0.195 |
| `ort_qoperator_gemm` | 22 | 5459 | 3387 | 22 | -0.106 |
| `ort_qoperator_mlp` | 13 | 3637 | 2174 | 17 | 0.360 |
| `qonnx_mlp` | 19 | 6858 | 3345 | 49 | 0.185 |
| `ridge` | 6 | 261 | 168 | 1 | 0.610 |
| `scaler_linear` | 6 | 237 | 168 | 1 | 0.953 |
| `scaler_logistic` | 5 | 216 | 195 | 0 | 1.189 |
| `torch_bn_mlp` | 13 | 3070 | 2245 | 17 | 0.550 |
| `xgb_multiclass` | 7 | 2058 | 2019 | 0 | 0.535 |
| `xgb_small` | 5 | 814 | 694 | 0 | 0.728 |
| `xgb_small_v5` | 5 | 804 | 695 | 0 | 0.824 |

## Table mapping

The worst path in `ort_qoperator_gemm` starts at the sigmoid table's
RAMB18E2 output and ends in the following dense layer's constant arithmetic.
Vivado inferred eight block RAMs for a table that the compiler models as
logic. Asking Vivado to use distributed ROM for that table removes the
block RAM output delay without changing the one-cycle table latency.

The regenerated Hardcaml RTL uses `rom_style="distributed"` on table lookups.
Both rerouted models meet the same clock constraint:

| Model | Cycles | LUTs | Registers | DSPs | Setup slack (ns) |
|---|---:|---:|---:|---:|---:|
| `ort_qoperator_gemm` | 22 | 5838 | 3533 | 22 | 0.294 |
| `mlp_sigmoid` | 14 | 3357 | 2285 | 17 | 0.153 |

The GEMM model uses 379 more LUTs and 146 more registers, and frees eight
RAMB18 blocks. Its latency stays at 22 cycles. The sigmoid MLP's slack
improves from 0.097 to 0.153 ns at the same 14-cycle latency. These two
models were rerouted after the change; the other rows above remain baseline
measurements. Hardcaml formatting, lint and tests pass, along with the
19 focused Python tests covering elementwise behavior, latency proofs and
a bit-exact simulation sweep across all 22 corpus models.

## Reproduce

With Vivado on PATH and a valid license:

```sh
make measure-k26
```

For one model:

```sh
python3 tools/measure_k26.py ort_qoperator_gemm
```

Results go to `build/measure_k26/measured.json`. Each model's `vivado/`
directory contains `utilization_impl.rpt`, `timing_impl.rpt` and
`summary.json`. Inspect the timing report's hold and pulse-width checks as
well as setup slack. The reported `fmax_mhz` is calculated from one run's
setup slack; it is not a frequency sweep or board validation.
