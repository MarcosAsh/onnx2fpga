# 013 Tree feature and leaf widths

Status: implemented. The widths are chosen by hand; picking them from data is
an open question.

## Problem

[005](005-tree-ensembles.md) gives every tree ensemble 16 bit feature codes and
16 bit leaves. That keeps the integer model within a rounding of the float one,
but it is far more than the labels need, and it is most of the area. The leaf
select is a mux per leaf bit, the adder tree is as wide as the leaves, and the
comparators are as wide as the features. TreeLUT, the closest published tree
compiler, quantizes its leaves to a few bits.

We first tried a lossless cut: give each tree only the bits its own leaves
use, since later boosting rounds have smaller leaves. On the corpus that saves
one to three bits a tree, 13 to 16 bits against 16, which is not worth a
change to the unit.

## Behaviour

A tree ensemble's layer takes both widths through the API it already had:

```python
acc = otf.load("xgb_multiclass.onnx")
acc.layers["TreeEnsembleClassifier"].quantize(weights="int8", activations="int12")
acc.compile(device="k26", mode="fixed", clock_mhz=322.265625)
```

`weights` are the leaves and `activations` are the features, each `"int2"` to
`"int16"`, and either left out keeps 16 bits. The grids are built as in 005 at
the asked width: each feature's step is still a power of two, spanning its
thresholds with a code to spare, so it is 16 times coarser at 12 bits, and the
leaf scale is the largest leaf over the largest code of the narrower type.
Everything after that, the comparator sharing, the split pruning, the sum
width and the pipelining, is unchanged and works on the narrower codes. The
summary's quantization line names the widths, for example `features int12 on
per feature grids from the thresholds, leaves int8`.

For other layers `activations` is still refused, and `weights` still stops at
`"int8"`.

## Results

The corpus models on `k26` at 322.265625 MHz. Agreement is the share of the
256 held out labels that match ONNX Runtime; for a regressor it is the mean
absolute error against ONNX Runtime as a share of the output's standard
deviation. LUT is the compiler's estimate.

| Model | Features, leaves | Cycles | LUT | Agreement |
|---|---|---|---|---|
| `xgb_multiclass` | 16, 16 | 7 | 2788 | 100 percent |
| | 12, 8 | 7 | 1542 | 99.6 percent |
| | 8, 4 | 6 | 908 | 97.3 percent |
| `lgbm_multiclass` | 16, 16 | 7 | 2690 | 100 percent |
| | 12, 8 | 6 | 1425 | 98.4 percent |
| | 8, 4 | 6 | 719 | 73.0 percent |
| `forest_multiclass` | 16, 16 | 6 | 2282 | 100 percent |
| | 12, 8 | 6 | 1220 | 99.2 percent |
| | 8, 4 | 5 | 682 | 96.5 percent |
| `forest_binary` | 16, 16 | 4 | 747 | 100 percent |
| | 12, 8 | 4 | 397 | 100 percent |
| | 8, 4 | 4 | 216 | 99.2 percent |
| `xgb_small` | 16, 16 | 5 | 926 | error 0.25 percent |
| | 12, 8 | 5 | 511 | error 0.54 percent |
| | 8, 4 | 5 | 299 | error 7.6 percent |
| `lgbm_small` | 16, 16 | 5 | 920 | error 0.00 percent |
| | 12, 8 | 5 | 507 | error 0.25 percent |
| | 8, 4 | 5 | 296 | error 3.8 percent |

12 bit features and 8 bit leaves take 45 percent off every model and keep
every classifier above the 98 percent the corpus test asks for. 4 bit leaves
are where it breaks: LightGBM's 60 trees of 8 leaves fall to 73 percent,
because 16 leaf codes cannot tell its small late leaves apart. Three models
lose a cycle at some width: a narrower score lets the pick tree close without
its register.

The 100 tree regressor of 005 did not fit the Icepi Zero, at 20164 LUT4 sites
against 19430 allowed. At 12 and 8 bits it needs 13293 and compiles.

## Resources

yosys 0.52 `synth_xilinx -family xcup` on the generated `rtl/top.v`, against
the compiler's estimate, which was fitted at 16 bits only:

| Model | Features, leaves | Compiler LUT | yosys LUT |
|---|---|---|---|
| `xgb_multiclass` | 16, 16 | 2788 | 2902 |
| | 12, 8 | 1542 | 1364 |
| | 8, 4 | 908 | 704 |
| `forest_binary` | 16, 16 | 747 | 936 |
| | 12, 8 | 397 | 392 |
| | 8, 4 | 216 | 192 |

The estimate follows the narrowing and stays within 30 percent, high at narrow
widths and low at 16 bits on the forest, so it is not refitted here. Vivado
numbers come from `make measure-k26`, once it has a license.

## Verification

`tests/test_trees.py` compiles `xgb_multiclass` at 12 and 8 bits for `k26` at
322.265625 MHz and checks the unit's widths and the summary line, that the
estimate is under 60 percent of the 16 bit design's, that at least 99 percent
of the held out labels match ONNX Runtime, that 10000 random vectors simulate
bit exact under Verilator, and that the latency proof of
[001](001-fixed-latency.md) closes at the design's latency and fails one cycle
short. It also checks that `"int17"` features and `"int1"` leaves are refused.

## Errors

- `activations='int17' for TreeEnsembleClassifier, expected 'int2' to 'int16'`,
  and the same for `weights`.
- `activation precision cannot be set per layer` for any layer that is not a
  tree ensemble, as before.

## Done when

All three are met.

- [x] A tree layer takes feature and leaf widths from 2 to 16 bits, and the
  design simulates bit exact and proves its latency.
- [x] The corpus trees at 12 and 8 bits are recorded here against 16 bits.
- [x] yosys confirms the LUT saving on two models.

## Open questions

- The widths could be picked from the calibration data: the narrowest pair
  whose labels agree with the float model on, say, 99.5 percent of it. Trees
  take no calibration data today, and the default would change every number in
  005 and 012.
- Features are one width. A feature tested by two thresholds needs far fewer
  codes than one tested by forty, and a per feature width would shrink the
  comparators and the input port further.
- The leaf scale is shared by every tree. A scale per score, or per tree with
  a shift before the adder, would keep small late leaves at 4 bits.
