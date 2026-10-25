# 005 Tree ensembles

Status: partial. Frozen TreeEnsembleRegressor and binary TreeEnsembleClassifier
models from ai.onnx.ml opsets 1 and 3 build in fixed latency mode and are
verified. Missing: the TreeEnsemble operator of opset 5, multiclass and
multi target ensembles, reloadable trees ([003](003-reloadable-weights.md)
made dense layers reloadable and says what trees still need), and a resource
comparison with conifer, which needs Vivado. `clock_mhz=` from
[002](002-pipelined-matvec.md) adds register stages to the leaf select and the
adder tree. The 100
tree regressor compiles for the `k26` device from [006](006-kv260.md) in 7580
LUTs, 6.5 percent of the part.

## Problem

Gradient boosted trees are among the most used models on tabular, market style
data, and until now we could not compile a single tree. XGBoost and LightGBM
export to ONNX's tree ensemble operators in the `ai.onnx.ml` domain, so the
front door was already the right shape.

## Behaviour

A tree ensemble becomes one fixed latency unit, `otf_trees`, with the same
ports as any fixed latency design ([001](001-fixed-latency.md)). Every tree is
evaluated in parallel, and the pipeline has three parts:

1. Every distinct split, a feature code against a threshold, is one comparator,
   and all of them are registered in the first cycle. Splits that test the same
   feature against the same threshold share a comparator, so the 1495 splits of
   the example regressor need 871.
2. Each tree picks its leaf with a mux tree that follows the tree's own shape,
   driven by those registered bits, and the leaf is registered.
3. An adder tree sums the leaves and a constant bias in binary levels, with a
   register after every second level, which is fan in 4 a stage.

The latency is 2 plus the levels of that adder tree for the trees and the bias,
so it depends only on the tree count: 3 cycles for up to 3 trees, 6 for 64 to
255. The 100 tree example answers in 6 cycles, 18.6 ns at 322.265625 MHz.

The regressor's output is the sum of leaf codes plus the bias, on a grid whose
scale is the leaf scale. For a classifier the logistic is monotone, so the
design decides on the margin: the bias carries minus the decision threshold
minus one, and the output is one bit, set when the sum is not negative. That
costs no extra cycle and no comparator. The bit is the label's index in
`classlabels`. ONNX Runtime labels a binary classifier with one score on the
raw score, against 0 when any weight is negative and against 0.5 otherwise, and
the import follows it.

## Quantization

Features are 16 bit codes and every feature gets its own grid,
`real = scale * (code - zero_point)`. The grid comes from the model, not from
calibration data: its step is a power of two and it spans the feature's
thresholds with one code to spare at each end. A value past the last threshold
is decided the same way as the clamped code, so clamping loses nothing, and the
power of two step puts round thresholds such as 0 and 0.5 exactly on a code,
which `BRANCH_EQ` needs. A `calibration=` argument is not needed and does not change the grids.

Each split then becomes `code <= T` or `code == T` on the codes. For `x <= t`,
T is the largest code whose real value is at most t; for `x < t` it is the
largest whose value is below t. `>=`, `>` and `!=` are the same tests with the
branches swapped. An equality split whose threshold is not on the grid can
never be taken, and a split no code can take one way, because its threshold is
past the grid, is cut out and replaced by its live branch. So on every grid
point the integer comparison agrees with the float one by construction, and
`tests/test_trees.py` checks it exhaustively: for every split of every fixture,
on all 65536 codes of its feature.

Leaves are 16 bit codes on one scale, the largest leaf magnitude over 32767,
and the base value and decision threshold are rounded onto it. The sum width is
whatever the leaf ranges and the bias need, 21 bits for the example regressor.

The integer model in `TreeEnsembleUnit.execute` is the reference the hardware
must match bit for bit. It differs from the float model only where a raw
feature falls between a threshold and the code next to it, and in the leaf
rounding. On 10000 inputs drawn from N(0, 1.5) on each feature, every one of the
100 trees picks the same leaf as the float model on 99.79 percent of inputs for
the regressor and 99.89 percent for the classifier, the classifier's label
agrees on all 10000, and the regressor's mean absolute error is 0.013 against
an output standard deviation of 188. With 8 bit features the classifier agreed
on 99.67 percent and every tree agreed on 83 percent, which is why features
are 16 bits; comparators that wide are cheap.

## Import

TreeEnsembleRegressor with one target and TreeEnsembleClassifier with two
classes scored by one weight list, from ai.onnx.ml opsets 1 and 3, the latter
with `_as_tensor` attributes. These are what onnxmltools 1.16 and skl2onnx 1.20
emit for XGBoost and scikit-learn. Split modes are LEQ, LT, GTE, GT, EQ and NEQ.
Post transforms are NONE for a regressor and NONE or LOGISTIC for a
classifier, and aggregate functions SUM and AVERAGE. A classifier's
probabilities output and a ZipMap after it are dropped, since the hardware
computes the label. NaN features are not supported, so
`nodes_missing_value_tracks_true` is ignored.

The repo's own protobuf decoder already read the node domain and the float,
int, string and tensor attributes these operators use, so it needed no change.

## API

```python
acc = otf.load("gbdt.onnx")        # exported with onnxmltools or skl2onnx
acc.compile(device="vu9p", mode="fixed")
acc.latency                        # 6
acc.summary()                      # trees, depth, comparators, latency, LUT
acc.evaluate(x, y)                 # label agreement, or mean_abs_error for a regressor
acc.simulate()                     # 10000 random vectors, one per cycle
```

The example regressor's summary row reads
`100/4  871  6  7580  0`: 100 trees of depth 4, 871 comparators, 6 cycles,
and the estimated LUT and DSP. `predict` returns the regression value, or the
label index as a float, shaped `(N, 1)`.

## Errors

Every error names the node or the device.

- An option we do not support raises `UnsupportedOpError` with one sentence
  naming it, for example `TreeEnsembleClassifier 'TreeEnsembleClassifier' has
  post_transform SOFTMAX, and only NONE or LOGISTIC is supported`. The same
  goes for more than two classes, more than one target, MIN or MAX aggregation
  and more than one base value.
- TreeEnsemble from ai.onnx.ml opset 5 raises `UnsupportedOpError`: it is not
  supported yet, so export TreeEnsembleRegressor or TreeEnsembleClassifier.
- An ensemble that does not fit the device raises `BudgetError` with the tree
  count and depth: the example regressor on `icepi25` says `the tree ensemble
  needs 20164 LUT, past the 19430 allowed on icepi25 (100 trees of depth 4)`.
  Depth is not limited separately; the leaf select of a depth 6 tree is a 64 to
  1 mux in one cycle, and whether deeper trees close timing at 322 MHz waits for
  Vivado.
- A tree ensemble compiled without `mode="fixed"` raises `ConfigError`.

## Verification

`simulate()` on the 100 tree, depth 4 XGBoost regressor sends 10000 random
vectors spanning the whole 16 bit grid of every feature, one per cycle, and all
10000 answers arrive exactly 6 cycles after their inputs, in 10006 cycles, and
match the integer model. The sweep at 50 and 10 percent valid passes all six
runs. The classifier does the same, with 6484 zeros and 3516 ones among the
expected labels. The latency proof from 001 runs on the generated Verilog of
the regressor and closes by k-induction in about 6 seconds, and a latency of 5
fails.

The float import is checked against the frameworks: the regressor's margins
match `XGBRegressor.predict` within 1.3e-4 on 256 inputs, the classifier's
labels match `XGBClassifier.predict` on all of them, a skl2onnx gradient
boosting classifier matches scikit-learn, and a hand built opset 3 model with
one tree per split mode and double tensors matches ONNX Runtime exactly. These
reference outputs are stored next to the fixtures in `tests/fixtures`, so the
tests need neither xgboost nor onnx.

The Hardcaml unit has expect tests on the Lws harness in
`hardware/hardcaml/test/test_otf_trees.ml`: six random ensembles, from one leaf
to 100 trees of depth 4 and 64 of depth 6, with a decision output and with
equality splits, each run 300 cycles at 100, 60 and 5 percent valid from a
random initial state and compared with an OCaml model of the trees every cycle;
every input of a 2 feature, 4 bit ensemble with and without a decision; and a
waveform of two stumps.

## Resources

yosys 0.52 on the generated `rtl/top.v`, out of context, with the commands
from [001](001-fixed-latency.md):

| Model | Flow | LUT | Wide muxes | Carry | Flip flops | Run time |
|---|---|---|---|---|---|---|
| Regressor | `synth_xilinx -family xcup` | 7530 LUT1 to LUT6 | 905 MUXF7, 4 MUXF8 | 1055 CARRY4 | 3041 | 37 s |
| Regressor | `synth_ecp5` | 6202 LUT4 | 409 PFUMX, 7 L6MUX21 | 6877 CCU2C | 3092 | 51 s |
| Classifier | `synth_xilinx -family xcup` | 6780 LUT1 to LUT6 | 926 MUXF7, 3 MUXF8 | 679 CARRY4 | 2747 | 40 s |
| Classifier | `synth_ecp5` | 5330 LUT4 | 370 PFUMX | 4185 CCU2C | 2824 | 30 s |

No DSPs are used. The compiler's estimates are fitted to these runs: 7580 and
6636 LUT on UltraScale+ against 7530 and 6780, and 20165 and 14028 LUT4 sites
on ECP5, counting a CCU2C as two, against 19956 and 13700. On ECP5 yosys builds
each comparison against a constant as a carry chain of 8 CCU2C, which is most
of the ECP5 area, and the regressor at 82 percent of the Icepi Zero's LUTs is
past the 80 percent the compiler allows.

conifer 1.10 installs in a scratch environment and converts the same XGBoost
model. Its VHDL backend states a latency of `1 + max_depth + ceil(log2(trees))`
cycles in `backends/vhdl/writer.py`, 12 for this model against our 6, because
it advances one tree level per cycle and adds pairs. Its resources need Vivado
or Vivado HLS, and writing its project needs Xilinx's `ap_fixed` headers, so
there is no resource comparison yet.

## Done when

One of the two is met.

- [x] An XGBoost model with 100 trees of depth 4 compiles and simulates bit
  exact against the integer model, with a latency of 6 cycles, 18.6 ns at
  322 MHz.
- [ ] We compare latency and resources with conifer on the same model. Latency
  is compared; resources wait for Vivado.

## Open questions

Three are left for the reloadable version and for timing.

- With reloadable trees ([003](003-reloadable-weights.md)) the shape is fixed
  and everything else can be swapped: tree count, maximum depth with smaller
  trees padded to complete ones, feature count and widths. The per node feature
  index, threshold and leaf values live in the register banks, because a
  retrained ensemble changes which feature each node tests. A reloadable feature
  index costs an F to 1 mux per node, and the frozen comparator sharing above
  is lost. The device model should budget that and fall back to frozen trees.
- Feature grids come from the thresholds of one model. A retrained model has
  other thresholds, so reloadable trees need a grid fixed up front, probably
  from calibration data, and the upstream feature logic has to produce codes on
  it.
- Can the comparisons and the leaf select share one cycle at 322 MHz, for a
  latency of 5? That needs Vivado timing.
