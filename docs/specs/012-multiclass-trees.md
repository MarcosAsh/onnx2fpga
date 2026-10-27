# 012 Multiclass tree classifiers

Status: implemented. Multi target regressors are still refused.

## Problem

A trading model that says up, flat or down is a three class classifier, and
the usual way to train one on tabular features is XGBoost, LightGBM or a
random forest. [005](005-tree-ensembles.md) built a tree regressor and a
binary classifier decided on the sign of one score, and refused every
classifier with more classes, as [011](011-reductions-and-softmax.md) noted:
`otf_trees` had one sum, and a multiclass label needs one per class.

## Behaviour

A `TreeEnsembleClassifier` with K classes keeps one score per class. Its
label is the class with the largest score, the first of a tie, which is what
ONNX Runtime computes: it takes the label from the raw scores before the post
transform, and every transform ONNX defines (NONE, LOGISTIC, SOFTMAX,
SOFTMAX_ZERO and PROBIT) is monotone anyway. None of them is built, and the
probabilities output is dropped as it is for the other classifiers. Every
class must have a score; an ensemble that scores only some of its classes is
refused. A binary classifier with one score is still decided on its sign as
in 005. One that scores both of its two classes is refused: ONNX Runtime does
not label it by the larger score. A random forest of ours, cut to two classes
with both scored, came out as class 1 on all 512 calibration inputs, although
class 0 had the larger score on 37 percent of them. No exporter we tried
writes that form: skl2onnx's random forest and onnxmltools' XGBoost model
score one class of a binary classifier.

The float model stores a row of leaf values per class. XGBoost and LightGBM
give each tree one class, so the row has one nonzero entry, while a
scikit-learn random forest gives every leaf a probability for every class.
At lowering each class gets its own copy of every tree that scores it, with
that class's leaf codes. The splits are shared, so a forest of 10 trees over
3 classes builds 30 leaf selects and still one comparator per distinct split.

`otf_trees` now takes a list of scores, each its trees and a bias. With one
score it is the unit of 005, cycle for cycle. With more, every score has its
own adder tree, all as deep as the one with the most trees, and a compare and
select tree over the scores carries the index of the larger at each node,
keeping the left on a tie. The output is an index of `ceil(log2 K)` bits on a
grid of scale 1 and zero point 0, so `predict()` returns the class's index in
`classlabels`, as for the linear classifier, and a `Cast` of it to an integer
type is allowed.

The leaf codes and feature grids are unchanged. A score's width is the
smallest signed type that holds the sum of its trees' extreme leaves plus its
bias. Finding it turned up a bug from 005: when that range never went
negative, the compiler took the unsigned width and made it signed, one bit
short. The regressors in the tests always had negative leaves, so nothing
showed it until the random forest, whose probability leaves are all positive:
its scores reach 315989, 20 bits signed, and the unit was built with 19, so
242 of 10000 simulated answers wrapped.

## Pipelining

The pick tree is a new knob, `PICK_STAGES`, cuts spread over its
`ceil(log2 K)` levels, after the register that always closes it. The delay
model prices a level as one compare of the score's width and a select, as the
ArgMax of 011 does. At 322.265625 MHz on `k26` the three class models take
one cut, which costs a cycle. The summary shows the stages as
select/sum/pick and names the pick tree.

| Model | Trees, depth | Comparators | Cycles | At 322 MHz | Predicted fmax there | LUT |
|---|---|---|---|---|---|---|
| `xgb_multiclass` | 60, 3 | 289 | 6 | 7, 21.7 ns | 460.8 MHz | 2788 |
| `lgbm_multiclass` | 60, 6 | 224 | 6 | 7, 21.7 ns | 413.6 MHz | 2690 |
| `forest_multiclass` | 30, 4 | 144 | 5 | 6, 18.6 ns | 460.8 MHz | 2282 |
| `forest_binary` | 10, 4 | 146 | 4 | 4, 12.4 ns | 464.8 MHz | 747 |

The forest's 30 trees are its 10 trees copied for each class. The binary
forest scores one class, so it is a 005 design. LUT counts are the
compiler's, which now adds a compare and two selects a bit at each of the
K - 1 nodes of the pick tree, unfitted.

## Errors

Each is one sentence and names the node.

- `TreeEnsembleClassifier 'op' has 3 classes and scores 2 of them, and every
  class needs a score`.
- `TreeEnsembleClassifier 'op' has 4 base values for 3 scores`.
- `TreeEnsembleClassifier 'op' scores both of its 2 classes, and ONNX Runtime
  does not label that by the larger score`.
- A binary classifier with one score and a SOFTMAX transform is refused as
  before.

## Verification

`tests/fixtures/make_corpus.py` adds four models trained on the corpus's
market features with the target cut into thirds, or halves for the binary
forest: XGBoost 3.4 with 20 rounds of depth 3, LightGBM 4.7 with 20 rounds of
8 leaves, and scikit-learn 1.9 random forests of 10 trees of depth 4 exported
by skl2onnx 1.20. `tests/test_corpus.py` compiles each for `k26`, simulates
10000 random vectors bit exact under Verilator, and requires 98 percent label
agreement with ONNX Runtime; all four agree on every one of the 256 held out
inputs.

`tests/test_trees.py` checks that the float import of the three multiclass
models gives ONNX Runtime's labels on the held out inputs, that the integer
label agrees with the float one on 99 percent of 4000 raw inputs, that no
score can overflow its width, and the refusal. `xgb_multiclass` at
322.265625 MHz builds with one pick stage in 7 cycles, simulates bit exact
with all three labels among the answers, and the latency proof of
[001](001-fixed-latency.md) closes at 7 cycles and fails at 6.

The Hardcaml unit has expect tests against an OCaml model of the scores and
the first largest index: two, three and ten classes at 100, 60 and 5 percent
valid from a random initial state, a ten class ensemble with 0 to 3 pick
stages, each adding one cycle and keeping every answer, and every input of a
small classifier with two bit leaves for 2, 3 and 5 classes, where 105, 48
and 172 of the 256 inputs tie. The tests of 005 give the same answers and
the same waveform as before.

## Resources

yosys 0.52 `synth_xilinx -family xcup` on `xgb_multiclass` at 322.265625 MHz
gives 2910 LUTs, 562 CARRY4, 70 MUXF7 and 2016 flip flops, against the
compiler's 2788 LUTs.

## Done when

All four are met.

- [x] XGBoost, LightGBM and random forest classifiers with three classes
  compile in fixed mode for `k26`, simulate bit exact and agree with ONNX
  Runtime on at least 98 percent of held out labels.
- [x] The Hardcaml unit has expect tests against an OCaml model, ties
  included.
- [x] The latency of a multiclass design is proved on the generated Verilog.
- [x] The cycles with and without `clock_mhz=322.265625` on `k26` are
  recorded here.

## Open questions

- A multi target regressor needs the scores as a vector output, which the
  unit could give by skipping the pick tree; nothing in the corpus needs it
  yet.
- The pick tree compares full width scores. Comparing only the top bits after
  the sum would shorten it, at the cost of ties the rounding creates.
