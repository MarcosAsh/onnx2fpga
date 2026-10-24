# 005 Tree ensembles

Status: planned

## Problem

Gradient boosted trees are among the most used models on tabular, market style
data, and we cannot compile a single tree today. XGBoost and LightGBM export to
ONNX's tree ensemble operators in the `ai.onnx.ml` domain, so the front door is
already the right shape.

## Behaviour

A tree ensemble becomes a fixed latency block. Every tree is evaluated in
parallel: each node is one comparison of a feature against a threshold, all
comparisons happen in the first cycle, and each tree's leaf is picked by a
small mux tree. The leaf values are summed in an adder tree. Latency is a
handful of cycles and does not depend on the input.

Thresholds and leaf values are quantized to the same fixed point grid as the
features, and the integer model is the golden reference, exactly as for neural
networks.

With reloadable weights ([003](003-reloadable-weights.md)) the shape is fixed
and everything else can be swapped. The shape is the tree count, the maximum
depth (smaller trees are padded to complete ones), the feature count and the
widths. The per node feature index, threshold and leaf values live in the
register banks. That matters because a retrained ensemble changes which feature
each node tests, so keeping the structure fixed would not survive the first
retrain. A reloadable feature index costs an F to 1 mux per node, and the
device model budgets it and falls back to frozen trees when it does not fit.

We write our own generator instead of wrapping conifer. conifer has no Verilog
or Hardcaml output, its fixed latency backends cannot reload, and its ONNX
importer reads only TreeEnsembleClassifier.

## API

```python
acc = otf.load("gbdt.onnx")        # exported with onnxmltools or skl2onnx
acc.compile(device="k26", mode="fixed", clock_mhz=322)
acc.summary()                      # trees, depth, comparators, latency
```

## Errors

- A tree deeper than the device or clock allows: `BudgetError` with the depth.
- An ensemble operator option we do not support, such as a post transform
  other than none or logistic: `UnsupportedOpError` naming it.

## Done when

- An XGBoost model with 100 trees of depth 4 compiles and simulates bit exact
  against the integer model, with a latency of about 6 cycles, 19 ns at 322 MHz.
- We compare latency and resources with conifer, the hls4ml team's tree tool,
  on the same model.

## Open questions

- Import order is settled: TreeEnsembleRegressor and TreeEnsembleClassifier v1
  and v3 first, since onnxmltools 1.16 and skl2onnx still emit those, then
  TreeEnsemble v5. Splits LEQ, LT, GTE, GT and EQ; post transforms NONE and
  LOGISTIC, deciding on the margin since the logistic is monotone.
