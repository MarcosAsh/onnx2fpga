# 009 Operator coverage

Status: partial. Constant folding, live layout ops as wiring or views,
BatchNormalization, LinearRegressor, binary LinearClassifier, Scaler,
Binarizer and the quantized MatMul, Conv and Add forms are built, and all 13
corpus models compile in fixed latency mode for `k26`, simulate bit exact and
track ONNX Runtime; the multiclass classifier and the L1 Normalizer came with
[011](011-reductions-and-softmax.md). Missing: a `Concat` whose pieces are
not one Gemm's worth of the same tensor; `QGemm` and the other contrib
quantized ops; and a float reference for ORT models whose Relu lives only in
an output grid.

## Problem

The importer took Conv, Gemm, MatMul, Relu, MaxPool, Add, Flatten, Reshape,
Squeeze, the QDQ pair, Identity and the tree ensembles, and nothing else. Real
exporters wrap a model in more than that. skl2onnx writes `LinearRegressor`,
`LinearClassifier`, `Scaler` and a `Cast` on the label. PyTorch writes
`BatchNormalization` and shape arithmetic built from `Shape`, `Gather`,
`Unsqueeze` and `Concat`. ONNX Runtime's static quantizer writes
`QLinearMatMul` and `QLinearAdd`. Every one of those models failed on the
first unknown op, although none of them needs hardware we do not have: the
extra ops are either computable at compile time or fold into a dense layer.

## Behaviour

Nothing in this spec adds hardware, and no OCaml changed. Every op below is
evaluated at compile time, becomes wiring, or folds into the weights of a Gemm
or Conv that already exists. Ops that need new arithmetic on live data, such
as Sigmoid, Mul of two live tensors or Softmax, belong to spec 010 and later
work, and still raise `UnsupportedOpError`.

### Constant folding

The importer keeps every initializer as a numpy array and walks the nodes in
order. A node whose inputs are all constants is evaluated in numpy and its
outputs become constants. That covers the shape plumbing exporters emit:
`Constant`, `Shape`, `Size`, `Cast`, `Unsqueeze`, `Squeeze`, `Reshape`,
`Flatten`, `Transpose`, `Slice`, `Gather`, `Concat`, `Expand`,
`ConstantOfShape`, `Range`, `Identity` and `Dropout`, plus the arithmetic that
appears in shape computations (`Add`, `Sub`, `Mul`, `Div`, `Neg`, `Sqrt`,
`Reciprocal`, `Pow`, `MatMul`, `Equal`, `Less`, `Greater`, `Not`, `Where`,
`Max`, `Min`, `Abs`, `Floor`, `Ceil`). `Shape` and `Size` fold even on live data, since every live shape is
static; the batch is 1. A layout op on a quantized constant keeps the
constant's grid.

### Live data

On live data the layout ops are relabellings when the elements stay in the
same order. Inside the compiler a 4D tensor is channel-last (NHWC) while ONNX
means NCHW, so the test is made on the element order, not on the op:
`Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Transpose`, `Slice`, `Gather`,
`Expand`, `Dropout` and `ArrayFeatureExtractor` are evaluated on the element
indices of their input. Same order is wiring: an alias, or a Flatten node
where a map becomes a vector. `Identity` and a `Cast` to a float type are
aliases, and so is a `Cast` of a classifier's label to an integer type.

When the order changes, the result becomes a view, and so does a constant
`Add`, `Sub`, `Mul` or `Div` that broadcasts along the feature axis. A view is
`y[j] = mul[j] * x[index[j]] + add[j]` over the elements of one live tensor x,
and it costs nothing when a Gemm reads it: the Gemm's weight rows are gathered
and scaled, and `add @ W` joins its bias. Layout ops on a view compose into
another view, at any rank. That is how the CNN example's Flatten after a 4
channel map now means what ONNX means: the Flatten node stays as wiring and
the classifier's weight rows are permuted from NCHW to channel-last order.
Before, they were read as if ONNX flattened channel-last, which no exporter
does. A view read by anything other than a Gemm, or a graph output, is built
on its own. One that only scales and shifts each feature becomes a `Mul`,
`Add` or `Sub` from [010](010-lookup-tables-and-elementwise.md), a one cycle
table. Any other becomes a Gemm with one weight per output, which in fixed
latency mode costs one dense layer, and a view that is not a feature vector
by then raises `UnsupportedOpError`.

A constant `Add`, `Sub`, `Mul` or `Div` right after a Gemm or Conv whose output
nothing else reads folds into that layer instead: the weights of each output
channel are scaled and the bias shifted. This is how skl2onnx's `MatMul` then
`Add` becomes one layer.

`Concat` of live feature vectors is built when every input is a constant, a
view of one tensor, the tensor itself, or a Gemm on that tensor that nothing
else reads. The pieces then become one Gemm whose weights are the pieces side
by side, or a view if no piece is a Gemm. In fixed latency mode that keeps the
design a chain of dense layers with one latency. Any other live `Concat`
raises `UnsupportedOpError`.

### BatchNormalization

Inference batch normalization is `y = mul * x + add` per channel, with
`mul = scale / sqrt(var + epsilon)` and `add = B - mean * mul`. After a Gemm
or Conv it folds into that layer like a constant `Mul` and `Add`. On a feature
vector anywhere else it is a view, so a `BatchNorm1d` on a network's raw
inputs folds into the first layer. On a 4D map with no Conv before it, it
becomes a 1 by 1 Conv with a diagonal kernel, which the streaming mode builds.

### ONNX-ML operators

`LinearRegressor` is a Gemm with `coefficients` as its weights, for any number
of targets; its `post_transform` must be NONE. `Scaler` is a view,
`(x - offset) * scale`.

`LinearClassifier` with two classes is decided on the margin, as the tree
classifier of [005](005-tree-ensembles.md) is. The label is the class with the
larger score, and every post transform ONNX defines is monotone, so the
design computes the score difference as a one output Gemm and the label is
one bit, set when that difference is positive. The rescale of that layer is a
multiplier of 1 and a clamp to 0 and 1, so the bit costs no comparator. The
bit is the label's index in `classlabels`. The probabilities output is not
built, and a `Normalizer` or `ZipMap` reading it is dropped with it. With more
than two classes the label is an ArgMax of the scores, from
[011](011-reductions-and-softmax.md). `ZipMap` on a kept tensor is an alias.

`Binarizer` on live data becomes a dense layer with identity weights, a bias
of minus the floor of the threshold's code, and a clamp to 0 and 1, so on
every input code it agrees with `x > threshold`; `tests/test_operators.py`
checks all 256 codes on three feature grids. Its output grid is scale 1 and
zero point 0, so a Gemm after it reads the bits as they are.
`Normalizer` with an L1 norm is built by [011](011-reductions-and-softmax.md),
and the MAX and L2 norms are refused.

### Quantized operators

These are rewritten into the QDQ form the importer already reads, where a
quantized tensor is the real tensor with a grid attached.

- `QLinearMatMul` and `QLinearConv` attach their input and output grids,
  dequantize their weights with the weight grid, and become a MatMul or Conv.
  `QLinearConv`'s int32 bias is dequantized with the product of the input and
  weight scales.
- `QLinearAdd` from ONNX Runtime's contrib domain is an Add between grids. With
  a constant operand after a Gemm it folds into the Gemm's bias, and the Gemm
  takes the Add's output grid, so the intermediate grid ORT keeps before the
  bias is not built.
- `MatMulInteger` and `ConvInteger` compute `(a - a_zero) @ (b - b_zero)` in
  int32. When `a` carries a grid of scale `s`, that equals the real input times
  `(b - b_zero) / s`, which becomes the layer's weights. The `Cast` and the
  constant `Mul` that follow fold into the layer as above.
- `DequantizeLinear` of any integer constant, including an int32 bias, is
  evaluated at compile time; it used to pass int32 biases through undequantized.

`DynamicQuantizeLinear` computes its scale from the data at run time, which a
fixed point design cannot do, and is refused. That rules out ONNX Runtime's
dynamic quantization, whose `MatMulInteger` reads such a tensor.

### Input grids

The dense path quantizes a tensor with one scale. A model that scales its own
inputs, through a `Scaler`, a `BatchNormalization` or a constant `Sub` and
`Div` folded into the first layer, is telling us its features have different
sizes, and one scale for all of them would lose the small ones. When a
calibrated model's input reaches only Gemms and one of them absorbed such a
view, the input gets a grid per feature, asymmetric and fitted to that
feature's range in the calibration data. The matrix unit folds the per
feature scale into its weights before they are quantized and the zero points
into its bias, so the hardware is unchanged. Models without a scaler keep one
grid, so existing designs do not change.

The zero point now joins the bias as `-z @ W` on the weights before rounding,
where it used to be `-z @ round(W)`. The two agree when the model supplied the
weights' grid, since then `W` is already on it, and the QDQ fixtures give the
same biases either way. For a calibrated layer the old form multiplied each
weight's rounding error by the zero point, which for a price near 100 is
several thousand codes: on the scaler pipeline below it put the mean error at
0.29, against 0.025 now, with an output standard deviation of 1.93.

## Errors

An op with no handler raises `UnsupportedOpError` naming it, as before. An op
we know whose case we cannot build raises it with one sentence naming the
case, for example `Normalizer 'Normalizer' has norm MAX, and only L1 is
supported` or `DynamicQuantizeLinear 'q' computes its scale from the data at
run time, which a fixed point design cannot do`.

## Verification

`tests/fixtures/make_corpus.py` exports thirteen small real models with
scikit-learn 1.9, skl2onnx 1.20, XGBoost 3.4, LightGBM 4.7, onnxmltools 1.16,
ONNX Runtime 1.31 and PyTorch 2.12, and stores each `.onnx` in
`tests/fixtures/corpus` with 512 calibration inputs, 256 held out inputs and
ONNX Runtime's outputs on them. The largest file is 13 KB. Features are eight
columns of different sizes; the scaler pipelines and the PyTorch model get
them at the sizes of prices, volumes and ratios, so a price sits near 100 and
a volume near 5000.

`tests/test_corpus.py`, part of `make test-api`, compiles each for `k26` in
fixed latency mode, sends 10000 random vectors over the whole input grid
through Verilator and requires every answer to match the integer model bit for
bit. It then requires the integer model's mean absolute error against the
stored ONNX Runtime outputs to stay under 2.5 percent of the output's standard
deviation, or the labels to agree on at least 98 percent of the held out
inputs. When onnxruntime is importable it also checks the stored outputs are
still what ONNX Runtime computes; the main test environment does not have it,
and the repo does not depend on it.

| Model | Exporter and ops | Compiles | Latency | Bit exact | Against ONNX Runtime |
|---|---|---|---|---|---|
| `linear_regression` | skl2onnx `LinearRegressor` | yes | 3 | yes | mean error 1.9 % of std |
| `ridge` | skl2onnx `LinearRegressor` | yes | 3 | yes | 1.9 % |
| `logistic_binary` | skl2onnx `LinearClassifier` LOGISTIC, `Cast`, `ZipMap` | yes | 3 | yes | labels agree on 98.8 % |
| `scaler_linear` | skl2onnx `Scaler`, `LinearRegressor` | yes | 3 | yes | 1.3 % |
| `scaler_logistic` | skl2onnx `Scaler`, `LinearClassifier`, `Cast`, `ZipMap` | yes | 3 | yes | labels agree on 100 % |
| `mlp_regressor` | skl2onnx `Cast`, `MatMul`, `Add`, `Relu`, `Reshape` | yes | 6 | yes | 1.9 % |
| `torch_bn_mlp` | PyTorch `BatchNormalization` twice, `Gemm`, `Relu` | yes | 6 | yes | 1.3 % |
| `ort_qoperator_mlp` | ORT static `QLinearMatMul`, `QLinearAdd` | yes | 6 | yes | 0.6 %, worst 2 output steps |
| `ort_qdq_mlp` | ORT static QDQ, int32 bias | yes | 6 | yes | 0.6 %, worst 2 output steps |
| `xgb_small` | onnxmltools `TreeEnsembleRegressor`, 20 trees | yes | 5 | yes | 0.25 % |
| `lgbm_small` | onnxmltools `TreeEnsembleRegressor`, `Identity` | yes | 5 | yes | under 0.001 % |
| `logistic_multiclass` | skl2onnx `LinearClassifier` SOFTMAX, `Normalizer`, with [011](011-reductions-and-softmax.md) | yes | 4 | yes | labels agree on 98.8 % |
| `mlp_sigmoid` | skl2onnx `Sigmoid`, a table from [010](010-lookup-tables-and-elementwise.md) | yes | 7 | yes | 1.7 % |

Latencies are at the device model's clock. At `clock_mhz=322.265625` the
logistic regression takes 5 cycles, 15.5 ns, and the PyTorch MLP 13 cycles,
40.3 ns, in 4654 LUT and 17 DSP. The quantized MLPs differ from ONNX Runtime by
up to two steps of the output grid because ORT rounds the product onto the
`QLinearMatMul` output grid before adding the bias and rounds again after,
and we round once.

`tests/test_operators.py`, in `make test-unit`, builds small models with the
repo's own `ModelBuilder` and checks each float graph against numpy's reading
of the ONNX model: PyTorch style shape plumbing that folds to one Gemm, the
Flatten of a map with and without an NHWC transpose before it, BatchNorm on
inputs, after a Gemm and after a Conv, Concat of the input, a Gemm on it and a
Gather of it, `QLinearMatMul` with `QLinearAdd`, `MatMulInteger` with its
`Cast` and `Mul`, `QLinearConv`, `ConvInteger`, the Binarizer on every code,
and the refusals.

## Done when

- [x] Every corpus model that needs no op from spec 010 or later compiles in
  fixed latency mode for `k26` and simulates bit exact.
- [x] The integer model tracks ONNX Runtime within the stated tolerance on
  every compiled corpus model.
- [x] The repo gains no runtime dependency on onnx, onnxruntime or
  scikit-learn.

## Open questions

- A `Concat` whose pieces have different latencies, such as `x` beside
  `relu(W x)`, needs a delay line in the fixed latency top. Is it worth
  building, or is the merged Gemm enough for real models?
- Should every calibrated fixed latency model get per feature input grids, not
  only those with a scaler? It would change the grids of existing designs and
  of the tick to trade path, whose features unit produces codes on one grid.
- ONNX Runtime's quantizer drops a Relu whose output grid already clamps at
  zero, so the float graph of `ort_qoperator_mlp` has no Relu and
  `evaluate()` compares the integer model with the wrong float model. The
  integer model is right, as the table shows; the float reference should take
  the clamp from the grid.
