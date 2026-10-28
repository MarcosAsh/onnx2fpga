# 009 Operator coverage

Status: implemented. Constant folding, live layout ops as wiring or views,
BatchNormalization, LinearRegressor, binary LinearClassifier, Scaler,
Binarizer and ONNX Runtime's QDQ and QOperator forms are built, and all 20
corpus models compile in fixed latency mode for `k26`, simulate bit exact and
track ONNX Runtime; the multiclass classifier and the L1 Normalizer came with
[011](011-reductions-and-softmax.md). A `Concat` of several live tensors
builds in fixed latency mode, its earlier pieces waiting in registers.

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

Apart from a `Concat` of several live tensors, nothing in this spec adds
hardware. Every other op below is evaluated at compile time, becomes wiring,
or folds into the weights of a Gemm or Conv that already exists. Ops that
need new arithmetic on live data, such as Sigmoid, Mul of two live tensors or
Softmax, belong to [010](010-lookup-tables-and-elementwise.md) and later
specs.

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
design a chain of dense layers with one latency.

Any other live `Concat` joins several tensors, such as a wide and deep model's
raw features beside its hidden layer, `concat(x, relu(W x))`. Its pieces
arrive at different cycles, so in fixed latency mode it becomes an
`otf_concat` unit, and the fixed latency top already delays a unit's earlier
inputs in plain registers until its latest one arrives, as it does for the
elementwise units of 010. The pieces also sit on different grids, and the
output gets one grid from calibration, so each lane goes through a table from
its piece's grid onto the output's, the `otf_table` of 010 behind a join of
its inputs, lowest lane first. That costs one cycle. A piece on a grid per
feature, such as a model input with a scaler, gets a table per feature.
Streaming mode refuses the unit with `ConfigError`, and a constant beside
pieces of more than one live tensor is refused at import.

On a model with 16 raw features, a Sigmoid of them and an 8 unit hidden
layer joined into 40 lanes and read by one output, compiled for `k26` at
322.265625 MHz, the hidden layer takes 7 cycles, so the features wait 7 and
the Sigmoid 6. The design takes 15 cycles, 46.5 ns, and the join costs 1280
LUTs of tables and 1985 flip flops, most of them the 1664 that hold the early
pieces. The integer model's mean error is 2.5 percent of the output's
standard deviation, it simulates bit exact on 10000 vectors, and the latency
proof of [001](001-fixed-latency.md) closes at 15 cycles and fails at 14.

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
`Normalizer` is built by [011](011-reductions-and-softmax.md) with any of
its three norms.

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
- `QGemm` is a Gemm whose weights are dequantized along their output axis,
  which `transB` decides. Its int32 bias is on the grid
  `alpha * a_scale * b_scale`, as ORT defines it, and without an output grid
  its output is float.
- `QLinearMul`, `QLinearSigmoid`, `QLinearLeakyRelu`, `QLinearSoftmax` and
  `QLinearReduceMean` become their float op between their grids, so the
  units of [010](010-lookup-tables-and-elementwise.md) and
  [011](011-reductions-and-softmax.md) build them. A `QLinearMul` with a
  constant operand folds like a constant `Mul`. `QLinearConcat` gives its
  output grid first and then a tensor, scale and zero point per piece, and is
  a `Concat`. `QLinearAveragePool` and
  `QLinearGlobalAveragePool` are refused, as their float forms are.
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
case, for example `TreeEnsemble 'op' has a BRANCH_MEMBER node, which is not
supported` or `DynamicQuantizeLinear 'q' computes its scale from the data at
run time, which a fixed point design cannot do`.

## Verification

`tests/fixtures/make_corpus.py` exports 22 small real models with
scikit-learn 1.9, skl2onnx 1.20, XGBoost 3.4, LightGBM 4.7, onnxmltools 1.16,
ONNX Runtime 1.31, PyTorch 2.12, qonnx 1.0 and Brevitas 0.13, and stores each `.onnx` in
`tests/fixtures/corpus` with 512 calibration inputs, 256 held out inputs and
ONNX Runtime's outputs on them, or qonnx's for `qonnx_mlp` and `brevitas_mlp`. The largest file is 34 KB. Features are eight
columns of different sizes; the scaler pipelines and the PyTorch model get
them at the sizes of prices, volumes and ratios, so a price sits near 100 and
a volume near 5000.

`tests/test_corpus.py`, part of `make test-api`, compiles each for `k26` in
fixed latency mode, sends 10000 random vectors over the whole input grid
through Verilator and requires every answer to match the integer model bit for
bit. It then requires the integer model's mean absolute error against the
stored ONNX Runtime outputs to stay under 2.5 percent of the output's standard
deviation, or the labels to agree on at least 98 percent of the held out
inputs. ONNX Runtime's quantizer drops a Relu whose output grid already clamps
at zero, so the float graph of both ORT MLPs has no Relu. The float reference
that `evaluate()` uses therefore clips every tensor whose grid the model
states to that grid's range, as a QDQ pair saturates in ORT. Before, it sat 70
percent of the output's standard deviation away from ORT on those two models;
now it sits 1.7 percent away, and the test requires every ORT model's float
reference to stay under the same 2.5 percent. When onnxruntime is importable
it also checks the stored outputs are still what ONNX Runtime computes; the
main test environment does not have it, and the repo does not depend on it.

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
| `ort_qoperator_gemm` | ORT static `QGemm`, `QLinearSigmoid`, `QLinearSoftmax` | yes | 12 | yes | 0.01 %, worst 1 output step |
| `xgb_small` | onnxmltools `TreeEnsembleRegressor`, 20 trees | yes | 5 | yes | 0.25 % |
| `lgbm_small` | onnxmltools `TreeEnsembleRegressor`, `Identity` | yes | 5 | yes | under 0.001 % |
| `logistic_multiclass` | skl2onnx `LinearClassifier` SOFTMAX, `Normalizer`, with [011](011-reductions-and-softmax.md) | yes | 4 | yes | labels agree on 98.8 % |
| `mlp_sigmoid` | skl2onnx `Sigmoid`, a table from [010](010-lookup-tables-and-elementwise.md) | yes | 7 | yes | 1.7 % |
| `xgb_multiclass` | onnxmltools `TreeEnsembleClassifier` SOFTMAX, 3 classes, with [012](012-multiclass-trees.md) | yes | 6 | yes | labels agree on 100 % |
| `lgbm_multiclass` | onnxmltools `TreeEnsembleClassifier` SOFTMAX, `Identity`, `Cast`, `Mul` | yes | 6 | yes | labels agree on 100 % |
| `forest_multiclass` | skl2onnx random forest, every leaf scoring 3 classes | yes | 5 | yes | labels agree on 100 % |
| `forest_binary` | skl2onnx random forest, one score | yes | 4 | yes | labels agree on 100 % |
| `xgb_small_v5` | `xgb_small` rewritten as ai.onnx.ml 5 `TreeEnsemble` | yes | 5 | yes | 0.25 % |
| `normalizer_logistic` | skl2onnx L2 `Normalizer`, `LinearClassifier`, with [011](011-reductions-and-softmax.md) | yes | 8 | yes | labels agree on 99.2 % |
| `qonnx_mlp` | QONNX `Quant` at 4 and 8 bits, `MatMul`, `Add`, `Relu`, with [014](014-qonnx.md) | yes | 9 | yes | equal to qonnx's outputs |
| `brevitas_mlp` | Brevitas `export_qonnx`, `Quant`, `Gemm`, `Relu`, a float output | yes | 9 | yes | equal to qonnx's outputs |

Latencies are at the device model's clock. At `clock_mhz=322.265625` the
logistic regression takes 5 cycles, 15.5 ns, and the PyTorch MLP 13 cycles,
40.3 ns, in 4654 LUT and 17 DSP. The quantized MLPs differ from ONNX Runtime by
up to two steps of the output grid because ORT rounds the product onto the
`QLinearMatMul` output grid before adding the bias and rounds again after,
and we round once. `QGemm` adds its bias before it rounds, as we do, and the
three class `ort_qoperator_gemm` is never more than one step of its 1/256
output grid from ORT, with the most likely class the same on every held out
input. It takes 22 cycles at 322.265625 MHz, 68.3 ns, in 7536 LUT and 22 DSP.

`tests/test_operators.py`, in `make test-unit`, builds small models with the
repo's own `ModelBuilder` and checks each float graph against numpy's reading
of the ONNX model: PyTorch style shape plumbing that folds to one Gemm, the
Flatten of a map with and without an NHWC transpose before it, BatchNorm on
inputs, after a Gemm and after a Conv, Concat of the input, a Gemm on it and a
Gather of it, the input beside a Relu of a Gemm on it, `QLinearMatMul` with
`QLinearAdd`, `MatMulInteger` with its `Cast` and `Mul`, `QLinearConv`,
`ConvInteger`, `QGemm` with `transB` and `alpha`, `QLinearMul`,
`QLinearSigmoid`, `QLinearLeakyRelu`, `QLinearSoftmax`, `QLinearReduceMean`,
`QLinearConcat`, the Binarizer on every code, and the refusals.

`tests/test_elementwise.py`, in `make test-api`, compiles the wide and deep
model of the Concat section and checks the waits, the latency, the error
against float, the simulation, the proof and the refusal in streaming mode.

## Done when

- [x] Every corpus model that needs no op from spec 010 or later compiles in
  fixed latency mode for `k26` and simulates bit exact.
- [x] The integer model tracks ONNX Runtime within the stated tolerance on
  every compiled corpus model.
- [x] The repo gains no runtime dependency on onnx, onnxruntime or
  scikit-learn.

## Open questions

- The join's tables cost a cycle. When a Gemm reads the join, as in the wide
  and deep model, the move onto one grid could fold into that Gemm's weights
  per feature, as the input grids above do, and a table that feeds a piece
  could fold into the join's; both would save the cycle.
- Should every calibrated fixed latency model get per feature input grids, not
  only those with a scaler? It would change the grids of existing designs and
  of the tick to trade path, whose features unit produces codes on one grid.
