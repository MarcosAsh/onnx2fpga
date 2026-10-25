# 011 Reductions and softmax

Status: implemented. AveragePool, GlobalAveragePool, the MAX and L2
Normalizer and the multiclass tree classifier are left out on purpose, and
the section on them says why.

## Problem

A classifier ends in a choice. A PyTorch model ends in Softmax or LogSoftmax
and an ArgMax, scikit-learn's multiclass `LinearClassifier` ends in a label and
a probability vector, and a model that pools its features ends in a
ReduceMean. Fixed latency mode ([001](001-fixed-latency.md)) worked lane by
lane, so every one of these was refused, and `logistic_multiclass` was the one
corpus model of [009](009-operator-coverage.md) that did not compile.

## Behaviour

Two new fixed latency units, both Hardcaml only, take a feature vector and
work across its lanes: `otf_reduce` for ArgMax, ArgMin, ReduceMax, ReduceMin,
ReduceSum and ReduceMean, and `otf_softmax` for Softmax, LogSoftmax and an L1
Normalizer. Every operator here works on the feature axis of a `(1, N)`
vector, the only live tensors fixed latency mode has.

### Trees

ArgMax and ArgMin are a binary tree of compare and select over the lanes. Each
node keeps the larger (or smaller) of its two inputs together with its index,
and on a tie it keeps the left one, so the tree returns the first index of the
maximum, which is what ONNX means by `select_last_index=0`. With
`select_last_index=1` a tie keeps the right one and the tree returns the last.
The output is an index code: an unsigned integer of `max(1, ceil(log2 N))`
bits on a grid of scale 1 and zero point 0, so `predict()` returns the index
itself. `keepdims` changes the shape of the output, `(1, 1)` or `(1,)`, and
nothing in the hardware. Only a graph output, or a Cast of one, may read the
index.

When the ArgMax is the only reader of a Gemm, the Gemm gives its scores in 11
bits instead of 8, on a grid 8 times finer: the rescale keeps its multipliers
and shifts 3 bits less. Two classes whose scores land on one int8 code would
otherwise tie and go to the lower index. On `logistic_multiclass` this moves
the labels from 98.4 to 98.8 percent agreement with ONNX Runtime, and on 4000
inputs drawn like the calibration data from 98.6 to 98.8 percent agreement
with the float model. More bits gain nothing measurable there, because the
inputs are 8 bit and the rest of the disagreement is theirs. Three bits is
the most for which the `k26` model still fits two compare levels into one
322 MHz cycle.

ReduceMax and ReduceMin are the same tree without the index, and their output
keeps the input's grid, since the largest of some codes is a code on the same
grid. ReduceSum and ReduceMean are an adder tree over the codes and the
constant `-N * zero_point`, registered, then the rescale of
[the rescale contract](../hardware/requant.md) onto the output grid.
ReduceMean is ReduceSum with `1 / N` folded into the rescale multiplier, so the
mean costs nothing the sum does not.

The axes come from the `axes` attribute or, from opset 13 for ReduceSum and 18
for the rest, from a constant second input. No axes means every axis, which on
a `(1, N)` vector is the feature axis again. An axis list without the feature
axis is refused, and so is `noop_with_empty_axes`.

### Softmax

Softmax over the lanes is built in four steps, each registered:

1. A max tree, the one ReduceMax uses, gives `m` while the codes wait for it.
2. Each lane looks up `E[m - x]` in a constant table of 256 entries, where
   `E[d] = round(2^12 * exp(-s * d))` and `s` is the input scale. The
   difference never goes below zero, so the table has no negative side, and
   `E[0]` is exactly `2^12`. It is the Exp table of
   [010](010-lookup-tables-and-elementwise.md) with the subtraction of the
   maximum folded into its index and 13 bit entries in place of int8 codes.
3. An adder tree sums the lanes into `S`, which lies between `2^12` and
   `N * 2^12`.
4. The division `p = E / S` onto the output grid.

For the division we normalise `S` and look up a reciprocal. A leading one
detector finds the highest set bit `k` of `S`, the 10 bits below it are a
mantissa `m`, and one table of 1024 entries gives `R[m] = round(K * 2^P / (1 +
(m + 0.5) / 1024))`, where `K` is one over the output scale and `P` is the
largest shift that keeps every entry in an 18 bit signed multiplier. Each
lane's `E` is shifted left by `kmax - k`, which loses nothing, and the rescale
contract does the rest with the multiplier `R[m]` and the constant shift
`P + kmax`. The reciprocal table is shared by every lane, the shift is a small
mux per lane, and the multiply is the DSP48E2 every rescale already takes. The
worst relative error of the reciprocal is the mantissa's, `2^-11`, whatever
`N` is.

We also considered a restoring divider per lane. It gives the correctly
rounded quotient, but the `k26` delay model fits one 23 bit subtract and
select per cycle at 322 MHz, so 8 quotient bits cost 8 cycles, where the
normalisation, the table and the multiply cost 5 at that clock with the
rounding onto the output grid included. The exponential is rounded to 12
bits anyway, so an exact quotient of rounded numerators buys little.

Against float softmax rounded onto the same int8 grid, on 4000 vectors of
codes drawn with a spread of 50 codes, the unit is never more than one code
off, and it gives exactly the rounded float answer on 98.7 percent of outputs
for 3 lanes, 99.5 percent for 10 and 99.7 percent for 16. Without a clock it
takes 5 cycles; at 322.265625 MHz on `k26` it takes 8 cycles for 3 lanes and
10 for 10 or 16.

On a calibrated model the output grid of a Softmax is fixed at scale `1/127`
and zero point 0, since its range is known to be 0 to 1, and a model's own
grid is kept when it states one.

LogSoftmax is `s * (x - m) - ln(S / 2^12)`. The same unit computes it with a
second lane table, `B[d] = round(-d * s / so * 2^6)` for an output scale `so`,
and replaces the reciprocal with a logarithm: `ln(S / 2^12)` is the position
of the leading one times `ln 2` plus the logarithm of the mantissa, each from
a table, and added. Each lane subtracts it from `B` and the rescale rounds
away the 6 fraction bits. It is within one code of the rounded float answer
everywhere and equal on 99.2 percent of outputs for 10 lanes.

A Softmax or LogSoftmax whose only reader is an ArgMax or ArgMin is dropped at
import, since the exponential is monotone and the ArgMax of a softmax is the
ArgMax of its input. A classifier's label never needs the probabilities for
the same reason.

### Normalizer

`Normalizer` with `norm="L1"` divides each lane by the sum of the absolute
values. Read from a Softmax it is the identity, since the probabilities sum
to 1 already, and the import makes it an alias. On any other live vector it
is the Softmax unit without the max tree, with `|x - z|` as the summed table
and `x - z` as the numerator. That is the same divider: the sum now ranges
from 0 to 255 times `N`, and the normalisation and the shift of the numerator
handle any range at the same precision. A sum of 0 gives the output zero
point, as scikit-learn leaves a zero vector alone. The output grid is `1/127`,
since the result lies between -1 and 1. It is within one code of the rounded
float answer and equal on 99.2 percent of outputs for 6 lanes.

### Classifiers

A `LinearClassifier` with three or more classes now builds its label as the
binary one does: a Gemm gives the class scores and an ArgMax picks one, and
the label is the class's index in `classlabels`. Every post transform ONNX
defines is monotone, so NONE, LOGISTIC and SOFTMAX give the same label and
none of them is built. The probabilities output is not built, and a
`Normalizer` or `ZipMap` reading it is dropped with it, as for two classes.
Before this spec a multiclass classifier with post transform NONE gave its
scores and dropped the label; it now gives the label, ONNX's first output.

`logistic_multiclass` compiles for `k26` into a 3 class Gemm and an ArgMax,
4 cycles without a clock and 7 at 322.265625 MHz, 21.7 ns, in 862 LUT and 3
DSP by the compiler's count. It simulates bit exact on 10000 random vectors,
and its labels agree with ONNX Runtime's on 98.8 percent of the 256 held out
inputs, 3 of them wrong, against the 98 percent the corpus test asks for.

### Left out

AveragePool and GlobalAveragePool are not built. In fixed latency mode the
only 4D tensor is the model's input, and pooling it is a constant linear map
that a Gemm reading it would have to absorb, which the view machinery of 009
cannot express since every output reads many inputs. In streaming mode
neither maps onto a unit we have without writing a new one. A multiclass
tree classifier needs one tree sum per class and `otf_trees` has one, so it
is still refused; it is a new unit, not a small step. The MAX Normalizer
needs a max tree in place of the sum, and L2 a square root, and both are
refused.

## Pipelining

The register stages are set by `clock_mhz` through the delay model, the way
[002](002-pipelined-matvec.md) sets a dense layer's. `otf_reduce` has two
knobs: `TREE_STAGES`, cuts spread evenly over the levels of the comparison or
adder tree, and for the sum `RESCALE_STAGES`, the cuts inside `Otf_requant`.
`otf_softmax` has five: `MAX_STAGES` in the max tree, `TABLE_STAGES` between
the subtraction and the lane tables, `SUM_STAGES` in the adder tree,
`NORMALIZE_STAGES` between the leading one detector and the scale table, and
`RESCALE_STAGES`.

Predicted by the `k26` model at 322.265625 MHz, a period of 3.10 ns, for one
unit after a dense layer on 10 lanes, unverified until a Vivado run:

| Operator | Unit | Cycles without a clock | Cycles at 322 MHz | Stages | Slowest path at 322 MHz | Without stages |
|---|---|---|---|---|---|---|
| ArgMax, ArgMin | `otf_reduce` | 1 | 2 | 1 | 3.08 ns, the tree | 5.96 ns |
| ReduceMax, ReduceMin | `otf_reduce` | 1 | 2 | 1 | 2.96 ns, the tree | 5.74 ns |
| ReduceSum, ReduceMean | `otf_reduce` | 2 | 5 | 1/2 | 2.02 ns, the tree | 5.78 ns |
| Softmax | `otf_softmax` | 5 | 10 | 1/0/1/1/2 | 2.96 ns, the max tree | 6.38 ns |
| LogSoftmax | `otf_softmax` | 5 | 9 | 1/0/1/1/1 | 3.06 ns, the table | 5.74 ns |
| Normalizer L1 | `otf_softmax` | 4 | 8 | 0/1/1/2 | 2.72 ns, the normalisation | 6.42 ns |

Stages are tree/rescale for `otf_reduce` and max/table/sum/normalize/rescale
for `otf_softmax`, which has no max knob for the Normalizer. With 3 lanes the
arg and extreme ops take 1 cycle at 322 MHz, the sum 4, Softmax 8, LogSoftmax
7 and the Normalizer 7; with 16 lanes the numbers are those of 10. Without
stages every unit is limited by its rescale's multiply, rounding and clamp, or
by its tree; with them the slowest step anywhere is LogSoftmax's lane table,
a 9 bit subtraction and two LUT levels with the fanout of 27 table bits.

## Streaming mode

Both units are fixed latency only. Streaming mode refuses these operators with
`ConfigError` before calibration, as it refuses the units of 010, and the
SystemVerilog library stays frozen, as [001](001-fixed-latency.md) says.

## API

Nothing new to call. A model with these operators compiles with
`mode="fixed"`, and the summary shows a row per unit with `argmax`, `max`,
`sum`, `softmax`, `logsoftmax` or `normalizer` in the arithmetic column and
its stages. An ArgMax unit's row reads `10 -> 1`.

## Errors

Each is one sentence and names the node.

- An operator here in streaming mode: `mode='stream' cannot build op (Softmax);
  it needs mode='fixed'`.
- An axis other than the feature axis: `ArgMax 'op' works on axis 0, and only
  the feature axis of a (1, N) vector is supported`, and a tensor that is not
  such a vector says so too.
- `Normalizer 'op' has norm L2, and only L1 is supported`.
- `ReduceSum 'op' has noop_with_empty_axes, which is not supported`.
- An index read by another unit raises `ConfigError`: `am (ArgMax) gives an
  index, which only a graph output can take`.

## Verification

`tests/test_reductions.py`, in `make test-api`, checks each unit's integer
model against float. Softmax on 3, 10 and 16 lanes, LogSoftmax and the
Normalizer must be within one code of the float answer rounded onto the output
grid on 4000 vectors, and Softmax and LogSoftmax equal on at least 98 percent;
the measured shares are above. ArgMax and ArgMin, with and without
`select_last_index`, match numpy on codes squeezed to five values so that ties
are common. ReduceMax and ReduceMin match exactly, and ReduceSum and ReduceMean
within a code. The import tests cover the dropped Softmax, the Normalizer
alias, axes from the attribute, from an input and from neither, the three
class classifier and the refusals.

Two designs compile for `k26` and simulate bit exact under Verilator, the
first with and without a clock, each on 10000 random vectors, one per cycle,
every answer exactly `latency` cycles later:

- An ensemble: two dense heads on one input, a Softmax on each, an Add of the
  probabilities and an ArgMax. 11 cycles without a clock, and 22 at
  322.265625 MHz, 68.3 ns, with 325.8 MHz predicted.
- A pooled model at 322.265625 MHz: a dense layer, then LogSoftmax and
  ReduceMax, the L1 Normalizer and ReduceSum, ReduceMin and ReduceMean (axes
  as an opset 18 input), joined by Sub and Add into one number. 27 cycles,
  83.8 ns, 323.9 MHz predicted.

The latency proof from [001](001-fixed-latency.md) runs on the generated
Verilog of the ensemble, where two softmax units meet at the Add, and closes
at 11 cycles; the same check at 10 fails. The
corpus test, `tests/test_corpus.py`, now compiles and simulates all 13 models.

The Hardcaml units have expect tests on the Lws harness, from a random
initial state, on a shared bench that checks every answer arrives exactly
`latency` cycles after its input. `test_otf_reduce.ml` runs ArgMax and ArgMin
first and last of a tie on 3 bit codes, Max on 16 lanes, Min on one lane and
Sum with its rescale, each without stages and with the most it takes, 200
cycles at 100 and 40 percent valid, against an OCaml model of
`ReduceUnit.execute`, and a waveform shows an argmax over four lanes with a
tie. `test_otf_softmax.ml` does the same for Softmax on 5 and 2 lanes,
LogSoftmax and L1 against a model of `SoftmaxUnit.execute`, with codes over
the whole range and within one code of the L1 zero point, starting with a
vector whose L1 sum is 0.

## Resources

yosys 0.52 `synth_xilinx -family xcup` on a Softmax over 10 lanes at
322.265625 MHz gives 970 LUTs, 101 MUXF7, 115 CARRY4, 865 flip flops and 130
SRL16E, 10 DSP48E2 and one RAMB18E2, which yosys chose for the reciprocal
table since a register follows it. The compiler counts 2504 LUT and 10 DSP: it
prices each lane's table at a LUT6 per 64 entries of each bit, and the long
run of zeros at the end of the exponential costs yosys nothing. An ArgMax over 10 int8
lanes at the same clock comes to 125 LUTs, 9 CARRY4 and 35 flip flops, where
the compiler counts 180.

## Done when

All five are met.

- [x] Every operator above compiles in fixed mode, and a model using ArgMax
  and Softmax simulates bit exact against the integer model.
- [x] `logistic_multiclass` compiles, simulates bit exact and its labels agree
  with ONNX Runtime's on at least 98 percent of the held out inputs.
- [x] Both units have Hardcaml expect tests against an OCaml model.
- [x] The latency of a design with both units is proved on the generated
  Verilog.
- [x] The cycles each operator costs with and without `clock_mhz=322.265625`
  on `k26`, and the slowest path, are recorded here.

## Open questions

- The rescale after the reciprocal costs 2 of Softmax's 10 cycles at 322 MHz.
  Folding the leading one detector and the table into the cycle of the sum
  tree's last level, where the paths are short, would save one; is it worth
  a less regular unit?
- yosys put the reciprocal table in block RAM. Vivado may do the same, and a
  block RAM read is slower than the LUT table the delay model prices; the
  normalize knob may need another cut once a timing report exists.
- A multiclass tree classifier would reuse the ArgMax as it is; it needs
  `otf_trees` to sum each class's trees apart.
