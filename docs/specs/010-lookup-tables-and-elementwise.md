# 010 Lookup tables and elementwise operators

Status: implemented. Streaming mode refuses these operators by design, and a
table right after a dense layer still costs a cycle of its own; both are below.

## Problem

Fixed latency mode ([001](001-fixed-latency.md)) built dense layers and tree
ensembles, and the only activation it knew was a Relu fused into the dense
layer before it. Real models put a Sigmoid, a Tanh or a Gelu between layers,
add a skip connection, gate a feature by multiplying it with another, or clip
an output. Every one of those raised `ConfigError`.

## Behaviour

Two new fixed latency units cover them: `otf_table` for anything with one live
input and `otf_eltwise` for two or three. Both are Hardcaml only.

### Lookup tables

Every activation in the compiler is an int8 code, so a function of one
activation has 256 possible inputs. The compiler takes each code, turns it into
its real value on the input grid, evaluates the float function there, and
rounds the result onto the output grid, clamped to int8. The hardware is a
constant table indexed by the code, one per output element, followed by a
register: one cycle, whatever the function. NaN, which Log and Sqrt return
below zero, becomes the output zero point, and infinities clamp. The tables
are muxes over constants, as `Sv_int.table` explains: Hardcaml compares
memories by size alone when it shares modules, so two units with different
contents in a `rom` would be built as one.

Elements that use the same function share one table. A constant that differs
per element, such as a PRelu slope per channel or an Add of a constant vector,
gives one table per distinct value, and each element picks its own. Any
operator with exactly one live input becomes a table this way, so the unit
covers Sigmoid, Tanh, HardSigmoid, HardSwish, Elu, Selu, Gelu (exact and
tanh), Softplus, Softsign, Exp, Log, Sqrt, Reciprocal, Erf, Mish, LeakyRelu,
PRelu with a constant slope, a Relu that could not be fused, Neg, Abs, Not and
Clip with constant bounds. Add, Sub, Mul, Div, Max, Min, Equal, Less, Greater,
LessOrEqual, GreaterOrEqual, And and Or with one constant operand are tables
too, and so is Mul of a tensor by itself, and Where with one live operand.

Two tables in a row, where the first feeds nothing else, are composed into one
table at compile time. Sigmoid, then Mul by 2, then Tanh costs one cycle, and
the composed table is the same integer function as the three in a row on all
256 codes of all 16 lanes.

### Elementwise units

`otf_eltwise` takes two live tensors of the same shape, or three for Where,
and works lane by lane. With `a` and `b` the input codes, `za` and `zb` their
zero points and `s` the scales:

| Operator | Arithmetic |
|---|---|
| Add, Sub | `acc = (a - za) * ma + (b - zb) * mb`, then the rescale with multiplier 1 |
| Mul | `acc = (a - za) * (b - zb)`, then the rescale with `sa * sb / so` |
| Less, LessOrEqual, Equal | `acc = (a - za) * ma - (b - zb) * mb`, then the sign of `acc` |
| Max, Min | each operand looked up in a table that puts it on the output grid, then the larger or smaller |
| And, Or | `a != za` and `b != zb`, combined |
| Where | the condition `c != zc` picks between its operands, each put on the output grid by a table |

Add, Sub and Mul go through the arithmetic of
[the rescale contract](../hardware/requant.md), the `Otf_requant` circuit the
dense layers use. For Add `ma` and `mb` are `sa / so` and `sb / so` sharing one
shift, as the streaming `otf_add` does, and Sub negates `mb`. For a comparison
they are the two input scales over the larger one, so the comparison is exact
when the scales are equal and otherwise as close as an 18 bit multiplier gets
their ratio. Greater and GreaterOrEqual swap the operands.

Max, Min and Where move each operand onto the output grid on its own before
they choose, and that is a function of one code, so it is a table as above
rather than a multiply. The answer is the same as rescaling after choosing,
since rounding onto a grid never reverses an order, and it is the correctly
rounded value. At the trading clock the multiply would have cost Max and Min
four cycles and Where three; the tables cost two and one. Where also takes a
constant condition, or a constant operand, which is then a code on the output
grid already.

Without `clock_mhz`, Add, Sub, Mul and the comparisons take two cycles, since
they register their accumulator, and Max, Min, Where, And and Or take one.

### Booleans

The outputs of comparisons, And, Or and Not are int8 codes 0 and 1 on a grid
of scale 1 and zero point 0, whatever calibration measured. A code counts as
true when it differs from its tensor's zero point, the code of a real 0, so a
boolean read from any grid means what the float model means by nonzero.

### Branches

Fixed latency designs were chains. A binary operator needs both its inputs to
come from the same feature vector, so `Fixed_top` now takes any graph whose
nodes are listed in topological order, works out when each tensor arrives, and
puts plain registers on the earlier inputs of each unit until they line up.
The valid bit is delayed with them, and the bank tag of
[003](003-reloadable-weights.md) too. The latency of a design is the longest
path from the input to the output, and the generator refuses a design.json
whose stated latency is not that number. `x * Sigmoid(x)` takes three cycles:
one for the table while `x` waits in a register, then two for the Mul.

### Reloadable weights

Tables and elementwise units are frozen. A model with `weights="reloadable"`
reloads its dense layers around them: the grids on both sides of every frozen
unit stay as compiled when the new model is calibrated, so the tables still
mean what they meant, and a new model whose tables or constants would differ,
such as one with another PRelu slope, raises `ConfigError` naming the unit.

## Widths

The compiler builds tables for int8 activations, and for uint8 ones carried on
the signed grid as [quantization](../guides/quantization.md) describes: 256
entries of 8 bits. The Hardcaml unit takes any input width from 1 to 12 bits
and any output width, since its tables come from design.json, and its tests
run 1, 3, 4, 8 and 10 bit inputs, but the compiler has no activations wider
than 8 bits to give it.

## Pipelining

The table is one path, from the input register through the table's logic to
the output register. The delay model prices it as a level of LUTs and wide
muxes per LUT's worth of address bits, two levels for 8 bits on both
UltraScale+ and ECP5, plus the fanout of each input bit to the LUTs of every
output bit. It has no stage to add. An `otf_eltwise` unit has two knobs, set by
`clock_mhz` the way [002](002-pipelined-matvec.md) sets a dense layer's:

- **Products.** A register after the multiplies of Add, Sub, Mul or a
  comparison, before the accumulator, which a DSP48E2 takes as MREG.
- **Rescale.** For Add and Mul the cuts inside `Otf_requant`, after its
  multiply, after the rounding shift and after the zero point; Add has the
  last two, since its rescale multiplies by 1. For Max, Min and Where, a
  register after the tables.

Each knob takes the fewest registers whose path fits the clock, and the
design.json parameters are `PRODUCT_STAGES` and `RESCALE_STAGES`.

Predicted by the `k26` model at 322.265625 MHz, a period of 3.10 ns, for one
unit on 8 lanes, unverified until a Vivado run:

| Operator | Unit | Cycles without a clock | Cycles at 322 MHz | Stages | Slowest path at 322 MHz | Without stages |
|---|---|---|---|---|---|---|
| Any table operator | `otf_table` | 1 | 1 | none | 2.05 ns, the table | 2.05 ns |
| Add, Sub | `otf_eltwise` | 2 | 4 | 1/1 | 2.84 ns, the multiplies | 4.66 ns |
| Mul | `otf_eltwise` | 2 | 5 | 1/2 | 2.84 ns, the multiplies | 5.97 ns |
| Less, Equal and the rest | `otf_eltwise` | 2 | 3 | 1/0 | 2.84 ns, the multiplies | 4.66 ns |
| Max, Min | `otf_eltwise` | 1 | 2 | 0/1 | 2.05 ns, the tables | 3.43 ns |
| Where | `otf_eltwise` | 1 | 1 | 0/0 | 2.67 ns, a table and the select | 2.67 ns |
| And, Or | `otf_eltwise` | 1 | 1 | none | 0.81 ns | 0.81 ns |

Stages are products/rescale. Every pipelined `otf_eltwise` unit ends up limited
by the same step, the zero point subtract and a DSP48E2 multiply with MREG, which
the model puts at 2.84 ns.

## Streaming mode

Both units are fixed latency only. Streaming mode keeps the two operators it
already had, an Add of two streams and a Relu that cannot be fused, and
refuses the rest with `ConfigError` before calibration. A streaming table
would need a new Hardcaml streaming unit and a backend that refuses it under
`hdl="sv"`, so we left it out. The SystemVerilog library stays frozen, as
[001](001-fixed-latency.md) says, and neither unit has a SystemVerilog twin.

## API

Nothing new to call. A model with these operators compiles with
`mode="fixed"`, and the summary shows a row per unit: its lanes, `table` or
the operator, its multipliers, latency, stages and predicted fmax. A composed
table is named after its last operator and lists them all:

```
 tanh (Sigmoid+Mul+Tanh)          16 -> 16       table           0        1             606      512      0
```

That is on `vu9p`, whose model predicts 606 MHz for the table.

From Python, `lookup_table` in `compiler/onnx2fpga/p4_quantize/elementwise.py`
gives the codes of any numpy function between two grids, along the last axis,
so a per lane constant shaped `(lanes, 1)` gives a table per lane.
`TableUnit.per_lane` in `compiler/onnx2fpga/p3_ops/elementwise.py` turns such
rows into a unit, keeping each distinct row once. An operator built on an
exponential, such as a Softmax, can take its table from there:

```python
rows = lookup_table(np.exp, in_scale, in_zero_point, out_scale, out_zero_point, INT8)
unit = TableUnit.per_lane(name, [source], [target], np.broadcast_to(rows, (lanes, 256)),
                          INT8, INT8)
```

## Errors

Each is one sentence and names the node.

- An operator other than Add or Relu in streaming mode: `mode='stream' cannot
  build op (Sigmoid); it needs mode='fixed'`.
- A Div, PRelu or Clip whose divisor, slope or bound is live raises
  `UnsupportedOpError` at load, for example `Div 'op' needs a constant
  divisor`, and so does a Max or Min of more than two inputs.
- Live operands of different shapes raise `ConfigError`: `op (Max) takes
  tensors of shapes (1, 8) and (1, 1), and live operands must have one shape`.
  So does a constant that would broadcast the output past its live operand's
  shape, and shapes that do not broadcast at all raise `ModelError`.

## Verification

`tests/test_elementwise.py` checks every table exhaustively. Each of the 20
one input operators, and 16 cases with constants (PRelu per channel, Clip,
the tanh Gelu, LeakyRelu with its own slope, Add, Sub, Mul, Div, Max, Min,
Greater, LessOrEqual, Equal, And and Where with constants, and Mul of a tensor
by itself), is lowered from a one node model and every entry of every lane is
compared with a reference written with Python's `math` module, apart from the
numpy definitions the compiler uses. Every entry matches, allowing one code
only where the real value lands within 1e-6 of a rounding tie.

Two models compile in fixed mode and simulate bit exact under Verilator, and
in each `simulate()` sends 10000 random vectors, one per cycle, and all 10000
answers arrive exactly `latency` cycles later and match the integer model:

- A gate model: a dense layer, `h * Sigmoid(h)`, then 21 table operators on
  branches (all of the list above apart from Not, with PRelu per channel and
  Clip), joined pairwise by Add, Sub, Max, Min and Mul down to one tensor and
  added to `h`. On `k26` at 322.265625 MHz it has 45 units, 22 of them
  tables, and a latency of 37 cycles, 114.8 ns, with 352.6 MHz predicted.
- A logic model: two dense layers, Less, Greater, LessOrEqual and
  GreaterOrEqual between them, Equal, Not, And and Or on the results, and
  Where with a live condition, a constant branch and a constant condition,
  12 cycles without a clock and 20 at 322.265625 MHz.

The latency proof from [001](001-fixed-latency.md) runs on the generated
Verilog of the logic model, where the branches wait for each other, and closes
at 12 cycles; the same check at 11 fails. The composed table above simulates
bit exact too.

The Hardcaml units have expect tests on the Lws harness, from a random initial
state. `test_otf_table.ml` sends every input code to every lane of five
configurations, from a 1 bit input to a 10 bit one and six lanes over three
tables, at 100 and 40 percent valid, and checks each answer one cycle later;
a waveform shows two lanes and a gap. `test_otf_eltwise.ml` runs every
operator, Mul of a port by itself, and Where with a constant condition and
with a constant branch, each without stages and with the most it takes, 150
cycles at 100 and 40 percent valid and again with codes near zero so that
Equal and the logic operators see both answers, against an OCaml model of
`EltwiseUnit.execute`. `test_fixed_top.ml` builds a design where a table of the
features waits five cycles for two dense layers before a Mul joins them, and
checks 400 cycles at 70 percent valid, latency 8.

## Resources

yosys 0.52 `synth_xilinx -family xcup` on the composed table above, 16 lanes
sharing one table, gives 368 LUTs (305 of them LUT6), 176 MUXF7, 80 MUXF8 and
81 flip flops: 23 LUTs a lane. The compiler counts 32 a lane, a LUT6 per 64
entries of each output bit, which is what a table with no structure takes. The
`otf_eltwise` estimates, and every ECP5 figure for both units, are not fitted
to a synthesis run yet.

## Done when

All four are met.

- [x] Every operator listed above compiles in fixed mode and simulates bit
  exact against the integer model.
- [x] Both units have Hardcaml expect tests against an OCaml model.
- [x] A design with branches has its latency proved on the generated Verilog.
- [x] `clock_mhz` pipelines both units, and the cycles and predicted delay at
  322.265625 MHz on `k26` are recorded here.

## Open questions

- A table right after a dense layer could sit in the layer's output stage and
  save a cycle, at the price of a longer rescale path. That needs a table input
  on `otf_dense` and the delay model to charge it, which we left for later.
- A comparison of two live tensors takes three cycles at 322 MHz. For each
  code of one operand the other's codes split at a threshold, so a table of
  thresholds and one comparator would take one cycle and be exact for any two
  scales.
- Is one cycle for a 256 entry table at 322 MHz real on the KV260? The delay
  model says 2.05 ns of a 3.10 ns period, and only Vivado can say for sure.
