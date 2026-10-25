# 004 Weight specialised arithmetic

Status: partial. Constant arithmetic is built for the dense layers of fixed
latency mode, is the default there, and is proved equal to the multiplier
version on the example MLP's first and last layers. Nobody has timed it at the
322.265625 MHz trading clock: the `k26` device model from [006](006-kv260.md)
now estimates it, but the timing needs Vivado.

## Problem

With frozen weights ([003](003-reloadable-weights.md)) every weight is known when
the hardware is generated, yet each still gets a general multiplier. A
multiplier by a constant is smaller and faster as a few shifts and adds, and a
zero weight needs no hardware at all.

## Behaviour

In fixed latency mode ([001](001-fixed-latency.md)) every multiplier sees one
weight. With constant arithmetic the first stage of each `otf_dense` unit
builds logic for that weight instead. A zero weight disappears, plus or minus
one becomes an add or a subtract, a power of two becomes a shift, and anything
else becomes a short sum of shifts. No weight is pruned unless the user asks.

The unit splits each weight into a sign, a power of two and an odd magnitude,
so -40 is the input times 5, shifted left by 3, then subtracted. For each input
it builds every odd magnitude its weights need once, as a sum of canonical
signed digits, and registers it. Every output channel that needs that product
reads the same register. The digits are taken in pairs, and a pair is the input
times 2^j + 1 or 2^j - 1, shifted, which is also built once per input and shared
by all its products. A product of int8 weights is never more than two adders
deep, the same as plain canonical signed digits. The signs and shifts go into
the second stage's adder tree, which subtracts where a weight is negative. The
tree starts with the bias, which is added, so it never needs a negation.

Values never change, only cost and timing. Both versions compute modulo
2^acc_bits, so the result is exact. The latency stays at three cycles a layer,
since the stages are the same. The multiplier version now also multiplies by
the magnitude and lets the same tree subtract, so the two share everything
after the first stage. That change lets yosys merge a product by w with one by
-w on the same input, which takes the multiplier version of the example MLP
from the 2853 DSP48E2 measured in 001 to 2660.

On the first layer of the example MLP, 32 by 64, Hardcaml counts 2105
multipliers in the multiplier version and 64 with constant arithmetic, which
are the rescale. Adders and subtractors go from 2105 to 3280, and registered
bits fall from 34451 to 17786, because the first stage keeps one word per input
and magnitude instead of one per weight.

`"auto"` prices both versions of each layer with the device's cost model and
takes constant arithmetic unless its busiest resource, as a share of the
budget, is fuller than with multipliers. On `vu9p` all three layers of the
example MLP take constant arithmetic. On the `sim` pseudo device, where every
budget is the same size, the multiplier version has fewer LUTs and wins.

The da4ml project already generates multiplier free arithmetic of this kind.
Its comparison with ours is below.

## API

```python
acc.layers["fc2"].arithmetic("constant")
acc.layers["fc2"].arithmetic("multiplier")
acc.layers["fc2"].arithmetic("auto")      # the default
```

The fixed latency summary gains an arithmetic column and a line with the
multipliers saved:

```
 Layer (kind)                    In -> out  Arithmetic Multipliers  Latency      LUT    DSP
===========================================================================================
 fc0 (Gemm+Relu)                  32 -> 64    constant           0        3    59364     64
 fc1 (Gemm+Relu)                  64 -> 16    constant           0        3    37190     16
 fc2 (Gemm)                       16 -> 10    constant           0        3     6064     10
===========================================================================================
 Total                                                           0        9   102618     90
 ...
 Constant arithmetic: 3232 multipliers saved
```

`design.json` gives each `otf_dense` node a `CONSTANT` parameter, and
`manifest.json` records each layer's arithmetic.

## Errors

A value other than the three raises `ConfigError`, and so does a layer without
weights. Compiling with `mode="stream"` while a layer is set to `"constant"`
raises `ConfigError: fc1 is folded in mode='stream', so arithmetic('constant')
needs mode='fixed'`. `arithmetic("constant")` with reloadable weights
([003](003-reloadable-weights.md)) raises `ConfigError` too, since those weights
are not constants, and `"auto"` picks multipliers there.

## Verification

ABC's `cec` proves that fc2 of the example MLP, a whole 16 by 10 layer with its
bias and rescale, is the same logic in both versions. The check reads the two
generated Verilog files, turns each flip flop into a wire, and runs SAT
sweeping on the result. That is sound because every path in both versions
crosses the same three registers, which the expect tests and the latency proof
below check. The test then flips one weight bit in the constant version and
checks `cec` finds a mismatch. Both checks take two minutes and run in
`tests/test_fixed_latency.py`. The same check on fc0, the 32 by 64 first layer,
ran for 2540 seconds as one network without an answer. Each output is independent,
so we proved fc0 one output channel at a time instead, with a wrapper that
keeps one channel's 8 bits, or the valid bit, and lets yosys drop the rest. All
65 runs say the networks are equivalent, in 11 to 30 seconds each and 1082
seconds in total, three and a half minutes on 16 cores. That run is too heavy
for the test suite, which keeps fc2.

The Hardcaml expect tests in `test_otf_dense.ml` run all six configurations of
001 in both versions, 300 cycles at 100, 60 and 5 percent valid, and compare
every output with the OCaml model exactly three cycles later. `test_fixed_top.ml`
builds a two layer design with one layer in each version.

The default build of the example MLP now uses constant arithmetic everywhere,
so the existing fixed mode checks run on it: `simulate()` sends 10000 random
vectors, one per cycle, and every answer arrives exactly 9 cycles later and
matches the integer model, then the 50 and 10 percent sweeps pass. The
SymbiYosys proof that `prediction_valid` is `features_valid` delayed by 9
cycles holds.

## Resources

We ran yosys 0.52 on the generated `rtl/top.v` of the example MLP, flattened,
out of context and without place and route. The stock `synth_xilinx -family
xcup` flow merges each adder tree into one `$macc` cell and maps it into LUT6
and wide multiplexers instead of carry chains. Leaving `alumacc` out of the
coarse step maps the same trees onto CARRY4 chains, so both are shown. Vivado
maps adders its own way, so neither LUT figure is a Vivado figure.

| Flow | Arithmetic | Multipliers | LUT | Wide mux | Carry | Flip flops |
|---|---|---|---|---|---|---|
| `synth_xilinx -family xcup` | multiplier | 2660 DSP48E2 | 103265 | 41210 | 7976 | 40044 |
| | constant | 90 DSP48E2 | 110645 | 39755 | 15299 | 28781 |
| same, without `alumacc` | multiplier | 2660 DSP48E2 | 52142 | 0 | 15954 | 40044 |
| | constant | 90 DSP48E2 | 66994 | 0 | 23268 | 28781 |
| `synth_ecp5` | multiplier | 2740 MULT18X18D | 64005 | 16280 | 16809 | 40044 |
| | constant | 170 MULT18X18D | 60450 | 14676 | 28933 | 28781 |

Constant arithmetic saves 2570 multipliers on both families. The ones left are
the rescale, one DSP48E2 or two MULT18X18D per output channel. On UltraScale+
without `alumacc` it costs 14852 more LUTs, 28 percent, and 7314 more CARRY4.
On ECP5 the LUT4 count falls by 3555 and CCU2C carry cells rise by 12124. Flip
flops fall by 11263 on both. The compiler's estimate, 102618 LUT, is higher
than either measurement.

## da4ml

da4ml 0.6.0 installed from PyPI into a scratch environment in under a minute,
and is not a dependency of this project. Its `solve` takes an integer matrix
and the input ranges, and `VerilogModel` writes Verilog for it. One trap:
`solve` reads the matrix memory as C ordered, so a transposed numpy view gives a
wrong kernel unless it is made contiguous first. We gave it fc0 of the example
MLP, 32 by 64 int8, and checked its Python model against the matrix product. It
splits the matrix into two stages that share sums of inputs across columns,
with a register at the input, between the stages and at the output.

Our side is the fc0 `otf_dense` module with the 64 rescale blocks subtracted,
measured the same way. It still adds the bias and registers every product, the
20 bit sums and the 8 bit outputs, which accounts for most of the flip flop
gap. The adder counts are add and subtract operations, without our 64 bias
adds.

| fc0, 32 by 64 | Adders | xcup LUT, stock | xcup LUT + CARRY4, no `alumacc` | ECP5 LUT4 + CCU2C, stock | ECP5 CCU2C, no `alumacc` | DSP | Flip flops |
|---|---|---|---|---|---|---|---|
| da4ml | 2495 | 32703 | 26044 + 7999 | 24878 + 9351 | 14689 | 0 | 2527 |
| ours, constant | 3136 | 149598 | 37448 + 12793 | 74052 + 7199 | 23213 | 0 | 15779 |
| ours, multiplier | 1977 | 153658 | 31176 + 9341 | 77791 + 624 | 17590 | 1517 | 23789 |

da4ml's adder graph is smaller, 2495 adders against 3136, because it shares
partial sums across inputs as well as within each input. Without `alumacc`,
which takes the tree mapping problem away from both, it needs 30 percent fewer
LUTs and 37 percent fewer carry cells on UltraScale+, and 37 percent fewer
CCU2C on ECP5. In the stock flows yosys maps its separate adder modules well
and our merged trees badly, so the gap there is mostly the mapping.

We keep our own arithmetic for now. It shares the three stage structure, the
rescale and the proof with the multiplier version, and its latency in cycles
is fixed by construction, where da4ml's pipeline depth comes from its own
latency model. The adder graph is the part da4ml does better.

## Done when

- [ ] hardcaml_verify proves the constant version equal to the multiplier
  version for the jet tagging sized MLP. ABC's `cec` proves it for fc2 in the
  test suite and for fc0, the full 32 by 64 first layer of the example MLP, one
  output channel at a time. No hardcaml_verify, and no jet tagging model.
- [ ] At the same clock, the constant version has fewer DSPs and no more
  latency. It has 2570 fewer multipliers and the same 9 cycles, but no clock
  has been measured.
- [x] We have a table of our arithmetic against da4ml's for the same model.

## Open questions

- Should fixed mode call da4ml for the first stage when it is installed? Its
  graphs need 20 percent fewer adders on fc0, but its pipeline would have to
  fit the three stage latency contract.
- How does Vivado map the adder trees, and does the two adder first stage close
  at 322.265625 MHz on the `k26` part?
