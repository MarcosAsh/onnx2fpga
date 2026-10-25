# 002 Pipelining and clock targets

Status: partial. In fixed latency mode `compile(clock_mhz=...)` adds the fewest
register stages each layer needs to reach the clock, by a per device delay
model, for dense layers (multiplier, constant arithmetic and reloadable) and
tree ensembles. Every stage is verified bit exact at the new latency, and the
latency and bank tag proofs close at the new depths. The ECP5 model is fitted
to nextpnr-ecp5 place and route. Missing: any Vivado run, so the UltraScale+
model and the timing closure on the KV260 are unverified; pipelining in
streaming mode; and the jet tagging model.

## Problem

Our fixed latency layers ([001](001-fixed-latency.md)) multiply, sum and
rescale in three register stages. The rescale alone is a multiply, two adds and
a clamp in one cycle. On the Icepi Zero the 4-8-1 MLP closes at 71 MHz, and the
trading clock on the Kria boards is 322.265625 MHz, the 10GBASE-R clock that a
32 bit MAC runs at. A latency quoted in cycles means nothing until it is quoted
in nanoseconds at a clock the design reaches.

## Behaviour

`clock_mhz=` asks for a clock. The compiler works out, for every path between
two registers, how long it takes with 0, 1, 2 or more registers added in it,
and gives each layer the fewest stages that bring every path under the clock
period. Each stage adds exactly one cycle of latency and the design still takes
a new input on every cycle. Without `clock_mhz` nothing changes: three stages a
dense layer, as in 001.

A dense layer has three places for stages, each with its own count:

- **Products.** The first stage puts a register straight after each multiply,
  where a DSP48E2 takes it as MREG; the existing product register becomes PREG.
  The second registers the multiplier inputs, which a DSP slice takes as AREG
  and BREG; in a reloadable layer that register sits between the bank
  multiplexer and the multiplier. Constant arithmetic
  ([004](004-weight-specialised-arithmetic.md)) has one, between the shared
  pairs of shifted inputs and their sum.
- **Adder tree.** The tree adds in binary levels, ceil(log2(inputs + 1)) of
  them for the bias and every weight, and k stages register every operand after
  k evenly spaced levels. A channel with fewer nonzero weights runs out of
  levels early and carries its sum through the remaining registers, so every
  channel has the same latency.
- **Rescale.** Up to three registers inside the rescale of
  [the rescale contract](../hardware/requant.md): after its multiply (MREG),
  after the rounding shift, and after the zero point. The arithmetic is the
  same `Otf_requant` circuit, cut into steps, so the result is bit for bit the
  same.

A tree ensemble ([005](005-tree-ensembles.md)) has two:

- **Leaf select.** k stages cut each tree's mux tree at k evenly spaced depths.
  A node's select bit is delayed by the number of cuts below it, so shallow
  trees and leaves at any depth arrive with the rest.
- **Adder tree.** The leaves and the bias are summed in binary levels with a
  register after every second level, as before. Each stage registers one more
  level, until every level has its register.

The comparators are one stage that cannot be split: a carry chain from a
feature that fans out to every split on it.

Hardcaml builds every stage from `Stages` in `otf_dense` and `otf_trees`, set
from design.json parameters (`PRODUCT_STAGES`, `SUM_STAGES` and
`RESCALE_STAGES` on `otf_dense`, `SELECT_STAGES` and `SUM_STAGES` on
`otf_trees`). The valid chain and, in a reloadable layer, the bank tag of
[003](003-reloadable-weights.md) are pipelines as long as the layer's latency.
Each bank pick reads the tag from the stage its data is in: the weights the tag
that arrived with the features, the bias the tag one product stage later, the
rescale multiplier the tag that reaches the rescale with the sum.

## Delay model

A path is the source register's clock to out and the setup at the far end,
then for each step a routed net and a cell: a carry chain for an adder or a
comparison, a LUT, a multiplier, or a level of a mux tree. A net from a
register that fans out to many cells costs more per doubling of its fanout. An
adder tree between two registers is one carry chain plus a fixed cost per
further level, since synthesis merges the adds into compressors, and the
rescale's clamp is one LUT level, since its comparisons fold into the zero
point add. Each device gives the numbers in `compiler/onnx2fpga/targets`.

**ECP5, speed 6.** Fitted to place and route: the three example models below,
each built at several clocks and again with one stage count swept at a time
while every other was at its most, 54 distinct designs through yosys 0.52 and
nextpnr-ecp5 0.9 with `--25k --package CABGA256 --speed 6`. The carry bits
(0.035 ns, nextpnr's 0.07 ns a CCU2C), the carry chain's own cost (0.30 ns)
and a mux level (1.2 ns, from a leaf select path nextpnr reported) are held;
the rest is fitted to the measured clock periods by least squares on their
logarithms: 1.21 ns of clock to out, setup and final net, 1.17 ns a net, 0.39
ns a LUT, 1.80 ns an adder tree level, 0.53 ns a fanout doubling, and 4.31 ns
through a MULT18X18D, which is nextpnr's 3.93 ns and a longer net out of the
DSP column. The RMS error is 8.4 percent. yosys puts no registers inside a
MULT18X18D, so on ECP5 the product stages never help and the compiler never
picks them. The scheduler keeps 10 percent of the period free, about the
model's spread, so a design it accepts closes.

**UltraScale+ (`k26`, -2LV).** AMD's DS925 (v1.30) publishes DSP48E2 maximum
frequencies only: at VCCINT 0.72 V, which is the K26's -2LV and DS925's -2LE
column, 644 MHz with every register used and 440 MHz without MREG; at 0.85 V
(-2) 775 and 544 MHz. The model takes its multiplier delays from those. DS925
gives no CLB, carry or routing delays at all; those live only in Vivado's
speed files. So the fabric figures are our assumptions for a -2 part at 0.85 V
(clock to out and setup 0.15 ns, a net 0.40 ns, a LUT 0.10 ns, a carry chain
0.10 ns plus 0.015 ns a bit, a mux level 0.30 ns, an adder tree level 0.80 ns,
0.10 ns a fanout doubling), scaled by 1.24 for 0.72 V, the ratio of the two DSP
figures. `vu9p`, `vu47p` and `zu7ev` use the 0.85 V model. The UltraScale+
model is unverified until a Vivado run.

nextpnr-himbaechel in oss-cad-suite has only the gatemate and gowin uarches
and no Xilinx chip database, so there is no open cross check for UltraScale+.

## API

```python
acc.compile(device="k26", mode="fixed", clock_mhz=322.265625)
acc.latency            # 22 for the example MLP, with utilisation=0.9
acc.result.fmax_mhz    # what the delay model predicts
acc.summary()
```

The summary gains two columns, the stages added per layer and the predicted
fmax, and a `Clock:` line:

```
 Layer (kind)                    In -> out  Arithmetic Multipliers  Latency   Stages   Fmax      LUT    DSP
===========================================================================================================
 fc0 (Gemm+Relu)                  32 -> 64    constant           0        7    0/2/2    343    59364     64
 fc1 (Gemm+Relu)                  64 -> 16    constant           0        8    0/3/2    372    37190     16
 fc2 (Gemm)                       16 -> 10    constant           0        7    0/2/2    390     6064     10
===========================================================================================================
 Total                                                           0       22             343   102618     90
 Budget on k26                                                                                117120   1248
 Used                                                                                          87.6%   7.2%
 Stages: registers added at the multiplies/adder tree/rescale
 Latency: 22 cycles, 68.3 ns at 322 MHz, with a new input every cycle
 Clock: 322.265625 MHz asked, 343.2 MHz predicted by the k26 delay model (DSP48E2 from
 DS925 at 0.72 V (-2LE), fabric assumed and scaled; unverified until a Vivado run)
```

`manifest.json` and design.json carry the latency and every stage count.

## Errors

Each is one sentence.

- A clock no amount of pipelining reaches raises `BudgetError` naming the
  slowest path and the fastest clock:
  `2000 MHz is out of reach on k26: the slowest path, through the adder tree
  of fc0, takes 1.92 ns however it is pipelined, so the fastest clock is 520.3
  MHz`
- `clock_mhz` with `mode="stream"`: `ConfigError: clock_mhz needs mode='fixed'`.
- A device without a delay model, such as `z7020`: `ConfigError: z7020 has no
  delay model, so clock_mhz needs one of icepi25, k26, vu47p, vu9p, zu7ev`.

## Verification

The Hardcaml expect tests run a 32 by 16 layer, both arithmetics, with its
product stages, one and five adder tree stages, one and two rescale stages, and
everything at its most, 200 cycles at 100 and 30 percent valid, and compare
every output with the OCaml model exactly `latency` cycles later. A reloadable 8 by 5 layer with two product,
three adder tree and three rescale stages takes a random bank every cycle and
checks each answer and its tag 11 cycles later. A 100 tree ensemble runs with
one and three leaf select stages, three adder tree stages, and both.

On the Python side the stated latency must equal the sum over the Hardcaml
units or the generator refuses design.json, so every compile checks the
compiler's number against the hardware. Then:

- The example MLP at 322.265625 MHz on `vu9p`, `fc1` on multipliers and the
  other two layers constant, takes 17 cycles. `simulate()` sends 10000
  random vectors, one per cycle, and every answer arrives exactly that many
  cycles later and matches the integer model. SymbiYosys proves the latency on
  the generated Verilog, and the same check one cycle shorter fails.
- The example MLP reloadable at the same clock takes 18 cycles. A
  second model written into the idle bank and committed on cycle 5000 gives
  5001 answers from the compiled bank and 4999 from the new one, every one
  exact. The bank tag proof closes at 18 and fails at 17.
- The 100 tree regressor at 500 MHz on `k26` registers every adder tree level,
  9 cycles. Simulation is bit exact and the latency proof closes at 9 and
  fails at 8.
- `tests/test_ecp5.py` builds the 4-8-1 MLP for 120 MHz on `icepi25`, places
  and routes it, and checks that it meets 120 MHz and that the prediction is
  within 10 percent of nextpnr's.

## Measured

`python tools/fmax_table.py icepi25` builds each model for each clock, then
yosys and nextpnr-ecp5 place and route it with `--freq` at that clock. Stages
are per layer as products/adder tree/rescale, or leaf select/adder tree, and
the latency is quoted at the clock asked for.

| Model | Asked | Stages | Latency | Predicted | nextpnr | Error | Meets |
|---|---|---|---|---|---|---|---|
| 4-8-1 MLP | 60 MHz | 0/0/0, 0/0/0 | 6, 100.0 ns | 72.6 MHz | 71.3 MHz | +2% | yes |
| 4-8-1 MLP | 80 MHz | 0/0/1, 0/0/1 | 8, 100.0 ns | 114.8 MHz | 105.4 MHz | +9% | yes |
| 4-8-1 MLP | 100 MHz | 0/0/1, 0/0/1 | 8, 80.0 ns | 114.8 MHz | 105.4 MHz | +9% | yes |
| 4-8-1 MLP | 120 MHz | 0/1/2, 0/1/2 | 12, 100.0 ns | 149.5 MHz | 141.0 MHz | +6% | yes |
| 16-8-4 MLP | 60 MHz | 0/0/1, 0/0/0 | 7, 116.7 ns | 72.6 MHz | 70.8 MHz | +3% | yes |
| 16-8-4 MLP | 80 MHz | 0/1/1, 0/0/1 | 9, 112.5 ns | 102.4 MHz | 104.7 MHz | -2% | yes |
| 100 tree classifier | 60 to 120 MHz | 0/0 | 6 | 149.3 MHz | 138.2 MHz | +8% | yes |

Every design the compiler accepted meets its clock, and every prediction is
within 9 percent of nextpnr. The rest raise `BudgetError`. The 4-8-1 MLP at 140
MHz stops at the rescale's multiply, 6.69 ns, so 135.9 MHz with the margin. The
16-8-4 MLP at 100 MHz and above stops at the same multiply, 9.45 ns because
its 19 bit accumulator takes a second MULT18X18D and an adder, so 96.1 MHz. The
classifier at 140 MHz stops at its comparators, 6.70 ns, so 135.7 MHz. The
100 tree regressor does not fit the Icepi Zero at all. The classifier's
comparators cannot be split, so asking it for 60 or 120 MHz builds the same
design.

## UltraScale+ at the trading clock

Predicted by the `k26` model at 322.265625 MHz, unverified until a Vivado run.
Stages are listed per layer as products/adder tree/rescale, or leaf select/adder
tree.

| Model | Stages | Latency | Predicted fmax |
|---|---|---|---|
| Example MLP, 32-64-16-10, constant arithmetic | 0/2/2, 0/3/2, 0/2/2 | 22 cycles, 68.3 ns | 343.2 MHz |
| 100 tree regressor, depth 4 | 0/0 | 6 cycles, 18.6 ns | 456.9 MHz |
| Tick to trade 4-8-1 MLP | 0/1/2, 0/1/2 | 12 cycles, 37.2 ns | 402.2 MHz |

The example MLP needs 88 percent of the K26's LUTs on the estimate, so it is
compiled with `utilisation=0.9`. The regressor needs no stages: its slowest
path is two levels of its 21 bit adder tree, and its comparators, 16 bit carry
chains from features that drive up to 131 of them each, predict 506 MHz. In the tick to trade path ([007](007-tick-to-trade.md)) the model
would go from 6 cycles to 12.

## Done when

- [ ] The jet tagging sized MLP (16-64-32-32-5) closes timing at 322.265625 MHz
  on the KV260 in Vivado. Needs Vivado and the model.
- [ ] The predicted fmax is within 10 percent of Vivado's on the example
  models. Needs Vivado; against nextpnr-ecp5 it is, see Measured.
- [x] Latency in cycles in simulation equals the compiler's number.

## Open questions

- How far off are the assumed fabric delays? One Vivado run of the example
  models at a few clocks would fit them the way the ECP5 ones were fitted.
- Should streaming mode take `clock_mhz` too? Its units would gain latency and
  keep their throughput, and the FIFO sizing already reads each unit's output
  schedule.
- Can two int8 products share a DSP48E2 with MREG and PREG on, as the resource
  estimate assumes? That is a Vivado question.
