# 001 Fixed latency mode

Status: partial. Dense models (Gemm and MatMul, a fused Relu, Flatten and
Reshape) build and are verified. Conv, MaxPool, Add and a Relu that cannot be
fused raise `ConfigError` in this mode for now.

## Problem

Today every unit talks over a valid and ready handshake. A unit can stall, a
FIFO can fill, and the time from input to answer depends on what happened in
the cycles before. That is fine for throughput. A trading path cannot use it. A
strategy needs to know that a model answers exactly N cycles after the features
arrive, every time, and that it takes a new input on every cycle.

## Behaviour

`mode="fixed"` builds a different kind of design from the same model. Every
layer is unrolled and pipelined so it accepts one input vector per cycle. There
is no ready signal anywhere and no FIFO. The design has one input, a features
bus with a valid bit, and one output, the prediction with a valid bit that rises
exactly `latency` cycles after the input's.

The compiler works out the latency in cycles and prints it. The number does not
depend on the data, and we prove that rather than hope it.

The streaming mode stays. It is still the right shape for large models that
cannot be unrolled.

## Pipeline

Each dense layer becomes one `otf_dense` unit with three register stages, so it
adds exactly three cycles:

1. One multiplier per weight, `x[k] * w[k][c]`, registered at its natural width.
2. One adder tree per output channel over the products and the bias, registered
   at the accumulator width the streaming unit already uses.
3. The rescale from [the rescale contract](../hardware/requant.md), the same
   `Otf_requant` circuit the streaming units use, registered at the output width.

Flatten and Reshape are wiring and cost nothing. The latency of a model is three
times its number of dense layers. The example MLP, 32-64-16-10, has three, so
`acc.latency` is 9.

The valid bit travels down a matching chain of flip flops, three per layer, and
those are the only registers `clear` touches. The data registers have no reset
and no enable, so the datapath runs every cycle whatever the valid bit says.
Where to place more registers to reach a clock is [002](002-pipelined-matvec.md).

Weights are constants in the generated Verilog. A general multiplier per weight
is the baseline that [004](004-weight-specialised-arithmetic.md) will specialise.

## API

```python
acc = otf.load("examples/models/mlp.onnx")
acc.compile(device="vu9p", mode="fixed", calibration=x_train[:64])
acc.latency            # 9
acc.layers["fc0"].latency   # 3
acc.summary()          # per layer multipliers and latency, then the total in cycles and ns
acc.simulate()         # 10000 random vectors, one per cycle
```

`clock_mhz=` arrives with [002](002-pipelined-matvec.md). Until then the
nanosecond figure uses the device model's clock, 300 MHz for `vu9p`, which puts
the example MLP at 30.0 ns. At the 322.265625 MHz trading clock it would be
27.9 ns, if the design closes timing there.

`target`, `target_fps`, `target_cycles` and `Layer.fold` pins have no meaning
when every layer is unrolled, and passing any of them raises `ConfigError`.

## Interface of the generated block

| Port | Width | Meaning |
|---|---|---|
| `clock`, `clear` | 1 | Synchronous, active high clear. |
| `features` | inputs x bits | One feature vector, element 0 in the low bits. |
| `features_valid` | 1 | `features` holds a vector this cycle. |
| `prediction` | outputs x bits | The model's output, element 0 in the low bits. |
| `prediction_valid` | 1 | High exactly `latency` cycles after `features_valid`. |

For the example MLP that is a 256 bit `features` bus and an 80 bit
`prediction`. A `clear` drops whatever is in flight: no `prediction_valid` rises
for an input taken before it. Weight reload ports are added by
[003](003-reloadable-weights.md).

`design.json` carries `"mode": "fixed"` and `"latency"`, and the Hardcaml
generator refuses a file whose stated latency is not the sum over its units.
`manifest.json` records the total and each layer's latency.

## Backends

Fixed latency units are Hardcaml only. The SystemVerilog library stays frozen
as the lockstep reference for the existing streaming units, where it has
already caught a real bug, and is not extended with fixed latency twins.
`mode="fixed"` with `hdl="sv"`, or with `ONNX2FPGA_HDL=sv`, raises
`ConfigError`. Fixed mode needs the Hardcaml toolchain, and raises
`ToolchainError` without it.

The Icepi Zero flow in `build(board="icepi_zero")` wraps a design in a UART that
streams frames, so it refuses a fixed design with `ConfigError`. Fixed designs
get a top of their own on the Kria boards, behind AXI4-Lite registers; see
[006](006-kv260.md).

## Errors

A model too large to unroll on the device raises `BudgetError`, naming the
largest layer and how many multipliers it needs against how many there are.
The example MLP on `icepi25` says:

```
the model does not fit icepi25 unrolled: it needs 3216 DSP against the 28 allowed,
and its largest layer fc0 alone has 2048 multipliers; use a larger device or the
streaming mode
```

A layer fixed mode cannot build yet raises `ConfigError` before calibration,
naming it: `mode='fixed' cannot build conv0 (Conv+Relu) yet, only Gemm and
MatMul with an optional fused Relu, Flatten and Reshape`. So does a Gemm that
multiplies more than one vector per input.

## Verification

`simulate()` in fixed mode drives the Verilator model through
`runtime/sim/fixed_harness.cpp`. On the example MLP it sends 10000 random
vectors spanning the whole int8 input grid, one per cycle, and finishes in 10009
cycles. All 10000 answers arrive exactly 9 cycles after their inputs and match
the integer model in `reference/runner.py`. `sweep=True` runs the same vectors
again with `features_valid` high on 50 and then 10 percent of cycles, three
seeds each, and all six pass. The harness fails on a `prediction_valid` with no
input waiting, an input with no answer after `latency` cycles, or any differing
answer, and we checked that telling it the latency is 8 or 10 makes it fail.

The latency is proved with SymbiYosys on the generated Verilog of the whole
example MLP, not on a model of it. A wrapper keeps its own shift register of
`features_valid`, cleared by `clear`, and asserts that after the first clear
`prediction_valid` equals that register delayed by `latency` cycles, for every
input sequence. `prep -flatten` and `opt_clean` drop the datapath, since the
property never reads `prediction`, and k-induction at depth 11 closes the
proof; the prove and cover tasks take about 12 seconds together. A cover task shows `prediction_valid` does rise, at
step 10, so the assertion is not vacuous, and the same check with a latency of
8 fails. `tests/test_fixed_latency.py` runs all three.

The Hardcaml unit has expect tests on the Lws harness in
`hardware/hardcaml/test/test_otf_dense.ml`. Six configurations, from the MLP's
32 by 64 first layer down to a 1 by 1 layer, a fused Relu at a nonzero zero
point, no shift, a shift past the product and 4 bit activations, each run 300
cycles at 100, 60 and 5 percent valid, and every output is compared with an
OCaml model of the rescale contract exactly three cycles later. One run starts
from `~random_initial_state:`All` to show the clear is enough, and a waveform
shows back to back vectors and a gap. `test_fixed_top.ml` builds a two layer
design from `design.json` and checks 500 cycles of answers six cycles late, the
port names above, and that a wrong stated latency is refused.

## Resources

We ran yosys 0.52 on the generated `rtl/top.v` of the example MLP, out of
context and without place and route, since Vivado is not installed here:

```
yosys -p "read_verilog rtl/top.v; synth_xilinx -family xcup -top otf_top; stat"
yosys -p "read_verilog rtl/top.v; synth_ecp5 -top otf_top; stat"
```

| Flow | Multipliers | LUT | Carry | Flip flops | Run time |
|---|---|---|---|---|---|
| `synth_xilinx -family xcup` | 2853 DSP48E2 | 249435 LUT1 to LUT6, plus 198474 MUXF7/F8/F9 | 1483 CARRY4 | 43748 FDRE | 4 min 16 s |
| `synth_ecp5` | 2933 MULT18X18D | 62904 LUT4, plus 14888 PFUMX/L6MUX21 | 16894 CCU2C | 43748 TRELLIS_FF | 5 min 1 s |

Each multiplier by a weight gets a whole DSP slice. Of the 3232 weights,
yosys turns the zeros and powers of two into wiring and leaves 2763
multipliers on UltraScale+, and each layer's rescale takes one more slice per
channel, 90 in all. The compiler's estimate of 1706 DSPs assumes two int8
products share a DSP48E2, which yosys does not do. On ECP5 the estimate is 3216
multipliers against 2933 measured, and either way the model is 100 times too
big for the 28 on an Icepi Zero.

The UltraScale+ LUT count is the surprise. In the smallest layer, 16 by 10, the
160 two input adders of the adder trees are still separate cells when DSPs are
mapped, and come out of `synth_xilinx` as 8592 LUT6 and 4252 MUXF7 with only 57
CARRY4. The same Verilog through `synth_ecp5` builds its adders on carry
chains, which is why the ECP5 column is a quarter of the size. Vivado will map
the trees its own way, so the UltraScale+ LUT figure waits for a Vivado run,
and the flip flop count is the only number here we expect to carry over.

## Done when

Three of the four are met.

- [x] `simulate()` sends a random input on every cycle for 10000 cycles and
  every output arrives exactly `latency` cycles later and matches the integer
  model.
- [x] A formal check proves `prediction_valid` is `features_valid` delayed by
  `latency` cycles, for all inputs.
- [x] Lockstep keeps the SystemVerilog and Hardcaml versions identical, or the
  backend policy says why not. The policy above says why not.
- [ ] Conv, MaxPool and Add in fixed mode.

## Open questions

- Should the input be one wide bus, or a few beats over a narrower one when the
  feature vector is wider than the source can supply in one cycle? The 256 bit
  bus of the example MLP is fine; a model with a thousand features is not.
- How does Vivado map the adder trees? yosys builds them from LUTs and wide
  muxes for UltraScale+ instead of carry chains. If Vivado does the same, the
  trees need restructuring before 002.
- Three stages per layer is a fixed choice. Should a layer small enough to close
  timing in one or two stages take fewer, leaving it to [002](002-pipelined-matvec.md)
  to add registers only where the clock needs them?
