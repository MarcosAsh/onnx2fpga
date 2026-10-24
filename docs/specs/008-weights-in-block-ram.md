# 008 Weights in block RAM

Status: partial. The streaming matrix unit `otf_mvau` keeps large weight tables
in block RAM on both backends, cycle for cycle identical, and the folding
allocator budgets the blocks. On the Icepi Zero the example MLP's LUTs fall by
more than half and yosys's DP16KD count equals the estimate on the MLP and the
CNN. Two things are missing. The CNN's LUTs fall by 18 percent, not a third,
because its weight tables were under a quarter of its logic. Nobody has run a
block RAM design on the board yet.

## Problem

Weight tables were read combinationally, so synthesis built them out of LUTs.
On the Icepi Zero they were the largest LUT cost and the 56 block RAMs sat
unused. This capped model size long before the multipliers did.

## Behaviour

Each matrix unit's weight table goes in block RAM (EBR on ECP5, BRAM on AMD)
when it is large enough to be worth a block, and stays in LUTs when it is not.
Biases and per channel multipliers stay in LUTs. They have one entry per
output channel, so they are small next to the weights.

A block RAM read is registered, so the unit gains one cycle of latency.
Throughput is unchanged: one weight word per cycle, as before. The output
schedule each unit hands to `p5_schedule/tokens.py` carries the extra cycle, so
the FIFO sizing sees it. On the residual example a block RAM on one branch
leaves every FIFO depth as it was, and the design runs without deadlock on
both backends.

`summary()` shows the block RAM of each unit, and the folding allocator
charges it against the device budget like any other resource.

## Hardware

With `BLOCK_RAM=1` the unit splits into two stages. The first issues a step:
it advances the fold counters, takes or replays the input word and presents
the weight address. The second multiplies. Between them sit the block RAM's
output register and a register for the input word, the first and last step
flags and the neuron fold that picks the bias and the multiplier. A step's
last multiply writes the result register, so each result leaves one cycle
later than in LUT mode. The pipeline stalls as a whole when the last step of a
fold is waiting in the second stage and the result register is still full.

The table is one word per step holding every PE lane, lane 0 in the low bits,
so a unit has one memory with one read port however wide it folds. It is
marked `ram_style = "block"` in both backends. yosys left a 160 x 8 table in
LUTs without the attribute, and with it the compiler decides and the count is
predictable. Hardcaml spells its own Vivado attribute `RAM_STYLE`, which yosys
ignores, so the unit names it in lower case.

Small tables keep the old single stage unit, unchanged.

## When a table is worth a block

On ECP5 a table goes in block RAM when each DP16KD it takes saves at least 128
LUTs, counting LUTs with the same ROM model the estimate uses. A 160 x 8 table
costs 125 LUTs and one block, so it stays in LUTs, which is also what yosys
picks for it on its own. A 128 x 128 table costs about 1500 LUTs and four
blocks, so it moves. Tables of 32 words or fewer never qualify. On AMD a table
moves when it is deeper than 64 words, where the AMD cost model already
charged block RAM. No Vivado run has checked that rule yet.

## API

Automatic by default. Per layer override:

```python
acc.layers["fc0"].memory("block")   # force block RAM
acc.layers["fc0"].memory("lut")     # force LUTs
acc.layers["fc0"].memory("auto")    # back to the compiler's choice
```

A table of one word is a constant and stays in logic whatever the setting.

## Errors

- `memory("block")` on a device without block RAM:
  `ConfigError: fc1 cannot go in block RAM because no_ram has none`.
- Forced block RAM that does not fit: `BudgetError: block RAM for fc0 takes the
  design to 58 ebr, past the 56 on icepi25`.
- `memory()` with anything other than the three values, on a layer without
  weights, or with `mode="fixed"`: `ConfigError`.

## Measured

The example MLP and CNN compiled for `icepi25` with `hdl="hardcaml"`, then
yosys 0.52 `synth_ecp5` and `nextpnr-ecp5 --pack-only`. LUT is nextpnr's
TRELLIS_COMB count. "fit" is `compile` with no target.

| design | LUT before | LUT after | change | DP16KD estimate | DP16KD yosys |
|---|---|---|---|---|---|
| MLP, fit (256 cycles) | 4793 | 2126 | -56% | 5 | 5 |
| MLP, `target_cycles=512` | 4325 | 1549 | -64% | 2 | 2 |
| CNN, fit (400 cycles) | 3901 | 3191 | -18% | 1 | 1 |
| CNN, `target_cycles=512` (500) | 3801 | 3111 | -18% | 1 | 1 |

The folding and the multipliers are the same before and after, 26, 12, 25 and
23. Flip flops rise because the second stage holds the input word: 287 to 463
on the MLP at fit, where `fc0` takes 16 int8 inputs a beat. In the MLP at fit,
`fc0` (128 words of 128 bits) takes four blocks and `fc1` one, and `fc2` (160
words of 8 bits) stays in LUTs. In the CNN, `fc1` takes one block and the
convolution's four word table stays in LUTs. The rest of the CNN is the
sliding windows, FIFOs and the multiply datapath, which this spec does not
touch.

## Done when

- The example MLP and CNN on `icepi25` use block RAM for their weights, and
  yosys's DP16KD count matches the estimate. Done, see the table.
- LUT use for those designs drops by at least a third. Done for the MLP; the
  CNN drops by 18 percent.
- Lockstep, simulation sweeps and the board run are all bit exact. Lockstep
  and the sweeps are, on both backends: `tests/test_hardcaml_lockstep.py` pairs
  seven block RAM shapes against the SystemVerilog unit, and the CNN at 400
  cycles on `vu9p` puts `fc1` in block RAM for the whole design comparison.
  The board run is not done.

## Open questions

- Should the line buffers of the sliding window units, which are now the
  largest LUT cost in the CNN, move to block RAM too?
- The 128 LUT threshold was set against yosys on a handful of shapes. A model
  that runs out of blocks before multipliers may want it higher.
