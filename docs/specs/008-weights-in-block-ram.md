# 008 Weights in block RAM

Status: planned (lower priority since the HFT pivot; reloadable weights in [003](003-reloadable-weights.md) may use block RAM)

## Problem

Weight tables are read combinationally, so synthesis builds them out of LUTs.
On the Icepi Zero they are the largest LUT cost and the 56 block RAMs sit
unused. This caps model size long before the multipliers do.

## Behaviour

- Each matrix unit's weight table goes in block RAM (EBR on ECP5, BRAM on AMD)
  when it is large enough to be worth a block, and stays in LUTs when it is not.
- A block RAM read is registered, so the unit gains one cycle of latency.
  Throughput is unchanged: one weight word per cycle, as now.
- Both backends change together and stay cycle for cycle identical.
- The FIFO sizing in `p5_schedule/tokens.py` accounts for the extra cycle.
- `summary()` shows block RAM per unit, and the folding allocator budgets it.

## API

Automatic by default. Per layer override:

```python
acc.layers["fc0"].memory("block")   # force block RAM
acc.layers["fc0"].memory("lut")     # force LUTs
acc.layers["fc0"].memory("auto")    # back to the compiler's choice
```

## Errors

- `memory("block")` on a device without block RAM: `ConfigError`.
- Forced block RAM that does not fit: `BudgetError` naming the layer.

## Done when

- The example MLP and CNN on `icepi25` use block RAM for their weights, and
  yosys's DP16KD count matches the estimate.
- LUT use for those designs drops by at least a third.
- Lockstep, simulation sweeps and the board run are all bit exact.

## Open questions

- What size makes a table worth a block? Start with "more than one 18 kbit
  block's worth of bits, or more than N LUTs", and tune against yosys.
- Should biases and per channel multipliers move too, or only weights?
