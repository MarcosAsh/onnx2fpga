# 003 Reloadable weights

Status: planned

## Problem

Weights are baked into the bitstream as constants. Firms do not publish how
often they retrain, but the products that do publish, Myrtle's VOLLO for one,
load a new model in under a second, and a Vivado rebuild takes hours. A model
that needs a rebuild to change is a model nobody will update.

## Behaviour

`weights="reloadable"` keeps every weight, bias and rescale multiplier in
registers or block RAM that a host can write while the design runs. There are
two banks. The host writes the inactive bank at its own pace, then flips one
bit. The bank bit is sampled together with `features_valid` and travels down
the pipeline as a tag beside the data, so every inference uses one bank from
start to finish even though a new one starts every cycle. No inference ever
sees half old and half new weights. Latency is the same in both banks and the same
as before the reload.

`weights="frozen"` is today's behaviour and the only mode where weight
specialised arithmetic ([004](004-weight-specialised-arithmetic.md)) applies.
It is smaller and faster, and changing a weight means a rebuild.

## API

```python
acc.compile(device="kv260", mode="fixed", weights="reloadable")
acc.build(board="kv260")
acc.deploy()
acc.load_weights("model_2026-10-25.onnx")   # writes the idle bank, then swaps
```

`load_weights` checks the new model has the same architecture and quantization
grid as the compiled one, and refuses it otherwise.

## Interface

A memory mapped register port, AXI4-Lite on AMD boards, holding the two banks,
a commit register for the active bank, a version counter and a CRC of the
active bank that the host reads back to confirm the swap. On Kria the ARM cores
drive it; on an Alveo card it would sit behind a PCIe BAR. The map is written into `manifest.json` so other software can drive it.

## Errors

- A model with a different shape or grid: `ConfigError` naming the first
  mismatch.
- Reloadable weights that do not fit on chip: `BudgetError`.

## Done when

- In simulation, a weight swap in the middle of a stream of inputs changes the
  outputs on exactly the first inference after the swap, and every output
  matches the integer model of whichever weights were active.
- On a Kria board, `load_weights` takes under a second for the example models and
  needs no rebuild.

## Open questions

- Registers or block RAM for the banks? Registers keep latency lowest; block
  RAM scales. Probably both, chosen by size like [008](008-weights-in-block-ram.md).
- Quantization scales and rescale multipliers are reloadable too, since a
  retrain moves them. The grid's bit widths stay fixed.
