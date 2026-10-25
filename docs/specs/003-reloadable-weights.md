# 003 Reloadable weights

Status: partial. Dense layers in fixed latency mode are reloadable: two
register banks, the commit register, the bank tag, the version counter and the
CRCs are built in Hardcaml, `load_weights` and the mid-stream swap in
`simulate` are verified, and the tag is proved. Missing: a run on a Kria board,
which needs Vivado, the board and a Kria runtime for `deploy(host=...)`;
reloadable tree ensembles (see Trees below); and block RAM banks for models
past the 8192 word window.

## Problem

Weights are baked into the bitstream as constants. Firms do not publish how
often they retrain, but the products that do publish, Myrtle's VOLLO for one,
load a new model in under a second, and a Vivado rebuild takes hours. A model
that needs a rebuild to change is a model nobody will update.

## Behaviour

`weights="reloadable"` keeps every weight, bias and rescale multiplier of every
dense layer in registers that a host can write while the design runs. There
are two banks. The host writes the idle bank at its own pace, then writes a
commit register, which makes the idle bank the active one. The active bank bit
is sampled in the cycle the features arrive and travels down the pipeline as a
tag beside the data, so every inference uses one bank from start to finish even
though a new one starts every cycle. No inference ever sees half old and half
new weights, and the host can never write the bank in use.

`weights="frozen"` is today's behaviour and stays the default. It is the only
mode where constant arithmetic ([004](004-weight-specialised-arithmetic.md))
applies, since reloadable weights are not constants.

Tree ensembles ([005](005-tree-ensembles.md)) stay frozen only for now; see
the end of this page.

## Pipeline

A reloadable dense layer keeps the three stages of
[001](001-fixed-latency.md) and their latency, three cycles. Each value has two
registers, one per bank, and a 2 to 1 multiplexer picks one with the tag the
stage holds:

1. Every weight goes through its multiplexer into a general multiplier,
   `x[k] * w[k][c]`, using the tag that arrived with the features.
2. The adder tree adds the products and the bias, picked by the tag one cycle
   later.
3. The rescale multiplies by the multiplier picked by the tag two cycles
   later.

The tag leaves the layer three cycles after it entered and goes on to the next
layer with the data, so a layer always sees its own tag. Zero weights get a
multiplier too, since a reload can make them nonzero. The multiplexer sits in
front of the multiplier in the same cycle, so the latency equals the frozen
multiplier version's: 9 cycles for the example MLP, not the 6 cycles of a
reloadable linear model against 4 frozen that the research notes budgeted.
Whether that extra LUT level closes at 322.265625 MHz needs Vivado; if it does
not, a register between the multiplexer and the multiplier costs one cycle per
layer, the same in both banks.

The tag registers have no reset, like the rest of the datapath, and neither do
the banks. A `clear` drops the inferences in flight and keeps the loaded model.

## Quantization grid

Everything the hardware hard wires stays fixed across reloads; everything a
retrain moves is in the banks. Fixed: the shapes, every bit width, the input
and output grids (scale and zero point), each layer's output zero point and
clamp, and each layer's rescale shift. In the banks: the weights, the biases
and the per channel rescale multipliers, which absorb any change in a weight
scale or an internal activation scale.

So that a retrained model fits, a reloadable layer sizes for the grid, not for
the weights it was compiled with. The accumulator is wide enough for any int
weights on the grid, every product at its largest, plus a bias as large again,
22 bits for the example MLP's first layer against 20 frozen. The rescale shift
is four less than the compiled weights would choose, so a new model's
multipliers can be up to sixteen times the compiled ones and still fit the
18 bits. The compiled multipliers lose those four bits, which leaves each
layer's largest one 13 significant bits. On 2000 random inputs to the example
MLP, 98.3 percent of the output codes equal the frozen design's, none differ by
more than 3, and the argmax agrees with the float model on 95.10 percent of
inputs against 95.15 frozen.

`load_weights` quantizes the new model onto that grid. A float model is
calibrated on the samples the design was compiled with, then its input and
output grids are pinned to the compiled ones. A model with QDQ nodes states its
own grid, which must match. Weights are quantized per channel as at compile,
biases on the new scales, and multipliers at the compiled shift.

## Register map

The reload sits in the AXI4-Lite window of [006](006-kv260.md). Its control
registers take the free offsets after `cycles`, and the banks take the window
that 006 reserved at 0x8000, which holds 8192 words. The window did not need
widening.

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x0020 | `weights_words` | r | Words in one bank, 0 when the weights are frozen. |
| 0x0024 | `weights_commit` | w | Writing 1 makes the idle bank active and counts a version. |
| 0x0028 | `weights_bank` | r | The active bank, 0 or 1. |
| 0x002C | `weights_version` | r | Commits since configuration. |
| 0x0030 | `weights_crc` | r | CRC-32 of the active bank. |
| 0x0034 | `weights_idle_crc` | r | CRC-32 of the idle bank. |
| 0x8000 | `weights` | w | The idle bank, one value per word. |

A bank holds, layer after layer in pipeline order, the weights output channel
by channel (`w[k][c]` at `c * inputs + k`), then one bias per channel, then one
multiplier per channel. Each value takes one 32 bit word, masked to its width
with the upper bits zero, and reads of the window return 0. The example MLP
needs 3412 words.

Each bank keeps a CRC-32, zlib's, of the words written into it since the last
write to its word 0, in write order. Both banks start with the CRC of the
compiled image. A host that writes the whole bank once in address order, as
`load_weights` does, reads back `zlib.crc32` of the image it wrote, and a
dropped or repeated write shows up as a different CRC before the commit. The
restart at word 0 lets a host that saw a bad CRC simply write the bank again;
a first version restarted the CRC at each commit instead, and the test of a
lost write showed that a retry could then never match. A write in the same
cycle as a commit is dropped; the shell cannot issue both in one cycle.

`manifest.json` gets a `weights` section with the mode, the bank size, the CRC
of the compiled image and each layer's offsets, counts, widths and shift, and
`build(board=...)` adds the registers above to its `registers` map.

## API

```python
acc.compile(device="vu9p", mode="fixed", weights="reloadable")
acc.load_weights("model_2026-10-25.onnx", registers=regs)
acc.simulate(reload="model_2026-10-25.onnx", commits=[5000])
```

`load_weights` checks the model, then writes the idle bank, reads
`weights_idle_crc` and stops before committing if it differs from the CRC of
what it wrote, commits, and checks that `weights_version` went up by one, that
`weights_bank` flipped and that `weights_crc` is the new CRC. After that,
`predict` and `evaluate` use the new model. `registers` is anything with
`read(offset)` and `write(offset, value)` on byte offsets into the window.
`onnx2fpga.simulate.SimulatedRegisters` is `kria_regs` under Verilator, and the
board's own register file arrives with the Kria runtime, which `deploy(host=)`
still waits for.

`simulate(reload=model, commits=cycles)` writes the new model into the idle
bank one word per cycle from the first cycle while the inputs stream, and
commits on each given cycle, flipping between the two banks. Every answer is
checked against the integer model of the bank it is tagged with.

## Errors

Every error is one sentence.

- `weights` other than `"frozen"` or `"reloadable"`, or `"reloadable"` with
  `mode="stream"` or a tree ensemble: `ConfigError`.
- `arithmetic("constant")` with reloadable weights: `ConfigError: fc0 is set
  to arithmetic('constant'), which needs weights='frozen'`. `"auto"` picks
  multipliers.
- A model with a different architecture or grid: `ConfigError` naming the
  first mismatch, such as `fc1 has weights shaped (64, 16) in the compiled
  model but (64, 32) in the new one` or `fc2 needs a rescale multiplier of
  1048576, past the 18 bits the design holds`.
- `load_weights` or `simulate(reload=...)` on a design with frozen weights:
  `ConfigError`.
- Reloadable weights that do not fit the device, or more than the 8192 words
  of the window: `BudgetError`.
- `load_weights` without `registers` and without a deployed board:
  `NotDeployedError`. A CRC or version that does not read back: `BoardError`.

## Verification

`simulate(reload=..., commits=[5000])` on the example MLP sends 10000 random
vectors, one per cycle, writes a second model into the idle bank one word per
cycle during the first 3412 cycles, and commits on cycle 5000. All 10000
answers arrive exactly 9 cycles after their inputs: the 5001 whose features
arrived up to the commit cycle are tagged bank 0 and match the compiled
integer model, and the 4999 after it are tagged bank 1 and match the new one.
The two models give different answers on more than 95 percent of the vectors,
so a swap one inference early or late fails, and the same run with the two
expectations exchanged does fail. A second run commits on 42 random cycles
between 3412 and 10000, two of them back to back, and passes at full rate and
with `features_valid` high on 50 and 10 percent of cycles. Each run also checks
the active bank every cycle and, at the end, the version and both CRCs.

`load_weights` runs against `kria_regs` itself, generated by the Hardcaml
`kria-regs` command and built with Verilator, driven one AXI4-Lite transfer at
a time over a pipe. The test loads a second model, reads four predictions back
through the registers and compares them with the new integer model, then
drops one write of a third model and gets `BoardError` before any commit, with
the version still 1, and finally loads the compiled model again: bank 0,
version 2, and the CRC of the compiled image from `manifest.json`, with
`predict` giving the compiled answers again. The architecture and grid
mismatches above are tested with a wider layer, a missing layer, a narrower
input and a last layer scaled by 1000, which needs a multiplier past 18 bits.

SymbiYosys proves on the generated Verilog of the whole example MLP that, once
9 cycles have passed, `prediction_bank` equals `weights_bank` delayed by 9
cycles for every input sequence, commits and writes included. k-induction at
depth 11 closes it, a cover task shows the tag changing at step 9, and the same
property with 8 cycles fails. The prove and cover tasks take 42 seconds
together, most of it reading the 8 MB of Verilog.

The Hardcaml expect tests are on the Lws harness. `test_otf_dense.ml` writes a
second bank into an 8 by 5 layer, with writes to the words on either side that
must not land, then sends 300 vectors at 80 percent valid with a random bank
on every cycle and checks each output and its tag three cycles later, from a
random initial state. `test_fixed_top.ml` streams a vector every cycle through
the two layer fixture while it writes the idle bank, commits 41 times, on
cycles 60 and 61 and at random after them, and checks every answer against the model of
its tagged bank, the active bank each cycle, the version and both CRCs. A
second test writes half the bank and streams 100 vectors: every answer uses
the compiled bank and the idle CRC is that of the half, not of the new model,
and writing the whole bank again from word 0 gives the right CRC.
`test_kria_regs.ml` loads the new bank over AXI4-Lite with inferences between
the writes, checks the CRCs before and after the commit, and predicts with the
new weights. The CRC matches zlib's on `12345678`, 0x9ae0daaf, and on the
fixture's two images.

## Resources

yosys 0.52 on the generated `rtl/top.v` of the example MLP, flattened, out of
context and without place and route, with the commands of
[004](004-weight-specialised-arithmetic.md). The frozen rows are 004's; their
Verilog is byte for byte the same today.

| Weights | Arithmetic | DSP48E2 | LUT, stock | Wide mux | CARRY4 | LUT, no `alumacc` | CARRY4, no `alumacc` | Flip flops |
|---|---|---|---|---|---|---|---|---|
| frozen | multiplier | 2660 | 103265 | 41210 | 7976 | 52142 | 15954 | 40044 |
| frozen | constant | 90 | 110645 | 39755 | 15299 | 66994 | 23268 | 28781 |
| reloadable | multiplier | 3322 | 144422 | 33759 | 9014 | 100495 | 18710 | 113457 |

Reloadable takes one DSP48E2 for each of the 3232 weights and one per channel
for the rescale. The frozen multiplier version needs 662 fewer because yosys
folds zeros and powers of two into wiring. The banks are 29462 bits, so they
add 58924 flip flops, and their multiplexers and write decoders most of the
48353 extra LUTs over the frozen multiplier version without `alumacc`. The
compiler estimates 113342 flip flops and 101430 LUTs; its 1706 DSPs assume two
int8 products per DSP48E2, which yosys does not do. So the example MLP does
not fit reloadable on the K26 (`BudgetError`, 1706 DSPs against 998); the
16-32-8 MLP of the tests needs 424 DSPs and 24096 LUTs on the estimate.

The bank of the example MLP is 3412 words, so a reload is 3412 AXI4-Lite
writes and a few reads. How long that takes from the ARM cores has not been
measured.

## Done when

One of the two is met.

- [x] In simulation, a weight swap in the middle of a stream of inputs changes
  the outputs on exactly the first inference after the swap, and every output
  matches the integer model of whichever weights were active.
- [ ] On a Kria board, `load_weights` takes under a second for the example
  models and needs no rebuild. Needs the board, Vivado and the Kria runtime;
  the example MLP also needs a larger part or block RAM banks to fit.

## Trees

Reloadable trees are not built. The plan stands as [005](005-tree-ensembles.md)
describes it: fix the tree count, the depth padded to complete trees, the
feature count and the widths, and put each node's feature index, threshold
and split kind and each leaf in the banks, which costs an F to 1 multiplexer
per node. What blocks it is the feature grid. 005 takes each feature's grid
from the compiled model's thresholds, and a retrained model has other
thresholds, so reloadable trees need a grid fixed up front, from calibration
data, before the bank layout means anything.

## Open questions

Three are open.

- Registers or block RAM for the banks? Registers keep latency lowest; block
  RAM scales. Probably both, chosen by size like [008](008-weights-in-block-ram.md).
- Should the commit wait for a CRC the host writes first, so the hardware
  itself refuses a bank that did not arrive whole?
- Can two int8 products share a DSP48E2 when both weights are registers, as
  the estimate assumes? That would halve the DSPs and let the example MLP fit
  the K26.
