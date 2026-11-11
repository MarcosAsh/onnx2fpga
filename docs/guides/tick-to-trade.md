# Tick to trade replay

Status: implemented in simulation, see [007](../specs/007-tick-to-trade.md).

The replay generates a day of synthetic NASDAQ ITCH 5.0 traffic, wraps it in
MoldUDP64 packets, and streams it through the Hardcaml tick to trade path in
Verilator, 8 bytes a cycle. A Python model of the same path runs over the same
packets, and every book, feature vector and decision the RTL produces has to
match it exactly.

```
python3 -m onnx2fpga.trading.replay --messages 100000
```

Run it from a checkout with `PYTHONPATH=compiler`. It needs the Hardcaml
toolchain (see [backends](backends.md)) and Verilator 5, and takes about 15
seconds:

```
100000 messages in 4934 packets, 410437 beats; 65151 tracked messages, all identical
orders: 6627 buy, 6942 sell; 471 adds dropped on a full slot, 2570 levels forgotten
last byte to decision: 15 cycles, 46.5 ns at 322.27 MHz target, every message (moldudp64 1, itch_parser 1, order_book 3, features 3, model 6, decision 1)
first byte to decision: min 17, median 19, p99 20, max 20 cycles
```

`--seed` picks another day, `--gap 30` leaves 30 percent of cycles idle
between beats, and `--out` names the build directory, `build/tick_to_trade` by
default. A difference stops the run and prints the message, the RTL's output
and the model's.

## K26 clock target

The default replay keeps the six-cycle model compiled for `vu9p`. To pipeline
the model for the K26 trading clock:

```sh
PYTHONPATH=compiler python3 -m onnx2fpga.trading.replay \
  --device k26 --clock-mhz 322.265625 --messages 100000 --out build/tick_k26
```

The model takes 12 cycles and the full path takes 22. The 100000-message
replay matches all 65151 tracked messages, including the book, features and
orders. The printed nanoseconds use the requested clock, not a measured
clock. The whole pipeline does not yet meet 322.265625 MHz.

For out-of-context place and route, run from the generated directory:

```sh
cd build/tick_k26
OTF_PART=xck26-sfvc784-2LV-c OTF_TOP=tick_to_trade OTF_RTL=tick_to_trade.v \
OTF_PERIOD=3.103 OTF_CLOCK=clock OTF_IMPL=1 OTF_OUT=vivado \
  vivado -mode batch -nojournal -nolog -source ../../hardware/synth/synth_unit.tcl
```

The implementation run saves `vivado/routed.dcp`. From the same build
directory, refine its placement and routing with:

```sh
OTF_PART=xck26-sfvc784-2LV-c OTF_TOP=tick_to_trade OTF_PERIOD=3.103 \
OTF_CHECKPOINT=vivado/routed.dcp OTF_OUT=vivado_refined \
  vivado -mode batch -nojournal -nolog -source ../../hardware/synth/synth_unit.tcl
```

The script checks the checkpoint's part and clock period, runs aggressive
physical optimization, then reroutes with the Explore directive. Reports and
the refined checkpoint go to `vivado_refined`.

The order table stores payloads in block RAM and live-state flags in
distributed RAM. Reads take two cycles. Forwarding makes simultaneous writes
and pending flag updates visible to reads. The book accepts one event every
two cycles and produces its snapshot after four cycles.

The book compares insertion prices before compacting removed levels, then
moves those comparison bits with the levels. A register separates removal
from insertion. `tests/test_tick_to_trade.py` proves the result equals the
four-level RTL from revision `5244d34`, delayed one cycle, for all inputs.
The reference is in `tests/fixtures/book_side_baseline.v`.

Vivado 2026.1 out-of-context routed results on `xck26-sfvc784-2LV-c`,
with a 3.103 ns constraint:

| Variant | Cycles | LUTs | Registers | BRAM36 equivalents | URAMs | Setup slack (ns) |
|---|---:|---:|---:|---:|---:|---:|
| Original | 14 | 7153 | 4450 | 1 | 2 | -6.903 |
| K26 model, block RAM | 20 | 7148 | 4865 | 14.5 | 0 | -5.692 |
| Parallel book comparisons | 20 | 7643 | 4815 | 14.5 | 0 | -3.386 |
| Pipelined book update | 21 | 6434 | 5674 | 14.5 | 0 | -2.759 |
| Early table address | 21 | 6443 | 5674 | 14.5 | 0 | -2.352 |
| Earlier book register | 21 | 6565 | 5739 | 14.5 | 0 | -1.948 |
| Parallel feature normalization | 21 | 6701 | 5763 | 13.5 | 0 | -2.052 |
| Stored framer offset | 21 | 6299 | 5777 | 13.5 | 0 | -1.910 |
| Early payload writes | 21 | 17825 | 5977 | 0 | 0 | -1.564 |
| Pipelined table, fixed header lanes | 22 | 7402 | 6104 | 13.5 | 0 | -1.110 |
| Narrow payload banks, early lookup | 22 | 7787 | 6778 | 13.5 | 0 | -0.569 |
| Post-route refinement | 22 | 7787 | 6778 | 13.5 | 0 | -0.472 |
| Early crossing, separate removal comparison | 22 | 8353 | 6741 | 13.5 | 0 | -0.348 |
| Post-route refinement of early crossing | 22 | 8353 | 6741 | 13.5 | 0 | -0.299 |

The first change includes both the model pipeline and the RAM mapping.
Reordering the book comparisons alone improves the timing-derived frequency
estimate from 113.70 to 154.11 MHz. Pipelining the update reaches 170.59 MHz
with one more cycle. Selecting the RAM address from the event phase
instead of the write decision reaches 183.32 MHz at the same latency.
Moving the book register before removal reaches 197.98 MHz. At that stage,
the critical path runs from an order-table RAM output to its write enable.
These are routed estimates, without board I/O constraints or a physical board measurement.
The [measurement data](../measurements/tick-to-trade-k26-2026-11-11.json)
records forty-four runs, including the intermediate memory layouts.

Feature normalization selects the four-bit window directly and keeps the
imbalance lookup in distributed ROM. Its stage slack improves from -1.127
ns to -0.415 ns in that revision.
`tests/fixtures/features_baseline.v` holds the four-level feature RTL from `402ec54`.
The sequential equivalence check covers all inputs, outputs and the
three-cycle latency.

The framer stores its byte offset instead of subtracting the remaining
length on every beat. Its proof checks that the offset stays equal to
`length - skip` during the body and that framing still matches the serial
reference. Together these changes reach 199.48 MHz at the same latency.

Early payload writes reach 214.27 MHz with the same 21-cycle latency, using
17825 LUTs and no block RAM. The wide payload write no longer waits for
price acceptance. Rejected writes may change an empty slot's payload;
the live flags still follow acceptance. Both crossing comparisons run
before side selection.

Two-cycle payload reads and registered flag writes reach 237.36 MHz with
22 cycles of total latency. The book separates removal from insertion, and
the framer reads the packet count from fixed header lanes. Its proof checks
that header states have 20, 12 or 4 bytes left. The design uses 7402 LUTs and
13.5 block-RAM equivalents. At the routed frequency estimate, 22 cycles take
92.7 ns, compared with 98.0 ns for the 21-cycle, 214.27 MHz revision.

Payload banks of at most nine bits reach 272.33 MHz at the same 22-cycle
latency, or 80.8 ns at the routed frequency estimate. The banks limit BRAM
cascading. Entry matching and quantity reduction use the first read stage;
bid and ask updates run in parallel. Removal shares its subtraction with
the empty-level check, and event delays stay in flip-flops. This revision
uses 7787 LUTs, 6778 flip-flops and 13.5 block-RAM equivalents. Its worst
slack remains -0.569 ns against the 322.265625 MHz target.

Post-route refinement reaches 279.72 MHz with the same RTL and resource
counts. The 22-cycle path takes 78.7 ns at that estimate. Hold slack is
positive at 0.033 ns; setup slack is still -0.472 ns against the target.

Registering the crossing check before the book update and comparing removed
shares separately from the subtraction reach 289.77 MHz at 22 cycles, or
75.9 ns at the routed estimate. The crossing check forwards the preceding
book update when events for the same symbol arrive two cycles apart. The
minimum-gap tests cover rejected crosses and deletes that allow the next add.
This revision uses 8353 LUTs and 6741 flip-flops, with the same DSP and BRAM
counts. Setup slack remains -0.348 ns against the target. Post-route
refinement of this revision reaches 293.94 MHz, or 74.8 ns at 22 cycles,
with unchanged resource counts. Setup slack is -0.299 ns and hold slack
is positive at 0.029 ns.

The table induction check covers a four-slot instance for arbitrary legal
operation sequences, including pending flag writes, simultaneous reads and
writes, and initialization. It checks both read stages and every payload bank.
Extra payload writes preserve live entries.
A combinational proof checks crossing comparisons for all inputs. The
maximum-rate book tests cover rejected same-slot replacements and occupied
targets. The 100000-message K26 replay still matches every tracked message.

## What runs

The generator in `compiler/onnx2fpga/trading/itch.py` simulates six stocks
with a price-time priority book each. About 42 percent of messages are adds
near the touch (6 percent of them F messages with an attribution), 30 percent
deletes, 12 percent replaces, 9 percent executions at the best level (15
percent of those as C with a price), 5 percent partial cancels and 2 percent
trade messages the book ignores, after a system event and a stock directory.
Adds never cross the other side, as on the real feed. Packets hold up to 40
messages and 1400 bytes, with a heartbeat now and then and an end of session
packet last. The same seed gives the same bytes.

The design tracks four of the six stocks. The model is a 4 by 8 by 1 MLP with
random weights, calibrated on the stream's own features and compiled in fixed
latency mode; the buy and sell thresholds sit at the 90th and 10th percentile
of its predictions, so about one decision in five sends an order. The whole
configuration, model included, is written to `tick_to_trade.json` in the build
directory, and the generator turns it into `tick_to_trade.v`:

```
hardware/hardcaml/_build/default/bin/generate.exe tick-to-trade tick_to_trade.json
```

`-stages` prints each unit's latency instead, and `-unit moldudp64_step` or
`-unit book_side` emits one unit alone, which is what the proofs read.

## The ports

| Port | Width | Meaning |
|---|---|---|
| `clock`, `clear` | 1 | One clock domain; `clear` is synchronous. |
| `data`, `keep`, `valid`, `last` | 64, 8, 1, 1 | UDP payload bytes, byte 0 in the low bits. |
| `ready` | 1 | Low for 4096 cycles after a clear while the order table empties. |
| `book_*` | | The updated symbol's book, three cycles after each tracked message. |
| `features_valid`, `features` | 1, 32 | Spread, top imbalance, depth imbalance, trade flow, as int8. |
| `decision_valid`, `action`, `prediction` | 1, 2, 8 | One per tracked message: hold, buy or sell. |
| `order_valid`, `order` | 1, 376 | A 47 byte OUCH 5.0 Enter Order when the action is not hold. |
| `t_first`, `t_last`, `t_out` | 32 each | The cycle counter at the message's first byte, its last byte, and now. |

## Tests

`make test-sim` runs `tests/test_tick_to_trade.py`: the generator's framing
and determinism, the 100000 message replay with and without idle cycles, and
the two SymbiYosys proofs when `sby` is installed. `make test-hardcaml` runs
the expect tests for each unit.
