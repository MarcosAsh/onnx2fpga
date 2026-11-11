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

The model takes 12 cycles and the full path takes 21. The 100000-message
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

The order table explicitly requests block RAM. The book compares insertion
prices before compacting removed levels, then moves those comparison bits
with the levels. `tests/test_tick_to_trade.py` proves this update equivalent
to the four-level RTL from revision `5244d34` for all inputs, using
`tests/fixtures/book_side_baseline.v`. The existing invariant proofs and
replay checks still apply. A register separates removal and insertion; the
sequential proof checks that its result equals the original update delayed
one cycle. The table still writes in its second cycle, and the book accepts
one event every two cycles.

Vivado 2026.1 out-of-context routed results on `xck26-sfvc784-2LV-c`,
with a 3.103 ns constraint:

| Variant | Cycles | LUTs | Registers | BRAM36 equivalents | URAMs | Setup slack (ns) |
|---|---:|---:|---:|---:|---:|---:|
| Original | 14 | 7153 | 4450 | 1 | 2 | -6.903 |
| K26 model, block RAM | 20 | 7148 | 4865 | 14.5 | 0 | -5.692 |
| Parallel book comparisons | 20 | 7643 | 4815 | 14.5 | 0 | -3.386 |
| Pipelined book update | 21 | 6434 | 5674 | 14.5 | 0 | -2.759 |

The first change includes both the model pipeline and the RAM mapping.
Reordering the book comparisons alone improves the timing-derived frequency
estimate from 113.70 to 154.11 MHz. Pipelining the update reaches 170.59 MHz
with one more cycle. The remaining critical path starts at an order-table
RAM output and ends at the new book pipeline register. These are routed
estimates, without board I/O constraints or a physical board measurement.
The [measurement data](../measurements/tick-to-trade-k26-2026-11-11.json)
records all four runs.

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
