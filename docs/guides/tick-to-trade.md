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
last byte to decision: 14 cycles, 43.4 ns at 322.27 MHz, every message (moldudp64 1, itch_parser 1, order_book 2, features 3, model 6, decision 1)
first byte to decision: min 16, median 18, p99 19, max 19 cycles
```

`--seed` picks another day, `--gap 30` leaves 30 percent of cycles idle
between beats, and `--out` names the build directory, `build/tick_to_trade` by
default. A difference stops the run and prints the message, the RTL's output
and the model's.

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
| `book_*` | | The updated symbol's book, two cycles after each tracked message. |
| `features_valid`, `features` | 1, 32 | Spread, top imbalance, depth imbalance, trade flow, as int8. |
| `decision_valid`, `action`, `prediction` | 1, 2, 8 | One per tracked message: hold, buy or sell. |
| `order_valid`, `order` | 1, 376 | A 47 byte OUCH 5.0 Enter Order when the action is not hold. |
| `t_first`, `t_last`, `t_out` | 32 each | The cycle counter at the message's first byte, its last byte, and now. |

## Tests

`make test-sim` runs `tests/test_tick_to_trade.py`: the generator's framing
and determinism, the 100000 message replay with and without idle cycles, and
the two SymbiYosys proofs when `sby` is installed. `make test-hardcaml` runs
the expect tests for each unit.
