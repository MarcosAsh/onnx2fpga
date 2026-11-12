# 007 Tick to trade demo

Status: partial. Fabric DDR replay through the trading pipeline matches the
software model on every book update, feature vector and decision for 100,000
synthetic messages and all 1,519,865 available AAPL messages in a recorded sample.
The K26 pipeline takes 22 cycles from a
message's last byte to its decision; the original unpipelined design takes 14.
Unit proofs cover selected invariants; replay checks each bounded book snapshot.
Host DMA ownership and recovery pass simulation. A validated routed Kria trading
build, physical deployment and all of stage B remain unfinished. The recorded
source ends without a session terminator, so its completeness is unverified.

## Goal

One design that shows the whole path a trading FPGA runs: a market data packet
comes in on the wire, a decision goes out, and we measure how many cycles that
took. Everything is replayed from data anyone can generate or download, so
anyone can run it.

## Two stages

Stage A runs on either Kria board. A replayer in the fabric reads ITCH messages
from DDR and feeds them into the parser at line rate, so the whole path from
parser to decision runs and gets measured with no network at all. Today stage
A runs in Verilator, with the RTL replayer connected to a C++ DDR model.

The standalone `ddr_replay` unit reads 128-bit records through an AXI4 master.
Each record holds a 64-bit input beat in bits 63:0, `keep` in 71:64, `last` in
bit 72 and zero padding above it. The Python `trading.replay_records` codec uses that
layout. A buffer must end at a packet boundary.

The base address is 256-byte aligned. Bursts contain at most 16 records, so
they cannot cross a 4 KiB boundary. Only one burst is outstanding. The engine
holds address requests under backpressure and waits for the consumer before
accepting data. Bad AXI responses, masks, padding or burst endings stop output
and drain the current burst. Abort also drains it before accepting another
start. Hardware clear must reset the AXI interconnect too; software abort must
not clear outstanding bus state.

Randomized simulation covers address, data and consumer stalls across a 4 KiB
boundary. Directed cases cover early/late `RLAST`, response and record errors,
abort during address/data transfer, empty buffers, misalignment and address
overflow.

`ddr_trading` connects that engine to the complete trading pipeline. A start
clears the parser and book, waits for table initialization, then starts DDR reads.
Completion waits for the final decisions to leave the pipeline. Abort drains
the current burst and pending decisions; a subsequent start clears the book
again. Simulation compares every book update, feature vector and decision to
the software model, including repeated runs and abort followed by restart.

To reproduce the 100,000-message synthetic DDR replay:

```sh
PYTHONPATH=compiler:. python3 tools/replay_ddr.py --out build/ddr_trading_100k
```

The report records input/model/RTL hashes, source hashes, simulated cycle counts
and comparison results. The AXI memory model stalls address and data channels;
it does not model physical DDR bandwidth or a Linux DMA allocation. The KR260
project connects the reader to the PS HP0 port; its routed result and physical
operation still need validation.

The [100,000-message record](../measurements/ddr-trading-2026-11-12.json) covers
410,694 DDR records and 65,267 decisions per run. Four runs include restart,
with 0% and 65% randomized stalls. All match the software model and retain the
22-cycle last-byte-to-decision latency.

### KR260 replay control

After replay simulation, generate or build the board project with:

```sh
python3 tools/build_ddr_kr260.py build/ddr_trading_100k --generate-only
python3 tools/build_ddr_kr260.py build/ddr_trading_100k
```

The builder checks the report's input/configuration hashes and regenerates the
trading core to confirm it matches the simulated RTL. The board adds an AXI4-Lite
control window at `0xA0000000` and a 128-bit read path to PS HP0. That path maps
only low DDR, addresses `0` through `0x7fffffff`; OCM is excluded. A replay buffer
must belong to a DMA allocation with the required cache synchronization and
lifetime. The host runtime and allocation helper are described below; physical
DDR access still requires board validation.

| Byte offset | Register | Meaning |
|---|---|---|
| `0x00` | ID | `0x4f544432` (`OTD2`) |
| `0x04` | Version | `1` |
| `0x10` | Control | Bit 0 starts; bit 1 aborts and takes priority |
| `0x14` | Status | Bits 0 through 3: busy, done, error, aborted |
| `0x20`, `0x24` | Buffer address | Low 32 and high 8 bits; unused high bits must be zero |
| `0x28` | Records | Number of 16-byte DDR records |
| `0x30` | Delivered | Records delivered to the parser |
| `0x34`, `0x38` | Decisions, orders | Completed decisions and generated orders |
| `0x3c` | Cycles | Start through initialization and final pipeline drain |
| `0x40` | Latency | Last byte to decision, in pipeline cycles |
| `0x44` | Latest result | Prediction in bits 7:0, action in 9:8, order-valid in bit 10 |
| `0x48`, `0x4c`, `0x50` | Timestamps | First byte, last byte and output for the latest decision |
| `0x100` through `0x12c` | Latest order | Twelve little-endian words; the final byte is padding |

Configuration writes honor byte strobes. Write address and data may arrive in
either order. A start while busy returns `SLVERR`; unknown writes also return
`SLVERR`, and unknown reads return `DECERR`. Decision snapshots clear on start
and update on each decision. Read them after completion for a consistent
snapshot. Exposing the prediction and complete order keeps synthesis from
discarding their data paths when building the board design.

Stage B needs the KR260, the board with the SFP+ cage. It adds a 10G MAC and
PCS so real frames come in from a PC with a 10G card and decisions go back out.

## The path

Each step is one Hardcaml unit with a fixed number of register stages and no
handshake.

1. A 10G Ethernet frame arrives on the KR260's SFP+ port, or in stage A, the
   replayer hands over the UDP payload.
2. `moldudp64` pulls NASDAQ TotalView ITCH 5.0 messages out of MoldUDP64
   packets.
3. `itch_parser` decodes add (A and F), execute (E and C), cancel (X), delete
   (D) and replace (U) messages for a short list of stock locates.
4. `order_book` keeps the top price levels of each side for each symbol, over a
   table of the orders behind them.
5. `features` turns the updated book into four int8 features: spread, top of
   book imbalance, depth imbalance and recent trade flow.
6. The model, compiled by onnx2fpga in fixed latency mode, scores them.
7. `decision` applies a threshold and fills in a prebuilt order message.

A cycle counter stamps each message at its first and last byte, and the
decision carries both stamps and the counter at its own output. We publish the
distribution, not one best case.

## Stage A as built

### Input

The path takes the UDP payload 64 bits a cycle, byte 0 in the low bits, with
`valid`, an 8 bit `keep` and `last` on the final beat of a packet. `keep` is
all ones except on that last beat, where it is a run of ones from bit 0. There
is no `ready`: the path takes a beat whenever one comes.

### Framer

A MoldUDP64 packet is a 20 byte header (session, sequence number, message
count) followed by that many blocks, each a 2 byte big endian length and the
message. The framer handles all eight lanes in one step. ITCH 5.0's shortest
message is 12 bytes, so a block is longer than a beat and a beat holds the end
of at most one message and the start of at most one length field; the step
only needs the position of that boundary, not a byte by byte walk. It keeps the
first 40 bytes of each message, enough for the longest one we decode, and
emits it one cycle after the beat with its last byte.

A count of 0xffff, end of session, reads as zero. Bytes after the counted
blocks are ignored. A length under 12 drops the rest of the packet, as does a
packet that ends inside a message.

### Parser

The parser checks the type, the length the specification gives for it, and the
stock locate against its list, up to eight locates. A message that passes
becomes one event carrying the order reference, the new reference for a
replace, the side, the shares and the price. One register stage.

### Book

Each symbol has `levels` price levels per side, best first, as registers.
Every order on a tracked symbol is kept in a table of `2^order_bits` slots
indexed by the low bits of its reference number, with the other bits as a tag,
so a lookup never matches the wrong order. NASDAQ assigns reference numbers in
order through the day, so the low bits spread evenly. The demo uses 4 symbols,
4 levels and 4096 slots, which yosys maps to 14 block RAMs.

An add whose slot already holds a live order is dropped, order and level both.
So is an add that would cross the other side's best price, which the real feed
never sends. Together these keep the levels counting only orders the table
holds. A level pushed past the last one is forgotten, and removing more shares
than a level holds empties it. With 4096 slots the 100000 message replay drops
471 adds, and 2570 levels fall off the end; a book with more levels or slots
loses fewer, and the software model makes the same choices, so the comparison
stays exact.

The table is a true dual port RAM. Port A reads the order an event names and
port B the new reference of a replace, in the cycle the event arrives; the next
cycle the levels update and the ports write back. That needs events at least
two cycles apart, and the framer guarantees it: the shortest tracked message,
a delete, is 19 bytes, and with its length field takes more than two beats.
The proof below checks that a message of 15 bytes or more never completes on
the beat after another.

Executions also feed a per symbol trade flow: a signed sum of executed shares,
each capped at 65535, positive when a resting sell order traded, that loses an
eighth of itself on every execution. It fits in 20 bits without saturating.

After a clear the book spends 4096 cycles writing every slot empty and holds
`ready` low until it is done.

### Features

Four signed bytes, with no division. The spread is the best ask minus the best
bid shifted right by 6, capped at 127, or 127 when a side is empty. The two
imbalances, of the best level's shares and of all levels' shares, shift both
counts right until the larger fits in four bits and look up
`127 * (bid - ask) / (bid + ask)` in a 256 entry table. The flow is shifted
right by 5 and clamped to 127 either way. Three register stages.

### Model and decision

The model is any fixed latency design from [001](001-fixed-latency.md) that
takes four int8 features and gives one int8 prediction. The replay compiles a
4 by 8 by 1 MLP, 6 cycles, calibrated on the stream's own features.

A prediction at or above the buy threshold buys at the best ask, and one at or
below the sell threshold sells at the best bid; with no level on that side the
decision holds. The order is a 47 byte OUCH 5.0 Enter Order, immediate or
cancel, prebuilt except for the side, symbol, price and a token that counts
orders and serves as both UserRefNum and ClOrdID. One register stage.

### Latency

Measured from the cycle the beat holding a message's last byte is presented to
the cycle its decision is on the output.

| Unit | Cycles | Target |
|---|---|---|
| `moldudp64` | 1 | 8 for parsing |
| `itch_parser` | 1 | |
| `order_book` | 2 | 8 |
| `features` | 3 | 3 |
| model, 4-8-1 MLP | 6 | |
| `decision` | 1 | 3 |
| total | 14, 43.4 ns at 322.265625 MHz | 32 |

The model is 6 of the 14. The count does not depend on the data, and the replay
checks it on every one of 65151 decisions. From the message's first byte the
time is 16 to 19 cycles when beats arrive back to back, since a tracked
message spans 3 to 6 beats.

These are cycle counts. yosys puts 19 cells on the longest path through the
framer step and 26 through the book side update, counting LUTs, wide muxes and
carry cells, and at 3.1 ns a cycle both will need more register stages once
Vivado reports timing. The budget leaves room for them.

## Verification

`tests/test_tick_to_trade.py` generates 100000 messages, compiles the MLP,
generates the RTL and replays the stream through Verilator in about 15
seconds. All 65151 tracked messages give the same book snapshot, the same
features, the same prediction and the same order bytes as the Python model in
`compiler/onnx2fpga/trading/reference.py`, and the message stamps land on the
beats the model expects. A second run leaves 30 percent of cycles idle and
matches too. We checked that the comparison fails when the model's spread
shift changes, when it keeps three levels instead of four, or when it expects 13
cycles.

Two proofs run with SymbiYosys on the generated Verilog, each over every input
in one step, so k-induction closes at depth 1. The framer's beat step equals
eight steps of a byte at a time framer written separately in SystemVerilog, on
every field of the next state and on the emitted message, from every state that
satisfies the framer's invariant, and keeps that invariant. Since the
registers start in such a state, the framer emits exactly the messages the
byte at a time framer would, so it drops none. This takes about 64 seconds
with Bitwuzla. The book side update keeps a side sorted strictly best first,
gap free, with empty levels zero, and uncrossed against the other side's best;
the registers start empty and change only through it, so every reachable book
has those properties. That takes about 2 seconds. Each proof fails when a
bug is put in on purpose: a wrong lane in the framer's timestamp logic, or an
add allowed to cross.

Each unit has Hardcaml expect tests in `hardware/hardcaml/test`: messages
whose length fields and bodies straddle beats, malformed packets, every ITCH
kind and the ones dropped, book side corner cases, collisions and replaces in
the same slot, features against an OCaml model on 300 random books, the order
frame, and the whole path on a short packet.

## Resources

yosys 0.52, `synth_xilinx -family xcup`, out of context on the replay's
`tick_to_trade.v` with the 4-8-1 MLP, in about 15 seconds. Both of the model's
layers take constant arithmetic ([004](004-weight-specialised-arithmetic.md)),
the default, so its only DSPs are the 9 requantisers.

| Unit | LUT | MUXF7/8/9 | CARRY4 | FF | DSP | RAMB36 |
|---|---|---|---|---|---|---|
| `moldudp64` | 2219 | 251 | 307 | 812 | | |
| `itch_parser` | 125 | 44 | | 263 | | |
| `order_book` without the table | 5023 | 2184 | 137 | 3005 | | |
| order table | 246 | | | 244 | | 14 |
| `features` | 747 | 219 | 34 | 215 and 133 SRL16 | | |
| model | 1258 | 299 | 253 | 596 | 9 | |
| `decision` | 85 | 7 | 24 | 240 | | |
| cycle counter, context held past the model | | | 8 | 32 and 133 SRL16 | | |
| total | 9703 | 3004 | 763 | 5407 and 266 SRL16 | 9 | 14 |

On the K26 that is 8 percent of the LUTs, 2 percent of the flip flops, under 1
percent of the DSPs and 10 percent of the block RAM. As in 001, yosys builds
comparators and adders for UltraScale+ out of LUTs and wide muxes, so the LUT
count waits for Vivado.

## Data

NASDAQ publishes [sample ITCH 5.0 days](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/)
of several gigabytes. The default replay uses
`compiler/onnx2fpga/trading/itch.py`: six stocks with
a price-time priority book each, adds near the touch, executions at the best
level, cancels, deletes and replaces in roughly the mix of a NASDAQ day, plus
messages the book ignores, in MoldUDP64 packets of up to 40 messages with
heartbeats and an end of session packet. The same seed gives the same bytes.
To replay recorded data, supply an uncompressed or gzip-compressed
[BinaryFILE](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/binaryfile.pdf):

```sh
PYTHONPATH=compiler:. python3 tools/replay_ddr.py \
  --itch build/itch_samples/12302019.NASDAQ_ITCH50.gz \
  --allow-eof --symbol AAPL --messages 2000000 \
  --order-bits 15 --order-hash xor --out build/ddr_recorded_aapl
```

The reader finds the symbol's stock locate in its directory entry, retains its
first requested messages in source order, and builds new MoldUDP64 packets.
Payload bytes stay unchanged. The limit can select a prefix or exceed the file's
message count to replay every available message for that symbol. Replay uses
simulated stalls; it does not reproduce the original network packets or market arrival times.
The report records the source-file and scanned-message hashes, selection count,
and whether scanning reached the session terminator or exhausted the file.
A full scan rejects a missing terminator by default. For captures that end
between records, `--allow-eof` accepts the file while leaving
`source_session_complete` false. Truncated records are always rejected.

The software comparison uses the same bounded order table and price levels as
the hardware. Reports include table collisions and forgotten price levels;
agreement with this model does not establish a complete exchange order book.

The [recorded AAPL result](../measurements/ddr-recorded-aapl-2026-11-12.json)
uses all 1,519,865 AAPL messages found after scanning 268,744,780 source records
in Nasdaq's 30 December 2019 sample. It covers 6,005,842 DDR records and
1,512,179 decisions per run. All four runs,
with zero/65 percent stalls and restart, match the software books, features
and decisions at 22 cycles from the parser's last byte to its decision.
The 32,768-slot, four-level reference book uses XOR-folded order IDs and records
328,699 table collisions and 152,775 forgotten levels. These losses limit this configuration
for reconstructing that market session.

The source gzip passed its integrity check. The record includes the complete
compressed-file SHA-256 and source URL. Nasdaq's advertised `.md5sum` URL returned
HTTP 404, so no publisher checksum was verified. The scan exhausted the source
file at a record boundary without finding a BinaryFILE session terminator.
The result covers all available AAPL messages, with source-session completeness
explicitly false.

## Host buffer ownership

`trading.runtime.ReplaySession` controls the DDR registers through a `read`/`write`
transport. Opening verifies the device ID, version and latency, then aborts any
previous replay and waits for its outstanding reads to drain. `run(records)`
validates packed records, copies them to the buffer, transfers ownership to the
device and starts replay. It returns counters and the final decision snapshot.
Timeouts and transport failures require `abort()` before retrying.

`trading.dma.DmaBuffer` maps an existing Linux
[u-dma-buf allocation](https://github.com/ikwzm/udmabuf#sync_for_device).
The allocation must be noncoherent, 256-byte aligned and entirely within the
board design's low 2 GiB DDR window. Its DMA address must map directly to the
physical address used by HP0; translated IOMMU addresses are unsupported.
The adapter uses explicit `DMA_TO_DEVICE`
cache synchronization and holds an exclusive advisory lock on the device.
It refuses writes or close while the device owns the buffer. The native
`kria_regs` transport also holds an exclusive advisory lock on its register
device. Other programs accessing these devices must honor those locks.

The caller owns both resources: close the replay session before closing the
buffer and register transport. If abort cannot establish that the engine is
idle, keep the allocation reserved until the fabric has been stopped. This
adapter does not allocate arbitrary physical memory, unload the DMA driver,
or program the FPGA. Provision the allocation on the board before opening it.

Host tests cover ownership transitions, repeated replay, timeouts, lost start
responses, failed aborts and exclusive access. The host-to-RTL test drives the
generated registers through stalled DDR reads, compares decision snapshots and
counters with the software model, and checks timeout recovery and DDR errors.
Cache coherency and physical DDR access still require board validation.

### Provision Linux for replay

The board image needs the out-of-tree
[u-dma-buf driver](https://github.com/ikwzm/udmabuf#compile), built for its running
kernel and installed in its module search path. This is distinct from Linux's
`udmabuf` driver. Install the native register helper from this checkout on the
board:

```bash
make -C runtime bin/kria_regs
sudo install -m 0755 runtime/bin/kria_regs /usr/local/bin/kria_regs
sudo python3 tools/provision_ddr.py --records records.bin
```

The provisioning helper rounds the file size to a page boundary. It loads an
installed `u-dma-buf` module with `udmabuf0=<bytes>` and `dma_mask_bit=31`, then
checks the allocation's size, alignment, address range, device identity and
cache synchronization attributes. It rejects coherent allocations and an
attached IOMMU group. The board image must still provide direct DMA addresses
for HP0; these checks do not prove that mapping on an untested image.

An existing `udmabuf0` is checked and retained. The helper does not resize it,
unload a module, change permissions or touch buffer contents. If the module
is already loaded without that device, loading fails. If allocation fails,
check the kernel log and available contiguous memory on the board; the helper
does not edit boot configuration or reserve arbitrary physical addresses.
`--bytes 0x8000000` requests 128 MiB without a replay file. All allocation and
module-loading tests use fixtures and mocked commands; no kernel module was
loaded during these tests.

On an already programmed board with the package's `manifest.json`, a provisioned
`udmabuf0` and the native register helper installed, run:

```bash
python3 tools/run_ddr.py manifest.json records.bin --buffer udmabuf0 --timeout 30
```

The command prints counters and the last decision as JSON. It runs locally on
the board and does not program it or change device permissions.

## What is ours and what is borrowed

The parser, order book, features and model are written in Hardcaml in this
repo. For the MAC and PCS, taxi has a KR260 10G example that needs no licensed
IP and builds with free Vivado. Its CERN-OHL-S licence is reciprocal while this
repo is Apache-2.0, so it stays a separate build dependency, never copied in.
The older MIT licensed verilog-ethernet is the fallback.

Everything runs in one 322.265625 MHz clock domain, the MAC's own, so no clock
crossing FIFO adds cycles on the way in or out. taxi's low latency MAC is 32
bits wide; on a 64 bit path at the same clock it would fill every other beat.

## Targets

We measure the way STAC-T0 does, from the last byte needed to the first byte
out, and publish the distribution. The first version aims for at most 32
cycles, 99 ns, inside the fabric: about 8 for the parser, 8 for the book, 3 for
features, the model's own cycles, and 3 for the decision and a prebuilt order
template. 20 cycles is the stretch goal. Wire to wire, with the MAC, PCS and
transceiver included, is reported separately.

For scale: the fastest published tick to trade is 13.9 ns with no model in the
path, and the ML products that publish numbers sit at 1 to 3 us because the
model runs behind the host. A model inside the path at 10 to 40 ns is the gap
this demo is aimed at.

## Done when

- [ ] Replaying a sample ITCH day through the design gives decisions identical
  to a software model of the same pipeline run on the same file. All available
  AAPL messages in the sample pass in Verilator; source completeness is unverified.
- [x] We report tick to trade latency in cycles and ns, with the model's share
  of it broken out.
- [x] The parser has a formal proof that no message is dropped.
- [ ] The book after each message equals the software book, by proof. The
  replay checks it on every message, and the book's invariants are proved.
- [ ] Stage A on a Kria board, with Vivado's timing at 322.265625 MHz.

## Open questions

- How many symbols, levels and table slots the first board build keeps. Four,
  four and 4096 cost 14 block RAMs and about 5000 flip flops; the table is the
  first thing to grow, into UltraRAM.
- Whether the framer should take 32 bit beats to match taxi's MAC directly.
- How to pipeline the framer and the book side for 322.265625 MHz. Compiled
  with `clock_mhz=322.265625` for the `k26` ([002](002-pipelined-matvec.md)),
  the 4-8-1 MLP takes 12 cycles instead of 6 and the delay model predicts 402
  MHz for it, which would make the path 20 cycles, 62.1 ns. The framer step and
  the book side update are another matter. At the `k26` model's 0.62 ns for a
  net and a LUT, their 19 and 26 cells come to about 12 and 16 ns, 83 and 61
  MHz. Both are recurrences: the framer's next state and the book's next levels
  depend on the current ones within the cycle, so a register cannot go in
  without changing how often they take a beat or an event. The book has room,
  since events arrive at least two cycles apart and it could spread its update
  over both; the framer would need to work on 32 bit beats or split its lane
  logic. The replay keeps the unpipelined model and its 14 cycles until both
  are done.
