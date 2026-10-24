# 007 Tick to trade demo

Status: planned

## Goal

One design that shows the whole path a trading FPGA runs: a market data packet
comes in on the wire, a decision goes out, and we measure how many cycles that
took. Everything is replayed from public data, so anyone can run it.

## Two stages

Stage A runs on either Kria board. A replayer in the fabric reads ITCH messages
from DDR and feeds them into the parser at line rate, so the whole path from
parser to decision runs and gets measured with no network at all.

Stage B needs the KR260, the board with the SFP+ cage. It adds a 10G MAC and
PCS so real frames come in from a PC with a 10G card and decisions go back out.

## The path

1. A 10G Ethernet frame arrives on the KR260's SFP+ port, or in stage A, the
   replayer hands over the same bytes.
2. A UDP and MoldUDP64 parser pulls out NASDAQ TotalView ITCH 5.0 messages.
3. An ITCH parser decodes add, execute, cancel and delete messages for a short
   list of symbols.
4. A small order book keeps the top few price levels per symbol.
5. A feature block turns the book into a feature vector: spread, imbalance,
   recent trade flow.
6. The model, compiled by onnx2fpga in fixed latency mode, scores it.
7. A decision block applies a threshold and emits an order message, for the
   demo just a frame sent back out of the port.

A cycle counter stamps the frame at the first byte in and the first byte out,
and the design reports the difference for every decision. We publish the
distribution, not one best case.

## Data

NASDAQ publishes sample ITCH 5.0 files. A host program replays one at line rate
into the KR260's SFP+ port from a PC with a 10G card. Stage A replays the same
file from DDR instead.

## What is ours and what is borrowed

The parser, order book, features and model are written in Hardcaml in this
repo. For the MAC and PCS, taxi has a KR260 10G example that needs no licensed
IP and builds with free Vivado. Its CERN-OHL-S licence is reciprocal while this
repo is Apache-2.0, so it stays a separate build dependency, never copied in.
The older MIT licensed verilog-ethernet is the fallback.

Everything runs in one 322.265625 MHz clock domain, the MAC's own, so no clock
crossing FIFO adds cycles on the way in or out.

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

- Replaying a sample ITCH day through the design gives decisions identical to
  a software model of the same pipeline run on the same file.
- We report tick to trade latency in cycles and ns, with the model's share of
  it broken out.
- The parser and order book have formal properties: no message is dropped, and
  the book after each message equals the software book.

## Open questions

- How many symbols and price levels to keep on chip for the first version.
