# Stream contract

Status: implemented

Every unit talks to its neighbours through the same handshake, AXI-Stream style.

## Signals

| Signal | Direction | Meaning |
|---|---|---|
| `tdata` | producer to consumer | One beat. |
| `tvalid` | producer to consumer | `tdata` holds a beat. |
| `tready` | consumer to producer | The consumer takes the beat this cycle. |

A beat moves on a cycle where `tvalid` and `tready` are both high.

## Rules

1. Once `tvalid` is high it stays high, with `tdata` unchanged, until the beat
   moves. A unit that relies on this (the fork, `otf_dup`) says so.
2. `tvalid` must not depend combinationally on `tready`.
3. `tready` may depend combinationally on `tvalid`.
4. Reset is synchronous. `rst_n` is active low at the top level only; inside,
   units see an active high clear.

## Beats

A beat packs several elements. Element 0 is in the least significant bits.
Tensors are channel last, so a beat holds consecutive channels of one pixel.
The number of elements per beat is set by folding; width converters
(`otf_dwc`) sit between units that disagree.

## Buffers

FIFOs on each edge are sized by analysis, not guessed. On a chain a couple of
beats is enough. Across a fork that rejoins, the short branch must hold the
whole skew between the branches, or the design deadlocks. The compiler refuses
any stream with two consumers; forks get an explicit `otf_dup`.

## Timing is part of the contract

The FIFO sizing assumes each unit's exact cycle behaviour. Changing when a unit
asserts `tready` or `tvalid` is an interface change: update the token analysis
in `p5_schedule/tokens.py` and keep both backends identical.
