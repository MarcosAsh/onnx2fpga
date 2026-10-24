# 002 Pipelining and clock targets

Status: planned

## Problem

Our matrix unit reads a weight, multiplies, sums and rescales in one cycle. On
the Icepi Zero that closes at about 23 MHz. Trading designs on UltraScale+ run
at 300 MHz and up, and a latency quoted in cycles means nothing until it is
quoted in nanoseconds at a real clock.

## Behaviour

Registers go after each multiply, between levels of the adder tree, and around
the rescale. The compiler places the fewest registers that meet the requested
clock. In fixed latency mode ([001](001-fixed-latency.md)) each register adds a
cycle of latency. In streaming mode it adds latency and leaves throughput alone.

The multiplies map onto DSP48E2 slices with their internal pipeline registers
used, since that is where the speed on UltraScale+ comes from. On ECP5 the same
design simply runs slower.

We calibrate a per device delay model against Vivado's timing reports for
UltraScale+ and against nextpnr for ECP5, so `summary()` can predict fmax before
anyone runs place and route.

## API

```python
acc.compile(device="k26", mode="fixed", clock_mhz=350)
acc.summary()        # clock, register stages per layer, latency in ns
```

## Errors

A clock no amount of pipelining reaches raises `BudgetError` with the slowest
path and the fastest clock that is reachable.

## Done when

- The jet tagging sized MLP (16-64-32-32-5) closes timing at 350 MHz or better
  on the KV260 in Vivado.
- The predicted fmax is within 10 percent of Vivado's on the example models.
- Latency in cycles in simulation equals the compiler's number.

## Open questions

- Is 350 MHz the right first target, or should we aim for what the trading
  cards run at? The HFT research will set this.
