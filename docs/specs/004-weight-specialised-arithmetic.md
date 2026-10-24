# 004 Weight specialised arithmetic

Status: planned

## Problem

With frozen weights ([003](003-reloadable-weights.md)) every weight is known when
the hardware is generated, yet each still gets a general multiplier. A
multiplier by a constant is smaller and faster as a few shifts and adds, and a
zero weight needs no hardware at all.

## Behaviour

Where a multiplier always sees one weight, which is every multiplier in fixed
latency mode ([001](001-fixed-latency.md)), the compiler replaces it with logic
for that constant:

- a zero weight disappears;
- plus or minus one becomes an add or a subtract;
- a power of two becomes a shift;
- anything else becomes a short sum of shifts, sharing partial sums across the
  layer.

Values never change. This only changes cost and timing. No weight is pruned
unless the user asks.

The da4ml project already generates multiplier free arithmetic of this kind.
Before writing our own, we run its output through the same flow and compare. If
its adder graphs beat ours, we call da4ml for this step instead of competing
with it.

## API

```python
acc.layers["fc2"].arithmetic("constant")
acc.layers["fc2"].arithmetic("multiplier")
acc.layers["fc2"].arithmetic("auto")      # default: whichever gives lower latency at the clock
```

## Errors

`arithmetic("constant")` with reloadable weights raises `ConfigError`, since
those weights are not constants.

## Done when

- hardcaml_verify proves the constant version equal to the multiplier version
  for the jet tagging sized MLP.
- At the same clock, the constant version has fewer DSPs and no more latency.
- We have a table of our arithmetic against da4ml's for the same model.

## Open questions

- Timebox: two weeks, then decide between our code and da4ml.
