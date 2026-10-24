# Folding and budgets

Status: implemented

Every layer becomes its own hardware unit, and streams chain the units
together. A frame moves through all of them at once, like cars on an assembly
line. That has one big consequence: the slowest unit sets the speed for the
whole design. Speeding up anything else buys nothing.

A matrix unit takes `SIMD` inputs and produces `PE` outputs each cycle, so one
input vector costs

```
cycles = (inputs / SIMD) * (outputs / PE)
```

Raise `SIMD` or `PE` and you spend more multipliers to save cycles. Picking
those two numbers for every unit is what we call folding.

The compiler does it greedily. It finds the bottleneck, buys it the cheapest
step that makes it faster, and repeats. It stops when it meets your target,
when the next step would blow a budget, or when the bottleneck cannot fold any
further. Then it tells you which:

```
fold: fc1 held at 256 cycles by the dsp budget: its next step would take the design to 30 dsp, past the 28 allowed on icepi25
```

Multipliers and block RAM can be used to the last one. LUTs stop at 80 percent
by default, since routing a nearly full chip gets slow and then fails. Change
that with `compile(utilisation=0.9)` if you want to gamble. On the Icepi Zero
you will almost always run out of multipliers first, because yosys gives every
int8 multiply a whole 18x18 block.

When you ask for a speed the budget cannot hold, the default is to raise
`BudgetError` with the allocator's reason and a suggestion. Pass `strict=False`
and you get the fastest balanced design that does fit instead.

Pins from `acc.layers[name].fold(...)` win over the allocator, even when they
make that unit the bottleneck. A pin that pushes the design over budget raises
`BudgetError` and names the pin.
