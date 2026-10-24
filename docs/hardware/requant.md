# Rescale contract

Status: implemented

Every unit that changes scale ends in the same fixed point rescale:

```
out = clamp(((acc * mult + 2^(SHIFT-1)) >> SHIFT) + OUT_ZP, OUT_MIN, OUT_MAX)
```

- `acc` is the signed accumulator, `mult` a signed per channel multiplier.
- `>>` is an arithmetic shift, so the rounding is half up.
- `OUT_ZP` is the output zero point.
- One shift is shared by all channels of a unit; accuracy per channel comes
  from `mult`.

When `SHIFT >= ACC_BITS + MULT_BITS` the result is zero for every input, and the
hardware builds that constant directly rather than widening the datapath.

## Where it lives

The same arithmetic is written four times, on purpose:

| Where | Role |
|---|---|
| `compiler/onnx2fpga/p4_quantize/requantize.py` | The definition. |
| `runtime/include/otf/requantize.hpp` | C++, for fast dataset sweeps. |
| `hardware/rtl/otf_requant.sv` | SystemVerilog backend. |
| `hardware/hardcaml/src/otf_requant.ml` | Hardcaml backend. |

Tests check all four agree. A change to one is a change to the contract and
must be made in all four.
