# 014 QONNX Quant nodes

Status: implemented for weights and activations of 2 to 8 bits. Activations
still travel on 8 bit streams; `BipolarQuant`, `Trunc` and `FloatQuant` are
refused.

## Problem

hls4ml and FINN build their smallest designs from networks trained at 2 to 6
bits with Brevitas, QKeras or HGQ, and the format they share is QONNX. A
QONNX model states its quantization with one custom node, `Quant`, in the
domain `qonnx.custom_op.general`, where a QDQ model would use a
`QuantizeLinear` and `DequantizeLinear` pair. Before this, every QONNX model
failed at import with an unknown op, so the models those tools are measured on
could not be compiled here at all.

## Behaviour

`Quant` takes a tensor, a scale, a zero point and a bit width, with the
attributes `signed`, `narrow` and `rounding_mode`, and gives

```
scale * (clip(round(x / scale + zero_point), low, high) - zero_point)
```

where low and high are the integer range of the bit width: -8 to 7 at 4 bits
signed, -7 to 7 with `narrow`, 0 to 15 unsigned and 0 to 14 unsigned with
`narrow`. That is the qonnx 1.0 reference, `int_quant`. `IntQuant`, the name
newer qonnx releases use, is the same node.

On a constant, the importer computes this at import, so the weights the
compiler sees are already on their grid, and records the grid. Weights of up to
8 bits keep their width: a dense layer whose weights a `Quant` put on 4 bits is
built with 4 bit weights, on per channel scales if the model gave them.
Constants of up to 32 bits, such as the bias Brevitas quantizes onto the
product of the input and weight scales, are rounded the same way and need no
grid of their own.

On anything else the node is an alias carrying its grid, as a QDQ pair is in
[009](009-operator-coverage.md). An unsigned grid moves onto the signed type of
the same width, with the zero point lowered by half the range: 128 at 8 bits,
as before, and 8 at 4 bits. A grid narrower than the stream clamps where it is
made: the requant of the dense layer, convolution or activation that produces
it gets that grid's range as its output clamp, which
[001](001-fixed-latency.md)'s units already take as `OUT_MIN` and `OUT_MAX`.
A Relu between a layer and its `Quant` still fuses, and clamps at the zero
point inside that range. A unit with no requant, such as a max pool, may carry
a narrow grid only if its input had it; anything else raises.

The summary names what was read, for example `weights int4, activations int8
on 4 bit grids, from the model's Quant nodes`.

## Verification

`tests/fixtures/make_corpus.py` builds `qonnx_mlp` the way Brevitas exports a
network: a scikit-learn MLP of 8, 32, 16 and 1 trained on the corpus's market
features, an 8 bit `Quant` on the input, 4 bit narrow `Quant`s on the weights
per output channel, 24 bit `Quant`s on the biases, 4 bit unsigned `Quant`s
after each Relu, and an 8 bit `Quant` on the output. Its stored outputs come
from qonnx 1.0's own executor, since ONNX Runtime cannot run `Quant`.

`tests/test_qonnx.py` compiles it for `k26` in fixed mode. All 256 held out
outputs equal qonnx's, not within a step but exactly, every dense unit has
4 bit weights, the hidden layers clamp to -8 and 7, 10000 random vectors
simulate bit exact under Verilator, and the latency proof closes at 9 cycles
and fails at 8. The corpus test compiles it with the other 20 models. The
refusals below are each checked on a changed copy of the model.

| Layer | Weights | Latency | LUT | DSP |
|---|---|---|---|---|
| 8 to 32, Relu | int4 | 3 | 3508 | 32 |
| 32 to 16, Relu | int4 | 3 | 7833 | 16 |
| 16 to 1 | int4 | 3 | 360 | 1 |
| Total | | 9, 27.9 ns at 322 MHz | 11701 | 49 |

## Errors

Each names the node.

- `Quant 'q' rounds by FLOOR, and only ROUND is supported`, for any rounding
  mode but ROUND and HALF_EVEN, which are the same.
- `Quant 'q' quantizes to 12 bits, and activations take 2 to 8`, and for a
  constant, `2 to 32`.
- `Quant 'q' has a bit width that is not one integer`.
- `Quant 'q' has a zero point that is not an integer`.
- `Quant 'q' has a scale per channel, which an activation cannot have yet`.
- `Quant 'q' has a narrow range, which an activation cannot have yet`.
- `cannot lower this model yet: the model gives tensor 'm0' no grid`, at
  compile, when an op needs a grid the model did not state, such as a Sigmoid
  straight after a `MatMul`. A QDQ model gets the same message, where it used
  to get a bare `KeyError`.
- A unit with no requant, such as a table or an Add, that makes a grid
  narrower than the stream is refused at compile with `cannot lower this model
  yet` and the unit's name.

## Done when

All three are met.

- [x] A QONNX MLP with 4 bit weights and activations compiles in fixed mode
  and matches qonnx's executor exactly.
- [x] Weights keep the model's width in the built units.
- [x] The design simulates bit exact and its latency is proved.

## Open questions

- The activations are 4 bit codes on 8 bit streams. Giving each stream its own
  width would make the second layer's multiplies 4 by 4 bits: the estimate
  drops from 7833 to 6148 LUTs, and from 360 to 256 for the last. That needs
  converters between units, which the fixed latency top could add as plain
  wires.
- HGQ and QKeras networks usually quantize their input to 16 bits, past the
  8 that activations take.
- A real Brevitas export, and the hls4ml jet tagger in QONNX, would test what
  hand building cannot: Brevitas's own node order, its `Transpose`s, and
  `Trunc` after average pooling.
