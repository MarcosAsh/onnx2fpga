# Sliding window unit

Status: implemented

`otf_swu_line` feeds a convolution. It holds only the pixels a window can still
reach, the rows above the current one plus the current partial row, and it
captures new pixels in the same cycles it replays old ones. So a convolution
costs its replay time alone, and the buffer is a span of the image, not the
whole image. On a 224x224 layer that is 49 times less memory than buffering a
frame.

## Parallel columns

`PAR` emits that many adjacent kernel columns per beat instead of one. A first
layer with one input channel cannot fold on channels, so without `PAR` its
replay time would set the speed of the whole pipeline. Reading `PAR` pixels at
once needs `PAR` read ports, and we get them by keeping `PAR` copies of the
buffer. The writer writes every copy and copy j reads `j * DIL_W` pixels
further along. That costs `PAR` times the memory but keeps each read address a
plain bit slice, which a banked layout would not. `PAR > 1` reads whole pixels,
so it cannot be combined with channel folding.

## Sizing the buffer

The buffer is circular over pixel indices. One window covers

```
SPAN = (KH-1)*DIL_H*IFM_W + (KW-1)*DIL_W + 1
```

pixels in raster order, and you would expect `SPAN` to be enough. It is not.
With top padding, two output rows can start on the same input row, so when the
reader wraps from the end of one output row to the start of the next it walks
backwards to a pixel it already passed. The buffer has to hold the span plus
that backward reach, which the code calls `TAIL`. For the same reason the
oldest pixel still needed is the minimum over the rest of this output row and
the start of the next one, not just the current position.

Get either of those wrong and the first window of every output row comes out
corrupted, silently, but only on padded images that are not square. The
sliding window tests sweep 18 geometries for exactly this reason.

Capacity is rounded up to a power of two so addressing is a bit slice instead
of a modulo. Pixel counters run free and are compared by signed difference, so
they wrap harmlessly and one frame can overlap the next without a reset.
