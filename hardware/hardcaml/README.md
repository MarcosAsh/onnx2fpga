# Hardcaml units

The streaming units of `hardware/rtl`, ported to Hardcaml one module per
SystemVerilog module and cycle for cycle: same registers, same combinational
ready and valid paths, same reset values. The compiler's buffer sizing assumes
that timing, so either backend can carry a compiled design.

```
dune build @runtest                         expect tests, cyclesim
dune exec bin/generate.exe -- design FILE   Verilog for a design.json
dune exec bin/generate.exe -- icepi FILE    the same behind a UART, for an Icepi Zero
dune exec bin/generate.exe -- tick-to-trade FILE   the ITCH to order path around a fixed model
```

The UART side is `uart_rx`, `uart_tx` and `uart_bridge`, which turns bytes into beats
of whatever width the design has and back; `uart_top` puts a design behind them and
`icepi_top` adds the board's PLL and reset. `hardware/boards/icepi_zero` builds a
bitstream from it.

The tick to trade path is `moldudp64`, `itch_parser`, `order_book` (with
`book_side`), `features` and `decision`, put together with a fixed latency model by
`tick_to_trade`; see `docs/guides/tick-to-trade.md` for the replay that checks it.

`onnx2fpga compile --hdl hardcaml` writes the `design.json` and runs the
generator for you, producing a single `rtl/top.v` with the weights built in.
`tests/test_hardcaml_lockstep.py` in the repository root runs every unit here
beside the module it ports in one Verilator model and compares every output on
every cycle.

Needs the `5.2.0+ox` opam switch with hardcaml, ppx_hardcaml, hardcaml_test_harness,
hardcaml_waveterm, jsonaf and core_unix.
