# Backends

Status: implemented

We can emit the same compiled design in two languages.

The Hardcaml backend uses units written in Jane Street's OCaml HDL, under
`hardware/hardcaml`. A generator reads `design.json`, elaborates the whole
design and writes a single Verilog file, with the weights baked in as
constants. ECP5 builds need it, because yosys cannot parse the SystemVerilog
library.

The SystemVerilog backend instantiates hand written modules from
`hardware/rtl` and loads weights with `$readmemh`. It needs nothing but Python
to emit, and Vivado reads it happily.

We keep the two identical cycle for cycle. A lockstep test runs every unit from
both backends side by side in one Verilator model and compares every output,
every cycle, for 63 configurations. When they disagree, somebody has a real
bug. That test has already caught one in Hardcaml itself: two units with the
same parameters but different weights were being merged into one module.

Fixed latency units ([001](../specs/001-fixed-latency.md)) exist only in
Hardcaml. The SystemVerilog library is frozen as the lockstep reference for the
streaming units, so `mode="fixed"` with `hdl="sv"` raises `ConfigError`.

## Choosing one

It works like `KERAS_BACKEND`. `compile(hdl=...)` wins. After that comes the
`ONNX2FPGA_HDL` environment variable. With neither set we use Hardcaml if this
machine can build the generator and SystemVerilog if it cannot. A Hardcaml
project that is present but broken falls back to SystemVerilog with a warning.

## Writing Hardcaml here

We follow Jane Street's own conventions to the letter: `profile=janestreet`
formatting, `dune build @fmt @lint @runtest` clean, an `.mli` for every `.ml`,
`Config` and `Make` functors, `let%hw` names, and expect tests on the Lws
harness. When a convention and ocamlformat disagree, ocamlformat wins.
