# Top level targets; only Python, numpy and Verilator are needed.

PYTHON      ?= python3
PYTHONPATH  := compiler:tests:examples
export PYTHONPATH

.PHONY: help models test test-unit test-sim test-cpp test-hardcaml test-api runtime lint \
        clean demo vivado synth synth-impl synth-dry measure-k26 test-flaky test-flaky-all \
        assets wheel icepi

help:
	@echo "make models     build the example ONNX models"
	@echo "make test       run everything (unit, C++ contract, Verilator)"
	@echo "make test-unit  fast tests only, no simulation"
	@echo "make test-sim   Verilator end to end, both example models and the tick to trade replay"
	@echo "make test-api   the Python API, end to end, streaming, fixed latency, elementwise, reductions, trees, reloads, corpus"
	@echo "make test-cpp   C++ kernels against the Python model"
	@echo "make test-hardcaml  the Hardcaml library's own tests (needs dune)"
	@echo "make runtime    build the host tools in runtime/bin"
	@echo "make wheel      stage assets and build an installable wheel"
	@echo "make lint       lint every hand written RTL module"
	@echo "make test-flaky run the suite repeatedly, shuffled, to catch flakiness"
	@echo "make demo       compile the MLP and run it through simulation"
	@echo "make vivado     what this machine can compile, simulate and measure"
	@echo "make synth      measure real resources and fmax (needs Vivado)"
	@echo "make synth-dry  print the Vivado commands without running them"
	@echo "make measure-k26  place and route every corpus model on the K26 (needs Vivado)"
	@echo "make icepi      the example MLP as an Icepi Zero bitstream (needs dune, yosys, nextpnr)"

models:
	$(PYTHON) examples/make_models.py

test: test-unit test-cpp test-sim test-hardcaml test-api

test-unit:
	$(PYTHON) -m unittest -v \
	  test_average_pool.Arithmetic test_layernorm.Arithmetic test_recorded_itch \
	  test_trading_board test_trading_runtime test_trading_dma \
	  test_run_ddr test_replay_records test_replay_compare \
	  test_kria_package test_kria_install test_provision_ddr \
	  test_artix test_compatibility test_corpus_report \
	  test_cnn.Cnn test_ingest test_numerics \
	  test_pipeline test_determinism test_quantized_import \
	  test_operators test_characterize test_packaging \
	  test_synth_script test_hardcaml_backend test_ecp5

# API tests skip what this machine cannot run and name what is missing.
test-api: models
	$(PYTHON) -m unittest -v \
	  test_average_pool.Simulation test_cnn.Simulation test_api \
	  test_fixed_latency test_elementwise test_reductions \
	  test_layernorm.Hardware test_kria_board test_kria_runtime \
	  test_arty_board test_trees test_reload \
	  test_corpus test_qonnx

test-cpp: runtime
	$(PYTHON) -m unittest -v \
	  test_cpp_contract test_register_transport

test-flaky:
	$(PYTHON) tools/flakecheck.py --repeat 5 --fast

test-flaky-all:
	$(PYTHON) tools/flakecheck.py --repeat 3

test-sim:
	$(PYTHON) -m unittest -v \
	  test_trading_host_rtl test_ddr_trading test_residual \
	  test_sliding_window test_simulation test_hardcaml_lockstep \
	  test_icepi_board test_tick_to_trade

# Hardcaml tests skip where dune is missing.
test-hardcaml:
	@if command -v dune >/dev/null 2>&1; then \
	  cd hardware/hardcaml && dune build @runtest; \
	else \
	  echo "dune not on PATH, skipping the Hardcaml tests"; \
	fi

runtime:
	$(MAKE) -C runtime

# Stage the RTL and C++ runtime inside the package so a wheel carries them.
assets:
	$(PYTHON) tools/build_wheel.py --stage-only

# build_wheel.py stages the assets itself, so a wheel is correct without make assets.
wheel:
	$(PYTHON) tools/build_wheel.py

lint:
	@for module in hardware/rtl/otf_*.sv; do \
	  name=$$(basename $$module .sv); \
	  deps=""; \
	  if [ "$$name" != "otf_requant" ]; then deps=hardware/rtl/otf_requant.sv; fi; \
	  verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDPARAM \
	    $$deps $$module --top-module $$name >/dev/null \
	    && echo "  $$name clean" || exit 1; \
	done

demo: models
	$(PYTHON) -m onnx2fpga compile examples/models/mlp.onnx \
	  --out build/demo --device vu9p --target-cycles 64
	$(MAKE) -C build/demo run

# Then 'make -C build/mlp_icepi_zero/icepi_zero prog' loads it; see hardware/boards/icepi_zero/README.md.
icepi: models
	$(PYTHON) -m onnx2fpga bitstream examples/models/mlp.onnx --board icepi_zero \
	  --target-cycles 256

MODEL  ?= examples/models/cnn.onnx
DEVICE ?= z7020
VIVADO ?= vivado
export VIVADO

vivado:
	@-$(PYTHON) -m onnx2fpga doctor

synth: models
	$(PYTHON) -m onnx2fpga synth $(MODEL) --device $(DEVICE) --vivado "$(VIVADO)"

synth-impl: models
	$(PYTHON) -m onnx2fpga synth $(MODEL) --device $(DEVICE) --vivado "$(VIVADO)" \
	  --impl --measurements build/characterize/measured.json

measure-k26:
	$(PYTHON) tools/measure_k26.py --vivado "$(VIVADO)"

synth-dry: models
	$(PYTHON) -m onnx2fpga synth $(MODEL) --device $(DEVICE) --dry-run

clean:
	rm -rf build runtime/bin
	find . -name __pycache__ -type d -exec rm -rf {} +
