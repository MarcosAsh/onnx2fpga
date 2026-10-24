# Top level targets; only Python, numpy and Verilator are needed.

PYTHON      ?= python3
PYTHONPATH  := compiler:tests:examples
export PYTHONPATH

.PHONY: help models test test-unit test-sim test-cpp test-hardcaml test-api runtime lint \
        clean demo vivado synth synth-impl synth-dry test-flaky test-flaky-all \
        assets wheel icepi

help:
	@echo "make models     build the example ONNX models"
	@echo "make test       run everything (unit, C++ contract, Verilator)"
	@echo "make test-unit  fast tests only, no simulation"
	@echo "make test-sim   Verilator end to end, both example models"
	@echo "make test-api   the Python API, end to end, streaming, fixed latency and trees"
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
	@echo "make icepi      the example MLP as an Icepi Zero bitstream (needs dune, yosys, nextpnr)"

models:
	$(PYTHON) examples/make_models.py

test: test-unit test-cpp test-sim test-hardcaml test-api

test-unit:
	$(PYTHON) -m unittest discover -s tests -p "test_ingest.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_numerics.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_pipeline.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_determinism.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_quantized_import.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_characterize.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_packaging.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_synth_script.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_hardcaml_backend.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_ecp5.py" -v

# API tests skip what this machine cannot run and name what is missing.
test-api: models
	$(PYTHON) -m unittest discover -s tests -p "test_api.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_fixed_latency.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_trees.py" -v

test-cpp: runtime
	$(PYTHON) -m unittest discover -s tests -p "test_cpp_contract.py" -v

test-flaky:
	$(PYTHON) tools/flakecheck.py --repeat 5 --fast

test-flaky-all:
	$(PYTHON) tools/flakecheck.py --repeat 3

test-sim:
	$(PYTHON) -m unittest discover -s tests -p "test_residual.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_sliding_window.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_simulation.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_hardcaml_lockstep.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_icepi_board.py" -v

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

synth-dry: models
	$(PYTHON) -m onnx2fpga synth $(MODEL) --device $(DEVICE) --dry-run

clean:
	rm -rf build runtime/bin
	find . -name __pycache__ -type d -exec rm -rf {} +
