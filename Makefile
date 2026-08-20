# Top level entry points. Every target works with Python, numpy and Verilator
# alone; nothing else is required to build or test this repository.

PYTHON      ?= python3
PYTHONPATH  := compiler:tests:examples
export PYTHONPATH

.PHONY: help models test test-unit test-sim test-cpp runtime lint clean demo \
        vivado synth synth-impl synth-dry test-flaky test-flaky-all \
        assets wheel

help:
	@echo "make models     build the example ONNX models"
	@echo "make test       run everything (unit, C++ contract, Verilator)"
	@echo "make test-unit  fast tests only, no simulation"
	@echo "make test-sim   Verilator end to end, both example models"
	@echo "make test-cpp   C++ kernels against the Python model"
	@echo "make runtime    build the host tools in runtime/bin"
	@echo "make wheel      stage assets and build an installable wheel"
	@echo "make lint       lint every hand written RTL module"
	@echo "make test-flaky run the suite repeatedly, shuffled, to catch flakiness"
	@echo "make demo       compile the MLP and run it through simulation"
	@echo "make vivado     what this machine can compile, simulate and measure"
	@echo "make synth      measure real resources and fmax (needs Vivado)"
	@echo "make synth-dry  print the Vivado commands without running them"

models:
	$(PYTHON) examples/make_models.py

test: test-unit test-cpp test-sim

test-unit:
	$(PYTHON) -m unittest discover -s tests -p "test_ingest.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_numerics.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_pipeline.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_determinism.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_quantized_import.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_characterize.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_packaging.py" -v
	$(PYTHON) -m unittest discover -s tests -p "test_synth_script.py" -v

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

runtime:
	$(MAKE) -C runtime

# Stage the RTL and C++ runtime inside the package so a wheel carries them.
# A checkout does not need this: the locator finds them one level up.
assets:
	rm -rf compiler/onnx2fpga/assets
	mkdir -p compiler/onnx2fpga/assets/hardware
	cp -r hardware/rtl hardware/synth compiler/onnx2fpga/assets/hardware/
	cp -r runtime compiler/onnx2fpga/assets/runtime
	rm -rf compiler/onnx2fpga/assets/runtime/bin
	@echo "staged $$(find compiler/onnx2fpga/assets -type f | wc -l) files"

wheel: assets
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
