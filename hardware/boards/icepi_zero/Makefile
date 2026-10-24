# A compiled design on an Icepi Zero; usage: make -f this DESIGN=build/m/design.json [prog|sim] (480 / MHZ must be whole).

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
HARDCAML := $(abspath $(HERE)../../hardcaml)

DESIGN ?= $(error set DESIGN to a design.json from onnx2fpga compile --hdl hardcaml)
BUILD ?= $(dir $(DESIGN))icepi_zero
REV ?= v1.4
MHZ ?= 20
BAUD ?= 1000000
RX_DEPTH ?= 2048
TX_DEPTH ?= 2048
SEEDS ?= 1 2 3 4 5 6 7 8
UART ?= usb
ifeq ($(filter usb header,$(UART)),)
$(error UART is usb or header, not $(UART))
endif

OSS_CAD := $(firstword $(wildcard $(HOME)/.local/opt/oss-cad-suite/bin /opt/oss-cad-suite/bin))
tool = $(or $(shell command -v $(1) 2>/dev/null),$(if $(OSS_CAD),$(wildcard $(OSS_CAD)/$(1))),$(1))
YOSYS ?= $(call tool,yosys)
NEXTPNR ?= $(call tool,nextpnr-ecp5)
ECPPACK ?= $(call tool,ecppack)
LOADER ?= $(call tool,openFPGALoader)
VERILATOR ?= verilator
DUNE ?= dune
GENERATE ?= $(HARDCAML)/_build/default/bin/generate.exe

TOP_V = $(BUILD)/icepi_top.v
JSON = $(BUILD)/icepi_top.json
NAME = $(REV)$(if $(filter header,$(UART)),-header)
CONFIG = $(BUILD)/$(NAME).config
BIT = $(BUILD)/$(NAME).bit
PARAMS = $(BUILD)/params
BOARD_JSON = $(BUILD)/board.json
CLOCKS_PER_BIT = $(shell echo $$(( ($(MHZ) * 1000000 + $(BAUD) / 2) / $(BAUD) )))

bit: $(BIT) $(BOARD_JSON)

# Rewritten only when a parameter changes, so a new MHZ regenerates only what depends on it.
PARAMS_LINE = REV=$(REV) UART=$(UART) MHZ=$(MHZ) BAUD=$(BAUD) RX_DEPTH=$(RX_DEPTH) TX_DEPTH=$(TX_DEPTH)
$(PARAMS): FORCE
	@mkdir -p $(BUILD)
	@echo "$(PARAMS_LINE)" | cmp -s - $@ || echo "$(PARAMS_LINE)" > $@

# What host.py needs to know about the link; the beats come from the manifest.
$(BOARD_JSON): $(PARAMS)
	printf '{"board": "icepi_zero", "rev": "%s", "uart": "%s", "mhz": %s, "baud": %s, "clocks_per_bit": %s, "rx_depth": %s, "tx_depth": %s}\n' \
	  $(REV) $(UART) $(MHZ) $(BAUD) $(CLOCKS_PER_BIT) $(RX_DEPTH) $(TX_DEPTH) > $@

$(GENERATE):
	$(DUNE) build --root $(HARDCAML) ./bin/generate.exe

$(TOP_V): $(DESIGN) $(PARAMS) | $(GENERATE)
	$(GENERATE) icepi $(DESIGN) -mhz $(MHZ) -baud $(BAUD) -rx-depth $(RX_DEPTH) \
	  -tx-depth $(TX_DEPTH) > $@.tmp
	mv $@.tmp $@

$(JSON): $(TOP_V)
	$(YOSYS) -q -l $(BUILD)/yosys.log -p "read_verilog $<; synth_ecp5 -top icepi_top -json $@"

LPFS = $(HERE)icepi_zero.lpf $(HERE)uart-$(UART).lpf $(HERE)$(REV).lpf

$(CONFIG): $(JSON) $(LPFS)
	rm -f $@ $(BUILD)/nextpnr-seed*.log
	@for seed in $(SEEDS); do \
	  $(NEXTPNR) --25k --package CABGA256 --speed 6 --json $< \
	    $(addprefix --lpf ,$(LPFS)) --freq $(MHZ) --seed $$seed \
	    --textcfg $@ --report $(BUILD)/report.json --log $(BUILD)/nextpnr-seed$$seed.log -q \
	    && echo "seed $$seed meets $(MHZ) MHz" && cp $(BUILD)/nextpnr-seed$$seed.log $(BUILD)/nextpnr.log \
	    && break; \
	  rm -f $@; \
	  if grep -q "Unable to place" $(BUILD)/nextpnr-seed$$seed.log; then \
	    grep -h "^ERROR" $(BUILD)/nextpnr-seed$$seed.log; \
	    echo "the design does not fit the LFE5U-25F: compile it with a larger --target-cycles"; \
	    exit 1; \
	  fi; \
	done; \
	if [ ! -f $@ ]; then \
	  grep -h "Max frequency" $(BUILD)/nextpnr-seed*.log; \
	  echo "no seed met $(MHZ) MHz: try a lower MHZ that divides 480 (20, 16, 15, 12, 10)"; \
	  exit 1; \
	fi
	@$(MAKE) --no-print-directory -f $(HERE)Makefile report DESIGN=$(DESIGN) BUILD=$(BUILD)

$(BIT): $(CONFIG)
	$(ECPPACK) --compress --input $< --bit $@

report:
	@grep -h "Max frequency for clock" $(BUILD)/nextpnr.log | tail -1
	@grep -h -E "(TRELLIS_COMB|TRELLIS_FF|TRELLIS_RAMW|MULT18X18D|DP16KD):" $(BUILD)/nextpnr.log

# Loads the SRAM only, so a power cycle brings back the flash contents.
prog: $(BIT)
	$(LOADER) -b icepi-zero $<

SEND ?= $(error set SEND to a file of hex bytes)
SIM = $(BUILD)/sim

$(SIM)/Vicepi_top: $(TOP_V) $(HERE)sim/ehxplll.v $(HERE)sim/uart_bench.cpp
	$(VERILATOR) --cc --exe --build -j 0 -O2 -Wno-fatal -Wno-lint -Wno-style -Wno-COMBDLY \
	  --top-module icepi_top -Mdir $(SIM) $(TOP_V) $(HERE)sim/ehxplll.v \
	  $(HERE)sim/uart_bench.cpp -o Vicepi_top >/dev/null

sim: $(SIM)/Vicepi_top $(BOARD_JSON)
	$(SIM)/Vicepi_top --clocks-per-bit $(CLOCKS_PER_BIT) --send $(SEND) \
	  --receive $(BUILD)/received.hex $(SIM_FLAGS)

clean:
	rm -rf $(BUILD)

.PHONY: bit report prog sim clean FORCE
