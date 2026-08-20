# Host tools and the cross language kernel checks.
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude
BIN      := bin

.PHONY: all clean

all: $(BIN)/check_matvec $(BIN)/otf_run

$(BIN):
	mkdir -p $(BIN)

$(BIN)/check_matvec: kernels/check_matvec.cpp include/otf/kernels.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/otf_run: host/otf_run.cpp include/otf/sim_accelerator.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BIN)
