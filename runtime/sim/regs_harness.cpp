// Verilator driver for kria_regs alone: one AXI4-Lite transfer per stdin line, "w OFFSET VALUE" or "r OFFSET" in hex.

#include <cstdint>
#include <iostream>
#include <string>

#include "Vkria_regs.h"
#include "verilated.h"

namespace {

// Waits for ready before a rising edge and takes that edge; the caller reads outputs first through peek.
template <typename Ready, typename Peek>
bool handshake(Vkria_regs& regs, Ready ready, Peek peek) {
    for (int waited = 0; waited < 100; ++waited) {
        regs.clock = 0;
        regs.eval();
        const bool done = ready();
        if (done) peek();
        regs.clock = 1;
        regs.eval();
        if (done) return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vkria_regs regs;
    regs.clear = 1;
    for (int edge = 0; edge < 4; ++edge) {
        regs.clock = edge & 1;
        regs.eval();
    }
    regs.clear = 0;
    regs.s_axi_bready = 1;
    regs.s_axi_rready = 1;
    const auto nothing = [] {};
    std::string command;
    while (std::cin >> command) {
        uint32_t offset = 0, value = 0;
        bool ok = false;
        if (command == "w") {
            std::cin >> std::hex >> offset >> value;
            regs.s_axi_awaddr = offset;
            regs.s_axi_wdata = value;
            regs.s_axi_awvalid = 1;
            regs.s_axi_wvalid = 1;
            ok = handshake(regs, [&] { return regs.s_axi_awready; }, nothing);
            regs.s_axi_awvalid = 0;
            regs.s_axi_wvalid = 0;
            ok = ok && handshake(regs, [&] { return regs.s_axi_bvalid; }, nothing);
        } else if (command == "r") {
            std::cin >> std::hex >> offset;
            regs.s_axi_araddr = offset;
            regs.s_axi_arvalid = 1;
            ok = handshake(regs, [&] { return regs.s_axi_arready; }, nothing);
            regs.s_axi_arvalid = 0;
            ok = ok && handshake(regs, [&] { return regs.s_axi_rvalid; },
                                 [&] { value = regs.s_axi_rdata; });
            std::cout << std::hex << value << std::endl;
        }
        if (!ok) {
            std::cerr << "no AXI4-Lite handshake for " << command << " 0x" << std::hex << offset
                      << "\n";
            return 1;
        }
    }
    regs.final();
    return 0;
}
