// Register protocol and shared DDR memory for the complete trading pipeline.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "Vddr_regs_sim.h"
#include "verilated.h"

namespace {
constexpr uint64_t base = 0x100000;

class Memory {
public:
    explicit Memory(const char* path) {
        const int fd = open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open simulated DDR");
        struct stat info {};
        if (fstat(fd, &info) || info.st_size <= 0) {
            close(fd);
            throw std::runtime_error("invalid simulated DDR size");
        }
        size = info.st_size;
        void* mapped = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mapped == MAP_FAILED) throw std::runtime_error("cannot map simulated DDR");
        data = static_cast<const unsigned char*>(mapped);
    }
    ~Memory() { munmap(const_cast<unsigned char*>(data), size); }
    const unsigned char* data = nullptr;
    size_t size = 0;
};

class Simulation {
public:
    Simulation(const char* path, unsigned stall, bool error)
        : memory(path), stall(stall), error(error) {
        if (stall >= 100) throw std::runtime_error("invalid DDR stall percentage");
        dut.clock = 0;
        dut.clear = 1;
        dut.s_axi_bready = 1;
        dut.s_axi_rready = 1;
        dut.s_axi_wstrb = 15;
        step();
        step();
        dut.clear = 0;
    }

    void prepare() {
        dut.clock = 0;
        dut.arready = left == 0 && rng() % 100 >= stall;
        presenting = left && (presenting || rng() % 100 >= stall);
        dut.rvalid = presenting;
        dut.rlast = left == 1;
        dut.rresp = error ? 2 : 0;
        if (presenting) {
            for (unsigned word = 0; word < 4; ++word) {
                uint32_t value = 0;
                for (unsigned byte = 0; byte < 4; ++byte)
                    value |= uint32_t(memory.data[next + word * 4 + byte]) << (8 * byte);
                dut.rdata[word] = value;
            }
        }
        dut.eval();
    }

    void edge() {
        const bool take = dut.rvalid && dut.pipeline__024replay__024rready;
        if (dut.pipeline__024replay__024arvalid && dut.arready) {
            const uint64_t address = dut.pipeline__024replay__024araddr;
            const size_t count = dut.pipeline__024replay__024arlen + 1;
            if (left || address < base || (address - base) % 16 || count > 16 ||
                address - base + count * 16 > memory.size ||
                (address & 4095) + count * 16 > 4096 ||
                dut.pipeline__024replay__024arsize != 4 ||
                dut.pipeline__024replay__024arburst != 1)
                throw std::runtime_error("invalid simulated DDR request");
            next = address - base;
            left = count;
        }
        if (take) {
            next += 16;
            --left;
            presenting = false;
        }
        dut.clock = 1;
        dut.eval();
        dut.clock = 0;
        dut.eval();
    }

    void step() { prepare(); edge(); }

    void write(uint32_t address, uint32_t value) {
        dut.s_axi_awaddr = address;
        dut.s_axi_wdata = value;
        dut.s_axi_awvalid = 1;
        dut.s_axi_wvalid = 1;
        bool aw = false, w = false;
        for (unsigned cycle = 0; cycle < 1000; ++cycle) {
            prepare();
            aw = aw || (dut.s_axi_awvalid && dut.s_axi_awready);
            w = w || (dut.s_axi_wvalid && dut.s_axi_wready);
            const bool response = dut.s_axi_bvalid;
            const auto code = dut.s_axi_bresp;
            edge();
            if (aw) dut.s_axi_awvalid = 0;
            if (w) dut.s_axi_wvalid = 0;
            if (response) {
                if (code) throw std::runtime_error("AXI register write rejected");
                return;
            }
        }
        throw std::runtime_error("AXI register write timed out");
    }

    uint32_t read(uint32_t address) {
        // Advance the fabric between host polls, including during abort drains.
        for (unsigned cycle = 0; cycle < 256; ++cycle) step();
        dut.s_axi_araddr = address;
        dut.s_axi_arvalid = 1;
        for (unsigned cycle = 0; cycle < 1000; ++cycle) {
            prepare();
            const bool accepted = dut.s_axi_arvalid && dut.s_axi_arready;
            const bool response = dut.s_axi_rvalid;
            const auto value = dut.s_axi_rdata;
            const auto code = dut.s_axi_rresp;
            edge();
            if (accepted) dut.s_axi_arvalid = 0;
            if (response) {
                if (code) throw std::runtime_error("AXI register read rejected");
                return value;
            }
        }
        throw std::runtime_error("AXI register read timed out");
    }

private:
    Vddr_regs_sim dut;
    Memory memory;
    std::mt19937 rng {71};
    unsigned stall;
    bool error;
    size_t next = 0, left = 0;
    bool presenting = false;
};
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        if (argc != 4) throw std::runtime_error("usage: Vddr_regs_sim memory stall error");
        Simulation simulation(argv[1], std::stoul(argv[2]), std::stoi(argv[3]) != 0);
        std::cout << "OTFREG1\n" << std::flush;
        std::string command;
        uint32_t address, value;
        while (std::cin >> command >> std::hex >> address) {
            if (command == "w" && (std::cin >> std::hex >> value)) {
                simulation.write(address, value);
                std::cout << "ok\n" << std::flush;
            } else if (command == "r") {
                value = simulation.read(address);
                std::cout << "ok " << std::hex << value << '\n' << std::flush;
            } else throw std::runtime_error("invalid register command");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
