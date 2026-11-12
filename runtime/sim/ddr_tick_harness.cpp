// DDR replay through the full trading pipeline, with a stalled AXI memory model.

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vddr_trading.h"
#include "verilated.h"
#include "otf/ports.hpp"

namespace {
using Record = std::array<uint32_t, 4>;
constexpr uint64_t base = 0x10000f00;

std::vector<Record> load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open replay records");
    std::vector<Record> records;
    unsigned char bytes[16];
    while (file.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) {
        Record record {};
        for (size_t word = 0; word < 4; ++word)
            for (size_t byte = 0; byte < 4; ++byte)
                record[word] |= uint32_t(bytes[word * 4 + byte]) << (byte * 8);
        records.push_back(record);
    }
    if (file.gcount() != 0) throw std::runtime_error("truncated replay record");
    return records;
}

template <typename T>
std::string hex(const T& port, size_t bits) {
    return otf::HexVectors::render(otf::port_read(port, (bits + 31) / 32), (bits + 3) / 4);
}
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        if (argc != 6 && argc != 7)
            throw std::runtime_error("usage: Vddr_trading records output runs seed stall-percent [abort-after]");
        const auto records = load(argv[1]);
        if (records.size() > UINT32_MAX) throw std::runtime_error("too many replay records");
        std::ofstream out(argv[2]);
        if (!out) throw std::runtime_error("cannot open replay output");
        const int runs = std::stoi(argv[3]), stall = std::stoi(argv[5]);
        const size_t abort_after = argc == 7 ? std::stoull(argv[6]) : 0;
        if (runs < 1 || stall < 0 || stall >= 100) throw std::runtime_error("invalid replay options");
        std::mt19937 rng(std::stoul(argv[4]));
        Vddr_trading dut;
        uint64_t cycle = 0;
        auto tick = [&] {
            dut.clock = 1; dut.eval();
            dut.clock = 0; dut.eval();
            ++cycle;
        };
        dut.clock = 0; dut.clear = 1; dut.start = 0; dut.abort_request = 0;
        dut.arready = 0; dut.rvalid = 0; dut.rresp = 0; dut.rlast = 0;
        dut.base = base; dut.records = records.size();
        dut.eval(); tick(); dut.clear = 0;
        for (int run = 0; run < runs; ++run) {
            out << "run " << run << '\n';
            size_t left = 0, next = 0, received = 0, decisions = 0, orders = 0;
            bool presenting = false, first = true;
            dut.abort_request = 0;
            dut.start = 1; dut.eval(); tick(); dut.start = 0;
            const uint64_t transactions = records.size() + (records.size() + 15) / 16;
            const uint64_t limit = cycle + (1 << 20) + 1024
                + transactions * 8 * ((100 + 99 - stall) / (100 - stall));
            for (;;) {
                if (cycle > limit) throw std::runtime_error("DDR replay timed out");
                dut.abort_request = run == 0 && abort_after && received >= abort_after;
                dut.arready = left == 0 && int(rng() % 100) >= stall;
                presenting = left != 0 && (presenting || int(rng() % 100) >= stall);
                dut.rvalid = presenting;
                dut.rlast = left == 1;
                if (presenting)
                    for (size_t word = 0; word < 4; ++word) dut.rdata[word] = records.at(next)[word];
                dut.eval();
                if (dut.replay__024arvalid && dut.arready) {
                    if (dut.replay__024arsize != 4 || dut.replay__024arburst != 1 || left != 0)
                        throw std::runtime_error("invalid AXI read request");
                    const uint64_t address = dut.replay__024araddr;
                    const size_t count = dut.replay__024arlen + 1;
                    if (address < base || (address - base) % 16 || count > 16 ||
                        (address & 4095) + count * 16 > 4096)
                        throw std::runtime_error("invalid AXI burst boundary");
                    next = (address - base) / 16;
                    if (next != received || next + count > records.size())
                        throw std::runtime_error("AXI read outside replay buffer");
                    left = count;
                }
                if (dut.replay__024valid && first) {
                    out << "beat0 " << cycle << '\n'; first = false;
                }
                if (dut.trading__024book_valid)
                    out << "B " << int(dut.trading__024book_symbol) << ' '
                        << hex(dut.trading__024book_bids, sizeof(dut.trading__024book_bids) * 8) << ' '
                        << hex(dut.trading__024book_asks, sizeof(dut.trading__024book_asks) * 8) << ' '
                        << hex(dut.trading__024book_flow, 20) << '\n';
                if (dut.trading__024features_valid) out << "F " << hex(dut.trading__024features, 32) << '\n';
                if (dut.trading__024decision_valid) {
                    ++decisions;
                    orders += dut.trading__024order_valid;
                    out << "D " << cycle << ' ' << int(dut.trading__024action) << ' '
                        << hex(dut.trading__024prediction, 8) << ' ' << int(dut.trading__024order_valid)
                        << ' ' << hex(dut.trading__024order, 376) << ' ' << dut.trading__024t_first
                        << ' ' << dut.trading__024t_last << ' ' << dut.trading__024t_out << '\n';
                }
                if (presenting && dut.replay__024rready) {
                    ++next; ++received; --left; presenting = false;
                }
                if (dut.done_) {
                    const bool aborted = run == 0 && abort_after && received >= abort_after;
                    if (dut.error || bool(dut.aborted) != aborted || left || dut.busy ||
                        (!aborted && (received != records.size() || dut.replay__024delivered != records.size())) ||
                        dut.decisions != decisions || dut.orders != orders)
                        throw std::runtime_error("DDR replay completed with inconsistent status");
                    out << "cycles " << dut.cycles << '\n';
                    break;
                }
                tick();
            }
        }
        dut.final();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
