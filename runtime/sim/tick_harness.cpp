// Replays a MoldUDP64 byte stream through tick_to_trade and writes every book, feature and decision output.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Vtick_to_trade.h"
#include "verilated.h"

#include "otf/ports.hpp"

namespace {

struct Beat {
    uint64_t data;
    uint8_t keep;
    bool last;
};

std::vector<Beat> load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::vector<Beat> beats;
    unsigned char record[10];
    while (file.read(reinterpret_cast<char*>(record), sizeof(record))) {
        uint64_t data = 0;
        for (int i = 7; i >= 0; --i) data = (data << 8) | record[i];
        beats.push_back({data, record[8], record[9] != 0});
    }
    return beats;
}

template <typename T>
std::string hex(const T& port, size_t bits) {
    const size_t words = (bits + 31) / 32;
    return otf::HexVectors::render(otf::port_read(port, words), (bits + 3) / 4);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::string beats_path, out_path;
    int gap = 0;
    unsigned seed = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--beats") beats_path = value;
        else if (flag == "--out") out_path = value;
        else if (flag == "--gap") gap = std::stoi(value);
        else if (flag == "--seed") seed = std::stoul(value);
        else {
            std::cerr << "unknown option " << flag << "\n";
            return 2;
        }
    }
    std::vector<Beat> beats;
    try {
        beats = load(beats_path);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "cannot write " << out_path << "\n";
        return 2;
    }

    Vtick_to_trade design;
    std::mt19937 rng(seed);
    const size_t levels_bits = sizeof(design.book_bids) * 8;
    long cycle = 0;
    auto tick = [&] {
        design.clock = 1;
        design.eval();
        design.clock = 0;
        design.eval();
        ++cycle;
    };

    design.clock = 0;
    design.clear = 1;
    design.valid = 0;
    design.eval();
    tick();
    design.clear = 0;
    design.eval();
    for (long wait = 0; !design.ready; ++wait) {
        if (wait > (1L << 22)) {
            std::cerr << "FAIL: ready never rose after clear\n";
            return 1;
        }
        tick();
    }

    // Inputs are applied and outputs sampled before each rising edge; the stream then drains.
    size_t next = 0, idle = 0;
    const size_t drain = 256;
    while (next < beats.size() || idle < drain) {
        const bool send = next < beats.size()
            && (gap <= 0 || static_cast<int>(rng() % 100) >= gap);
        design.valid = send;
        if (send) {
            if (next == 0) out << "beat0 " << cycle << "\n";
            design.data = beats[next].data;
            design.keep = beats[next].keep;
            design.last = beats[next].last;
            ++next;
        } else if (next >= beats.size()) {
            ++idle;
        }
        design.eval();
        if (design.book_valid) {
            out << "B " << static_cast<int>(design.book_symbol) << " "
                << hex(design.book_bids, levels_bits) << " "
                << hex(design.book_asks, levels_bits) << " "
                << hex(design.book_flow, 20) << "\n";
        }
        if (design.features_valid) out << "F " << hex(design.features, 32) << "\n";
        if (design.decision_valid) {
            out << "D " << cycle << " " << static_cast<int>(design.action) << " "
                << hex(design.prediction, 8) << " " << static_cast<int>(design.order_valid)
                << " " << hex(design.order, 376) << " " << design.t_first << " "
                << design.t_last << " " << design.t_out << "\n";
        }
        tick();
    }
    design.final();
    out << "cycles " << cycle << "\n";
    std::cout << "beats " << beats.size() << ", cycles " << cycle << "\n";
    return 0;
}
