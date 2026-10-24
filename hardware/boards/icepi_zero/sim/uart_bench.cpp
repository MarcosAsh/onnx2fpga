// Runs icepi_top in Verilator with a host model on the far end of its serial line.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Vicepi_top.h"
#include "verilated.h"

namespace {

struct Options {
    int clocks_per_bit = 0;
    std::string send_path;
    std::string receive_path;
    long expect = -1;
    double skew = 0.0;
    int gap = 0;
    int quiet_bits = 400;
    unsigned seed = 1;
    uint64_t timeout = 200'000'000;
    bool press = false;

    static Options parse(int argc, char** argv) {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    std::cerr << "missing value for " << flag << "\n";
                    std::exit(2);
                }
                return argv[++i];
            };
            if (flag == "--clocks-per-bit") o.clocks_per_bit = std::stoi(next());
            else if (flag == "--send") o.send_path = next();
            else if (flag == "--receive") o.receive_path = next();
            else if (flag == "--expect") o.expect = std::stol(next());
            else if (flag == "--skew") o.skew = std::stod(next());
            else if (flag == "--gap") o.gap = std::stoi(next());
            else if (flag == "--quiet-bits") o.quiet_bits = std::stoi(next());
            else if (flag == "--seed") o.seed = std::stoul(next());
            else if (flag == "--timeout") o.timeout = std::stoull(next());
            else if (flag == "--press") o.press = true;
            else {
                std::cerr << "unknown option " << flag << "\n";
                std::exit(2);
            }
        }
        if (o.clocks_per_bit < 4 || o.send_path.empty() || o.receive_path.empty()) {
            std::cerr << "need --clocks-per-bit (4 or more), --send and --receive\n";
            std::exit(2);
        }
        return o;
    }
};

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot read " << path << "\n";
        std::exit(2);
    }
    std::vector<uint8_t> bytes;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        bytes.push_back(static_cast<uint8_t>(std::stoul(line, nullptr, 16)));
    }
    return bytes;
}

// The level the host drives onto rxd, as a list of (cycle, level) changes.
class Line {
public:
    void hold(double cycles, int level) {
        changes_.push_back({static_cast<uint64_t>(std::llround(now_)), level});
        now_ += cycles;
    }
    double now() const { return now_; }
    int level_at(uint64_t cycle) {
        while (next_ < changes_.size() && changes_[next_].first <= cycle) {
            level_ = changes_[next_].second;
            ++next_;
        }
        return level_;
    }

private:
    std::vector<std::pair<uint64_t, int>> changes_;
    double now_ = 0.0;
    size_t next_ = 0;
    int level_ = 1;
};

// A UART receiver with the host's own bit period.
class Receiver {
public:
    explicit Receiver(double bit) : bit_(bit) {}
    // Returns true when a byte completed on this cycle.
    bool step(uint64_t cycle, int level, uint8_t& byte) {
        if (!busy_) {
            if (last_ == 1 && level == 0) {
                busy_ = true;
                start_ = static_cast<double>(cycle);
                index_ = 0;
                shift_ = 0;
            }
            last_ = level;
            return false;
        }
        last_ = level;
        const double sample = start_ + bit_ * (index_ + 0.5);
        if (static_cast<double>(cycle) < sample) return false;
        if (index_ == 0 && level != 0) {
            busy_ = false;
            ++glitches;
            return false;
        }
        if (index_ >= 1 && index_ <= 8) shift_ |= level << (index_ - 1);
        if (index_ == 9) {
            busy_ = false;
            if (level != 1) {
                ++framing_errors;
                return false;
            }
            byte = static_cast<uint8_t>(shift_);
            return true;
        }
        ++index_;
        return false;
    }
    long glitches = 0;
    long framing_errors = 0;

private:
    double bit_;
    bool busy_ = false;
    int last_ = 1;
    double start_ = 0.0;
    int index_ = 0;
    int shift_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const Options options = Options::parse(argc, argv);
    const std::vector<uint8_t> sent = read_bytes(options.send_path);
    const double bit = options.clocks_per_bit * (1.0 + options.skew / 100.0);
    std::mt19937 rng(options.seed);
    std::uniform_int_distribution<int> gap(0, options.gap);

    Line line;
    line.hold(64 * bit, 1);
    line.hold(40 * bit, 0);  // the break: twice what the board needs
    line.hold(4 * bit, 1);
    for (uint8_t byte : sent) {
        line.hold(bit, 0);
        for (int k = 0; k < 8; ++k) line.hold(bit, (byte >> k) & 1);
        line.hold(bit, 1);
        if (options.gap) line.hold(gap(rng) * bit, 1);
    }
    const uint64_t sent_by = static_cast<uint64_t>(line.now());
    line.hold(1, 1);

    Vicepi_top top;
    top.clk = 0;
    top.button_n = options.press ? 0 : 1;
    top.rxd = 1;
    top.eval();

    Receiver receiver(bit);
    std::vector<uint8_t> received;
    uint64_t cycle = 0;
    uint64_t last_activity = 0;
    const uint64_t quiet = static_cast<uint64_t>(options.quiet_bits * bit);
    for (; cycle < options.timeout; ++cycle) {
        if (options.press && cycle == 32) top.button_n = 1;
        top.rxd = line.level_at(cycle);
        top.clk = 1;
        top.eval();
        top.clk = 0;
        top.eval();
        uint8_t byte = 0;
        if (receiver.step(cycle, top.txd, byte)) {
            received.push_back(byte);
            last_activity = cycle;
        }
        if (options.expect >= 0 && static_cast<long>(received.size()) >= options.expect
            && cycle > sent_by)
            break;
        if (options.expect < 0 && cycle > sent_by && cycle > last_activity + quiet) break;
    }
    const bool timed_out = cycle >= options.timeout;
    // A few more bit times, to catch anything sent beyond what was expected.
    for (uint64_t extra = 0; extra < static_cast<uint64_t>(24 * bit); ++extra, ++cycle) {
        top.rxd = 1;
        top.clk = 1;
        top.eval();
        top.clk = 0;
        top.eval();
        uint8_t byte = 0;
        if (receiver.step(cycle, top.txd, byte)) received.push_back(byte);
    }
    top.final();

    std::ofstream out(options.receive_path);
    for (uint8_t byte : received) {
        out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte)
            << "\n";
    }
    std::cout << "sent " << sent.size() << " received " << received.size() << " cycles "
              << std::dec << cycle << " error_led " << ((top.led >> 1) & 1)
              << " framing_errors " << receiver.framing_errors << " glitches "
              << receiver.glitches << "\n";
    if (timed_out) std::cerr << "timed out\n";
    return timed_out ? 1 : 0;
}
