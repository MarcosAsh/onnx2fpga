// Verilator driver for a generated onnx2fpga design.
//
// This is the part of the project where C++ is doing real work: it is the
// cycle accurate simulation loop, and it runs once per clock edge for the
// whole test. It drives the input stream, applies randomised backpressure so
// handshake bugs surface, counts cycles, and checks every output beat against
// the golden vectors the compiler produced from its numpy model.
//
// Randomised backpressure matters. A stream pipeline that only ever sees
// tready held high will pass its tests and then deadlock on real hardware, so
// the default run perturbs both ends.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Votf_top.h"
#include "verilated.h"

#include "otf/hex_io.hpp"

namespace {

class Options {
public:
    std::string input_path = "golden/input.hex";
    std::string expected_path = "golden/expected.hex";
    int output_beats = 0;     // set instead of --expected to skip checking
    unsigned seed = 1;
    int input_duty = 100;   // percent of cycles the producer offers data
    int output_duty = 100;  // percent of cycles the consumer accepts data
    uint64_t timeout = 20'000'000;
    bool quiet = false;
    std::string dump_path;   // where to write the beats that came out

    static Options parse(int argc, char** argv) {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    std::cerr << "missing value for " << flag << "\n";
                    std::exit(2);
                }
                return argv[++i];
            };
            if (flag == "--input") options.input_path = next();
            else if (flag == "--expected") options.expected_path = next();
            else if (flag == "--output-beats") {
                options.output_beats = std::stoi(next());
                options.expected_path.clear();
            }
            else if (flag == "--seed") options.seed = std::stoul(next());
            else if (flag == "--input-duty") options.input_duty = std::stoi(next());
            else if (flag == "--output-duty") options.output_duty = std::stoi(next());
            else if (flag == "--timeout") options.timeout = std::stoull(next());
            else if (flag == "--dump") options.dump_path = next();
            else if (flag == "--quiet") options.quiet = true;
            else {
                std::cerr << "unknown option " << flag << "\n";
                std::exit(2);
            }
        }
        return options;
    }
};

// Verilator picks a port type from the signal width. These overloads let the
// harness treat every width the same way.
template <typename T>
inline void port_write(T& port, const otf::Beat& beat) {
    uint64_t value = 0;
    for (size_t word = 0; word < beat.size() && word < 2; ++word) {
        value |= static_cast<uint64_t>(beat[word]) << (32 * word);
    }
    port = static_cast<T>(value);
}

template <std::size_t N>
inline void port_write(VlWide<N>& port, const otf::Beat& beat) {
    for (std::size_t word = 0; word < N; ++word) {
        port[word] = word < beat.size() ? beat[word] : 0u;
    }
}

template <typename T>
inline otf::Beat port_read(const T& port, size_t words) {
    otf::Beat beat(words, 0u);
    const uint64_t value = static_cast<uint64_t>(port);
    for (size_t word = 0; word < words && word < 2; ++word) {
        beat[word] = static_cast<uint32_t>(value >> (32 * word));
    }
    return beat;
}

template <std::size_t N>
inline otf::Beat port_read(const VlWide<N>& port, size_t words) {
    otf::Beat beat(words, 0u);
    for (std::size_t word = 0; word < words && word < N; ++word) {
        beat[word] = port[word];
    }
    return beat;
}

bool beats_equal(const otf::Beat& left, const otf::Beat& right) {
    const size_t count = std::max(left.size(), right.size());
    for (size_t word = 0; word < count; ++word) {
        const uint32_t a = word < left.size() ? left[word] : 0u;
        const uint32_t b = word < right.size() ? right[word] : 0u;
        if (a != b) return false;
    }
    return true;
}

class Simulation {
public:
    Simulation(Votf_top& design, const Options& options)
        : design_(design), options_(options), rng_(options.seed) {}

    int run(const otf::HexVectors& stimulus, const otf::HexVectors& expected,
            size_t wanted_beats, size_t out_words) {
        reset();
        size_t sent = 0;
        std::vector<otf::Beat> collected;

        while (collected.size() < wanted_beats && cycles_ < options_.timeout) {
            design_.clk = 0;
            const bool offer = sent < stimulus.size() && gate(options_.input_duty);
            design_.s_tvalid = offer;
            if (offer) port_write(design_.s_tdata, stimulus[sent]);
            design_.m_tready = gate(options_.output_duty);
            design_.eval();

            const bool input_fires = design_.s_tvalid && design_.s_tready;
            const bool output_fires = design_.m_tvalid && design_.m_tready;
            otf::Beat sampled;
            if (output_fires) sampled = port_read(design_.m_tdata, out_words);

            design_.clk = 1;
            design_.eval();

            if (input_fires) ++sent;
            if (output_fires) {
                if (collected.empty()) first_output_cycle_ = cycles_;
                collected.push_back(std::move(sampled));
            }
            ++cycles_;
        }

        if (!options_.dump_path.empty()) {
            dump(collected, out_words * 8);
        }
        return report(stimulus, expected, collected, sent, wanted_beats);
    }

private:
    void dump(const std::vector<otf::Beat>& collected, size_t digits) const {
        otf::HexVectors vectors;
        vectors.set_digits(digits);
        for (const otf::Beat& beat : collected) vectors.push(beat);
        vectors.save(options_.dump_path);
    }

    void reset() {
        design_.rst_n = 0;
        design_.s_tvalid = 0;
        design_.m_tready = 0;
        for (int edge = 0; edge < 8; ++edge) {
            design_.clk = edge & 1;
            design_.eval();
        }
        design_.rst_n = 1;
    }

    bool gate(int duty) {
        if (duty >= 100) return true;
        return static_cast<int>(rng_() % 100) < duty;
    }

    int report(const otf::HexVectors& stimulus, const otf::HexVectors& expected,
               const std::vector<otf::Beat>& collected, size_t sent,
               size_t wanted_beats) {
        const bool checking = !expected.empty();
        size_t mismatches = 0;
        for (size_t index = 0; checking && index < collected.size(); ++index) {
            if (beats_equal(collected[index], expected[index])) continue;
            if (mismatches < 8) {
                std::cerr << "  beat " << index << " expected "
                          << otf::HexVectors::render(expected[index], expected.digits())
                          << " got "
                          << otf::HexVectors::render(collected[index], expected.digits())
                          << "\n";
            }
            ++mismatches;
        }

        const bool timed_out = collected.size() < wanted_beats;
        if (!options_.quiet) {
            std::cout << "input beats      " << sent << " / " << stimulus.size() << "\n"
                      << "output beats     " << collected.size() << " / "
                      << wanted_beats << "\n"
                      << "cycles           " << cycles_ << "\n"
                      << "latency          " << first_output_cycle_
                      << " cycles to first output\n"
                      << "backpressure     input " << options_.input_duty
                      << "%, output " << options_.output_duty << "%\n";
        }
        if (timed_out) {
            std::cerr << "FAIL: stalled after " << collected.size()
                      << " of " << wanted_beats << " output beats\n";
            return 1;
        }
        if (mismatches) {
            std::cerr << "FAIL: " << mismatches << " mismatched beats\n";
            return 1;
        }
        if (checking) {
            std::cout << "PASS: " << collected.size()
                      << " beats match the golden model\n";
        }
        return 0;
    }

    Votf_top& design_;
    const Options& options_;
    std::mt19937 rng_;
    uint64_t cycles_ = 0;
    uint64_t first_output_cycle_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const Options options = Options::parse(argc, argv);

    otf::HexVectors stimulus, expected;
    try {
        stimulus = otf::HexVectors::load(options.input_path);
        if (!options.expected_path.empty()) {
            expected = otf::HexVectors::load(options.expected_path);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
    if (expected.empty() && options.output_beats <= 0) {
        std::cerr << "give either --expected <file> or --output-beats <n>\n";
        return 2;
    }

    Votf_top design;
    const size_t wanted = expected.empty()
        ? static_cast<size_t>(options.output_beats) : expected.size();
    const size_t out_words = expected.empty() ? 1 : expected[0].size();
    Simulation simulation(design, options);
    const int status = simulation.run(stimulus, expected, wanted, out_words);
    design.final();
    return status;
}
