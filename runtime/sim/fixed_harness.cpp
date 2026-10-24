// Verilator driver for fixed latency designs: each answer is due exactly --latency cycles after its input.

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Votf_top.h"
#include "verilated.h"

#include "otf/ports.hpp"

namespace {

struct Options {
    std::string input_path = "golden/input.hex";
    std::string expected_path = "golden/expected.hex";
    long latency = -1;      // measured from the first answer when not given
    int input_duty = 100;   // percent of cycles features_valid is high
    unsigned seed = 1;
    bool quiet = false;

    static Options parse(int argc, char** argv) {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--quiet") {
                options.quiet = true;
                continue;
            }
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(2);
            }
            const std::string value = argv[++i];
            if (flag == "--input") options.input_path = value;
            else if (flag == "--expected") options.expected_path = value;
            else if (flag == "--latency") options.latency = std::stol(value);
            else if (flag == "--input-duty") options.input_duty = std::stoi(value);
            else if (flag == "--seed") options.seed = std::stoul(value);
            else {
                std::cerr << "unknown option " << flag << "\n";
                std::exit(2);
            }
        }
        return options;
    }
};

struct Sent {
    long cycle;
    size_t index;
};

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const Options options = Options::parse(argc, argv);
    otf::HexVectors stimulus, expected;
    try {
        stimulus = otf::HexVectors::load(options.input_path);
        expected = otf::HexVectors::load(options.expected_path);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
    if (stimulus.size() != expected.size() || stimulus.empty()) {
        std::cerr << "need one expected answer per input vector\n";
        return 2;
    }

    Votf_top design;
    std::mt19937 rng(options.seed);
    const size_t out_words = expected[0].size();
    long latency = options.latency;
    std::deque<Sent> in_flight;
    size_t sent = 0, answered = 0, mismatches = 0;
    long cycle = 0;

    design.clear = 1;
    design.features_valid = 0;
    for (int edge = 0; edge < 4; ++edge) {
        design.clock = edge & 1;
        design.eval();
    }
    design.clear = 0;

    // Inputs are applied and outputs sampled before each rising edge.
    const long give_up = static_cast<long>(stimulus.size()) * 200 + 1000;
    while ((sent < stimulus.size() || !in_flight.empty()) && cycle < give_up) {
        design.clock = 0;
        const bool offer = sent < stimulus.size()
            && (options.input_duty >= 100
                || static_cast<int>(rng() % 100) < options.input_duty);
        design.features_valid = offer;
        if (offer) otf::port_write(design.features, stimulus[sent]);
        design.eval();

        if (design.prediction_valid) {
            if (in_flight.empty()) {
                std::cerr << "FAIL: prediction_valid at cycle " << cycle
                          << " with no input waiting\n";
                return 1;
            }
            const Sent front = in_flight.front();
            in_flight.pop_front();
            if (latency < 0) latency = cycle - front.cycle;
            if (cycle - front.cycle != latency) {
                std::cerr << "FAIL: vector " << front.index << " answered after "
                          << cycle - front.cycle << " cycles, not " << latency << "\n";
                return 1;
            }
            const otf::Beat got = otf::port_read(design.prediction, out_words);
            if (got != expected[front.index]) {
                if (mismatches < 8) {
                    std::cerr << "  vector " << front.index << " expected "
                              << otf::HexVectors::render(expected[front.index],
                                                         expected.digits())
                              << " got " << otf::HexVectors::render(got, expected.digits())
                              << "\n";
                }
                ++mismatches;
            }
            ++answered;
        } else if (latency >= 0 && !in_flight.empty()
                   && cycle - in_flight.front().cycle >= latency) {
            std::cerr << "FAIL: vector " << in_flight.front().index
                      << " got no answer after " << latency << " cycles\n";
            return 1;
        }
        if (offer) in_flight.push_back({cycle, sent++});

        design.clock = 1;
        design.eval();
        ++cycle;
    }
    design.final();

    if (!options.quiet) {
        std::cout << "input vectors    " << sent << "\n"
                  << "answers          " << answered << "\n"
                  << "cycles           " << cycle << "\n"
                  << "latency          " << latency << " cycles, every answer\n"
                  << "input duty       " << options.input_duty << "%\n";
    }
    if (answered != stimulus.size()) {
        std::cerr << "FAIL: " << answered << " of " << stimulus.size() << " answered\n";
        return 1;
    }
    if (mismatches) {
        std::cerr << "FAIL: " << mismatches << " answers differ from the golden model\n";
        return 1;
    }
    std::cout << "PASS: " << answered << " answers match, each exactly " << latency
              << " cycles after its input\n";
    return 0;
}
