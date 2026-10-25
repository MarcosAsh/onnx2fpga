// Verilator driver for fixed latency designs, each answer due --latency cycles after its input; OTF_RELOAD adds a second bank.

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
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
    std::string weights_path;       // the idle bank's words, one hex word a line
    std::string expected_new_path;  // the answers with those weights
    std::set<long> commits;
    std::vector<uint32_t> crcs;     // zlib CRC-32 of the compiled bank, then of the new one

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
            else if (flag == "--weights") options.weights_path = value;
            else if (flag == "--expected-new") options.expected_new_path = value;
            else if (flag == "--commits") {
                std::stringstream list(value);
                for (std::string cycle; std::getline(list, cycle, ',');)
                    options.commits.insert(std::stol(cycle));
            } else if (flag == "--crc") {
                std::stringstream list(value);
                for (std::string crc; std::getline(list, crc, ',');)
                    options.crcs.push_back(static_cast<uint32_t>(std::stoul(crc, nullptr, 16)));
            }
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
    int bank;
};

std::vector<uint32_t> load_words(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<uint32_t> words;
    for (std::string line; in >> line;)
        words.push_back(static_cast<uint32_t>(std::stoul(line, nullptr, 16)));
    return words;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const Options options = Options::parse(argc, argv);
    otf::HexVectors stimulus, expected[2];
    std::vector<uint32_t> words;
    const bool reload = !options.weights_path.empty();
    try {
        stimulus = otf::HexVectors::load(options.input_path);
        expected[0] = otf::HexVectors::load(options.expected_path);
        expected[1] = expected[0];
        if (reload) {
            words = load_words(options.weights_path);
            expected[1] = otf::HexVectors::load(options.expected_new_path);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
    if (stimulus.size() != expected[0].size() || stimulus.size() != expected[1].size()
        || stimulus.empty()) {
        std::cerr << "need one expected answer per input vector\n";
        return 2;
    }
#ifndef OTF_RELOAD
    if (reload || !options.commits.empty()) {
        std::cerr << "--weights and --commits need a reloadable design\n";
        return 2;
    }
#endif
    if (!options.commits.empty() && *options.commits.begin() < static_cast<long>(words.size())) {
        std::cerr << "a commit comes before the idle bank is written\n";
        return 2;
    }

    Votf_top design;
    std::mt19937 rng(options.seed);
    const size_t out_words = expected[0][0].size();
    long latency = options.latency;
    std::deque<Sent> in_flight;
    size_t sent = 0, answered = 0, mismatches = 0, written = 0;
    size_t per_bank[2] = {0, 0};
    int bank = 0;
    long cycle = 0;
    const long last_commit = options.commits.empty() ? -1 : *options.commits.rbegin();

    design.clear = 1;
    design.features_valid = 0;
    for (int edge = 0; edge < 4; ++edge) {
        design.clock = edge & 1;
        design.eval();
    }
    design.clear = 0;

    // Inputs are applied and outputs sampled before each rising edge.
    const long give_up = static_cast<long>(stimulus.size()) * 200 + 1000;
    while ((sent < stimulus.size() || !in_flight.empty() || cycle <= last_commit)
           && cycle < give_up) {
        design.clock = 0;
        const bool offer = sent < stimulus.size()
            && (options.input_duty >= 100
                || static_cast<int>(rng() % 100) < options.input_duty);
        design.features_valid = offer;
        if (offer) otf::port_write(design.features, stimulus[sent]);
        const bool commit = options.commits.count(cycle) > 0;
#ifdef OTF_RELOAD
        design.weights_write = written < words.size();
        if (written < words.size()) {
            design.weights_address = static_cast<uint32_t>(written);
            design.weights_data = words[written++];
        }
        design.weights_commit = commit;
#endif
        design.eval();
#ifdef OTF_RELOAD
        if (design.weights_bank != bank) {
            std::cerr << "FAIL: bank " << int(design.weights_bank) << " is active at cycle "
                      << cycle << ", not " << bank << "\n";
            return 1;
        }
#endif

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
#ifdef OTF_RELOAD
            if (design.prediction_bank != front.bank) {
                std::cerr << "FAIL: vector " << front.index << " came out tagged with bank "
                          << int(design.prediction_bank) << ", not " << front.bank << "\n";
                return 1;
            }
#endif
            const otf::HexVectors& wanted = expected[front.bank];
            const otf::Beat got = otf::port_read(design.prediction, out_words);
            if (got != wanted[front.index]) {
                if (mismatches < 8) {
                    std::cerr << "  vector " << front.index << " in bank " << front.bank
                              << " expected "
                              << otf::HexVectors::render(wanted[front.index], wanted.digits())
                              << " got " << otf::HexVectors::render(got, wanted.digits())
                              << "\n";
                }
                ++mismatches;
            }
            ++per_bank[front.bank];
            ++answered;
        } else if (latency >= 0 && !in_flight.empty()
                   && cycle - in_flight.front().cycle >= latency) {
            std::cerr << "FAIL: vector " << in_flight.front().index
                      << " got no answer after " << latency << " cycles\n";
            return 1;
        }
        if (offer) in_flight.push_back({cycle, sent++, bank});

        design.clock = 1;
        design.eval();
        if (commit) bank ^= 1;
        ++cycle;
    }
    design.final();

    if (!options.quiet) {
        std::cout << "input vectors    " << sent << "\n"
                  << "answers          " << answered << "\n"
                  << "cycles           " << cycle << "\n"
                  << "latency          " << latency << " cycles, every answer\n"
                  << "input duty       " << options.input_duty << "%\n";
        if (reload) {
            std::cout << "commits          " << options.commits.size() << "\n"
                      << "answers by bank  " << per_bank[0] << " " << per_bank[1] << "\n";
        }
    }
#ifdef OTF_RELOAD
    const uint32_t version = design.weights_version;
    if (version != options.commits.size()) {
        std::cerr << "FAIL: the version reads " << version << " after "
                  << options.commits.size() << " commits\n";
        return 1;
    }
    if (options.crcs.size() == 2
        && (design.weights_crc != options.crcs[bank]
            || design.weights_idle_crc != options.crcs[bank ^ 1])) {
        std::cerr << std::hex << "FAIL: the CRCs read 0x" << design.weights_crc << " and 0x"
                  << design.weights_idle_crc << ", not 0x" << options.crcs[bank] << " and 0x"
                  << options.crcs[bank ^ 1] << "\n";
        return 1;
    }
#endif
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
