// Runs the golden vectors on a Kria board through the AXI4-Lite map in manifest.json; needs root for /dev/mem.

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "otf/hex_io.hpp"
#include "otf/json.hpp"

namespace {

class Registers {
public:
    Registers(const std::string& device, off_t base, size_t span) : span_(span) {
        const int fd = open(device.c_str(), O_RDWR | O_SYNC);
        if (fd < 0) throw std::runtime_error("cannot open " + device);
        void* mapped = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
        close(fd);
        if (mapped == MAP_FAILED) throw std::runtime_error("cannot map " + device);
        words_ = static_cast<volatile uint32_t*>(mapped);
    }
    ~Registers() { munmap(const_cast<uint32_t*>(words_), span_); }

    uint32_t read(int offset) const { return words_[offset / 4]; }
    void write(int offset, uint32_t value) { words_[offset / 4] = value; }

private:
    volatile uint32_t* words_ = nullptr;
    size_t span_;
};

otf::JsonValue load_json(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::ostringstream text;
    text << file.rdbuf();
    return otf::JsonValue::parse(text.str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: kria_run <build-dir> [--mem /dev/mem]\n";
        return 2;
    }
    const std::string build_dir = argv[1];
    std::string device = "/dev/mem";
    if (argc == 4 && std::string(argv[2]) == "--mem") device = argv[3];

    try {
        const otf::JsonValue registers = load_json(build_dir + "/manifest.json")["registers"];
        const otf::JsonValue& map = registers["map"];
        auto offset = [&](const char* name) { return map[name]["offset"].as_int(); };
        Registers bus(device, static_cast<off_t>(registers["base"].number()),
                      static_cast<size_t>(registers["span"].number()));

        const uint32_t id = bus.read(offset("id"));
        if (id != static_cast<uint32_t>(map["id"]["value"].number())) {
            std::cerr << "no onnx2fpga shell at the base address, the id reads 0x" << std::hex
                      << id << "\n";
            return 1;
        }
        const otf::HexVectors inputs = otf::HexVectors::load(build_dir + "/golden/input.hex");
        const otf::HexVectors expected = otf::HexVectors::load(build_dir + "/golden/expected.hex");
        const int prediction_words = map["prediction"]["words"].as_int();
        int mismatches = 0;
        for (size_t vector = 0; vector < inputs.size(); ++vector) {
            for (size_t word = 0; word < inputs[vector].size(); ++word) {
                bus.write(offset("features") + 4 * static_cast<int>(word), inputs[vector][word]);
            }
            bus.write(offset("control"), 1);
            const auto start = std::chrono::steady_clock::now();
            while ((bus.read(offset("status")) & 1) == 0) {
                if (std::chrono::steady_clock::now() - start > std::chrono::seconds(1)) {
                    std::cerr << "no prediction after a second\n";
                    return 1;
                }
            }
            for (int word = 0; word < prediction_words; ++word) {
                const uint32_t got = bus.read(offset("prediction") + 4 * word);
                const size_t index = static_cast<size_t>(word);
                const uint32_t want = index < expected[vector].size() ? expected[vector][index] : 0;
                if (got != want) ++mismatches;
            }
        }
        std::cout << inputs.size() << " vectors, " << mismatches << " mismatched words, "
                  << bus.read(offset("cycles")) << " cycles from start to prediction (latency "
                  << bus.read(offset("latency")) << ")\n";
        return mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
