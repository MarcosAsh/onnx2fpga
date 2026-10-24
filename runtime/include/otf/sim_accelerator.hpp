// Runs one frame through the Verilator model: writes hex stimulus, runs the harness and reads the beats back.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "otf/hex_io.hpp"
#include "otf/manifest.hpp"

namespace otf {

class SimulationAccelerator {
public:
    explicit SimulationAccelerator(const std::string& build_dir)
        : build_dir_(build_dir), manifest_(Manifest::load(build_dir + "/manifest.json")) {
        binary_ = "./obj_dir/V" + manifest_.name();
        if (!std::filesystem::exists(build_dir_ + "/" + binary_)) {
            throw std::runtime_error("simulation binary for " + manifest_.name() +
                                     " not built; run make in " + build_dir_);
        }
    }

    const Manifest& manifest() const { return manifest_; }

    std::vector<float> infer_float(const std::vector<float>& values) {
        const PortSpec& in = manifest_.input();
        const PortSpec& out = manifest_.output();
        if (values.size() != in.elements()) {
            throw std::runtime_error("expected " + std::to_string(in.elements()) +
                                     " input elements");
        }
        std::vector<int8_t> quantized;
        for (float value : values) {
            const double raw = std::nearbyint(value / in.scale) + in.zero_point;
            quantized.push_back(static_cast<int8_t>(std::clamp(raw, -128.0, 127.0)));
        }
        write_beats(build_dir_ + "/golden/host_input.hex", quantized, in);

        // $readmemh paths in the netlist are relative to the build directory, so run from there.
        const std::string command =
            "cd " + build_dir_ + " && " + binary_ +
            " --input golden/host_input.hex --output-beats " +
            std::to_string(out.beats) + " --quiet --dump golden/host_output.hex";
        if (std::system(command.c_str()) != 0) throw std::runtime_error("simulation run failed");

        std::vector<float> result;
        for (int8_t value : read_beats(build_dir_ + "/golden/host_output.hex", out)) {
            result.push_back(static_cast<float>((value - out.zero_point) * out.scale));
        }
        return result;
    }

private:
    static void write_beats(const std::string& path,
                            const std::vector<int8_t>& values, const PortSpec& port) {
        const int beat_bits = port.elems_per_beat * port.elem_bits;
        HexVectors vectors;
        vectors.set_digits((beat_bits + 3) / 4);
        for (size_t base = 0; base < values.size(); base += port.elems_per_beat) {
            Beat beat((beat_bits + 31) / 32, 0u);
            for (int lane = 0; lane < port.elems_per_beat; ++lane) {
                const int bit = lane * port.elem_bits;
                beat[bit / 32] |= static_cast<uint32_t>(static_cast<uint8_t>(values[base + lane]))
                                  << (bit % 32);
            }
            vectors.push(std::move(beat));
        }
        vectors.save(path);
    }

    static std::vector<int8_t> read_beats(const std::string& path, const PortSpec& port) {
        const HexVectors vectors = HexVectors::load(path);
        std::vector<int8_t> values;
        for (size_t index = 0; index < vectors.size(); ++index) {
            for (int lane = 0; lane < port.elems_per_beat; ++lane) {
                const int bit = lane * port.elem_bits;
                values.push_back(static_cast<int8_t>((vectors[index][bit / 32] >> (bit % 32)) & 0xffu));
            }
        }
        return values;
    }

    std::string build_dir_;
    std::string binary_;
    Manifest manifest_;
};

}  // namespace otf
