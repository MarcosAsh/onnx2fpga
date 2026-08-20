// Accelerator backed by the Verilator model of a compiled design.
//
// It writes the stimulus in the same hex form the simulation harness reads,
// invokes the harness, and reads the beats back, so the host path and the
// verification path exercise the same design and the same encoding.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "otf/accelerator.hpp"
#include "otf/hex_io.hpp"

namespace otf {

class SimulationAccelerator : public Accelerator {
public:
    explicit SimulationAccelerator(const std::string& build_dir)
        : build_dir_(build_dir),
          manifest_(Manifest::load(build_dir + "/manifest.json")) {
        binary_ = "./obj_dir/V" + manifest_.name();
        if (!std::filesystem::exists(build_dir_ + "/obj_dir/V" + manifest_.name())) {
            throw std::runtime_error(
                "simulation binary for " + manifest_.name() +
                " not built; run make in " + build_dir_);
        }
    }

    const Manifest& manifest() const override { return manifest_; }

    std::vector<int8_t> infer(const std::vector<int8_t>& input) override {
        const PortSpec& in = manifest_.input();
        const PortSpec& out = manifest_.output();
        if (input.size() != in.elements()) {
            throw std::runtime_error("expected " + std::to_string(in.elements()) +
                                     " input elements");
        }
        write_beats(build_dir_ + "/golden/host_input.hex", input, in);

        // $readmemh paths in the generated netlist are relative to the build
        // directory, so the model has to run from there.
        const std::string command =
            "cd " + build_dir_ + " && " + binary_ +
            " --input golden/host_input.hex --output-beats " +
            std::to_string(out.beats) + " --quiet --dump golden/host_output.hex";
        if (std::system(command.c_str()) != 0) {
            throw std::runtime_error("simulation run failed");
        }
        return read_beats(build_dir_ + "/golden/host_output.hex", out);
    }

private:
    static void write_beats(const std::string& path,
                            const std::vector<int8_t>& values, const PortSpec& port) {
        HexVectors vectors;
        vectors.set_digits(static_cast<size_t>(
            (port.elems_per_beat * port.elem_bits + 3) / 4));
        for (size_t base = 0; base < values.size();
             base += static_cast<size_t>(port.elems_per_beat)) {
            Beat beat((port.elems_per_beat * port.elem_bits + 31) / 32, 0u);
            for (int lane = 0; lane < port.elems_per_beat; ++lane) {
                const uint32_t raw =
                    static_cast<uint8_t>(values[base + static_cast<size_t>(lane)]);
                const int bit = lane * port.elem_bits;
                beat[static_cast<size_t>(bit / 32)] |= raw << (bit % 32);
            }
            vectors.push(std::move(beat));
        }
        vectors.save(path);
    }

    static std::vector<int8_t> read_beats(const std::string& path, const PortSpec& port) {
        const HexVectors vectors = HexVectors::load(path);
        std::vector<int8_t> values;
        values.reserve(vectors.size() * static_cast<size_t>(port.elems_per_beat));
        for (size_t index = 0; index < vectors.size(); ++index) {
            const Beat& beat = vectors[index];
            for (int lane = 0; lane < port.elems_per_beat; ++lane) {
                const int bit = lane * port.elem_bits;
                const uint32_t word = beat[static_cast<size_t>(bit / 32)];
                values.push_back(static_cast<int8_t>((word >> (bit % 32)) & 0xffu));
            }
        }
        return values;
    }

    std::string build_dir_;
    std::string binary_;
    Manifest manifest_;
};

}  // namespace otf
