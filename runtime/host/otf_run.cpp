// Host driver.
//
// Loads a compiled design, quantises a float input with the scales recorded in
// the manifest, runs one frame, and prints the dequantised result. Today the
// only backend is the Verilator model; an XRT backend slots in behind the same
// Accelerator interface without the caller changing.

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "otf/sim_accelerator.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: otf_run <build-dir> [--seed N]\n";
        return 2;
    }
    const std::string build_dir = argv[1];
    unsigned seed = 0;
    for (int index = 2; index + 1 < argc; index += 2) {
        if (std::string(argv[index]) == "--seed") seed = std::stoul(argv[index + 1]);
    }

    try {
        otf::SimulationAccelerator accelerator(build_dir);
        const otf::Manifest& manifest = accelerator.manifest();

        std::cout << "design      " << manifest.name() << " on " << manifest.device()
                  << "\ncycles      " << manifest.cycles_per_frame()
                  << " per frame\nthroughput  " << manifest.estimated_fps()
                  << " frames/s at " << manifest.fmax_mhz() << " MHz\n";

        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> input(manifest.input().elements());
        for (float& value : input) value = normal(rng);

        const std::vector<float> output = accelerator.infer_float(input);
        std::cout << "output      ";
        for (float value : output) std::cout << value << " ";
        std::cout << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
