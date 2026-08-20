// Cross language contract test.
//
// The Python compiler writes a fixture describing a matrix-vector layer and
// the exact integer output it expects. This program recomputes it with the C++
// kernel. If the two disagree, one of them has drifted from the written
// arithmetic contract, and the RTL is checked against the same contract, so
// catching it here is much cheaper than catching it in simulation.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "otf/kernels.hpp"

namespace {

class Fixture {
public:
    static Fixture load(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("cannot open " + path);
        Fixture fixture;
        file >> fixture.mw >> fixture.mh >> fixture.shift >> fixture.lower >>
            fixture.upper >> fixture.vectors >> fixture.zero_point;
        fixture.weights = read<int8_t>(file, fixture.mw * fixture.mh);
        fixture.bias = read<int32_t>(file, fixture.mh);
        fixture.multipliers = read<int32_t>(file, fixture.mh);
        fixture.input = read<int8_t>(file, fixture.vectors * fixture.mw);
        fixture.expected = read<int8_t>(file, fixture.vectors * fixture.mh);
        return fixture;
    }

    int mw = 0, mh = 0, shift = 0, vectors = 0;
    int64_t lower = 0, upper = 0, zero_point = 0;
    std::vector<int8_t> weights, input, expected;
    std::vector<int32_t> bias, multipliers;

private:
    template <typename T>
    static std::vector<T> read(std::istream& stream, int count) {
        std::vector<T> values(static_cast<size_t>(count));
        for (int index = 0; index < count; ++index) {
            long long raw = 0;
            if (!(stream >> raw)) throw std::runtime_error("fixture truncated");
            values[static_cast<size_t>(index)] = static_cast<T>(raw);
        }
        return values;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: check_matvec <fixture>\n";
        return 2;
    }
    try {
        const Fixture fixture = Fixture::load(argv[1]);
        const otf::MatVecKernel kernel(fixture.mw, fixture.mh, fixture.weights,
                                       fixture.bias, fixture.multipliers,
                                       fixture.shift, fixture.lower, fixture.upper,
                                       fixture.zero_point);
        std::vector<int8_t> produced(fixture.expected.size());
        kernel.run(fixture.input.data(), fixture.vectors, produced.data());

        size_t mismatches = 0;
        for (size_t index = 0; index < produced.size(); ++index) {
            if (produced[index] == fixture.expected[index]) continue;
            if (mismatches < 8) {
                std::cerr << "  element " << index << " expected "
                          << static_cast<int>(fixture.expected[index]) << " got "
                          << static_cast<int>(produced[index]) << "\n";
            }
            ++mismatches;
        }
        if (mismatches) {
            std::cerr << "FAIL: " << mismatches << " of " << produced.size()
                      << " elements differ from the Python model\n";
            return 1;
        }
        std::cout << "PASS: " << produced.size()
                  << " elements match the Python model\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
