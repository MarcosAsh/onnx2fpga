// Host side interface to a compiled design.
//
// Everything above this line is float and channel-last in the caller's layout;
// everything below is quantised beats on a stream. Backends differ only in
// where those beats go: a Verilator binary today, an XRT kernel on an F2
// instance once the shell integration lands.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "otf/manifest.hpp"

namespace otf {

// Converts between the caller's floats and the integer grid the design runs on.
class QuantizedIo {
public:
    explicit QuantizedIo(const Manifest& manifest) : manifest_(manifest) {}

    std::vector<int8_t> quantize_input(const std::vector<float>& values) const {
        const double scale = manifest_.input().scale;
        const double zero = manifest_.input().zero_point;
        std::vector<int8_t> out;
        out.reserve(values.size());
        for (float value : values) {
            const double raw =
                std::nearbyint(static_cast<double>(value) / scale) + zero;
            out.push_back(static_cast<int8_t>(std::max(-128.0, std::min(127.0, raw))));
        }
        return out;
    }

    std::vector<float> dequantize_output(const std::vector<int8_t>& values) const {
        const double scale = manifest_.output().scale;
        const int zero = manifest_.output().zero_point;
        std::vector<float> out;
        out.reserve(values.size());
        for (int8_t value : values) {
            out.push_back(static_cast<float>((static_cast<int>(value) - zero) * scale));
        }
        return out;
    }

private:
    const Manifest& manifest_;
};

class Accelerator {
public:
    virtual ~Accelerator() = default;

    // One frame in, one frame out, both channel-last integer buffers.
    virtual std::vector<int8_t> infer(const std::vector<int8_t>& input) = 0;

    virtual const Manifest& manifest() const = 0;

    virtual uint64_t last_cycle_count() const { return 0; }

    std::vector<float> infer_float(const std::vector<float>& values) {
        QuantizedIo io(manifest());
        return io.dequantize_output(infer(io.quantize_input(values)));
    }
};

}  // namespace otf
