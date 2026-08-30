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

    // The index is the feature position, which for a port on one scale makes
    // no difference and for a per-feature port is the whole point. Frames are
    // laid out one after another, so the column is the position within a
    // frame rather than within the vector handed in.
    std::vector<int8_t> quantize_input(const std::vector<float>& values) const {
        const PortSpec& port = manifest_.input();
        const double zero = port.zero_point;
        const size_t stride = port.elements() ? port.elements() : 1;
        std::vector<int8_t> out;
        out.reserve(values.size());
        for (size_t index = 0; index < values.size(); ++index) {
            const double scale = port.scale_at(index % stride);
            const double raw =
                std::nearbyint(static_cast<double>(values[index]) / scale) + zero;
            out.push_back(static_cast<int8_t>(std::max(-128.0, std::min(127.0, raw))));
        }
        return out;
    }

    std::vector<float> dequantize_output(const std::vector<int8_t>& values) const {
        const PortSpec& port = manifest_.output();
        const int zero = port.zero_point;
        const size_t stride = port.elements() ? port.elements() : 1;
        std::vector<float> out;
        out.reserve(values.size());
        for (size_t index = 0; index < values.size(); ++index) {
            const double scale = port.scale_at(index % stride);
            out.push_back(static_cast<float>(
                (static_cast<int>(values[index]) - zero) * scale));
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
