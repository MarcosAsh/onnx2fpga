// Bit exact twin of Requantizer.apply in the compiler and otf_requant.sv.
#pragma once

#include <algorithm>
#include <cstdint>

namespace otf {

class Requantizer {
public:
    Requantizer(int32_t multiplier, int shift, int64_t lower, int64_t upper,
                int64_t zero_point = 0)
        : multiplier_(multiplier), shift_(shift), lower_(lower), upper_(upper),
          zero_point_(zero_point) {}

    int64_t apply(int64_t accumulator) const {
        int64_t scaled = accumulator * static_cast<int64_t>(multiplier_);
        if (shift_ > 0) scaled = (scaled + (int64_t{1} << (shift_ - 1))) >> shift_;
        return std::clamp(scaled + zero_point_, lower_, upper_);
    }

private:
    int32_t multiplier_;
    int shift_;
    int64_t lower_;
    int64_t upper_;
    int64_t zero_point_;
};

}  // namespace otf
