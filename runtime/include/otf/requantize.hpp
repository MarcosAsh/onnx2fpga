// Bit exact counterpart of Requantizer.apply in the compiler and of
// otf_requant.sv in the hardware. All three must agree exactly; the harness
// and the kernel tests exist to prove they do.
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
        if (shift_ > 0) {
            scaled = (scaled + (int64_t{1} << (shift_ - 1))) >> shift_;
        }
        return std::clamp(scaled + zero_point_, lower_, upper_);
    }

    int32_t multiplier() const { return multiplier_; }
    int shift() const { return shift_; }
    int64_t zero_point() const { return zero_point_; }

private:
    int32_t multiplier_;
    int shift_;
    int64_t lower_;
    int64_t upper_;
    int64_t zero_point_;
};

// Sign extends the low `bits` of a two's complement value held in a wider word.
inline int64_t sign_extend(uint64_t raw, int bits) {
    if (bits >= 64) return static_cast<int64_t>(raw);
    const uint64_t mask = (uint64_t{1} << bits) - 1;
    const uint64_t value = raw & mask;
    const uint64_t sign = uint64_t{1} << (bits - 1);
    return static_cast<int64_t>((value ^ sign) - sign);
}

}  // namespace otf
