// Integer reference kernel, a third implementation of the arithmetic next to the numpy model and the RTL.
#pragma once

#include <cstdint>
#include <vector>

#include "otf/requantize.hpp"

namespace otf {

// Input is (vectors, mw), weights are (mw, mh), bias and multipliers are per output channel.
class MatVecKernel {
public:
    MatVecKernel(int mw, int mh, std::vector<int8_t> weights,
                 std::vector<int32_t> bias, std::vector<int32_t> multipliers,
                 int shift, int64_t lower, int64_t upper, int64_t zero_point = 0)
        : mw_(mw), mh_(mh), weights_(std::move(weights)), bias_(std::move(bias)),
          multipliers_(std::move(multipliers)), shift_(shift),
          lower_(lower), upper_(upper), zero_point_(zero_point) {}

    void run(const int8_t* input, int vectors, int8_t* output) const {
        for (int v = 0; v < vectors; ++v) {
            const int8_t* row = input + static_cast<size_t>(v) * mw_;
            for (int channel = 0; channel < mh_; ++channel) {
                int64_t accumulator = bias_[channel];
                for (int k = 0; k < mw_; ++k) {
                    accumulator += static_cast<int64_t>(row[k]) *
                                   weights_[static_cast<size_t>(k) * mh_ + channel];
                }
                const Requantizer rescale(multipliers_[channel], shift_, lower_,
                                          upper_, zero_point_);
                output[static_cast<size_t>(v) * mh_ + channel] =
                    static_cast<int8_t>(rescale.apply(accumulator));
            }
        }
    }

private:
    int mw_;
    int mh_;
    std::vector<int8_t> weights_;
    std::vector<int32_t> bias_;
    std::vector<int32_t> multipliers_;
    int shift_;
    int64_t lower_;
    int64_t upper_;
    int64_t zero_point_;
};

}  // namespace otf
