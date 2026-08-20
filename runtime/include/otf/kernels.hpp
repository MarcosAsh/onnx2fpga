// Integer reference kernels.
//
// These are a third independent implementation of the arithmetic, alongside
// the numpy model in the compiler and the RTL itself. Agreement between three
// implementations written against the same written contract is what makes a
// mismatch a real finding rather than a shared misreading. They are also fast
// enough to sweep a full dataset, which numpy is not.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "otf/requantize.hpp"

namespace otf {

struct Shape2d {
    int height = 0;
    int width = 0;
};

struct WindowSpec {
    Shape2d kernel;
    Shape2d stride{1, 1};
    Shape2d dilation{1, 1};
    int pad_top = 0;
    int pad_left = 0;

    Shape2d output_extent(Shape2d input, int pad_bottom, int pad_right) const {
        const int eh = (kernel.height - 1) * dilation.height + 1;
        const int ew = (kernel.width - 1) * dilation.width + 1;
        return {(input.height + pad_top + pad_bottom - eh) / stride.height + 1,
                (input.width + pad_left + pad_right - ew) / stride.width + 1};
    }
};

// Folded matrix vector product with per channel bias and rescale.
// Input is row major (vectors, mw); weights are (mw, mh).
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

    int mw() const { return mw_; }
    int mh() const { return mh_; }

private:
    int mw_;
    int mh_;
    std::vector<int8_t> weights_;
    std::vector<int32_t> bias_;
    std::vector<int32_t> multipliers_;
    int shift_;
    int64_t lower_;
    int64_t upper_;
    int64_t zero_point_ = 0;
};

// Channel last window extraction. Out of range positions emit the integer
// standing for real zero, which is the zero point.
class SlidingWindowKernel {
public:
    SlidingWindowKernel(Shape2d input, int channels, WindowSpec spec, Shape2d output,
                        int8_t pad_value = 0)
        : input_(input), channels_(channels), spec_(spec), output_(output),
          pad_value_(pad_value) {}

    std::vector<int8_t> run(const std::vector<int8_t>& image) const {
        const int window = spec_.kernel.height * spec_.kernel.width * channels_;
        std::vector<int8_t> out(static_cast<size_t>(output_.height) *
                                output_.width * window, pad_value_);
        size_t cursor = 0;
        for (int oh = 0; oh < output_.height; ++oh) {
            for (int ow = 0; ow < output_.width; ++ow) {
                for (int ky = 0; ky < spec_.kernel.height; ++ky) {
                    for (int kx = 0; kx < spec_.kernel.width; ++kx) {
                        const int row = oh * spec_.stride.height +
                                        ky * spec_.dilation.height - spec_.pad_top;
                        const int col = ow * spec_.stride.width +
                                        kx * spec_.dilation.width - spec_.pad_left;
                        const bool inside = row >= 0 && row < input_.height &&
                                            col >= 0 && col < input_.width;
                        for (int c = 0; c < channels_; ++c) {
                            out[cursor++] = inside
                                ? image[(static_cast<size_t>(row) * input_.width + col) *
                                        channels_ + c]
                                : pad_value_;
                        }
                    }
                }
            }
        }
        return out;
    }

private:
    Shape2d input_;
    int channels_;
    WindowSpec spec_;
    Shape2d output_;
    int8_t pad_value_ = 0;
};

// Max over each window. Scale preserving, so no rescale is needed.
class MaxPoolKernel {
public:
    MaxPoolKernel(int channels, int window) : channels_(channels), window_(window) {}

    std::vector<int8_t> run(const std::vector<int8_t>& windows) const {
        const size_t count = windows.size() / (static_cast<size_t>(window_) * channels_);
        std::vector<int8_t> out(count * channels_);
        for (size_t w = 0; w < count; ++w) {
            for (int c = 0; c < channels_; ++c) {
                int8_t best = std::numeric_limits<int8_t>::min();
                for (int k = 0; k < window_; ++k) {
                    best = std::max(best, windows[(w * window_ + k) * channels_ + c]);
                }
                out[w * channels_ + c] = best;
            }
        }
        return out;
    }

private:
    int channels_;
    int window_;
};

}  // namespace otf
