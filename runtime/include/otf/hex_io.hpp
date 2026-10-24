// Reads and writes $readmemh style vectors; a beat of any width is held as little endian 32 bit words.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace otf {

using Beat = std::vector<uint32_t>;

class HexVectors {
public:
    static HexVectors load(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("cannot open " + path);
        HexVectors vectors;
        std::string line;
        while (std::getline(file, line)) {
            const size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line.compare(first, 2, "//") == 0) continue;
            const std::string text = line.substr(first, line.find_last_not_of(" \t\r\n") - first + 1);
            vectors.beats_.push_back(parse(text));
            vectors.digits_ = std::max(vectors.digits_, text.size());
        }
        return vectors;
    }

    void save(const std::string& path) const {
        std::ofstream file(path);
        if (!file) throw std::runtime_error("cannot write " + path);
        for (const Beat& beat : beats_) file << render(beat, digits_) << '\n';
    }

    void push(Beat beat) { beats_.push_back(std::move(beat)); }

    const Beat& operator[](size_t index) const { return beats_[index]; }
    size_t size() const { return beats_.size(); }
    bool empty() const { return beats_.empty(); }
    size_t digits() const { return digits_; }
    void set_digits(size_t digits) { digits_ = digits; }

    static std::string render(const Beat& beat, size_t digits) {
        std::string text;
        for (size_t word = beat.size(); word-- > 0;) {
            char buffer[9];
            std::snprintf(buffer, sizeof(buffer), "%08x", beat[word]);
            text += buffer;
        }
        size_t start = 0;
        while (start + 1 < text.size() && text.size() - start > digits) ++start;
        text.erase(0, start);
        if (text.size() < digits) text.insert(0, digits - text.size(), '0');
        return text;
    }

private:
    static Beat parse(const std::string& text) {
        Beat beat;
        for (size_t end = text.size(); end > 0;) {
            const size_t take = std::min<size_t>(end, 8);
            beat.push_back(static_cast<uint32_t>(
                std::stoul(text.substr(end - take, take), nullptr, 16)));
            end -= take;
        }
        return beat;
    }

    std::vector<Beat> beats_;
    size_t digits_ = 1;
};

}  // namespace otf
