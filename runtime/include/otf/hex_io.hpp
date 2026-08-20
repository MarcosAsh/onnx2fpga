// Reading and writing the $readmemh style vectors the compiler emits.
// A beat of arbitrary width is held as little endian 32 bit words so the same
// container feeds a Verilator port, a C++ kernel or a DMA buffer.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace otf {

using Beat = std::vector<uint32_t>;

class HexVectors {
public:
    HexVectors() = default;

    static HexVectors load(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("cannot open " + path);
        HexVectors vectors;
        std::string line;
        while (std::getline(file, line)) {
            const std::string trimmed = strip(line);
            if (trimmed.empty()) continue;
            vectors.beats_.push_back(parse(trimmed));
            vectors.digits_ = std::max(vectors.digits_, trimmed.size());
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
    const std::vector<Beat>& beats() const { return beats_; }

    static std::string render(const Beat& beat, size_t digits) {
        std::ostringstream out;
        std::string text;
        for (size_t word = beat.size(); word-- > 0;) {
            char buffer[9];
            std::snprintf(buffer, sizeof(buffer), "%08x", beat[word]);
            text += buffer;
        }
        size_t start = 0;
        while (start + 1 < text.size() && text.size() - start > digits) ++start;
        text = text.substr(start);
        while (text.size() < digits) text.insert(text.begin(), '0');
        out << text;
        return out.str();
    }

private:
    static std::string strip(const std::string& line) {
        size_t begin = line.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) return "";
        size_t end = line.find_last_not_of(" \t\r\n");
        std::string body = line.substr(begin, end - begin + 1);
        if (body.rfind("//", 0) == 0) return "";
        return body;
    }

    static Beat parse(const std::string& text) {
        Beat beat;
        size_t position = text.size();
        while (position > 0) {
            const size_t take = position >= 8 ? 8 : position;
            beat.push_back(static_cast<uint32_t>(
                std::stoul(text.substr(position - take, take), nullptr, 16)));
            position -= take;
        }
        return beat;
    }

    std::vector<Beat> beats_;
    size_t digits_ = 1;
};

}  // namespace otf
