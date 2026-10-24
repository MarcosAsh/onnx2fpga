// Moves beats in and out of Verilator ports, whose C++ type depends on the signal width.
#pragma once

#include <cstdint>

#include "verilated.h"

#include "otf/hex_io.hpp"

namespace otf {

template <typename T>
void port_write(T& port, const Beat& beat) {
    uint64_t value = 0;
    for (size_t word = 0; word < beat.size() && word < 2; ++word) {
        value |= static_cast<uint64_t>(beat[word]) << (32 * word);
    }
    port = static_cast<T>(value);
}

template <std::size_t N>
void port_write(VlWide<N>& port, const Beat& beat) {
    for (std::size_t word = 0; word < N; ++word) port[word] = word < beat.size() ? beat[word] : 0u;
}

template <typename T>
Beat port_read(const T& port, size_t words) {
    Beat beat(words, 0u);
    const uint64_t value = static_cast<uint64_t>(port);
    for (size_t word = 0; word < words && word < 2; ++word) {
        beat[word] = static_cast<uint32_t>(value >> (32 * word));
    }
    return beat;
}

template <std::size_t N>
Beat port_read(const VlWide<N>& port, size_t words) {
    Beat beat(words, 0u);
    for (std::size_t word = 0; word < words && word < N; ++word) beat[word] = port[word];
    return beat;
}

}  // namespace otf
