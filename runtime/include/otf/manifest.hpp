// Typed view over the manifest.json the compiler writes next to the RTL.
// This is the contract between a compiled design and any host that drives it.
#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "otf/json.hpp"

namespace otf {

struct PortSpec {
    std::string tensor;
    std::vector<int> shape;
    int elem_bits = 8;
    int elems_per_beat = 1;
    int beats = 0;
    double scale = 1.0;
    int zero_point = 0;
    std::string source_layout = "channel-last";

    size_t elements() const {
        size_t count = 1;
        for (int extent : shape) count *= static_cast<size_t>(extent);
        return count;
    }
};

class Manifest {
public:
    static Manifest load(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("cannot open " + path);
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return from_json(JsonValue::parse(buffer.str()));
    }

    static Manifest from_json(const JsonValue& root) {
        Manifest manifest;
        manifest.name_ = root["name"].string();
        manifest.device_ = root["device"].string();
        manifest.fmax_mhz_ = root["fmax_mhz"].number();
        manifest.input_ = read_port(root["input"]);
        manifest.output_ = read_port(root["output"]);
        if (root.has("cycles_per_frame")) {
            manifest.cycles_per_frame_ = root["cycles_per_frame"].as_int();
        }
        return manifest;
    }

    const std::string& name() const { return name_; }
    const std::string& device() const { return device_; }
    double fmax_mhz() const { return fmax_mhz_; }
    int cycles_per_frame() const { return cycles_per_frame_; }
    const PortSpec& input() const { return input_; }
    const PortSpec& output() const { return output_; }

    double estimated_fps() const {
        if (cycles_per_frame_ <= 0) return 0.0;
        return fmax_mhz_ * 1e6 / cycles_per_frame_;
    }

private:
    static PortSpec read_port(const JsonValue& node) {
        PortSpec port;
        port.tensor = node["tensor"].string();
        for (const JsonValue& extent : node["shape"].array()) {
            port.shape.push_back(extent.as_int());
        }
        port.elem_bits = node["elem_bits"].as_int();
        port.elems_per_beat = node["elems_per_beat"].as_int();
        port.beats = node["beats"].as_int();
        if (node.has("scale")) port.scale = node["scale"].number();
        if (node.has("zero_point")) port.zero_point = node["zero_point"].as_int();
        if (node.has("source_layout")) port.source_layout = node["source_layout"].string();
        return port;
    }

    std::string name_;
    std::string device_;
    double fmax_mhz_ = 0.0;
    int cycles_per_frame_ = 0;
    PortSpec input_;
    PortSpec output_;
};

}  // namespace otf
