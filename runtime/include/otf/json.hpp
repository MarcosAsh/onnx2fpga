// Minimal JSON reader, enough for the manifest the compiler emits.
// Present so the host runtime has no third party dependency either.
#pragma once

#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace otf {

class JsonValue {
public:
    enum class Kind { Null, Bool, Number, String, Array, Object };

    JsonValue() = default;

    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::Null; }

    double number() const { expect(Kind::Number); return number_; }
    bool boolean() const { expect(Kind::Bool); return boolean_; }
    const std::string& string() const { expect(Kind::String); return string_; }
    const std::vector<JsonValue>& array() const { expect(Kind::Array); return array_; }

    const JsonValue& operator[](const std::string& key) const {
        expect(Kind::Object);
        auto found = object_.find(key);
        if (found == object_.end()) throw std::runtime_error("missing key " + key);
        return found->second;
    }

    bool has(const std::string& key) const {
        return kind_ == Kind::Object && object_.count(key) > 0;
    }

    int as_int() const { return static_cast<int>(number()); }

    static JsonValue parse(const std::string& text) {
        size_t cursor = 0;
        JsonValue value = parse_value(text, cursor);
        skip_space(text, cursor);
        return value;
    }

private:
    void expect(Kind wanted) const {
        if (kind_ != wanted) throw std::runtime_error("json type mismatch");
    }

    static void skip_space(const std::string& text, size_t& cursor) {
        while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor]))) {
            ++cursor;
        }
    }

    static JsonValue parse_value(const std::string& text, size_t& cursor) {
        skip_space(text, cursor);
        if (cursor >= text.size()) throw std::runtime_error("unexpected end of json");
        switch (text[cursor]) {
            case '{': return parse_object(text, cursor);
            case '[': return parse_array(text, cursor);
            case '"': return parse_string(text, cursor);
            case 't': cursor += 4; return make_bool(true);
            case 'f': cursor += 5; return make_bool(false);
            case 'n': cursor += 4; return JsonValue();
            default:  return parse_number(text, cursor);
        }
    }

    static JsonValue make_bool(bool value) {
        JsonValue node;
        node.kind_ = Kind::Bool;
        node.boolean_ = value;
        return node;
    }

    static JsonValue parse_number(const std::string& text, size_t& cursor) {
        size_t end = cursor;
        while (end < text.size() &&
               (std::isdigit(static_cast<unsigned char>(text[end])) ||
                text[end] == '-' || text[end] == '+' || text[end] == '.' ||
                text[end] == 'e' || text[end] == 'E')) {
            ++end;
        }
        JsonValue node;
        node.kind_ = Kind::Number;
        node.number_ = std::stod(text.substr(cursor, end - cursor));
        cursor = end;
        return node;
    }

    static JsonValue parse_string(const std::string& text, size_t& cursor) {
        ++cursor;
        std::string out;
        while (cursor < text.size() && text[cursor] != '"') {
            if (text[cursor] == '\\' && cursor + 1 < text.size()) {
                ++cursor;
                const char escape = text[cursor];
                out += (escape == 'n') ? '\n' : (escape == 't') ? '\t' : escape;
            } else {
                out += text[cursor];
            }
            ++cursor;
        }
        ++cursor;
        JsonValue node;
        node.kind_ = Kind::String;
        node.string_ = out;
        return node;
    }

    static JsonValue parse_array(const std::string& text, size_t& cursor) {
        JsonValue node;
        node.kind_ = Kind::Array;
        ++cursor;
        skip_space(text, cursor);
        if (cursor < text.size() && text[cursor] == ']') { ++cursor; return node; }
        while (cursor < text.size()) {
            node.array_.push_back(parse_value(text, cursor));
            skip_space(text, cursor);
            if (cursor < text.size() && text[cursor] == ',') { ++cursor; continue; }
            break;
        }
        if (cursor < text.size() && text[cursor] == ']') ++cursor;
        return node;
    }

    static JsonValue parse_object(const std::string& text, size_t& cursor) {
        JsonValue node;
        node.kind_ = Kind::Object;
        ++cursor;
        skip_space(text, cursor);
        if (cursor < text.size() && text[cursor] == '}') { ++cursor; return node; }
        while (cursor < text.size()) {
            skip_space(text, cursor);
            const JsonValue key = parse_string(text, cursor);
            skip_space(text, cursor);
            if (cursor < text.size() && text[cursor] == ':') ++cursor;
            node.object_[key.string_] = parse_value(text, cursor);
            skip_space(text, cursor);
            if (cursor < text.size() && text[cursor] == ',') { ++cursor; continue; }
            break;
        }
        if (cursor < text.size() && text[cursor] == '}') ++cursor;
        return node;
    }

    Kind kind_ = Kind::Null;
    bool boolean_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> array_;
    std::map<std::string, JsonValue> object_;
};

}  // namespace otf
