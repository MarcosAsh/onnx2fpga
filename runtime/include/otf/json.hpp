// Minimal JSON reader for the manifest, so the runtime has no third party dependency.
#pragma once

#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace otf {

// true, false and null all read as Null; the manifest has no use for them.
class JsonValue {
public:
    enum class Kind { Null, Number, String, Array, Object };

    double number() const { expect(Kind::Number); return number_; }
    int as_int() const { return static_cast<int>(number()); }
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

    static JsonValue parse(const std::string& text) {
        size_t cursor = 0;
        return parse_value(text, cursor);
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

    // Consumes one character if it is the wanted one.
    static bool eat(const std::string& text, size_t& cursor, char wanted) {
        skip_space(text, cursor);
        if (cursor < text.size() && text[cursor] == wanted) { ++cursor; return true; }
        return false;
    }

    static JsonValue parse_value(const std::string& text, size_t& cursor) {
        skip_space(text, cursor);
        if (cursor >= text.size()) throw std::runtime_error("unexpected end of json");
        switch (text[cursor]) {
            case '{': return parse_object(text, cursor);
            case '[': return parse_array(text, cursor);
            case '"': return parse_string(text, cursor);
            case 't': cursor += 4; return JsonValue();
            case 'f': cursor += 5; return JsonValue();
            case 'n': cursor += 4; return JsonValue();
            default:  return parse_number(text, cursor);
        }
    }

    static JsonValue parse_number(const std::string& text, size_t& cursor) {
        size_t end = cursor;
        while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) ||
                                     std::string("-+.eE").find(text[end]) != std::string::npos)) {
            ++end;
        }
        JsonValue node;
        node.kind_ = Kind::Number;
        node.number_ = std::stod(text.substr(cursor, end - cursor));
        cursor = end;
        return node;
    }

    static JsonValue parse_string(const std::string& text, size_t& cursor) {
        JsonValue node;
        node.kind_ = Kind::String;
        for (++cursor; cursor < text.size() && text[cursor] != '"'; ++cursor) {
            char c = text[cursor];
            if (c == '\\' && cursor + 1 < text.size()) {
                c = text[++cursor];
                if (c == 'n') c = '\n';
                else if (c == 't') c = '\t';
            }
            node.string_ += c;
        }
        ++cursor;
        return node;
    }

    static JsonValue parse_array(const std::string& text, size_t& cursor) {
        JsonValue node;
        node.kind_ = Kind::Array;
        ++cursor;
        if (eat(text, cursor, ']')) return node;
        do {
            node.array_.push_back(parse_value(text, cursor));
        } while (eat(text, cursor, ','));
        eat(text, cursor, ']');
        return node;
    }

    static JsonValue parse_object(const std::string& text, size_t& cursor) {
        JsonValue node;
        node.kind_ = Kind::Object;
        ++cursor;
        if (eat(text, cursor, '}')) return node;
        do {
            skip_space(text, cursor);
            const JsonValue key = parse_string(text, cursor);
            eat(text, cursor, ':');
            node.object_[key.string_] = parse_value(text, cursor);
        } while (eat(text, cursor, ','));
        eat(text, cursor, '}');
        return node;
    }

    Kind kind_ = Kind::Null;
    double number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> array_;
    std::map<std::string, JsonValue> object_;
};

}  // namespace otf
