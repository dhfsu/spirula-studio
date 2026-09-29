#pragma once

// 基于 Json.h 值模型实现实用 YAML 子集，并提供 JSON/YAML 写入器。
// 以 { 或 [ 开头的文档交给 JSON 解析；其余支持块映射、序列、单行流式集合、普通或引号标量、|/> 块标量、注释及起始 ---。
// 锚点、别名、标签、多文档和跨行流式集合明确报错；需要跨行流式集合时使用 JSON。

#include "data/Json.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace yaml_detail {

struct Line {
    std::string text;    // 已去除注释及右侧空白
    int indent = 0;      // 首个记号前的空格数
    int number = 1;      // 从 1 开始的行号，用于错误消息
    bool blank = true;
};

[[noreturn]] inline void fail(int line, const std::string& why) {
    throw std::runtime_error("YAML line " + std::to_string(line) + ": " + why);
}

// 未被引号包裹且位于行首或空白之后的 # 起始注释。
inline std::string strip_comment(const std::string& s) {
    char quote = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (quote) {
            if (c == '\\' && quote == '"' && i + 1 < s.size()) i++;
            else if (c == quote) quote = 0;
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '#' && (i == 0 || s[i - 1] == ' ' || s[i - 1] == '\t')) {
            return s.substr(0, i);
        }
    }
    return s;
}

inline std::string rtrim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
        s.pop_back();
    return s;
}

inline std::vector<Line> split(const std::string& text) {
    std::vector<Line> out;
    size_t i = 0;
    int n = 1;
    while (i <= text.size()) {
        size_t e = text.find('\n', i);
        if (e == std::string::npos) e = text.size();
        std::string raw = text.substr(i, e - i);
        Line l;
        l.number = n++;
        // 缩进中的制表符不符合 YAML 规范，必须报错以免静默误读。
        for (char c : raw) {
            if (c == ' ') continue;
            if (c == '\t') fail(l.number, "tab in indentation");
            break;
        }
        const std::string body = rtrim(strip_comment(raw));
        size_t ind = 0;
        while (ind < body.size() && body[ind] == ' ') ind++;
        l.indent = (int)ind;
        l.text = body.substr(ind);
        l.blank = l.text.empty();
        // 块标量保留原始行及其缩进。
        if (!l.blank) l.text = rtrim(l.text);
        out.push_back(std::move(l));
        if (e == text.size()) break;
        i = e + 1;
    }
    return out;
}

// 按 YAML 规则解析 null/~ /空值、布尔值、数值或文本。
inline JsonValue plain_scalar(const std::string& s, int line) {
    JsonValue v;
    if (s.empty() || s == "~" || s == "null" || s == "Null" || s == "NULL")
        return v;                                  // 空值
    if (s == "true" || s == "True" || s == "TRUE" ||
        s == "false" || s == "False" || s == "FALSE") {
        v.type = JsonValue::Type::Bool;
        v.b = s[0] == 't' || s[0] == 'T';
        return v;
    }
    // 整个记号都是数字时才转为数值；1.2.3 和 3 apples 保持文本。
    const char* b = s.c_str();
    char* endp = nullptr;
    const double d = std::strtod(b, &endp);
    if (endp && *endp == '\0' && endp != b) {
        v.type = JsonValue::Type::Number;
        v.num = d;
        return v;
    }
    (void)line;
    v.type = JsonValue::Type::String;
    v.str = s;
    return v;
}

inline std::string unescape_double(const std::string& s, int line) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '\\') { out += s[i]; continue; }
        if (++i >= s.size()) fail(line, "trailing backslash");
        switch (s[i]) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case '0': out += '\0'; break;
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            default: fail(line, std::string("unsupported escape \\") + s[i]);
        }
    }
    return out;
}

// 从已去除注释的 s 解析一个标量或单行流式集合。
inline JsonValue parse_inline(const std::string& s, int line);

// 仅按集合最外层的逗号拆分。
inline std::vector<std::string> flow_items(const std::string& s, int line) {
    std::vector<std::string> out;
    int depth = 0;
    char quote = 0;
    std::string cur;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (quote) {
            cur += c;
            if (c == '\\' && quote == '"' && i + 1 < s.size()) cur += s[++i];
            else if (c == quote) quote = 0;
            continue;
        }
        if (c == '"' || c == '\'') { quote = c; cur += c; continue; }
        if (c == '[' || c == '{') depth++;
        if (c == ']' || c == '}') depth--;
        if (c == ',' && depth == 0) { out.push_back(cur); cur.clear(); continue; }
        cur += c;
    }
    if (quote) fail(line, "unterminated quote");
    if (depth) fail(line, "unbalanced flow collection");
    // [] 和 {} 无元素；[a,] 只有一个元素。
    std::string tail = cur;
    size_t b = tail.find_first_not_of(" \t");
    if (b != std::string::npos || !out.empty()) out.push_back(cur);
    return out;
}

inline std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return {};
    size_t e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

// 解析后接冒号的映射键，可带引号；不以 key: 开头时返回 npos。
inline size_t key_end(const std::string& s) {
    char quote = 0;
    int depth = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (quote) {
            if (c == '\\' && quote == '"' && i + 1 < s.size()) i++;
            else if (c == quote) quote = 0;
            continue;
        }
        if (c == '"' || c == '\'') { quote = c; continue; }
        if (c == '[' || c == '{') depth++;
        if (c == ']' || c == '}') depth--;
        // 冒号后为空格或行尾时才结束键；a:b 保持普通标量，以支持 URL 和时间。
        if (c == ':' && depth == 0 && (i + 1 == s.size() || s[i + 1] == ' '))
            return i;
    }
    return std::string::npos;
}

inline JsonValue parse_inline(const std::string& s, int line) {
    const std::string t = trim(s);
    if (t.empty()) return JsonValue{};
    if (t.front() == '[' || t.front() == '{') {
        const char close = t.front() == '[' ? ']' : '}';
        if (t.back() != close)
            fail(line, "flow collection must close on the same line");
        JsonValue v;
        const std::string body = t.substr(1, t.size() - 2);
        if (close == ']') {
            v.type = JsonValue::Type::Array;
            for (const std::string& item : flow_items(body, line))
                v.arr.push_back(parse_inline(item, line));
        } else {
            v.type = JsonValue::Type::Object;
            for (const std::string& item : flow_items(body, line)) {
                const std::string it = trim(item);
                const size_t k = key_end(it);
                if (k == std::string::npos) fail(line, "flow mapping needs 'key: value'");
                JsonValue key = parse_inline(it.substr(0, k), line);
                v.obj.emplace_back(key.type == JsonValue::Type::String ? key.str
                                                                       : trim(it.substr(0, k)),
                                   parse_inline(it.substr(k + 1), line));
            }
        }
        return v;
    }
    if (t.front() == '"') {
        if (t.size() < 2 || t.back() != '"') fail(line, "unterminated \"");
        JsonValue v;
        v.type = JsonValue::Type::String;
        v.str = unescape_double(t.substr(1, t.size() - 2), line);
        return v;
    }
    if (t.front() == '\'') {
        if (t.size() < 2 || t.back() != '\'') fail(line, "unterminated '");
        JsonValue v;
        v.type = JsonValue::Type::String;
        std::string body = t.substr(1, t.size() - 2);
        for (size_t i = 0; i < body.size(); i++) {
            v.str += body[i];
            if (body[i] == '\'' && i + 1 < body.size() && body[i + 1] == '\'') i++;
        }
        return v;
    }
    return plain_scalar(t, line);
}

struct Reader {
    std::vector<Line> lines;
    size_t i = 0;

    bool at_end() {
        while (i < lines.size() && lines[i].blank) i++;
        return i >= lines.size();
    }
    const Line& cur() { return lines[i]; }

    // | 与 > 读取更深缩进的后续行，分别以换行或空格拼接。
    JsonValue block_scalar(char kind, int parent_indent) {
        i++;
        std::string out;
        int base = -1;
        for (; i < lines.size(); i++) {
            const Line& l = lines[i];
            if (l.blank) { out += '\n'; continue; }
            if (l.indent <= parent_indent) break;
            if (base < 0) base = l.indent;
            std::string t = l.text;
            if (l.indent > base) t.insert(0, (size_t)(l.indent - base), ' ');
            out += t;
            out += kind == '|' ? '\n' : ' ';
        }
        while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
        if (kind == '|') out += '\n';
        JsonValue v;
        v.type = JsonValue::Type::String;
        v.str = out;
        return v;
    }

    // key: 或 - 后的值可位于同一行剩余部分，或下方缩进块中。
    JsonValue value_after(const std::string& rest, int parent_indent) {
        const std::string t = trim(rest);
        if (t == "|" || t == ">") return block_scalar(t[0], parent_indent);
        if (!t.empty()) { i++; return parse_inline(t, lines[i - 1].number); }
        i++;
        if (at_end()) return JsonValue{};
        if (cur().indent > parent_indent) return parse_block(cur().indent);
        // 序列可与所属键同级缩进，兼容常见 YAML 写入器及 yaml_write。
        if (cur().indent == parent_indent &&
            (cur().text == "-" || cur().text.compare(0, 2, "- ") == 0))
            return parse_block(parent_indent);
        return JsonValue{};
    }

    JsonValue parse_block(int indent) {
        if (at_end()) return JsonValue{};
        if (cur().indent < indent) return JsonValue{};
        const bool seq = cur().text == "-" || cur().text.compare(0, 2, "- ") == 0;
        JsonValue v;
        v.type = seq ? JsonValue::Type::Array : JsonValue::Type::Object;
        while (!at_end() && cur().indent == indent) {
            const Line& l = cur();
            if (seq) {
                // 与键同级的序列在外层映射的下一个键处结束。
                if (l.text != "-" && l.text.compare(0, 2, "- ") != 0) break;
                const std::string rest = l.text.size() > 1 ? l.text.substr(2) : "";
                // - key: value 开始映射，后续键应与 key 对齐，而非与短横线对齐。
                const size_t k = key_end(trim(rest));
                if (k != std::string::npos && !trim(rest).empty() &&
                    trim(rest)[0] != '[' && trim(rest)[0] != '{') {
                    const int inner = indent + 2 + (int)(rest.size() - trim(rest).size());
                    // 重写该行，仅将键值对交给映射解析器。
                    lines[i].text = trim(rest);
                    lines[i].indent = inner;
                    v.arr.push_back(parse_block(inner));
                    continue;
                }
                v.arr.push_back(value_after(rest, indent));
                continue;
            }
            // 外层序列继续时，内层映射结束。
            if (l.text == "-" || l.text.compare(0, 2, "- ") == 0) break;
            const size_t k = key_end(l.text);
            if (k == std::string::npos) fail(l.number, "expected 'key: value'");
            JsonValue key = parse_inline(l.text.substr(0, k), l.number);
            const std::string name = key.type == JsonValue::Type::String
                                         ? key.str
                                         : trim(l.text.substr(0, k));
            const std::string rest = l.text.substr(k + 1);
            v.obj.emplace_back(name, value_after(rest, indent));
        }
        return v;
    }
};

}  // 命名空间 yaml_detail

// 文档以流式集合开头时按 JSON 解析，否则按 YAML 解析。
inline JsonValue yaml_parse(const std::string& text) {
    size_t p = text.find_first_not_of(" \t\r\n");
    if (p != std::string::npos && (text[p] == '{' || text[p] == '['))
        return json_parse(text);
    yaml_detail::Reader r{yaml_detail::split(text), 0};
    // 起始 --- 是文档标记，不属于内容。
    if (!r.at_end() && r.cur().text == "---") r.i++;
    if (r.at_end()) return JsonValue{};
    const int indent = r.cur().indent;
    JsonValue v = r.parse_block(indent);
    if (!r.at_end())
        yaml_detail::fail(r.cur().number,
                          r.cur().text == "---" ? "multiple documents are not supported"
                                                : "unexpected indentation");
    return v;
}

inline JsonValue yaml_parse_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string text(n > 0 ? (size_t)n : 0, '\0');
    size_t got = n > 0 ? std::fread(&text[0], 1, (size_t)n, f) : 0;
    std::fclose(f);
    text.resize(got);
    return yaml_parse(text);
}

// ---------------- 写入器 ----------------

namespace yaml_detail {

// 采用能往返恢复同一 double 的最短表示，使 0.1 保持为 0.1，而非 0.10000000000000001。
inline std::string number_text(double d) {
    if (d == (double)(long long)d && d > -1e15 && d < 1e15)
        return std::to_string((long long)d);
    char buf[40];
    for (int prec = 15; prec <= 17; prec++) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, d);
        if (std::strtod(buf, nullptr) == d) break;
    }
    return buf;
}

inline std::string quote_json(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", c);
                    out += b;
                } else {
                    out += c;
                }
        }
    }
    return out + "\"";
}

// 可按原文往返解析时使用普通标量，否则加引号。
inline std::string scalar_yaml(const std::string& s) {
    if (s.empty()) return "\"\"";
    static const char* kReserved = "#&*!|>'\"%@`,[]{}:";
    if (s.find_first_of(" \t\n") != std::string::npos ||
        std::strchr(kReserved, s[0]) || s.find(": ") != std::string::npos ||
        s.back() == ':' || s.find(" #") != std::string::npos)
        return quote_json(s);
    JsonValue probe = plain_scalar(s, 0);
    return probe.type == JsonValue::Type::String ? s : quote_json(s);
}

inline void write_json(const JsonValue& v, std::string& out, int indent) {
    const std::string pad((size_t)indent * 2 + 2, ' ');
    const std::string pad0((size_t)indent * 2, ' ');
    switch (v.type) {
        case JsonValue::Type::Null: out += "null"; break;
        case JsonValue::Type::Bool: out += v.b ? "true" : "false"; break;
        case JsonValue::Type::Number: out += number_text(v.num); break;
        case JsonValue::Type::String: out += quote_json(v.str); break;
        case JsonValue::Type::Array:
            if (v.arr.empty()) { out += "[]"; break; }
            out += "[\n";
            for (size_t i = 0; i < v.arr.size(); i++) {
                out += pad;
                write_json(v.arr[i], out, indent + 1);
                out += i + 1 < v.arr.size() ? ",\n" : "\n";
            }
            out += pad0 + "]";
            break;
        case JsonValue::Type::Object:
            if (v.obj.empty()) { out += "{}"; break; }
            out += "{\n";
            for (size_t i = 0; i < v.obj.size(); i++) {
                out += pad + quote_json(v.obj[i].first) + ": ";
                write_json(v.obj[i].second, out, indent + 1);
                out += i + 1 < v.obj.size() ? ",\n" : "\n";
            }
            out += pad0 + "}";
            break;
    }
}

// pad 用于各行前缀，首行改用 first，使映射首键可以接在序列短横线之后。
inline void write_yaml(const JsonValue& v, std::string& out,
                       const std::string& pad, const std::string& first) {
    switch (v.type) {
        case JsonValue::Type::Null:   out += first + "null\n"; return;
        case JsonValue::Type::Bool:   out += first + (v.b ? "true\n" : "false\n"); return;
        case JsonValue::Type::Number: out += first + number_text(v.num) + "\n"; return;
        case JsonValue::Type::String: out += first + scalar_yaml(v.str) + "\n"; return;
        case JsonValue::Type::Array:
            if (v.arr.empty()) { out += first + "[]\n"; return; }
            for (size_t i = 0; i < v.arr.size(); i++)
                write_yaml(v.arr[i], out, pad + "  ", (i ? pad : first) + "- ");
            return;
        case JsonValue::Type::Object:
            if (v.obj.empty()) { out += first + "{}\n"; return; }
            for (size_t i = 0; i < v.obj.size(); i++) {
                const auto& kv = v.obj[i];
                const std::string lead = (i ? pad : first) + scalar_yaml(kv.first);
                const bool block = (kv.second.type == JsonValue::Type::Object &&
                                    !kv.second.obj.empty()) ||
                                   (kv.second.type == JsonValue::Type::Array &&
                                    !kv.second.arr.empty());
                if (block) {
                    out += lead + ":\n";
                    // 序列与键同级缩进，映射则增加缩进。
                    const std::string inner =
                        kv.second.type == JsonValue::Type::Array ? pad : pad + "  ";
                    write_yaml(kv.second, out, inner, inner);
                } else {
                    write_yaml(kv.second, out, pad, lead + ": ");
                }
            }
            return;
    }
}

}  // 命名空间 yaml_detail

inline std::string json_write(const JsonValue& v) {
    std::string out;
    yaml_detail::write_json(v, out, 0);
    out += "\n";
    return out;
}

inline std::string yaml_write(const JsonValue& v) {
    std::string out;
    yaml_detail::write_yaml(v, out, "", "");
    return out;
}
