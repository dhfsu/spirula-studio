// 纯 C++17 的 SPIR-V 扫描、修饰与嵌入工具，构建无需 Python。
// discover 输出模块名、源、入口、宏与包含依赖，供 CMake 逐模块调用 slangc；embed 生成注册翻译单元，--sfm/--nn 采用对应库格式。
// nocontract 为浮点结果添加 NoContraction，避免驱动合并表达式破坏 DF 无误差变换；普通嵌入还筛选能力变体并审计能力泄漏。

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------- 功能变体：后缀顺序必须与运行时约定一致 ----------------
struct Feature { const char* suffix; const char* define; };
const Feature ATOMIC{".atomicadd", "-DSS_NATIVE_F32_ATOMIC_ADD"};
const Feature INT8{".int8", "-DSS_NATIVE_INT8"};
const Feature NOINT64{".noint64", "-DSS_EMULATE_INT64"};

// SPIR-V 能力操作数，OpCapability 操作码为 17。
constexpr uint32_t CAP_INT64 = 11;
constexpr uint32_t CAP_INT8 = 39;
constexpr uint32_t CAP_8BIT_STORAGE = 4437;  // StorageBuffer8BitAccess 能力

bool is_ident(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

std::string read_file(const std::string& path, bool* ok = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (ok) *ok = false; return {}; }
    if (ok) *ok = true;
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string basename_of(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}
std::string dirname_of(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}
bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return static_cast<bool>(f);
}

// ---------------- 着色器源扫描 ----------------

// 在每个 compute 着色器标记后查找首个 void 函数定义，允许中间包含属性和注释。
bool parse_void_name(const std::string& s, size_t from, std::string* name) {
    const std::string kw = "void";
    while (true) {
        size_t v = s.find(kw, from);
        if (v == std::string::npos) return false;
        // 必须匹配完整单词
        bool lok = (v == 0) || !is_ident(s[v - 1]);
        size_t p = v + kw.size();
        if (!lok || p >= s.size() || is_ident(s[p])) { from = v + 1; continue; }
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' ||
                                s[p] == '\r')) p++;
        size_t ns = p;
        while (p < s.size() && is_ident(s[p])) p++;
        if (p == ns) { from = v + 1; continue; }  // 没有标识符
        std::string id = s.substr(ns, p - ns);
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' ||
                                s[p] == '\r')) p++;
        if (p < s.size() && s[p] == '(') { *name = id; return true; }
        from = v + 1;
    }
}

// 发现 compute 标记函数及行首 DEF_<macro>(<entry>,...) 实例，忽略宏定义中的占位参数 NAME。
std::vector<std::string> find_entries(const std::string& src) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    auto add = [&](const std::string& n) {
        if (n != "NAME" && seen.insert(n).second) out.push_back(n);
    };
    // compute 标记后的 void NAME 形式
    const std::string marker = "[shader(\"compute\")]";
    size_t pos = 0;
    std::vector<std::string> shader_names;
    while ((pos = src.find(marker, pos)) != std::string::npos) {
        pos += marker.size();
        std::string n;
        if (parse_void_name(src, pos, &n)) shader_names.push_back(n);
    }
    // 行首 DEF_...(entry,...) 形式
    std::vector<std::string> macro_names;
    size_t line_start = 0;
    while (line_start <= src.size()) {
        size_t nl = src.find('\n', line_start);
        size_t end = (nl == std::string::npos) ? src.size() : nl;
        // 行以 DEF_ 开头
        if (end - line_start > 4 && src.compare(line_start, 4, "DEF_") == 0) {
            size_t p = line_start + 4;
            while (p < end && is_ident(src[p])) p++;  // 宏名称的其余部分
            if (p < end && src[p] == '(') {
                p++;
                while (p < end && (src[p] == ' ' || src[p] == '\t')) p++;
                size_t ns = p;
                while (p < end && is_ident(src[p])) p++;
                if (p > ns) macro_names.push_back(src.substr(ns, p - ns));
            }
        }
        if (nl == std::string::npos) break;
        line_start = nl + 1;
    }
    for (auto& n : shader_names) add(n);
    for (auto& n : macro_names) add(n);
    return out;
}

// 提取双引号 include 目标，允许前导空白。
std::vector<std::string> parse_includes(const std::string& src) {
    std::vector<std::string> out;
    size_t line_start = 0;
    while (line_start <= src.size()) {
        size_t nl = src.find('\n', line_start);
        size_t end = (nl == std::string::npos) ? src.size() : nl;
        size_t p = line_start;
        while (p < end && (src[p] == ' ' || src[p] == '\t')) p++;
        const std::string inc = "#include";
        if (end - p >= inc.size() && src.compare(p, inc.size(), inc) == 0) {
            p += inc.size();
            while (p < end && (src[p] == ' ' || src[p] == '\t')) p++;
            if (p < end && src[p] == '"') {
                size_t q = src.find('"', p + 1);
                if (q != std::string::npos && q <= end)
                    out.push_back(src.substr(p + 1, q - p - 1));
            }
        }
        if (nl == std::string::npos) break;
        line_start = nl + 1;
    }
    return out;
}

std::string resolve_include(const std::string& inc, const std::string& from_dir,
                            const std::vector<std::string>& incdirs) {
    std::vector<std::string> bases;
    bases.push_back(from_dir);
    for (auto& d : incdirs) bases.push_back(d);
    for (auto& b : bases) {
        std::string cand = b + "/" + inc;
        if (file_exists(cand)) return cand;
    }
    return {};
}

// 按当前文件目录与包含目录解析传递包含闭包，源文件本身在首位。
void include_closure(const std::string& path,
                     const std::vector<std::string>& incdirs,
                     std::vector<std::string>& seen) {
    if (std::find(seen.begin(), seen.end(), path) != seen.end()) return;
    bool ok = false;
    std::string src = read_file(path, &ok);
    if (!ok) return;
    seen.push_back(path);
    for (auto& inc : parse_includes(src)) {
        std::string r = resolve_include(inc, dirname_of(path), incdirs);
        if (!r.empty()) include_closure(r, incdirs, seen);
    }
}

// 检查完整 token 后是否紧接左括号，允许空白。
bool calls_token(const std::string& text, const std::string& token) {
    size_t pos = 0;
    while ((pos = text.find(token, pos)) != std::string::npos) {
        bool lok = (pos == 0) || !is_ident(text[pos - 1]);
        size_t p = pos + token.size();
        while (p < text.size() && (text[p] == ' ' || text[p] == '\t' ||
                                   text[p] == '\n' || text[p] == '\r')) p++;
        if (lok && p < text.size() && text[p] == '(') return true;
        pos += token.size();
    }
    return false;
}

// 检查源文件包含闭包是否调用 header 提供的任一 token，头文件自身定义不计。
bool closure_calls(const std::vector<std::string>& closure,
                   const std::string& header_base,
                   const std::vector<std::string>& tokens) {
    bool has_header = false;
    for (auto& f : closure)
        if (basename_of(f) == header_base) has_header = true;
    if (!has_header) return false;
    for (auto& f : closure) {
        if (basename_of(f) == header_base) continue;
        std::string t = read_file(f);
        for (auto& tok : tokens)
            if (calls_token(t, tok)) return true;
    }
    return false;
}

bool closure_has(const std::vector<std::string>& closure,
                 const std::string& header_base) {
    for (auto& f : closure)
        if (basename_of(f) == header_base) return true;
    return false;
}

// 按规范顺序枚举全部特性子集，空集优先。
std::vector<std::pair<std::string, std::vector<std::string>>>
variant_subsets(const std::vector<Feature>& feats) {
    std::vector<std::pair<std::string, std::vector<std::string>>> out;
    size_t n = feats.size();
    // 先按子集大小，再按规范索引顺序。
    for (size_t r = 0; r <= n; r++) {
        // 保持索引顺序枚举大小为 r 的组合
        std::vector<size_t> idx(r);
        for (size_t i = 0; i < r; i++) idx[i] = i;
        if (r == 0) { out.push_back({"", {}}); continue; }
        while (true) {
            std::string suffix;
            std::vector<std::string> defs;
            for (size_t i = 0; i < r; i++) {
                suffix += feats[idx[i]].suffix;
                defs.push_back(feats[idx[i]].define);
            }
            out.push_back({suffix, defs});
            // 下一个组合
            long i = (long)r - 1;
            while (i >= 0 && idx[i] == n - r + (size_t)i) i--;
            if (i < 0) break;
            idx[i]++;
            for (size_t j = i + 1; j < r; j++) idx[j] = idx[j - 1] + 1;
        }
    }
    return out;
}

// ---------------- SPIR-V 能力检查 ----------------
std::set<uint32_t> spirv_capabilities(const std::string& bytes) {
    std::set<uint32_t> caps;
    if (bytes.size() < 24) return caps;
    auto w = [&](size_t off) {
        return (uint32_t)(uint8_t)bytes[off] |
               ((uint32_t)(uint8_t)bytes[off + 1] << 8) |
               ((uint32_t)(uint8_t)bytes[off + 2] << 16) |
               ((uint32_t)(uint8_t)bytes[off + 3] << 24);
    };
    if (w(0) != 0x07230203u) return caps;
    size_t i = 20;
    while (i + 4 <= bytes.size()) {
        uint32_t word = w(i);
        uint32_t opcode = word & 0xffff, count = word >> 16;
        if (count == 0) break;
        if (opcode == 17 && i + 8 <= bytes.size()) {
            caps.insert(w(i + 4));  // OpCapability 指令
        } else if (opcode != 17) {
            break;  // 能力声明位于模块开头
        }
        i += (size_t)count * 4;
    }
    return caps;
}

// ---------------- NoContraction 模式 ----------------
// 以下为此模式使用的 SPIR-V 操作码与枚举。
enum : uint32_t {
    OpExtInstImport = 11,
    OpExtInst = 12,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpDecorationGroup = 73,
    OpGroupDecorate = 74,
    OpGroupMemberDecorate = 75,
    OpFNegate = 127,
    OpFAdd = 129,
    OpFSub = 131,
    OpFMul = 133,
    OpFDiv = 136,
    OpDecorateId = 332,
    OpDecorateString = 5632,
    OpMemberDecorateString = 5633,
    OpTypeVoid = 19,
    OpTypeForwardPointer = 39,
    kDecorationNoContraction = 42,
    kGlslStd450Fma = 50,
};

bool is_annotation(uint32_t op) {
    return op == OpDecorate || op == OpMemberDecorate || op == OpDecorationGroup ||
           op == OpGroupDecorate || op == OpGroupMemberDecorate || op == OpDecorateId ||
           op == OpDecorateString || op == OpMemberDecorateString;
}

// 注解必须位于类型、常量和全局变量之前；无现有注解时以此确定插入位置。
bool is_type_decl(uint32_t op) { return op >= OpTypeVoid && op <= OpTypeForwardPointer; }

int run_nocontract(const std::string& in, const std::string& out) {
    bool ok = false;
    std::string bytes = read_file(in, &ok);
    if (!ok || bytes.size() % 4) {
        std::fprintf(stderr, "nocontract: cannot read %s as a word-aligned module\n",
                     in.c_str());
        return 1;
    }
    std::vector<uint32_t> w(bytes.size() / 4);
    std::memcpy(w.data(), bytes.data(), bytes.size());
    if (w.size() < 5 || w[0] != 0x07230203u) {
        // slangc 输出主机字节序，交换字节序的模块视为损坏输入。
        std::fprintf(stderr, "nocontract: %s is not a host-order SPIR-V module\n",
                     in.c_str());
        return 1;
    }

    // 记录 GLSL.std.450 导入 ID，仅识别该指令集的 Fma，兼容同时导入多个扩展集。
    uint32_t glsl_set = 0;
    std::vector<uint32_t> targets;
    size_t annot_end = 0;    // 最后一条注解后的指令位置
    size_t types_begin = 0;  // 首个类型声明位置
    bool have_types_begin = false;

    for (size_t i = 5; i < w.size();) {
        uint32_t len = w[i] >> 16, op = w[i] & 0xffffu;
        if (len == 0 || i + len > w.size()) {
            std::fprintf(stderr, "nocontract: malformed instruction in %s\n", in.c_str());
            return 1;
        }
        if (op == OpExtInstImport && len >= 3) {
            if (std::strcmp((const char*)&w[i + 2], "GLSL.std.450") == 0) glsl_set = w[i + 1];
        } else if (op == OpFAdd || op == OpFSub || op == OpFMul || op == OpFDiv ||
                   op == OpFNegate) {
            targets.push_back(w[i + 2]);  // 操作数 1 为结果类型，2 为结果 ID
        } else if (op == OpExtInst && len >= 5 && w[i + 3] == glsl_set &&
                   w[i + 4] == kGlslStd450Fma) {
            targets.push_back(w[i + 2]);
        }
        if (is_annotation(op)) annot_end = i + len;
        if (!have_types_begin && is_type_decl(op)) {
            types_begin = i;
            have_types_begin = true;
        }
        i += len;
    }

    size_t pos = annot_end ? annot_end : types_begin;
    if (!pos) {
        std::fprintf(stderr, "nocontract: no annotation or type section in %s\n", in.c_str());
        return 1;
    }

    std::vector<uint32_t> outw;
    outw.reserve(w.size() + targets.size() * 3);
    outw.insert(outw.end(), w.begin(), w.begin() + (long)pos);
    for (uint32_t t : targets) {
        outw.push_back((3u << 16) | OpDecorate);
        outw.push_back(t);
        outw.push_back(kDecorationNoContraction);
    }
    outw.insert(outw.end(), w.begin() + (long)pos, w.end());

    std::ofstream f(out, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "nocontract: cannot write %s\n", out.c_str());
        return 1;
    }
    f.write((const char*)outw.data(), (std::streamsize)outw.size() * 4);
    return 0;
}

// ---------------- 入口发现模式 ----------------
int run_discover(const std::vector<std::string>& args) {
    std::vector<std::string> incdirs, sources;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "-I" && i + 1 < args.size())
            incdirs.push_back(args[++i]);            // -I 与目录分开传入
        else if (args[i].rfind("-I", 0) == 0)
            incdirs.push_back(args[i].substr(2));    // -I 与目录连写
        else
            sources.push_back(args[i]);
    }

    // 无入口文件若被其他源包含则正常，仅完全无人包含时警告。
    std::set<std::string> included;
    std::map<std::string, std::vector<std::string>> closures;
    for (auto& s : sources) {
        std::vector<std::string> cl;
        include_closure(s, incdirs, cl);
        closures[s] = cl;
        for (auto& f : cl) if (f != s) included.insert(f);
    }

    std::ostringstream out;
    for (auto& source : sources) {
        std::string stem = basename_of(source);
        if (stem.size() > 6 && stem.compare(stem.size() - 6, 6, ".slang") == 0)
            stem = stem.substr(0, stem.size() - 6);
        const auto& closure = closures[source];
        std::vector<std::string> entries = find_entries(read_file(source));
        if (entries.empty() && !included.count(source))
            std::fprintf(stderr, "warning: no compute entry points in %s\n",
                         source.c_str());

        std::vector<Feature> feats;
        if (closure_calls(closure, "atomic_float.slang", {"atomic_add_f32"}))
            feats.push_back(ATOMIC);
        if (closure_calls(closure, "int8_compat.slang",
                          {"u8_load", "s8_load", "u8_store"}))
            feats.push_back(INT8);
        // 保守生成可能经兼容层处理 int64 的 noint64 候选；嵌入时若基础模块无 Int64 能力则删除冗余变体。
        bool noint64_cand = closure_has(closure, "int64_compat.slang");

        std::string deps;
        for (size_t i = 0; i < closure.size(); i++)
            deps += (i ? " " : "") + closure[i];

        auto emit = [&](const std::string& entry, const std::string& suffix,
                        const std::vector<std::string>& defs) {
            std::string name = stem + "." + entry + suffix;
            std::string defstr;  // 用短横线占位，保持字段非空
            for (size_t i = 0; i < defs.size(); i++)
                defstr += (i ? " " : "") + defs[i];
            if (defstr.empty()) defstr = "-";
            out << name << '\t' << source << '\t' << entry << '\t' << defstr
                << '\t' << deps << '\n';
        };

        auto base_subsets = variant_subsets(feats);
        for (auto& entry : entries) {
            for (auto& [suffix, defs] : base_subsets) emit(entry, suffix, defs);
            if (noint64_cand) {
                for (auto& [suffix, defs] : base_subsets) {
                    std::vector<std::string> d = defs;
                    d.push_back(NOINT64.define);
                    emit(entry, suffix + NOINT64.suffix, d);
                }
            }
        }
    }
    std::fputs(out.str().c_str(), stdout);
    return 0;
}

// ---------------- 模块嵌入模式 ----------------
std::string strip_feature_suffixes(std::string name) {
    const char* sfx[] = {NOINT64.suffix, INT8.suffix, ATOMIC.suffix};
    bool changed = true;
    while (changed) {
        changed = false;
        for (const char* s : sfx) {
            size_t n = std::strlen(s);
            if (name.size() > n && name.compare(name.size() - n, n, s) == 0) {
                name.resize(name.size() - n);
                changed = true;
            }
        }
    }
    return name;
}

int run_embed(const std::vector<std::string>& args) {
    std::string out_cpp, listfile, nn_tag;
    bool sfm = false, nn = false;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--list" && i + 1 < args.size()) listfile = args[++i];
        else if (args[i] == "--sfm") sfm = true;
        else if (args[i] == "--nn" && i + 1 < args.size()) {
            nn = true;
            nn_tag = args[++i];
        } else if (out_cpp.empty()) out_cpp = args[i];
    }
    if (out_cpp.empty() || listfile.empty()) {
        std::fprintf(stderr, "embed: usage: embed <out.cpp> --list <file>\n");
        return 2;
    }
    // SfM 与 NN 注册格式不参与训练引擎的变体门限和能力审计。
    const bool plain = sfm || nn;

    std::vector<std::string> paths;
    {
        std::ifstream lf(listfile);
        std::string line;
        while (std::getline(lf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) paths.push_back(line);
        }
    }

    // 名称到字节数据及能力集合的映射
    std::map<std::string, std::string> blob_bytes;
    std::map<std::string, std::set<uint32_t>> blob_caps;
    for (auto& p : paths) {
        std::string name = basename_of(p);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".spv") == 0)
            name.resize(name.size() - 4);
        bool ok = false;
        std::string data = read_file(p, &ok);
        if (!ok) {
            std::fprintf(stderr, "embed: missing blob %s\n", p.c_str());
            return 1;
        }
        blob_bytes[name] = data;
        blob_caps[name] = spirv_capabilities(data);
    }

    // 保留基础和功能模块，仅基础声明 Int64 时保留 noint64 变体；SfM 无此类功能变体。
    std::vector<std::string> kept;
    for (auto& [name, bytes] : blob_bytes) {
        (void)bytes;
        if (plain) { kept.push_back(name); continue; }
        size_t n = std::strlen(NOINT64.suffix);
        bool is_noint64 = name.size() > n &&
            name.compare(name.size() - n, n, NOINT64.suffix) == 0;
        if (is_noint64) {
            std::string base = strip_feature_suffixes(name);
            auto it = blob_caps.find(base);
            if (it == blob_caps.end() || !it->second.count(CAP_INT64))
                continue;  // 基础模块没有 Int64，无需独立回退变体
        }
        kept.push_back(name);
    }
    // 按完整 .spv 文件名排序，使输出顺序不受变体后缀集合影响。
    std::sort(kept.begin(), kept.end(),
              [](const std::string& a, const std::string& b) {
                  return a + ".spv" < b + ".spv";
              });

    // 检查引擎变体的能力不变量，SfM 模块不参与。
    bool failed = false;
    for (auto& name : plain ? std::vector<std::string>{} : kept) {
        const auto& caps = blob_caps[name];
        size_t n = std::strlen(NOINT64.suffix);
        bool is_noint64 = name.size() > n &&
            name.compare(name.size() - n, n, NOINT64.suffix) == 0;
        if (is_noint64 && caps.count(CAP_INT64)) {
            std::fprintf(stderr, "  CAPABILITY LEAK: %s still declares Int64 "
                "(unguarded 64-bit integer use; route it through "
                "vulkan/int64_compat.slang)\n", name.c_str());
            failed = true;
        }
        if (name.find(INT8.suffix) == std::string::npos &&
            (caps.count(CAP_INT8) || caps.count(CAP_8BIT_STORAGE))) {
            std::fprintf(stderr, "  CAPABILITY LEAK: %s declares 8-bit "
                "capabilities outside the .int8 variant (route byte access "
                "through vulkan/int8_compat.slang)\n", name.c_str());
            failed = true;
        }
    }
    if (failed) return 1;

    // 生成嵌入翻译单元。
    std::ostringstream o;
    o << "// GENERATED by shaders/spirv_tool.cpp -- DO NOT EDIT.\n"
         "#include <cstddef>\n#include <cstdint>\n";
    if (sfm)
        o << "#include <cstring>\n\nnamespace sfm {\n\n";
    else if (nn)
        o << "#include \"nn/vk/EmbeddedSpirv.h\"\n\n"
             "namespace nn {\nnamespace vk {\n\n";
    else
        o << "\nnamespace backend {\nnamespace vk {\n\n"
             "struct SpirvBlob { const char* name; const uint32_t* data;"
             " size_t size_bytes; };\n\n";
    for (size_t i = 0; i < kept.size(); i++) {
        const std::string& data = blob_bytes[kept[i]];
        o << "static const uint32_t _blob" << i << "[] = {\n";
        size_t words = data.size() / 4;
        char buf[16];
        for (size_t wi = 0; wi < words; wi += 8) {
            o << "   ";
            for (size_t j = wi; j < wi + 8 && j < words; j++) {
                uint32_t v = (uint32_t)(uint8_t)data[j * 4] |
                             ((uint32_t)(uint8_t)data[j * 4 + 1] << 8) |
                             ((uint32_t)(uint8_t)data[j * 4 + 2] << 16) |
                             ((uint32_t)(uint8_t)data[j * 4 + 3] << 24);
                std::snprintf(buf, sizeof(buf), " 0x%08xu,", v);
                o << buf;
            }
            o << "\n";
        }
        o << "};\n";
    }
    if (sfm) {
        o << "\nstruct SpirvBlob { const char* name; const uint32_t* code;"
             " size_t words; };\n"
             "static const SpirvBlob kBlobs[] = {\n";
        for (size_t i = 0; i < kept.size(); i++)
            o << "    {\"" << kept[i] << "\", _blob" << i << ", sizeof(_blob" << i
              << ") / 4},\n";
        if (kept.empty())
            o << "    {nullptr, nullptr, 0},  // keep the array non-empty\n";
        o << "};\n\n"
             "const uint32_t* findSpirv(const char* name, size_t* words) {\n"
             "    for (const SpirvBlob& b : kBlobs)\n"
             "        if (b.name && std::strcmp(b.name, name) == 0) {\n"
             "            if (words) *words = b.words;\n"
             "            return b.code;\n"
             "        }\n"
             "    return nullptr;\n"
             "}\n\n"
             "const char* spirvBlobName(size_t i) {\n"
             "    return i < " << kept.size() << " ? kBlobs[i].name : nullptr;\n"
             "}\n\n"
             "}  // namespace sfm\n";
    } else if (nn) {
        // 每库独立表并显式注册，名称使用外部链接，tag 区分同二进制中的多个生成翻译单元。
        o << "\nextern const EmbeddedModule kEmbeddedModules_" << nn_tag << "[];\n"
             "const EmbeddedModule kEmbeddedModules_" << nn_tag << "[] = {\n";
        for (size_t i = 0; i < kept.size(); i++)
            o << "    {\"" << kept[i] << "\", _blob" << i << ", sizeof(_blob" << i
              << ") / 4},\n";
        if (kept.empty())
            o << "    {nullptr, nullptr, 0},  // keep the array non-empty\n";
        o << "};\n"
             "extern const size_t kEmbeddedModuleCount_" << nn_tag << ";\n"
             "const size_t kEmbeddedModuleCount_" << nn_tag << " = "
          << kept.size() << ";\n\n"
             "}  // namespace vk\n}  // namespace nn\n";
    } else {
        o << "\nextern const SpirvBlob g_spirv_blobs[];\n"
             "extern const size_t g_spirv_blob_count;\n"
             "const SpirvBlob g_spirv_blobs[] = {\n";
        for (size_t i = 0; i < kept.size(); i++)
            o << "    {\"" << kept[i] << "\", _blob" << i << ", "
              << blob_bytes[kept[i]].size() << "},\n";
        if (kept.empty())
            o << "    {nullptr, nullptr, 0},  // keep the array non-empty\n";
        o << "};\nconst size_t g_spirv_blob_count = " << kept.size() << ";\n\n"
             "}  // namespace vk\n}  // namespace backend\n";
    }

    // 嵌入步骤仅在 SPIR-V 改变时运行，始终重写更新时间，使 Ninja 的输出时间检查通过。
    std::string text = o.str();
    std::ofstream f(out_cpp, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "embed: cannot write %s\n", out_cpp.c_str());
        return 1;
    }
    f << text;
    return 0;
}

}  // 匿名命名空间

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s discover|embed|nocontract ...\n", argv[0]);
        return 2;
    }
    std::vector<std::string> args(argv + 2, argv + argc);
    std::string mode = argv[1];
    if (mode == "discover") return run_discover(args);
    if (mode == "embed") return run_embed(args);
    if (mode == "nocontract") {
        if (args.size() < 2) {
            std::fprintf(stderr, "nocontract: usage: nocontract <in.spv> <out.spv>\n");
            return 2;
        }
        return run_nocontract(args[0], args[1]);
    }
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
