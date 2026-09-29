// 区域与语言设置实现，参见 Locale.h。

#include "i18n/Locale.h"

#include "core/Env.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__APPLE__)
#  include <CoreFoundation/CoreFoundation.h>
#endif

namespace spirula {
namespace i18n {

namespace detail {
std::atomic<Lang> g_current{kDefaultLang};
}

void set_current(Lang l) {
    detail::g_current.store(l, std::memory_order_relaxed);
}

const char* code(Lang l) {
    switch (l) {
#define X(id, c, native, english) case Lang::id: return c;
        SS_LANGUAGES(X)
#undef X
    }
    return "en";
}

const char* native_name(Lang l) {
    switch (l) {
#define X(id, c, native, english) case Lang::id: return native;
        SS_LANGUAGES(X)
#undef X
    }
    return "English";
}

const char* english_name(Lang l) {
    switch (l) {
#define X(id, c, native, english) case Lang::id: return english;
        SS_LANGUAGES(X)
#undef X
    }
    return "English";
}

bool needs_cjk_font(Lang l) {
    switch (l) {
#define X(id) case Lang::id: return true;
        SS_LANGUAGES_CJK(X)
#undef X
        default: return false;
    }
}

namespace {

char lower(char c) {
    return (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
}

// zh_Hans_CN.UTF-8@modifier -> {zh, hans, cn}；统一小写，只需前三个子标签。
struct Subtags {
    std::string t[3];
    int n = 0;
};

Subtags split_locale(const char* s) {
    Subtags out;
    std::string cur;
    for (const char* p = s; *p; p++) {
        // 编码与修饰符不影响语言判定。
        if (*p == '.' || *p == '@') break;
        if (*p == '-' || *p == '_') {
            if (!cur.empty() && out.n < 3) out.t[out.n++] = cur;
            cur.clear();
            continue;
        }
        cur += lower(*p);
    }
    if (!cur.empty() && out.n < 3) out.t[out.n++] = cur;
    return out;
}

// 中文需判断字形，系统可能使用文字子标签、地区，或旧 Windows 的 CHS/CHT 标记。
bool resolve_chinese(const Subtags& st, Lang* out) {
    for (int i = 1; i < st.n; i++) {
        const std::string& t = st.t[i];
        if (t == "hans" || t == "chs") { *out = Lang::zh_hans; return true; }
        if (t == "hant" || t == "cht") { *out = Lang::zh_hant; return true; }
        // 澳门和香港采用繁体；新加坡和马来西亚采用简体。
        if (t == "cn" || t == "sg" || t == "my") { *out = Lang::zh_hans; return true; }
        if (t == "tw" || t == "hk" || t == "mo") { *out = Lang::zh_hant; return true; }
    }
    // 只有 zh 而无文字、地区时，繁体默认构建保持繁体，其余采用简体。
    *out = (kDefaultLang == Lang::zh_hant) ? Lang::zh_hant : Lang::zh_hans;
    return true;
}

}  // 匿名命名空间

bool parse_lang(const char* s, Lang* out) {
    if (!s || !*s) return false;

    const Subtags st = split_locale(s);
    if (st.n == 0) return false;

    // C 与 POSIX 表示未指定偏好，应继续尝试下一优先级，不能强行解释为英语。
    if (st.t[0] == "c" || st.t[0] == "posix") return false;

    if (st.t[0] == "zh") return resolve_chinese(st, out);

    // 接受规范语言码，以及 SS_DEFAULT_LANG 和设置文件使用的 zh_hans 等标识符形式。
    std::string joined = st.t[0];
    for (int i = 1; i < st.n; i++) joined += "-" + st.t[i];

#define X(id, c, native, english)                                             \
    {                                                                         \
        std::string canon(c);                                                 \
        for (auto& ch : canon) ch = lower(ch);                                \
        if (joined == canon || st.t[0] == canon) { *out = Lang::id; return true; } \
    }
    SS_LANGUAGES(X)
#undef X

    return false;
}

Lang detect_os_lang() {
    Lang l = kDefaultLang;

#if defined(_WIN32)
    // Windows 用户界面语言，如 zh-Hans-CN；Vista 起提供。
    wchar_t buf[LOCALE_NAME_MAX_LENGTH] = {0};
    if (GetUserDefaultLocaleName(buf, LOCALE_NAME_MAX_LENGTH) > 0) {
        // 区域名称均为 ASCII，窄字符复制不会损失信息。
        char narrow[LOCALE_NAME_MAX_LENGTH] = {0};
        size_t n = 0;
        for (; n + 1 < sizeof narrow && buf[n]; n++)
            narrow[n] = (buf[n] < 128) ? char(buf[n]) : '?';
        narrow[n] = '\0';
        if (parse_lang(narrow, &l)) return l;
    }
#elif defined(__APPLE__)
    // Finder 启动时通常没有 POSIX 区域变量，因此通过 CoreFoundation 查询；接口为 C，无需 Objective-C。
    if (CFLocaleRef loc = CFLocaleCopyCurrent()) {
        char buf[128] = {0};
        const bool ok = CFStringGetCString(CFLocaleGetIdentifier(loc), buf,
                                           sizeof buf, kCFStringEncodingUTF8);
        CFRelease(loc);
        if (ok && parse_lang(buf, &l)) return l;
    }
#endif

    // POSIX 及通用回退顺序：LC_ALL > LC_MESSAGES > LANG。
    for (const char* var : {"LC_ALL", "LC_MESSAGES", "LANG"})
        if (const char* v = std::getenv(var))
            if (parse_lang(v, &l)) return l;

    return kDefaultLang;
}

Lang init(const char* cli, const char* saved) {
    Lang l = kDefaultLang;

    auto try_set = [&](const char* s, const char* origin) {
        if (!s || !*s) return false;
        if (parse_lang(s, &l)) return true;
        std::fprintf(stderr, "warning: unknown language '%s' (%s); using %s\n"
                             "%s",
                     s, origin, code(kDefaultLang), language_list().c_str());
        return false;
    };

    if (try_set(cli, "--lang")) { set_current(l); return l; }
    if (try_set(env("LANG"), "SS_LANG")) { set_current(l); return l; }
    if (try_set(saved, "settings")) { set_current(l); return l; }

    l = detect_os_lang();
    set_current(l);
    return l;
}

namespace {
const char* g_lang_arg = nullptr;
}

const char* lang_arg() { return g_lang_arg; }

const char* take_lang_arg(int* argc, char** argv) {
    const char* value = nullptr;
    int out = 1;
    for (int i = 1; i < *argc; i++) {
        const char* a = argv[i];
        if (std::strcmp(a, "--lang") == 0 && i + 1 < *argc) {
            value = argv[++i];
            continue;
        }
        if (std::strncmp(a, "--lang=", 7) == 0) {
            value = a + 7;
            continue;
        }
        argv[out++] = argv[i];
    }
    argv[out] = nullptr;
    *argc = out;
    g_lang_arg = value;
    return value;
}

std::string language_list() {
    std::string s;
#define X(id, c, native, english)                                             \
    {                                                                         \
        std::string row = std::string("    ") + c;                            \
        row.resize(14, ' ');                                                  \
        s += row; s += english; s += "\n";                                    \
    }
    SS_LANGUAGES(X)
#undef X
    return s;
}

// ---------------- 位置参数替换，规则见 Message.h ----------------

Arg::Arg(int v)                : s(std::to_string(v)) {}
Arg::Arg(long v)               : s(std::to_string(v)) {}
Arg::Arg(long long v)          : s(std::to_string(v)) {}
Arg::Arg(unsigned v)           : s(std::to_string(v)) {}
Arg::Arg(unsigned long v)      : s(std::to_string(v)) {}
Arg::Arg(unsigned long long v) : s(std::to_string(v)) {}

Arg::Arg(double v) {
    // std::to_string(double) 固定使用 %f，会把 1e-7 变成 0.000000；界面数值使用 %g。
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    s = buf;
}

std::string format(const char* pattern, std::initializer_list<Arg> args) {
    std::string out;
    if (!pattern) return out;
    for (const char* p = pattern; *p;) {
        if (p[0] == '{' && p[1] == '{') { out += '{'; p += 2; continue; }
        if (p[0] == '}' && p[1] == '}') { out += '}'; p += 2; continue; }
        if (*p != '{') { out += *p++; continue; }

        const char* q = p + 1;
        size_t idx = 0;
        bool digits = false;
        while (*q >= '0' && *q <= '9') { idx = idx * 10 + size_t(*q - '0'); q++; digits = true; }
        if (!digits || *q != '}' || idx >= args.size()) {
            // 无效占位符或缺少参数时保留原文，使界面显露 {2} 等错误，避免静默丢失。
            out += *p++;
            continue;
        }
        out += (args.begin() + idx)->s;
        p = q + 1;
    }
    return out;
}

namespace {

// 覆盖译文常见的东亚宽字符、全角字符范围，并非完整 UAX #11 表。
bool wide_cp(unsigned int cp) {
    return (cp >= 0x1100 && cp <= 0x115F) ||    // 谚文字母
           (cp >= 0x2E80 && cp <= 0x303E) ||    // CJK 部首与标点
           (cp >= 0x3041 && cp <= 0x33FF) ||    // 假名、谚文兼容字符、CJK 兼容字符
           (cp >= 0x3400 && cp <= 0x4DBF) ||    // CJK 扩展 A
           (cp >= 0x4E00 && cp <= 0x9FFF) ||    // CJK 统一表意文字
           (cp >= 0xA000 && cp <= 0xA4CF) ||    // 彝文
           (cp >= 0xAC00 && cp <= 0xD7A3) ||    // 谚文音节
           (cp >= 0xF900 && cp <= 0xFAFF) ||    // CJK 兼容表意文字
           (cp >= 0xFE30 && cp <= 0xFE6F) ||    // CJK 兼容形式
           (cp >= 0xFF00 && cp <= 0xFF60) ||    // 全角形式
           (cp >= 0xFFE0 && cp <= 0xFFE6) ||
           (cp >= 0x20000 && cp <= 0x3FFFD);    // CJK 扩展 B 及后续区段
}

}  // 匿名命名空间

int display_width(const char* s) {
    // 解码 UTF-8，东亚宽字符与全角字符按两列计算。
    int w = 0;
    if (!s) return 0;
    for (const unsigned char* p = (const unsigned char*)s; *p;) {
        unsigned int cp = *p;
        int len = 1;
        if (cp >= 0xF0)      { cp &= 0x07; len = 4; }
        else if (cp >= 0xE0) { cp &= 0x0F; len = 3; }
        else if (cp >= 0xC0) { cp &= 0x1F; len = 2; }
        for (int i = 1; i < len; i++) {
            if ((p[i] & 0xC0) != 0x80) { len = i; break; }
            cp = (cp << 6) | (p[i] & 0x3F);
        }
        p += len;
        w += wide_cp(cp) ? 2 : 1;
    }
    return w;
}

namespace {

// UTF-8 字符的码点、字节长度及终端列宽。
struct Ch { unsigned int cp; int bytes; int width; };

Ch decode_one(const unsigned char* p) {
    Ch c{*p, 1, 1};
    if (c.cp >= 0xF0)      { c.cp &= 0x07; c.bytes = 4; }
    else if (c.cp >= 0xE0) { c.cp &= 0x0F; c.bytes = 3; }
    else if (c.cp >= 0xC0) { c.cp &= 0x1F; c.bytes = 2; }
    for (int i = 1; i < c.bytes; i++) {
        if ((p[i] & 0xC0) != 0x80) { c.bytes = i; break; }
        c.cp = (c.cp << 6) | (p[i] & 0x3F);
    }
    c.width = wide_cp(c.cp) ? 2 : 1;
    return c;
}

// 判断字符前是否允许断行；无空格书写的宽字符允许，禁止出现在行首的标点除外。
bool can_break_before(unsigned int cp) {
    if (!wide_cp(cp)) return false;
    switch (cp) {
        case 0x3001: case 0x3002:                        // 、。
        case 0xFF0C: case 0xFF0E: case 0xFF1A: case 0xFF1B:  // ，．：；
        case 0xFF01: case 0xFF1F:                        // ！？
        case 0x3009: case 0x300B: case 0x300D: case 0x300F:  // 〉》」』
        case 0x3011: case 0x3015: case 0xFF09: case 0xFF3D:  // 】〕）］
        case 0x30FC:                                     // ー
        case 0x3041: case 0x3043: case 0x3045: case 0x3047: case 0x3049:
        case 0x3063: case 0x3083: case 0x3085: case 0x3087:
        case 0x30A1: case 0x30A3: case 0x30A5: case 0x30A7: case 0x30A9:
        case 0x30C3: case 0x30E3: case 0x30E5: case 0x30E7:
            return false;
        default:
            return true;
    }
}

}  // 匿名命名空间

std::vector<std::string> wrap(const std::string& text, int columns) {
    std::vector<std::string> out;
    if (columns < 2) columns = 2;
    const unsigned char* p = (const unsigned char*)text.c_str();

    std::string line;          // 已确定写入当前行的内容
    int line_w = 0;
    std::string word;          // 上次可断行位置之后的连续内容
    int word_w = 0;

    // 断行处的空格不归属于任何一行。
    auto rtrim = [](std::string s) {
        while (!s.empty() && s.back() == ' ') s.pop_back();
        return s;
    };

    auto commit = [&]() {      // 将词并入当前行，必要时先换行
        if (word.empty()) return;
        if (line_w + word_w > columns && !line.empty()) {
            out.push_back(rtrim(line));
            line.clear();
            line_w = 0;
            // 断行位置的空格由换行消耗。
            while (!word.empty() && word[0] == ' ') { word.erase(0, 1); word_w--; }
        }
        line += word;
        line_w += word_w;
        word.clear();
        word_w = 0;
    };

    while (*p) {
        if (*p == '\n') {
            commit();
            out.push_back(rtrim(line));
            line.clear();
            line_w = 0;
            p++;
            continue;
        }
        const Ch c = decode_one(p);
        // 空格结束前一个词；宽字符既结束前词，也可自身作为行首。
        if (c.cp == ' ' || can_break_before(c.cp)) commit();
        word.append((const char*)p, (size_t)c.bytes);
        word_w += c.width;
        p += c.bytes;
        if (c.cp == ' ') commit();
    }
    commit();
    line = rtrim(line);
    if (!line.empty() || out.empty()) out.push_back(line);
    return out;
}

std::string pad_to(const std::string& s, int columns) {
    std::string out = s;
    out.append((size_t)std::max(0, columns - display_width(s)), ' ');
    return out;
}

// format 的逆操作：先拆出固定文本和占位符索引，再按顺序匹配文本，将间隙解析为参数；约定见 Message.h。
bool scan(const char* pattern, const std::string& text,
          std::vector<std::string>& out) {
    out.clear();
    if (!pattern) return false;

    std::vector<std::string> lits;   // lits[i] 位于 slot[i] 前，另含一段尾部文本
    std::vector<size_t> slot;        // 各间隙的占位符索引
    std::string cur;
    for (const char* p = pattern; *p;) {
        if (p[0] == '{' && p[1] == '{') { cur += '{'; p += 2; continue; }
        if (p[0] == '}' && p[1] == '}') { cur += '}'; p += 2; continue; }
        if (*p != '{') { cur += *p++; continue; }
        const char* q = p + 1;
        size_t idx = 0;
        bool digits = false;
        while (*q >= '0' && *q <= '9') { idx = idx * 10 + size_t(*q - '0'); q++; digits = true; }
        if (!digits || *q != '}') { cur += *p++; continue; }
        // 相邻占位符之间没有分隔符，无法可靠划分，必须拒绝匹配。
        if (!lits.empty() && cur.empty()) return false;
        lits.push_back(cur);
        slot.push_back(idx);
        cur.clear();
        p = q + 1;
    }
    lits.push_back(cur);              // 最后一个占位符之后的尾部文本

    if (text.compare(0, lits[0].size(), lits[0]) != 0) return false;
    size_t pos = lits[0].size();
    std::vector<std::string> got(slot.size());
    for (size_t i = 0; i < slot.size(); i++) {
        const std::string& next = lits[i + 1];
        size_t at;
        if (next.empty()) {
            // 模式以最后一个占位符结束时，将剩余文本全部作为其参数。
            if (i + 1 != slot.size()) return false;
            at = text.size();
        } else {
            at = text.find(next, pos);
            if (at == std::string::npos) return false;
        }
        got[i] = text.substr(pos, at - pos);
        pos = at + next.size();
    }
    if (pos != text.size()) return false;

    // 译文可重排占位符，因此按槽位编号写入结果。
    size_t n = 0;
    for (size_t s : slot) n = std::max(n, s + 1);
    out.assign(n, std::string());
    for (size_t i = 0; i < slot.size(); i++) out[slot[i]] = std::move(got[i]);
    return true;
}

}  // 命名空间 i18n
}  // 命名空间 spirula
