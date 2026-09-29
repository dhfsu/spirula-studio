#pragma once

// Msg 用类型表示完整译文，目录条目生成含各语言槽位的 constexpr 对象，并在编译期检查数量、重复标签及缺失译文。
// 标签携带槽位索引，因此顺序不影响结果；消息 id 保持稳定，语言切换不会改变 GUI 控件身份。
// 始终链接 13 种语言，每条消息约 104 字节指针加字符串，目录静态数据不足 1 MB，无运行期目录查找或资源文件依赖。

#include "i18n/Languages.h"

#include <atomic>
#include <initializer_list>
#include <string>
#include <vector>

namespace spirula {
namespace i18n {

enum class Lang : unsigned {
#define X(id, code, native, english) id,
    SS_LANGUAGES(X)
#undef X
};

inline constexpr unsigned kLangCount = 0
#define X(id, code, native, english) +1
    SS_LANGUAGES(X)
#undef X
    ;

// 带语言标签的译文，统一由 BeginCatalog.h 的 EN、JA、ZH_HANS 等宏构造。
template <Lang L>
struct Tr {
    const char* s;
};

struct Msg {
    const char* v[kLangCount] = {};
    const char* id = "";

    template <Lang... Ls>
    constexpr Msg(const char* id_, Tr<Ls>... t) : v{}, id(id_) {
        static_assert(sizeof...(Ls) == kLangCount,
                      "i18n: wrong number of translations -- one tag per "
                      "language in SS_LANGUAGES, no more and no fewer");
        ((v[unsigned(Ls)] = t.s), ...);   // 通过 C++17 折叠表达式填写带标签槽位
    }

    // 仅英文的过渡入口，参见 en_only()
    struct EnOnlyTag {};
    constexpr Msg(EnOnlyTag, const char* id_, const char* s) : v{}, id(id_) {
        for (unsigned i = 0; i < kLangCount; i++) v[i] = s;
    }

    constexpr bool complete() const {
        for (unsigned i = 0; i < kLangCount; i++)
            if (!v[i] || !*v[i]) return false;
        return true;
    }

    // 当前界面语言对应的字符串。
    const char* get() const;

    const char* in(Lang l) const { return v[unsigned(l)]; }
};

// 分阶段翻译时暂以英文填充全部槽位，使 complete() 仍成立；SS_MSG_EN 标识尚未完成的译文，便于统计。
constexpr Msg en_only(const char* id, const char* s) {
    return Msg(Msg::EnOnlyTag{}, id, s);
}

// ---------------- 当前语言 ----------------
// i18n::init() 初始化，GUI 可切换；工作线程同时格式化消息，因此使用原子变量。字符串常驻只读区，relaxed 足够，竞争最多使一帧或一行沿用旧语言。
namespace detail {
extern std::atomic<Lang> g_current;
}

inline Lang current() {
    return detail::g_current.load(std::memory_order_relaxed);
}
void set_current(Lang l);

inline const char* Msg::get() const { return v[unsigned(current())]; }

// ---------------- 位置参数替换 ----------------
// 完整句子使用一条消息及 {0}/{1} 占位符，禁止按英文顺序拼接片段；避免依赖复数变化，采用“图像：5”等标签形式。
// {{ 表示字面量 {；缺少参数的占位符原样保留，使错误可见而不会截断句子。
struct Arg {
    std::string s;

    Arg(const char* v) : s(v ? v : "") {}
    Arg(const std::string& v) : s(v) {}
    Arg(std::string&& v) : s(std::move(v)) {}
    Arg(int v);
    Arg(long v);
    Arg(long long v);
    Arg(unsigned v);
    Arg(unsigned long v);
    Arg(unsigned long long v);
    Arg(double v);           // 采用 %g；需要其他形式时由调用方自行格式化
    Arg(float v) : Arg(double(v)) {}
};

std::string format(const char* pattern, std::initializer_list<Arg> args);

inline std::string format(const Msg& m, std::initializer_list<Arg> args) {
    return format(m.get(), args);
}

// ---------------- 从日志反向解析参数 ----------------
// scan() 使用子进程打印时的同一 Msg 匹配整行，并提取各占位符，避免匹配英文片段而破坏本地化。
// 固定文本取允许后续继续匹配的最早位置；无分隔符的相邻占位符无法可靠解析，直接拒绝。
bool scan(const char* pattern, const std::string& text,
          std::vector<std::string>& out);

// UTF-8 字符串的终端列宽，东亚宽字符占两列，用于日志标签和表格对齐；覆盖常见译文范围，并非完整 UAX #11 表。
int display_width(const char* s);
inline int display_width(const std::string& s) { return display_width(s.c_str()); }

// 在 s 右侧补空格至 columns 列，不截断。
std::string pad_to(const std::string& s, int columns);

// 按终端列宽将 text 换行为最多 columns 列；有空格时按词断行，无空格的中日文按字符断行。
// 禁止闭合标点或小假名出现在行首。
std::vector<std::string> wrap(const std::string& text, int columns);

inline bool scan(const Msg& m, const std::string& text,
                 std::vector<std::string>& out) {
    return scan(m.get(), text, out);
}

}  // 命名空间 i18n
}  // 命名空间 spirula

// ---------------- 消息目录入口 ----------------
// 须位于 BeginCatalog.h 与 EndCatalog.h 之间，后者撤销短语言标签宏，避免污染普通代码。

#define SS_MSG(name, ...)                                                     \
    inline constexpr ::spirula::i18n::Msg name{#name, __VA_ARGS__};           \
    static_assert(name.complete(),                                            \
                  "i18n: '" #name "' is missing a translation")

#define SS_MSG_EN(name, s)                                                    \
    inline constexpr ::spirula::i18n::Msg name =                              \
        ::spirula::i18n::en_only(#name, s)
