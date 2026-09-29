// 日志实现，参见 Log.h。

#include "sfm/core/Log.h"

#include "i18n/catalog/Sfm.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

namespace msg = spirula::i18n::msg::sfm;

namespace sfm {
namespace slog {

namespace {

const spirula::i18n::Msg& tag_msg(Tag t) {
    switch (t) {
        case Tag::Run:     return msg::tag_run;
        case Tag::Extract: return msg::tag_extract;
        case Tag::Match:   return msg::tag_match;
        case Tag::Map:     return msg::tag_map;
        case Tag::Merge:   return msg::tag_merge;
        case Tag::Orient:  return msg::tag_orient;
        case Tag::Device:  return msg::tag_device;
    }
    return msg::tag_run;
}

constexpr int kNumTags = 7;

// 每种语言仅构建一次补齐宽度的标签前缀。
struct TagTable {
    spirula::i18n::Lang lang{};
    bool valid = false;
    std::string prefix[kNumTags];
};

std::mutex g_mu;
TagTable g_table;

const TagTable& table() {
    const spirula::i18n::Lang cur = spirula::i18n::current();
    if (g_table.valid && g_table.lang == cur) return g_table;
    int width = 0;
    for (int i = 0; i < kNumTags; i++)
        width = std::max(width, display_width(tag_msg((Tag)i).get()));
    for (int i = 0; i < kNumTags; i++) {
        const char* s = tag_msg((Tag)i).get();
        std::string p = "[";
        p += s;
        p += "]";
        p.append((size_t)std::max(0, width - display_width(s)), ' ');
        p += " ";
        g_table.prefix[i] = std::move(p);
    }
    g_table.lang = cur;
    g_table.valid = true;
    return g_table;
}

Sink g_sink;
// 逐行输出锁独立于保护标签表和接收端指针的 g_mu；接收端可能调用 prefix，持有 g_mu 回调会死锁。
std::mutex g_out_mu;

void print(const std::string& pfx, Level lv, const std::string& text) {
    std::FILE* f = lv == Level::Info ? stdout : stderr;
    if (lv == Level::Diag) {
        std::fprintf(f, "%s\n", text.c_str());
    } else {
        const char* word = lv == Level::Warning  ? msg::word_warning.get()
                           : lv == Level::Error  ? msg::word_error.get()
                                                 : nullptr;
        std::fprintf(f, "%s%s%s%s\n", pfx.c_str(), word ? word : "",
                     word ? " " : "", text.c_str());
    }
    // 及时刷新以保持 stdout/stderr 在父进程合并终端中的正确顺序。
    std::fflush(f);
}

void emit(Tag t, Level lv, const std::string& text) {
    Sink sink;
    std::string pfx;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        sink = g_sink;
        if (!sink) pfx = table().prefix[(int)t];
    }
    std::lock_guard<std::mutex> out(g_out_mu);
    if (sink) sink(t, lv, text);
    else print(pfx, lv, text);
}

}  // 匿名命名空间


int display_width(const char* s) { return spirula::i18n::display_width(s); }

void set_sink(Sink s) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_sink = std::move(s);
}

std::string prefix(Tag t) {
    std::lock_guard<std::mutex> lk(g_mu);
    return table().prefix[(int)t];
}

std::string num(double v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", std::max(0, decimals), v);
    return buf;
}

void out(Tag t, const spirula::i18n::Msg& m,
         std::initializer_list<spirula::i18n::Arg> args) {
    emit(t, Level::Info, spirula::i18n::format(m, args));
}

void err(Tag t, const spirula::i18n::Msg& m,
         std::initializer_list<spirula::i18n::Arg> args) {
    emit(t, Level::Note, spirula::i18n::format(m, args));
}

void warn(Tag t, const spirula::i18n::Msg& m,
          std::initializer_list<spirula::i18n::Arg> args) {
    emit(t, Level::Warning, spirula::i18n::format(m, args));
}

void fail(Tag t, const spirula::i18n::Msg& m,
          std::initializer_list<spirula::i18n::Arg> args) {
    emit(t, Level::Error, spirula::i18n::format(m, args));
}

void out_raw(Tag t, const std::string& text) { emit(t, Level::Info, text); }
void err_raw(Tag t, const std::string& text) { emit(t, Level::Note, text); }

void diag(Tag t, const char* fmt, ...) {
    std::va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string text;
    if (n > 0) {
        std::vector<char> buf((size_t)n + 1);
        std::vsnprintf(buf.data(), buf.size(), fmt, ap2);
        text.assign(buf.data(), (size_t)n);
    }
    va_end(ap2);
    emit(t, Level::Diag, text);
}

}  // 命名空间 slog
}  // 命名空间 sfm
