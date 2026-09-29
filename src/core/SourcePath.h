#pragma once

// SS_FILE 将编译器提供的绝对源路径裁剪为仓库相对路径，并统一分隔符。
// 处理在编译期完成，避免将构建机私人目录写入错误消息和二进制；仅含头文件，供独立工具共用。

#include <cstddef>

// 仓库根目录不含末尾分隔符，由 CMakeLists.txt 提供；为空时保留编译器原始路径。
#ifndef SS_SOURCE_ROOT
#define SS_SOURCE_ROOT ""
#endif

namespace spirula {
namespace detail {

// Windows 同一路径可能有多种写法，比较前对两侧规范化。
constexpr char path_fold(char c) {
    return c == '\\' ? '/' :
           (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
}

constexpr std::size_t root_len(const char *root) {
    std::size_t n = 0;
    while (root[n])
        ++n;
    while (n && (root[n - 1] == '/' || root[n - 1] == '\\'))
        --n;
    return n;
}

// 返回 p 中 root 前缀及后续分隔符的长度，不匹配时为 0；须在路径分量边界结束，防止将相邻目录误裁剪。
constexpr std::size_t root_prefix(const char *p, const char *root) {
    const std::size_t n = root_len(root);
    if (!n)
        return 0;
    for (std::size_t i = 0; i < n; ++i)
        if (path_fold(p[i]) != path_fold(root[i]))
            return 0;
    if (p[n] != '/' && p[n] != '\\')
        return 0;
    std::size_t i = n;
    while (p[i] == '/' || p[i] == '\\')
        ++i;
    return i;
}

// root 显式作为参数，便于测试当前平台不会生成的路径形式。
template <std::size_t N>
struct SourcePath {
    char v[N];
    constexpr SourcePath(const char (&p)[N], const char *root = SS_SOURCE_ROOT)
        : v{} {
        std::size_t j = 0;
        for (std::size_t i = root_prefix(p, root); i < N; ++i, ++j)
            v[j] = p[i] == '\\' ? '/' : p[i];
    }
};

}  // 命名空间 detail
}  // 命名空间 spirula

// 仅供主机代码使用；静态存储使裁剪后的字符串在表达式结束后仍有效。
#define SS_FILE                                                                \
    ([]() -> const char * {                                                    \
        static constexpr ::spirula::detail::SourcePath<sizeof(__FILE__)>       \
            _ss_source_path(__FILE__);                                         \
        return _ss_source_path.v;                                              \
    }())
