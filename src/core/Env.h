#pragma once

// 环境变量的统一读取入口，使用 SS_<suffix>，兼容 SSPLAT_<suffix> 并在每次运行中警告一次。
// 例如 spirula::env("VK_DEVICE")；仅含头文件，使独立工具无需链接引擎即可使用。

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace spirula {

// 返回 SS_<suffix>、兼容的 SSPLAT_<suffix>，或 nullptr。
inline const char* env(const char* suffix) {
    std::string name = "SS_";
    name += suffix;
    if (const char* v = std::getenv(name.c_str())) return v;

    name = "SSPLAT_";
    name += suffix;
    const char* v = std::getenv(name.c_str());
    if (v) {
        // 进程内仅提示一次，支持工作线程并发调用。
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            // 此头文件不依赖引擎或国际化库；环境变量拼写警告保留英文。
            std::fprintf(stderr,
                         "warning: %s is deprecated; use SS_%s "
                         "(SSPLAT_* is reported once per run)\n",
                         name.c_str(), suffix);
    }
    return v;
}

// 已设置且值不为 0 时视为启用。
inline bool env_on(const char* suffix) {
    const char* v = env(suffix);
    return v && v[0] && !(v[0] == '0' && v[1] == '\0');
}

}  // 命名空间 spirula
