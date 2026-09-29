#pragma once

// 统一将时长格式化为 hh:mm:ss.mmm，小时始终显示并补零，不限制在 24 以内；毫秒保留性能分析所需精度。
// 实现位于各调用方共同依赖的国际化层，保证日志、状态栏及命令行进度格式一致。

#include <cstdio>
#include <string>

namespace spirula {
namespace i18n {

// 负值表示时间未知，显示 --:--。
inline std::string format_duration(double seconds) {
    if (seconds < 0) return "--:--";
    // 先累计毫秒再进位，避免 59.9995 被错误显示为 00:00:60.000。
    const long long ms = (long long)(seconds * 1000.0 + 0.5);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%02lld:%02lld:%02lld.%03lld",
                  ms / 3600000, (ms / 60000) % 60, (ms / 1000) % 60, ms % 1000);
    return buf;
}

// 不含毫秒的同类格式，用于任务队列等粗略完成时间估计。
inline std::string format_duration_coarse(double seconds) {
    if (seconds < 0) return "--:--";
    const long long s = (long long)(seconds + 0.5);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%02lld:%02lld:%02lld", s / 3600,
                  (s / 60) % 60, s % 60);
    return buf;
}

}  // 命名空间 i18n
}  // 命名空间 spirula
