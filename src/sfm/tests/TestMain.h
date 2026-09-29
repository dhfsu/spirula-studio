#pragma once
// SfM 测试统一入口捕获异常并输出缺失设备能力等真实原因。
// 避免 MSVC 未捕获异常进入 Windows 错误报告，在无界面环境留下挂起的 WerFault 并丢失诊断。

#include <cstdio>
#include <exception>

inline int sfmTestMain(int argc, char** argv, int (*body)(int, char**)) {
    try {
        return body(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "FAIL: unknown exception\n");
        return 2;
    }
}
