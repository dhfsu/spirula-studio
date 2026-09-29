#pragma once

// 进程级 SfM 取消标记；一个进程同时运行一个任务，由驱动方设置。
// 检查时抛异常，逐层展开可执行 Vulkan 上下文析构并释放设备内存，无需层层传递状态码。

#include <atomic>
#include <exception>

namespace sfm {

// 用户请求取消，不视为故障，在任务边界捕获。
struct Cancelled : std::exception {
    const char* what() const noexcept override { return "sfm: cancelled"; }
};

namespace cancel {

// 空标记表示不能取消；标记生命周期必须长于运行。
inline std::atomic<const std::atomic<bool>*> g_flag{nullptr};

inline void set_token(const std::atomic<bool>* flag) {
    g_flag.store(flag, std::memory_order_relaxed);
}

inline bool requested() {
    const std::atomic<bool>* f = g_flag.load(std::memory_order_relaxed);
    return f && f->load(std::memory_order_relaxed);
}

// 在阶段边界及逐图像、图像对、配准、求解迭代处检查，使取消延迟小于一秒，同时保持原子读取开销可忽略。
inline void check() {
    if (requested()) throw Cancelled();
}

}  // 命名空间 cancel
}  // 命名空间 sfm
