// 受内存预算约束的并行图像解码；1700 万像素 JPEG 上解码约占提取阶段 66%，GPU SIFT 约每图 47 ms，因此 CPU 预解码与单线程 GPU 驱动重叠。
// 按输入顺序交付，工作线程受有限滑动窗口约束；线程数和窗口大小由机器预算及最大图像决定，峰值内存不随图像总数增长。
// 每次解码持有源 RGB 的 3 B/像素及缩小输出，不构造全尺寸 float；固定 1 GiB 曾使 2100 万像素数据仅能三线程，按物理内存分配后可利用 32 核并使提取受 GPU 限制。
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "sfm/core/HostMemory.h"
#include "sfm/core/Image.h"

namespace sfm {

struct ImageLoadOptions {
    int max_image_size = 3200;    // 最长边限制，采用 COLMAP 默认值
    int num_threads = 0;          // 0 使用 hardware_concurrency，1 在调用线程解码
    bool want_color = false;      // 同时解码缩小 RGB 供点云着色
    // 下列选项描述输入色彩空间，解码时统一转换为 sRGB。
    std::string gamut;
    std::optional<bool> is_linear;
    // 掩码路径与输入 paths 一一对应，空字符串表示该图无掩码，空列表完全跳过掩码。
    // 掩码与图像在池中并行解码，避免消费者线程串行 PNG 解码重新阻塞 GPU。
    std::vector<std::string> mask_paths;
    // 在解码池内反转掩码保留与忽略区域。
    bool flip_mask = false;
    // 按 EXIF Orientation 旋转图像。
    bool apply_exif_orientation = false;
    // 内存预算一半用于并发解码，一半用于就绪窗口，两者至少为 1，超预算单图仍可独占加载；0 根据机器估算。
    // 固定 1 GiB 对 32 核大图曾仅允许三线程，预算必须随机器调整。
    size_t memory_budget_bytes = 0;
};

// 默认物理内存的四分之一，限制在 1–8 GiB，为页缓存留空间；无法查询时使用 1 GiB。
inline size_t defaultDecodeBudget() {
    const size_t total = physicalRamBytes();
    if (total == 0) return 1ull << 30;
    return std::min<size_t>(std::max<size_t>(total / 4, 1ull << 30), 8ull << 30);
}

// 解码计划的实际决策，供日志显示。
struct ImageLoadPlan {
    int num_threads = 1;
    int window = 1;
    size_t decode_peak_bytes = 0;  // 最大图像的单次并发解码内存
    size_t held_bytes = 0;         // 窗口中单张就绪图像的内存
};

namespace detail {

// 按最长边限制后的尺寸。
inline void clampedSize(int w, int h, int max_image_size, int& dw, int& dh) {
    dw = w;
    dh = h;
    int longEdge = std::max(w, h);
    if (max_image_size > 0 && longEdge > max_image_size) {
        double scale = (double)max_image_size / longEdge;
        dw = std::max(1, (int)std::lround(w * scale));
        dh = std::max(1, (int)std::lround(h * scale));
    }
}

}  // 命名空间 detail

// 根据预算与最大输入推导线程数和窗口，dims 为各图像探测宽高，顺序不限。
inline ImageLoadPlan planImageLoad(const std::vector<std::pair<int, int>>& dims,
                                   const ImageLoadOptions& opt) {
    ImageLoadPlan plan;
    size_t maxPix = 1, maxOutPix = 1;
    for (const auto& d : dims) {
        maxPix = std::max(maxPix, (size_t)d.first * (size_t)d.second);
        int dw = 0, dh = 0;
        detail::clampedSize(d.first, d.second, opt.max_image_size, dw, dh);
        maxOutPix = std::max(maxOutPix, (size_t)dw * (size_t)dh);
    }
    // 峰值包括源 RGB 的 3 B/像素、输出灰度的 4 B/像素及可选 RGB 的 3 B/像素；掩码估计解码尺寸每像素 2 B，分别用于灰度和二值副本。
    // 掩码通常不大于源图，未额外探测其头部，预算仍由图像主导。
    const size_t mask_bytes = opt.mask_paths.empty() ? 0 : maxOutPix * 2;
    plan.decode_peak_bytes =
        maxPix * 3 + maxOutPix * (opt.want_color ? 7 : 4) + mask_bytes;
    plan.held_bytes = maxOutPix * (opt.want_color ? 7 : 4)  // 灰度 float，加可选 RGB uint8
                    + (opt.mask_paths.empty() ? 0 : maxOutPix);

    unsigned hc = std::thread::hardware_concurrency();
    int want = opt.num_threads > 0 ? opt.num_threads : (hc > 0 ? (int)hc : 1);
    const size_t budget =
        opt.memory_budget_bytes ? opt.memory_budget_bytes : defaultDecodeBudget();
    const size_t half = budget / 2;

    int by_mem = (int)std::max<size_t>(1, half / plan.decode_peak_bytes);
    plan.num_threads = std::max(1, std::min(want, by_mem));
    // 窗口必须覆盖所有工作线程，否则消费者等待的索引可能被窗口门限阻塞；额外窗口仅用于预解码。
    int win_by_mem = (int)std::max<size_t>(1, half / plan.held_bytes);
    plan.window = std::max(plan.num_threads, std::min(2 * plan.num_threads, win_by_mem));
    return plan;
}

// 工作池解码后，在调用线程按输入顺序执行 consume；坏文件经 on_error 报告后跳过，不中断整批。
// 传入已记录日志的 plan，保证展示与实际预算一致。
inline void loadImagesInOrder(const std::vector<std::string>& paths, const ImageLoadPlan& plan,
                              const ImageLoadOptions& opt,
                              const std::function<void(size_t, GrayImage&)>& consume,
                              const std::function<void(size_t, const std::string&)>& on_error
                                  = nullptr) {
    const size_t n = paths.size();
    if (n == 0) return;

    auto decodeOne = [&](size_t i, GrayImage& out, std::string& err) {
        static const std::string kNoMask;
        const std::string& mp = i < opt.mask_paths.size() ? opt.mask_paths[i] : kNoMask;
        try {
            out = loadGrayImage(paths[i], opt.max_image_size, opt.want_color, mp,
                                opt.gamut, opt.is_linear, opt.flip_mask,
                                opt.apply_exif_orientation);
        } catch (const std::exception& e) {
            err = e.what();
        }
    };

    if (plan.num_threads <= 1) {
        for (size_t i = 0; i < n; i++) {
            GrayImage img;
            std::string err;
            decodeOne(i, img, err);
            if (!err.empty()) {
                if (on_error) on_error(i, err);
                continue;
            }
            consume(i, img);
        }
        return;
    }

    std::vector<GrayImage> slots(n);
    std::vector<std::string> errors(n);
    std::vector<char> ready(n, 0);
    std::atomic<size_t> next_claim{0};
    std::atomic<bool> stop{false};
    size_t next_consume = 0;  // 由 mtx 保护
    std::mutex mtx;
    std::condition_variable cv_ready, cv_window;

    std::vector<std::thread> workers;
    workers.reserve(plan.num_threads);
    for (int t = 0; t < plan.num_threads; t++) {
        workers.emplace_back([&] {
            for (;;) {
                size_t i = next_claim.fetch_add(1);
                if (i >= n || stop) return;
                {   // 预解码门限，限制在途图像集合
                    std::unique_lock<std::mutex> lk(mtx);
                    cv_window.wait(lk, [&] {
                        return stop || i < next_consume + (size_t)plan.window;
                    });
                }
                if (stop) return;
                GrayImage img;
                std::string err;
                decodeOne(i, img, err);
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    slots[i] = std::move(img);
                    errors[i] = std::move(err);
                    ready[i] = 1;
                }
                cv_ready.notify_all();
            }
        });
    }

    // 消费者抛异常前必须汇合线程；析构仍可 join 的线程会触发 terminate，使真实错误来不及报告。
    auto join_workers = [&] {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv_window.notify_all();
        for (std::thread& w : workers) w.join();
    };

    try {
        for (size_t i = 0; i < n; i++) {
            GrayImage img;
            std::string err;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_ready.wait(lk, [&] { return ready[i] != 0; });
                img = std::move(slots[i]);
                slots[i] = GrayImage();  // 立即释放窗口槽位
                err = std::move(errors[i]);
                next_consume = i + 1;
            }
            cv_window.notify_all();
            if (!err.empty()) {
                if (on_error) on_error(i, err);
                continue;
            }
            consume(i, img);
        }
    } catch (...) {
        join_workers();
        throw;
    }
    join_workers();
}

}  // 命名空间 sfm
