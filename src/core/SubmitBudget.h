#pragma once

// 根据历史耗时调整单次 GPU 提交的工作量，避免超过驱动看门狗触发 VK_ERROR_DEVICE_LOST。
// Windows TDR 与 Linux 7.0 amdgpu 的超时均为 2 s；仅含头文件，供 SfM 与推理运行时共用。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "core/Env.h"

namespace spirula {

class SubmitBudget {
public:
    // 默认 0.25 s，相对 2 s 看门狗保留八倍余量以应对降频；SS_SUBMIT_BUDGET_MS 可覆盖，首次测量前使用 prior_rate。
    explicit SubmitBudget(double prior_rate = 0) : rate_(prior_rate) {
        target_ = 0.25;
        if (const char* v = env("SUBMIT_BUDGET_MS"))
            if (std::atof(v) > 0) target_ = std::atof(v) * 1e-3;
    }

    // 下一次提交的工作量，使用调用方单位；无测量且无先验时为 0，表示提交最小工作单元。
    double limit() const { return rate_ > 0 ? rate_ * target_ : 0; }
    double rate() const { return rate_; }
    double target() const { return target_; }
    bool measured() const { return measured_; }

    // 将 limit() 转成整单位启动大小；测量前采用 first，最大不超过 most，保持快速 GPU 原有批量。
    int64_t chunk(int64_t first, int64_t most) const {
        if (limit() <= 0) return std::max<int64_t>(1, std::min(first, most));
        return std::max<int64_t>(1, std::min<int64_t>((int64_t)limit(), most));
    }

    void record(double work, double seconds) {
        if (work <= 0 || seconds <= 0) return;
        const double r = work / seconds;
        // 足够长的测量直接替换先验，较慢估计立即生效；其他更新在对数空间折半，避免短提交的固定延迟压低后续批量。
        const bool solid = seconds > target_ / 16;
        if (rate_ <= 0 || (solid && (!measured_ || r < rate_))) rate_ = r;
        else rate_ = std::sqrt(rate_ * r);
        measured_ = measured_ || solid;
    }

private:
    double target_;
    double rate_;  // 每秒工作量
    bool measured_ = false;
};

}  // 命名空间 spirula
