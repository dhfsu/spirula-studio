// 通用 LO-RANSAC，调用方提供最小解、内点重拟合和平方残差；使用截断二次 MSAC 评分，并按最佳内点率调整试验数。
// SPRT 收益很小，因残差求值仅占少量开销；采用精确的内点数量上界提前退出，并在试验循环内局部优化以提升停止条件（D26）。
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

namespace sfm {

struct RansacOptions {
    double max_error = 4.0;          // 内点阈值，单位与残差一致，通常为像素
    double confidence = 0.999;
    int min_num_trials = 100;
    int max_num_trials = 10000;
    double min_inlier_ratio = 0.0;   // 明显达不到此内点数时提前停止
    unsigned seed = 0;
    int lo_iters = 10;               // 局部优化重拟合轮数
};

template <class Model>
struct RansacReport {
    Model model{};
    std::vector<char> inlier_mask;   // 逐点标记
    int num_inliers = 0;
    double score = 0;                // MSAC 代价，越小越好
    bool success = false;
    int trials = 0;
};

template <class Model>
using FitFn = std::function<std::vector<Model>(const std::vector<int>&)>;
template <class Model>
using ResidualFn = std::function<double(const Model&, int)>;

// 评分输出内点、数量与 MSAC 代价；res 为模板调用，避免高频残差的间接调用。
// 仅当 count+remaining<need 时精确提前退出并返回 count=-1，不能用 <=，以保留可能平局或胜出的候选。
template <class Model, class Res>
static double scoreModel(const Model& m, int n, double thr2, const Res& res,
                         std::vector<char>& inliers, int& count, int need = 0) {
    inliers.assign(n, 0);
    count = 0;
    double score = 0;
    for (int i = 0; i < n; i++) {
        double r2 = res(m, i);
        if (r2 < thr2) {
            inliers[i] = 1;
            count++;
            score += r2;
        } else {
            score += thr2;
            if (count + (n - 1 - i) < need) { count = -1; return score; }
        }
    }
    return score;
}

// 非最小求解器可接收当前模型作为精化初值，只有内点集合不足以描述其状态。
template <class Refit, class Model>
static auto refitModels(const Refit& refit, const std::vector<int>& idx, const Model& m) {
    if constexpr (std::is_invocable_v<Refit, const std::vector<int>&, const Model&>)
        return refit(idx, m);
    else
        return refit(idx);
}

// fit/refit/res 由类型推导以支持内联 lambda，也兼容类型擦除包装；Model 在调用处显式指定。
template <class Model, class Fit, class Refit, class Res>
RansacReport<Model> loransac(int n, int min_samples, const Fit& fit, const Refit& refit,
                             const Res& res, const RansacOptions& opt) {
    RansacReport<Model> best;
    best.score = 1e300;
    if (n < min_samples) return best;

    const double thr2 = opt.max_error * opt.max_error;
    std::mt19937 rng(opt.seed);
    std::uniform_int_distribution<int> uni(0, n - 1);

    // 跨试验复用掩码缓冲，胜出者与 best 交换，落败者存储重新作为临时区，避免反复分配。
    std::vector<char> inl;
    auto consider = [&](const Model& m) {
        int cnt = 0;
        double sc = scoreModel(m, n, thr2, res, inl, cnt, best.num_inliers);
        if (cnt < 0) return;  // 无法达到当前最佳内点数，提前退出
        if (cnt > best.num_inliers || (cnt == best.num_inliers && sc < best.score)) {
            best.model = m;
            best.inlier_mask.swap(inl);
            best.num_inliers = cnt;
            best.score = sc;
            best.success = cnt >= min_samples;
        }
    };

    // 局部优化最多 rounds 次或到无改善，既用于试验循环内提升内点率，也用于最终精化。
    std::vector<int> lo_idx;
    auto localOptimize = [&](int rounds) {
        for (int it = 0; it < rounds; it++) {
            lo_idx.clear();
            for (int i = 0; i < n; i++)
                if (best.inlier_mask[i]) lo_idx.push_back(i);
            if ((int)lo_idx.size() < min_samples) break;
            const int before = best.num_inliers;
            for (const Model& m : refitModels(refit, lo_idx, best.model)) consider(m);
            if (best.num_inliers <= before) break;  // 已收敛
        }
    };

    int max_trials = opt.max_num_trials;
    int trial = 0;
    std::vector<int> sample;
    sample.reserve(min_samples);
    for (; trial < max_trials; trial++) {
        // 随机抽取互不重复的最小样本
        sample.clear();
        while ((int)sample.size() < min_samples) {
            int idx = uni(rng);
            if (std::find(sample.begin(), sample.end(), idx) == sample.end()) sample.push_back(idx);
        }
        const int before = best.num_inliers;
        for (const Model& m : fit(sample)) consider(m);
        // 最佳模型一改善就做一轮局部优化，提高内点率以减少试验数并强化提前退出界限；完整优化留到循环结束。
        if (opt.lo_iters > 0 && best.num_inliers > before && best.success) localOptimize(1);

        // 按目前最佳内点率自适应停止
        if (best.num_inliers > 0) {
            double w = (double)best.num_inliers / n;
            double denom = std::log(1.0 - std::pow(w, min_samples));
            if (denom < 0) {
                double need = std::log(1.0 - opt.confidence) / denom;
                int dyn = (int)std::ceil(need);
                int cap = std::max(opt.min_num_trials, std::min(opt.max_num_trials, dyn));
                if (trial + 1 >= cap) { trial++; break; }
            }
        }
    }

    // 最终局部优化，refit 必须为有效可调用对象；没有非最小求解器时重复传入最小求解器。
    if (best.success) localOptimize(opt.lo_iters);
    best.trials = trial;
    if (opt.min_inlier_ratio > 0 && (double)best.num_inliers / n < opt.min_inlier_ratio)
        best.success = false;
    return best;
}

}  // 命名空间 sfm
