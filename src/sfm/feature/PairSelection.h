// GPU 两级图像对筛选：先对全部 N² 对做低成本双侧子集匹配，再对短名单执行可靠的单侧子集对完整特征评分，保留逐图 top-k。
// 完整穷举按 3.5 ms/对计，千图约需半小时；粗筛约便宜 30 倍，但双侧截断削弱真实对应及距离比判据，不能直接作为最终得分。
// 两级流程将昂贵部分降为近线性规模，千图测量中筛选从 54 s 降至个位数秒；更大数据才需考虑词汇树（D35/D56）。
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "sfm/core/Features.h"
#include "sfm/feature/Matcher.h"
#include "sfm/feature/Pairing.h"

namespace sfm {

struct PairSelectionOptions {
    uint32_t num_features = 512;  // 每图查询子集的最高等级特征数 K
    // 训练侧默认保留全部以保证评分可靠，train_features=0 表示不限制。
    uint32_t train_features = 0;
    uint32_t num_neighbors = 32;  // 每图保留得分最高的 k 个伙伴
    uint32_t min_score = 4;       // 最终得分低于此值不保留
    // 粗筛限制双侧描述子数并保留 coarse_neighbors 个候选，0 表示直接对全部图像对精确评分。
    uint32_t coarse_features = 256;
    uint32_t coarse_neighbors = 128;
    uint32_t coarse_min_images = 200;
    // 评分阶段独立的 Lowe 比值，只用于排序，可比完整匹配更宽松。
    float ratio = 0.8f;
    // 评分问题约为完整匹配的 1/32，每次提交可处理更多图像对。
    int batch_pairs = 256;
    int device = -1;
    std::string device_selector;   // 规范 uuid:<hex>，空值沿用共享优先级
};

// 显式按 FeatureSet::rank 收集 top-K，平局按索引；SIFT 用尺度，无尺度检测器用分数。
// K=0 或大于总数时仍按等级排序收集全部；不能假定学习特征的 scale 非零。
inline FeatureSet topScaleSubset(const FeatureSet& f, uint32_t K) {
    if (K == 0 || K > f.count()) K = f.count();
    std::vector<uint32_t> idx(f.count());
    for (uint32_t j = 0; j < f.count(); j++) idx[j] = j;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](uint32_t a, uint32_t b) {
        if (f.rank(a) != f.rank(b)) return f.rank(a) > f.rank(b);
        return a < b;
    });
    FeatureSet m;
    m.width = f.width;
    m.height = f.height;
    m.dim = f.dim;
    m.dtype = f.dtype;
    const uint32_t dsz = f.dim * dtypeSize(f.dtype);
    m.keypoints.reserve(K);
    m.descriptors.resize((size_t)K * dsz);
    for (uint32_t k = 0; k < K; k++) {
        m.keypoints.push_back(f.keypoints[idx[k]]);
        std::memcpy(&m.descriptors[(size_t)k * dsz], &f.descriptors[(size_t)idx[k] * dsz], dsz);
    }
    return m;
}

namespace detail {

// GPU 仅统计每个有序查询、训练对通过比值测试的数量；索引空间前 n 项为查询子集，后 n 项为训练侧，避免创建大量匹配列表。
inline std::vector<uint32_t> scoreOrderedPairs(
    const std::vector<const FeatureSet*>& sets,
    const std::vector<std::pair<uint32_t, uint32_t>>& pairs, const PairSelectionOptions& opt,
    size_t progress_base, size_t progress_total,
    const std::function<void(size_t, size_t)>& progress) {
    MatchOptions mo;
    mo.device = opt.device;
    mo.device_selector = opt.device_selector;
    mo.batch_pairs = opt.batch_pairs;
    mo.max_num_matches = 0;
    mo.max_ratio = opt.ratio;
    // 完整训练侧的比值测试已具选择性，跳过交叉检查及列结果；K=256 时每对回读由 K+n_train 降至 K，约节省 33 倍。
    mo.cross_check = false;
    BruteForceMatcher matcher(mo);

    std::vector<uint32_t> score(pairs.size(), 0);
    // 每次处理约列表百分之一以持续报告进度，匹配器内部再按预算分块。
    const size_t batch = (size_t)std::max(1, opt.batch_pairs);
    const size_t chunk = std::max(batch, std::min(batch * 64, pairs.size() / 100 + 1));
    std::vector<uint32_t> out;
    for (size_t b = 0; b < pairs.size(); b += chunk) {
        const size_t e = std::min(b + chunk, pairs.size());
        matcher.countBatch(sets, pairs, b, e, out);
        for (size_t p = b; p < e; p++) score[p] = out[p - b];
        if (progress) progress(progress_base + e, progress_total);
    }
    return score;
}

// 由无序加权边取各图像 top-k 伙伴的并集。
inline std::vector<std::pair<uint32_t, uint32_t>> topPartners(
    uint32_t n, const std::vector<std::pair<uint32_t, uint32_t>>& cand,
    const std::vector<uint32_t>& edge_score, uint32_t k, uint32_t min_score) {
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> adj(n);  // （分数，伙伴索引）
    for (size_t e = 0; e < cand.size(); e++) {
        const uint32_t s = edge_score[e];
        if (s < min_score) continue;
        adj[cand[e].first].push_back({s, cand[e].second});
        adj[cand[e].second].push_back({s, cand[e].first});
    }
    std::vector<std::pair<uint32_t, uint32_t>> sel;
    for (uint32_t i = 0; i < n; i++) {
        auto& a = adj[i];
        const size_t take = std::min<size_t>(k, a.size());
        std::partial_sort(a.begin(), a.begin() + take, a.end(),
                          [](const std::pair<uint32_t, uint32_t>& x,
                             const std::pair<uint32_t, uint32_t>& y) {
                              if (x.first != y.first) return x.first > y.first;
                              return x.second < y.second;
                          });
        for (size_t t = 0; t < take; t++) {
            const uint32_t j = a[t].second;
            sel.emplace_back(std::min(i, j), std::max(i, j));
        }
    }
    std::sort(sel.begin(), sel.end());
    sel.erase(std::unique(sel.begin(), sel.end()), sel.end());
    return sel;
}

}  // 命名空间 detail

// 候选评分后输出逐图 top-k 并集，按图像对字典序排列；子集与平局均按索引确定，保证可复现。
inline std::vector<std::pair<uint32_t, uint32_t>> prefilterPairs(
    const std::vector<FeatureSet>& feats, const PairSelectionOptions& opt,
    const std::function<void(size_t, size_t)>& progress = nullptr) {
    const uint32_t n = (uint32_t)feats.size();
    if (n < 2) return {};
    const size_t all_ordered = (size_t)n * (n - 1);
    const bool coarse = opt.coarse_features > 0 && n >= opt.coarse_min_images;

    // ---------------- 候选列表 ----------------
    // 使用全部图像对，或双侧小子集粗筛保留的短名单。
    std::vector<std::pair<uint32_t, uint32_t>> cand;
    size_t done_pairs = 0;
    if (coarse) {
        std::vector<FeatureSet> mini(n);
        std::vector<const FeatureSet*> sets(2 * (size_t)n);
        for (uint32_t i = 0; i < n; i++) {
            mini[i] = topScaleSubset(feats[i], opt.coarse_features);
            sets[i] = sets[(size_t)n + i] = &mini[i];
        }
        // 粗筛两侧仅数十 KB，可全部常驻显存，每对只需一个方向。
        std::vector<std::pair<uint32_t, uint32_t>> ordered;
        ordered.reserve(all_ordered / 2);
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = i + 1; j < n; j++) ordered.emplace_back(i, (uint32_t)(n + j));
        std::vector<uint32_t> s =
            detail::scoreOrderedPairs(sets, ordered, opt, 0, all_ordered, progress);
        for (auto& p : ordered) p.second -= n;
        // min_score 仅用于最终评分，不能在粗筛提前丢弃尚未可靠评估的图像对。
        cand = detail::topPartners(n, ordered, s, opt.coarse_neighbors, 1);
        done_pairs = all_ordered / 2;
    } else {
        cand.reserve(all_ordered / 2);
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = i + 1; j < n; j++) cand.emplace_back(i, j);
    }

    // ---------------- 最终评分 ----------------
    // 查询位于 [0,n)，训练位于 [n,2n)；完整训练描述子占主要内存，无上限时直接引用调用方数据，避免 GB 级复制。
    std::vector<FeatureSet> owned(n + (opt.train_features == 0 ? 0 : (size_t)n));
    std::vector<const FeatureSet*> sets(2 * (size_t)n);
    for (uint32_t i = 0; i < n; i++) {
        owned[i] = topScaleSubset(feats[i], opt.num_features);
        sets[i] = &owned[i];
        if (opt.train_features == 0) {
            sets[(size_t)n + i] = &feats[i];
        } else {
            owned[(size_t)n + i] = topScaleSubset(feats[i], opt.train_features);
            sets[(size_t)n + i] = &owned[(size_t)n + i];
        }
    }

    // 各候选双向评分并取最大值，避免共同内容只在一侧大尺度子集中显著而漏配。
    // 按训练图优先遍历，使每个完整训练集只需流过显存一次。
    std::vector<std::pair<uint32_t, uint32_t>> ordered;
    std::vector<size_t> edge_of;  // 与各方向对应的候选索引
    ordered.reserve(2 * cand.size());
    edge_of.reserve(2 * cand.size());
    {
        std::vector<std::vector<std::pair<uint32_t, size_t>>> by_train(n);  // （查询索引，边索引）
        for (size_t e = 0; e < cand.size(); e++) {
            by_train[cand[e].second].emplace_back(cand[e].first, e);
            by_train[cand[e].first].emplace_back(cand[e].second, e);
        }
        for (uint32_t t = 0; t < n; t++)
            for (const auto& q : by_train[t]) {
                ordered.emplace_back(q.first, (uint32_t)(n + t));
                edge_of.push_back(q.second);
            }
    }
    const size_t total = coarse ? done_pairs + ordered.size() : ordered.size();
    std::vector<uint32_t> s =
        detail::scoreOrderedPairs(sets, ordered, opt, done_pairs, total, progress);

    std::vector<uint32_t> edge_score(cand.size(), 0);
    for (size_t k = 0; k < s.size(); k++)
        edge_score[edge_of[k]] = std::max(edge_score[edge_of[k]], s[k]);
    return detail::topPartners(n, cand, edge_score, opt.num_neighbors, opt.min_score);
}

}  // 命名空间 sfm
