// 由已验证匹配构建逐（图像，特征）的对应图，供增量配准查找二维、三维对应及三角化扩展轨迹。
// 每图使用 CSR 偏移加连续对应数组，避免数百万特征各自分配向量及 24 字节头部，at 返回连续视图。
#pragma once

#include <cstdint>
#include <vector>

#include "sfm/core/Matches.h"

namespace sfm {

struct Correspondence {
    uint32_t image_id;
    uint32_t feature_idx;
};

// 单个（图像，特征）的连续对应区间。
struct CorrespondenceView {
    const Correspondence* first = nullptr;
    const Correspondence* last = nullptr;
    const Correspondence* begin() const { return first; }
    const Correspondence* end() const { return last; }
    size_t size() const { return (size_t)(last - first); }
    bool empty() const { return first == last; }
};

class CorrespondenceGraph {
public:
    // num_features 与 db.images 同序，保存逐图像特征数。
    void build(const MatchesDatabase& db, const std::vector<uint32_t>& num_features) {
        const size_t n = num_features.size();
        starts_.assign(n, {});
        data_.assign(n, {});
        for (size_t i = 0; i < n; i++) starts_[i].assign((size_t)num_features[i] + 1, 0);

        // 计数、前缀和、散射；计数写到下一槽，使前缀和直接成为最终偏移表。
        for (const TwoViewMatches& p : db.pairs)
            for (const FeatureMatch& m : p.matches) {
                if (m.idx1 + 1 >= starts_[p.image1].size() ||
                    m.idx2 + 1 >= starts_[p.image2].size())
                    continue;
                starts_[p.image1][m.idx1 + 1]++;
                starts_[p.image2][m.idx2 + 1]++;
            }
        std::vector<std::vector<uint32_t>> fill(n);
        for (size_t i = 0; i < n; i++) {
            std::vector<uint32_t>& s = starts_[i];
            for (size_t f = 1; f < s.size(); f++) s[f] += s[f - 1];
            data_[i].resize(s.empty() ? 0 : s.back());
            fill[i].assign(s.begin(), s.end() - (s.empty() ? 0 : 1));
        }
        for (const TwoViewMatches& p : db.pairs)
            for (const FeatureMatch& m : p.matches) {
                if (m.idx1 + 1 >= starts_[p.image1].size() ||
                    m.idx2 + 1 >= starts_[p.image2].size())
                    continue;
                data_[p.image1][fill[p.image1][m.idx1]++] = {p.image2, m.idx2};
                data_[p.image2][fill[p.image2][m.idx2]++] = {p.image1, m.idx1};
            }
    }

    CorrespondenceView at(uint32_t image, uint32_t feature) const {
        const std::vector<uint32_t>& s = starts_[image];
        const Correspondence* base = data_[image].data();
        return {base + s[feature], base + s[feature + 1]};
    }

    size_t numImages() const { return starts_.size(); }
    uint32_t numFeatures(uint32_t image) const {
        return starts_[image].empty() ? 0u : (uint32_t)(starts_[image].size() - 1);
    }

private:
    // 每图像保存 num_features+1 个偏移及其索引的对应数组。
    std::vector<std::vector<uint32_t>> starts_;
    std::vector<std::vector<Correspondence>> data_;
};

}  // 命名空间 sfm
