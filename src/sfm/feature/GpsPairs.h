// GPS 为半径内拍摄的图像补充匹配，提供户外闭环候选；仅处理有位置的图像，保留原配对列表。
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "sfm/core/PriorSource.h"

namespace sfm {

// 每张有位置图像选择 radius 米内最近的 max_per_image 张，输出排序去重的 i<j 图像对；positioned 返回有位置图像数。
inline std::vector<std::pair<uint32_t, uint32_t>> gpsProximityPairs(const PriorSource& src,
                                                                    uint32_t num_images,
                                                                    double radius,
                                                                    size_t max_per_image,
                                                                    size_t* positioned = nullptr) {
    std::vector<uint32_t> ids;
    std::vector<Vec3> pos;
    for (uint32_t i = 0; i < num_images; i++) {
        Vec3 p;
        if (!src.position(i, p)) continue;
        ids.push_back(i);
        pos.push_back(p);
    }
    if (positioned) *positioned = ids.size();
    std::vector<std::pair<uint32_t, uint32_t>> out;
    if (ids.size() < 2 || !(radius > 0)) return out;
    const double r2 = radius * radius;
    std::vector<std::pair<double, uint32_t>> nearby;
    for (size_t a = 0; a < ids.size(); a++) {
        nearby.clear();
        for (size_t b = 0; b < ids.size(); b++) {
            if (a == b) continue;
            const Vec3 d = pos[a] - pos[b];
            const double dd = d.dot(d);
            if (dd <= r2) nearby.push_back({dd, ids[b]});
        }
        if (nearby.size() > max_per_image) {
            std::partial_sort(nearby.begin(), nearby.begin() + (long)max_per_image, nearby.end());
            nearby.resize(max_per_image);
        }
        for (const auto& n : nearby)
            out.emplace_back(std::min(ids[a], n.second), std::max(ids[a], n.second));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

}  // 命名空间 sfm
