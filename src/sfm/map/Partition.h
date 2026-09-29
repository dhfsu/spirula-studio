// 验证视图图以图像为节点、内点数为边权，在重建前调用共享 GraphCut 切分，不依赖三维几何。
#pragma once

#include <cstdint>
#include <vector>

#include "core/GraphCut.h"
#include "sfm/core/Matches.h"

namespace sfm {

using ViewGraph = spirula::graph::WeightedGraph;

inline ViewGraph buildViewGraph(const MatchesDatabase& db) {
    const size_t n = db.images.size();
    std::vector<spirula::graph::Edge> edges;
    edges.reserve(db.pairs.size());
    for (const TwoViewMatches& p : db.pairs) {
        if (p.matches.empty() || p.image1 >= n || p.image2 >= n) continue;
        edges.push_back({p.image1, p.image2, (double)p.matches.size()});
    }
    return spirula::graph::build_graph(n, edges);
}

inline std::vector<std::vector<uint32_t>> connectedComponents(
    const ViewGraph& g, const std::vector<uint32_t>& nodes) {
    return spirula::graph::connected_components(g, nodes);
}

struct PartitionOptions {
    size_t leaf_max_images = 160;  // 持续切分直到每部分不超过此大小
    size_t overlap = 30;           // 每部分从相邻分区借入的图像数
    size_t min_part = 20;          // 小于此大小不值得单独建模
};

inline std::vector<std::vector<uint32_t>> bisect(const ViewGraph& g,
                                                 const std::vector<uint32_t>& nodes,
                                                 const PartitionOptions& opt) {
    return spirula::graph::bisect(g, nodes, {opt.overlap, opt.min_part, nullptr});
}

inline std::vector<std::vector<uint32_t>> partitionViewGraph(const ViewGraph& g,
                                                             const PartitionOptions& opt) {
    return spirula::graph::recursive_bisect(g, {opt.leaf_max_images, opt.overlap, opt.min_part});
}

}  // 命名空间 sfm
