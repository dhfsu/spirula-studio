#pragma once

// 加权无向图的谱划分：连通分量、基于 Fiedler 向量的归一化割及递归二分。
// 仅依赖标准库，供 SfM 视图图、数据集共视图及其他重建图共用。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace spirula {
namespace graph {

// CSR 格式的加权无向图。
struct WeightedGraph {
    std::vector<uint32_t> offs;   // n+1
    std::vector<uint32_t> adj;    // 相邻节点
    std::vector<double> w;        // 与 adj 一一对应
    size_t n() const { return offs.empty() ? 0 : offs.size() - 1; }
    double degree(uint32_t i) const {
        double d = 0;
        for (uint32_t k = offs[i]; k < offs[i + 1]; k++) d += w[k];
        return d;
    }
};

struct Edge {
    uint32_t a = 0, b = 0;
    double w = 0;
};

// 由边列表构造对称 CSR；重复边权相加，忽略自环与超出 n 的节点。
inline WeightedGraph build_graph(size_t n, const std::vector<Edge>& edges) {
    WeightedGraph g;
    g.offs.assign(n + 1, 0);
    for (const Edge& e : edges) {
        if (e.a == e.b || e.a >= n || e.b >= n || !(e.w > 0)) continue;
        g.offs[e.a + 1]++;
        g.offs[e.b + 1]++;
    }
    for (size_t i = 1; i <= n; i++) g.offs[i] += g.offs[i - 1];
    g.adj.resize(g.offs[n]);
    g.w.resize(g.offs[n]);
    std::vector<uint32_t> fill(g.offs.begin(), g.offs.end() - 1);
    for (const Edge& e : edges) {
        if (e.a == e.b || e.a >= n || e.b >= n || !(e.w > 0)) continue;
        g.adj[fill[e.a]] = e.b;
        g.w[fill[e.a]++] = e.w;
        g.adj[fill[e.b]] = e.a;
        g.w[fill[e.b]++] = e.w;
    }
    return g;
}

// 累加节点对权重，避免为共享点的所有图像对生成二次规模的临时边列表。
class EdgeAccumulator {
public:
    void add(uint32_t a, uint32_t b, double w) {
        if (a == b) return;
        if (a > b) std::swap(a, b);
        _acc[((uint64_t)a << 32) | b] += w;
    }
    void reserve(size_t n) { _acc.reserve(n); }
    size_t size() const { return _acc.size(); }
    WeightedGraph build(size_t n) const {
        std::vector<Edge> edges;
        edges.reserve(_acc.size());
        for (const auto& kv : _acc)
            edges.push_back({(uint32_t)(kv.first >> 32), (uint32_t)(kv.first & 0xffffffffu),
                             kv.second});
        // 哈希遍历顺序跨运行不稳定，图划分结果必须稳定。
        std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) {
            return x.a != y.a ? x.a < y.a : x.b < y.b;
        });
        return build_graph(n, edges);
    }

private:
    std::unordered_map<uint64_t, double> _acc;
};

// nodes 诱导子图的连通分量，按大小降序排列。
inline std::vector<std::vector<uint32_t>> connected_components(
    const WeightedGraph& g, const std::vector<uint32_t>& nodes) {
    std::vector<char> inside(g.n(), 0), seen(g.n(), 0);
    for (uint32_t v : nodes) inside[v] = 1;
    std::vector<std::vector<uint32_t>> out;
    std::vector<uint32_t> stack;
    for (uint32_t s : nodes) {
        if (seen[s]) continue;
        std::vector<uint32_t> comp;
        stack.assign(1, s);
        seen[s] = 1;
        while (!stack.empty()) {
            const uint32_t v = stack.back();
            stack.pop_back();
            comp.push_back(v);
            for (uint32_t k = g.offs[v]; k < g.offs[v + 1]; k++) {
                const uint32_t u = g.adj[k];
                if (inside[u] && !seen[u]) { seen[u] = 1; stack.push_back(u); }
            }
        }
        out.push_back(std::move(comp));
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
                         return a.size() > b.size();
                     });
    return out;
}

namespace detail {

// 对 M = D^-1/2 W D^-1/2 做幂迭代，并消去主特征向量 D^1/2 * 1，得到归一化拉普拉斯的 Fiedler 向量。
// 输出按 nodes 顺序排列，未收敛时为空。
inline std::vector<double> fiedler(const WeightedGraph& g, const std::vector<uint32_t>& nodes,
                                   const std::vector<uint32_t>& local_of, int iters = 300) {
    const size_t m = nodes.size();
    std::vector<double> deg(m, 0.0);
    for (size_t i = 0; i < m; i++) {
        const uint32_t v = nodes[i];
        for (uint32_t k = g.offs[v]; k < g.offs[v + 1]; k++)
            if (local_of[g.adj[k]] != UINT32_MAX) deg[i] += g.w[k];
    }
    std::vector<double> isq(m);
    for (size_t i = 0; i < m; i++) isq[i] = deg[i] > 0 ? 1.0 / std::sqrt(deg[i]) : 0.0;
    std::vector<double> v0(m);
    double n0 = 0;
    for (size_t i = 0; i < m; i++) { v0[i] = std::sqrt(deg[i]); n0 += v0[i] * v0[i]; }
    if (n0 <= 0) return {};
    n0 = std::sqrt(n0);
    for (double& x : v0) x /= n0;

    // 交替符号初值避免常量初值偶然与 Fiedler 向量正交的问题。
    std::vector<double> x(m), y(m);
    for (size_t i = 0; i < m; i++) x[i] = (i % 2 ? -1.0 : 1.0) + 1e-3 * (double)(i % 7);
    auto orthonormalize = [&](std::vector<double>& z) {
        double dot = 0;
        for (size_t i = 0; i < m; i++) dot += z[i] * v0[i];
        double nz = 0;
        for (size_t i = 0; i < m; i++) { z[i] -= dot * v0[i]; nz += z[i] * z[i]; }
        return std::sqrt(nz);
    };
    if (orthonormalize(x) <= 0) return {};
    {
        double nz = 0;
        for (double q : x) nz += q * q;
        nz = std::sqrt(nz);
        for (double& q : x) q /= nz;
    }
    // (M + I)/2 使目标特征向量成为正算子的主模态，避免迭代收敛到最负特征值。
    for (int it = 0; it < iters; it++) {
        std::fill(y.begin(), y.end(), 0.0);
        for (size_t i = 0; i < m; i++) {
            const uint32_t v = nodes[i];
            double acc = 0;
            for (uint32_t k = g.offs[v]; k < g.offs[v + 1]; k++) {
                const uint32_t j = local_of[g.adj[k]];
                if (j != UINT32_MAX) acc += g.w[k] * isq[j] * x[j];
            }
            y[i] = 0.5 * (isq[i] * acc + x[i]);
        }
        double nz = orthonormalize(y);
        if (nz <= 1e-300) return {};
        for (size_t i = 0; i < m; i++) y[i] /= nz;
        double diff = 0;
        for (size_t i = 0; i < m; i++) diff += std::fabs(y[i] - x[i]);
        x.swap(y);
        if (it > 10 && diff < 1e-7 * (double)m) break;
    }
    for (size_t i = 0; i < m; i++) x[i] *= isq[i];
    return x;
}

}  // 命名空间 detail

struct BisectOptions {
    size_t overlap = 0;    // 每侧从另一侧借入的节点数
    size_t min_part = 1;   // 任一侧小于此值则不接受切分
    // 可选逐节点代价与图节点对应，扫描按总代价平衡；空指针表示每节点代价为 1。
    const double* cost = nullptr;
    // 每侧至少占总代价的此比例；0 允许剥离连接较弱的小团，适合 SfM 原子分组，但不适合要求均衡大小的分区。
    double balance = 0.0;
};

// 沿归一化割二分 nodes，再为每侧加入另一侧连接最强的 overlap 个节点；不值得切分时返回空结果。
inline std::vector<std::vector<uint32_t>> bisect(const WeightedGraph& g,
                                                 const std::vector<uint32_t>& nodes,
                                                 const BisectOptions& opt) {
    if (nodes.size() < 2 * std::max<size_t>(1, opt.min_part)) return {};
    std::vector<uint32_t> local_of(g.n(), UINT32_MAX);
    for (size_t i = 0; i < nodes.size(); i++) local_of[nodes[i]] = (uint32_t)i;

    std::vector<double> f = detail::fiedler(g, nodes, local_of);
    std::vector<uint32_t> order(nodes.size());
    std::iota(order.begin(), order.end(), 0u);
    // 不连通或退化导致缺少 Fiedler 向量时采用输入顺序；视频数据通常即拍摄顺序。
    if (!f.empty())
        std::stable_sort(order.begin(), order.end(),
                         [&](uint32_t a, uint32_t b) { return f[a] < f[b]; });

    // 沿排序扫描 cut(S) * (1/vol(S) + 1/vol(V\S))，增量和使复杂度与边数线性；平衡项优先使用调用方提供的处理代价。
    const size_t m = nodes.size();
    std::vector<uint32_t> rank(m);
    for (size_t i = 0; i < m; i++) rank[order[i]] = (uint32_t)i;
    double total_vol = 0;
    std::vector<double> deg(m, 0.0);
    for (size_t i = 0; i < m; i++) {
        const uint32_t v = nodes[i];
        for (uint32_t k = g.offs[v]; k < g.offs[v + 1]; k++)
            if (local_of[g.adj[k]] != UINT32_MAX) deg[i] += g.w[k];
        total_vol += opt.cost ? std::max(opt.cost[v], 1e-9) : deg[i];
    }
    double vol = 0, cut = 0, best = 1e300;
    size_t best_split = 0;
    const size_t min_part = std::max<size_t>(1, opt.min_part);
    for (size_t i = 0; i + 1 < m; i++) {
        const uint32_t li = order[i];
        const uint32_t v = nodes[li];
        vol += opt.cost ? std::max(opt.cost[v], 1e-9) : deg[li];
        for (uint32_t k = g.offs[v]; k < g.offs[v + 1]; k++) {
            const uint32_t j = local_of[g.adj[k]];
            if (j == UINT32_MAX) continue;
            cut += rank[j] <= i ? -g.w[k] : g.w[k];
        }
        if (i + 1 < min_part || m - i - 1 < min_part) continue;
        const double other = total_vol - vol;
        if (vol <= 0 || other <= 0) continue;
        if (opt.balance > 0 && (vol < opt.balance * total_vol || other < opt.balance * total_vol))
            continue;
        const double score = cut * (1.0 / vol + 1.0 / other);
        if (score < best) { best = score; best_split = i + 1; }
    }
    if (best_split == 0) return {};

    std::vector<std::vector<uint32_t>> parts(2);
    std::vector<char> side(m, 1);
    for (size_t i = 0; i < m; i++) {
        const bool left = rank[i] < best_split;
        side[i] = left ? 0 : 1;
        parts[left ? 0 : 1].push_back(nodes[i]);
    }
    if (opt.overlap == 0) return parts;

    // 借入节点数必须受限，使子分区严格小于父分区，避免偏斜切分叠加重叠后递归无法结束。
    for (int s = 0; s < 2; s++) {
        const size_t room = m - 1 - std::min(m - 1, parts[s].size());
        std::vector<std::pair<double, uint32_t>> cross;
        for (size_t i = 0; i < m; i++) {
            if (side[i] == s) continue;
            double link = 0;
            const uint32_t v = nodes[i];
            for (uint32_t k = g.offs[v]; k < g.offs[v + 1]; k++) {
                const uint32_t j = local_of[g.adj[k]];
                if (j != UINT32_MAX && side[j] == s) link += g.w[k];
            }
            if (link > 0) cross.emplace_back(link, v);
        }
        std::sort(cross.begin(), cross.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        for (size_t k = 0; k < cross.size() && k < std::min(opt.overlap, room); k++)
            parts[s].push_back(cross[k].second);
        std::sort(parts[s].begin(), parts[s].end());
    }
    return parts;
}

struct RecursiveBisectOptions {
    size_t leaf_max = 160;   // 持续切分，直到每部分不超过此大小
    size_t overlap = 30;     // 每部分从另一侧借入的图像数
    size_t min_part = 20;    // 小于此值的分区不值得单独重建
};

// 递归二分直到叶节点数不超过 leaf_max，不计重叠；先分离连通分量，再进行有意义的内部切分。
inline std::vector<std::vector<uint32_t>> recursive_bisect(const WeightedGraph& g,
                                                           const RecursiveBisectOptions& opt) {
    std::vector<uint32_t> all(g.n());
    std::iota(all.begin(), all.end(), 0u);
    std::vector<std::vector<uint32_t>> queue = connected_components(g, all), leaves;
    const BisectOptions bo{opt.overlap, opt.min_part, nullptr};
    while (!queue.empty()) {
        std::vector<uint32_t> part = std::move(queue.back());
        queue.pop_back();
        // 叶大小检查不计借入重叠，否则每次借入都会再次触发切分。
        if (part.size() <= opt.leaf_max + opt.overlap) {
            if (part.size() >= 2) leaves.push_back(std::move(part));
            continue;
        }
        std::vector<std::vector<uint32_t>> halves = bisect(g, part, bo);
        // bisect 已限制借入数，仍检查子分区是否等于父分区，防止异常导致死循环。
        if (halves.size() != 2 || halves[0].size() >= part.size() ||
            halves[1].size() >= part.size()) {
            leaves.push_back(std::move(part));
            continue;
        }
        queue.push_back(std::move(halves[0]));
        queue.push_back(std::move(halves[1]));
    }
    std::stable_sort(leaves.begin(), leaves.end(),
                     [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
                         return a.size() > b.size();
                     });
    return leaves;
}

struct LabelCutOptions {
    // 大于 0 时持续二分最重分区，直到达到指定分区数；否则限制各部分代价不超过 leaf_max。
    size_t parts = 0;
    double leaf_max = 400;
    size_t min_part = 2;
    const double* cost = nullptr;   // 逐节点代价；空指针表示均为 1
    double balance = 0.3;           // 平衡约束见 BisectOptions::balance
    // 完成后节点数不足的部分并入连接最强的分区。
    size_t min_final = 2;
    int refine_passes = 3;
};

namespace detail {

// 累计 v 到邻居各标签的边权；own 保存同标签内部的权重。
inline void label_links(const WeightedGraph& g, const std::vector<int32_t>& label, uint32_t v,
                        std::vector<double>& into) {
    for (uint32_t k = g.offs[v]; k < g.offs[v + 1]; k++) {
        const int32_t l = label[g.adj[k]];
        if (l >= 0) into[(size_t)l] += g.w[k];
    }
}

// 若零散连通块占比不超过 max_share，则将最大连通块以外的部分移至连接最强的标签；真正分成两大块的分区留待后续切分。
inline void reconnect_parts(const WeightedGraph& g, std::vector<int32_t>& label, int n_labels,
                            double max_share = 1.0) {
    for (int l = 0; l < n_labels; l++) {
        std::vector<uint32_t> nodes;
        for (uint32_t v = 0; v < g.n(); v++)
            if (label[v] == l) nodes.push_back(v);
        std::vector<std::vector<uint32_t>> comps = connected_components(g, nodes);
        for (size_t c = 1; c < comps.size(); c++) {
            if ((double)comps[c].size() > max_share * (double)nodes.size()) continue;
            std::vector<double> into((size_t)n_labels, 0.0);
            for (uint32_t v : comps[c]) label_links(g, label, v, into);
            into[(size_t)l] = 0;
            int best = -1;
            for (int j = 0; j < n_labels; j++)
                if (into[(size_t)j] > 0 && (best < 0 || into[(size_t)j] > into[(size_t)best])) best = j;
            if (best < 0) continue;
            for (uint32_t v : comps[c]) label[v] = best;
        }
    }
}

}  // 命名空间 detail

// 为全部节点生成互斥且连续的分区标签，按代价降序排列；过小的独立连通分量保留自己的标签。
inline std::vector<int32_t> cut_labels(const WeightedGraph& g, const LabelCutOptions& opt) {
    auto cost_of = [&](const std::vector<uint32_t>& part) {
        double c = 0;
        for (uint32_t v : part) c += opt.cost ? std::max(opt.cost[v], 0.0) : 1.0;
        return c;
    };
    std::vector<uint32_t> all(g.n());
    std::iota(all.begin(), all.end(), 0u);
    std::vector<std::vector<uint32_t>> parts = connected_components(g, all);
    std::vector<char> final_(parts.size(), 0);
    const BisectOptions bo{0, opt.min_part, opt.cost, opt.balance};
    for (;;) {
        // 下一次切分仍可二分且代价最大的部分。
        int pick = -1;
        double pick_cost = -1;
        for (size_t i = 0; i < parts.size(); i++) {
            if (final_[i]) continue;
            const double c = cost_of(parts[i]);
            const bool wants = opt.parts > 0 ? parts.size() < opt.parts : c > opt.leaf_max;
            if (!wants) continue;
            if (c > pick_cost) { pick_cost = c; pick = (int)i; }
        }
        if (pick < 0) break;
        std::vector<std::vector<uint32_t>> halves = bisect(g, parts[(size_t)pick], bo);
        if (halves.size() != 2 || halves[0].empty() || halves[1].empty()) {
            final_[(size_t)pick] = 1;
            continue;
        }
        // 切分后零散块移至连接更强的另一侧，仅保留最大连通块；两侧仍包含于父分区，保证递归缩小。
        for (int side = 0; side < 2; side++) {
            std::vector<std::vector<uint32_t>> comps = connected_components(g, halves[side]);
            if (comps.size() < 2) continue;
            halves[side] = std::move(comps[0]);
            for (size_t c = 1; c < comps.size(); c++)
                halves[1 - side].insert(halves[1 - side].end(), comps[c].begin(), comps[c].end());
            std::sort(halves[1 - side].begin(), halves[1 - side].end());
        }
        parts[(size_t)pick] = std::move(halves[0]);
        parts.push_back(std::move(halves[1]));
        final_.push_back(0);
    }
    std::vector<int32_t> label(g.n(), -1);
    int n_labels = (int)parts.size();
    for (int k = 0; k < n_labels; k++)
        for (uint32_t v : parts[(size_t)k]) label[v] = k;
    auto node_cost = [&](uint32_t v) { return opt.cost ? std::max(opt.cost[v], 0.0) : 1.0; };
    auto sizes = [&]() {
        std::vector<double> sz((size_t)n_labels, 0.0);
        for (uint32_t v = 0; v < g.n(); v++)
            if (label[v] >= 0) sz[(size_t)label[v]] += node_cost(v);
        return sz;
    };

    // 边界节点可移向连接更强的分区，同时保持大小在均值的四分之一偏差内；随后修复孤立碎片。
    for (int pass = 0; pass < opt.refine_passes; pass++) {
        std::vector<double> sz = sizes();
        double mean = 0;
        for (double x : sz) mean += x;
        mean /= std::max<size_t>(1, sz.size());
        size_t moved = 0;
        std::vector<double> into((size_t)n_labels);
        for (uint32_t v = 0; v < g.n(); v++) {
            const int32_t l = label[v];
            if (l < 0) continue;
            std::fill(into.begin(), into.end(), 0.0);
            detail::label_links(g, label, v, into);
            int best = l;
            for (int j = 0; j < n_labels; j++)
                if (into[(size_t)j] > into[(size_t)best]) best = j;
            if (best == l || into[(size_t)best] <= 1.05 * into[(size_t)l]) continue;
            if (sz[(size_t)best] + node_cost(v) > 1.25 * mean) continue;
            if (sz[(size_t)l] - node_cost(v) < 0.5 * mean) continue;
            sz[(size_t)l] -= node_cost(v);
            sz[(size_t)best] += node_cost(v);
            label[v] = best;
            moved++;
        }
        if (moved == 0) break;
        detail::reconnect_parts(g, label, n_labels, 0.25);
    }

    // 过小分区并入连接最强的邻居；孤立部分保留，由调用方决定位置。
    for (;;) {
        std::vector<double> sz = sizes();
        int tiny = -1;
        for (int k = 0; k < n_labels; k++)
            if (sz[(size_t)k] > 0 && sz[(size_t)k] < (double)opt.min_final &&
                (tiny < 0 || sz[(size_t)k] < sz[(size_t)tiny]))
                tiny = k;
        if (tiny < 0) break;
        std::vector<double> into((size_t)n_labels, 0.0);
        for (uint32_t v = 0; v < g.n(); v++)
            if (label[v] == tiny) detail::label_links(g, label, v, into);
        into[(size_t)tiny] = 0;
        int best = -1;
        for (int j = 0; j < n_labels; j++)
            if (into[(size_t)j] > 0 && (best < 0 || into[(size_t)j] > into[(size_t)best])) best = j;
        if (best < 0) break;
        for (uint32_t v = 0; v < g.n(); v++)
            if (label[v] == tiny) label[v] = best;
    }

    // 标签连续编号，代价最大的部分优先。
    std::vector<double> sz = sizes();
    std::vector<int> order((size_t)n_labels);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return sz[(size_t)a] > sz[(size_t)b]; });
    std::vector<int32_t> remap((size_t)n_labels, -1);
    int next = 0;
    for (int k : order)
        if (sz[(size_t)k] > 0) remap[(size_t)k] = next++;
    for (int32_t& l : label)
        if (l >= 0) l = remap[(size_t)l];
    return label;
}

}  // 命名空间 graph
}  // 命名空间 spirula
