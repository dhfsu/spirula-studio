// 图像配对与描述子匹配独立；此处生成穷举与时序列表，内容预筛位于 PairSelection.h。
// loop-closure 与 prefilter-sequential 可互补时序和内容筛选的遗漏。
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "sfm/core/Sequence.h"

namespace sfm {

enum class PairMode { Exhaustive, Sequential, Prefilter };

// 每图与同序列后 overlap 张配对，可选再加入 i+2^k；窗口不跨文件夹，空分组视为单序列。
inline std::vector<std::pair<uint32_t, uint32_t>> sequentialPairs(
    uint32_t n, int overlap, bool quadratic, const std::vector<uint32_t>& run = {}) {
    const uint32_t ov = overlap > 0 ? (uint32_t)overlap : 10u;
    std::map<uint32_t, std::vector<uint32_t>> seqs;
    for (uint32_t i = 0; i < n; i++) seqs[run.size() == n ? run[i] : 0].push_back(i);
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    for (const auto& kv : seqs) {
        const std::vector<uint32_t>& s = kv.second;
        for (size_t a = 0; a < s.size(); a++) {
            for (uint32_t k = 1; k <= ov && a + k < s.size(); k++)
                pairs.emplace_back(s[a], s[a + k]);
            for (uint32_t k = 0; quadratic && k < ov && k < 31; k++) {
                const size_t b = a + ((size_t)1 << k);
                if (b >= s.size()) break;
                pairs.emplace_back(s[a], s[b]);
            }
        }
    }
    for (auto& p : pairs)
        if (p.first > p.second) std::swap(p.first, p.second);
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    return pairs;
}

// 图像父目录作为序列 ID。
inline std::vector<uint32_t> folderRuns(const std::vector<std::string>& names) {
    std::map<std::string, uint32_t> ids;
    std::vector<uint32_t> run(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        const size_t slash = names[i].find_last_of('/');
        const std::string dir = slash == std::string::npos ? std::string() : names[i].substr(0, slash);
        run[i] = ids.emplace(dir, (uint32_t)ids.size()).first->second;
    }
    return run;
}

// SequenceTable 中每成员独立形成时间链，rig 跨镜头连接由伙伴匹配补充。
inline std::vector<std::pair<uint32_t, uint32_t>> sequenceWindowPairs(const SequenceTable& st,
                                                                       int overlap,
                                                                       bool quadratic) {
    const uint32_t n = (uint32_t)st.seq.size();
    std::vector<uint32_t> order;
    for (uint32_t i = 0; i < n; i++)
        if (st.has(i)) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (st.seq[a] != st.seq[b]) return st.seq[a] < st.seq[b];
        if (st.member[a] != st.member[b]) return st.member[a] < st.member[b];
        return st.pos[a] != st.pos[b] ? st.pos[a] < st.pos[b] : a < b;
    });
    // 先按序列位置重编号，使按索引遍历的 sequentialPairs 得到正确时序。
    std::vector<uint32_t> chain_of(n, UINT32_MAX);
    for (uint32_t k = 0; k < order.size(); k++) chain_of[order[k]] = k;
    std::vector<uint32_t> runs(order.size());
    std::map<std::pair<int32_t, int32_t>, uint32_t> ids;
    for (uint32_t k = 0; k < order.size(); k++)
        runs[k] = ids.emplace(std::make_pair(st.seq[order[k]], st.member[order[k]]),
                              (uint32_t)ids.size()).first->second;
    std::vector<std::pair<uint32_t, uint32_t>> pairs =
        sequentialPairs((uint32_t)order.size(), overlap, quadratic, runs);
    for (auto& p : pairs) {
        p.first = order[p.first];
        p.second = order[p.second];
        if (p.first > p.second) std::swap(p.first, p.second);
    }
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

// Exhaustive 生成全部图像对，Sequential 生成无指数连接的单序列窗口。
inline std::vector<std::pair<uint32_t, uint32_t>> generatePairs(uint32_t n, PairMode mode,
                                                                int overlap = 10) {
    if (mode != PairMode::Exhaustive) return sequentialPairs(n, overlap, false);
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    pairs.reserve((size_t)n * (n - 1) / 2);
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = i + 1; j < n; j++) pairs.emplace_back(i, j);
    return pairs;
}

}  // 命名空间 sfm
