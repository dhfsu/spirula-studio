// 图像序列按用户声明的文件名顺序排列，建图优先信任相邻对应（D79）。
// 成员按 rig 相同的前缀规则定义，同相对路径共享一个序列位置，使多镜头属于同一时序。
#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "sfm/core/Rig.h"

namespace sfm {

constexpr int32_t kNoSequence = -1;

struct SequenceDef {
    std::vector<std::string> members;   // 路径前缀，空值覆盖整个数据集
};

// --sequence PREFIX[,PREFIX...]；点号表示图像根目录。
inline std::string parseSequenceArg(const std::string& arg, SequenceDef& out) {
    out = SequenceDef{};
    size_t i = 0;
    while (i <= arg.size()) {
        size_t c = arg.find(',', i);
        if (c == std::string::npos) c = arg.size();
        std::string p = arg.substr(i, c - i);
        while (!p.empty() && (p.back() == '/' || p.back() == '\\')) p.pop_back();
        if (p == ".") p.clear();
        out.members.push_back(p);
        if (c == arg.size()) break;
        i = c + 1;
    }
    if (out.members.empty()) return "--sequence '" + arg + "': empty";
    return {};
}

// 逐图像保存所属序列、成员及相对路径在序列不同路径中的排序位置。
struct SequenceTable {
    std::vector<int32_t> seq;      // 不属于任何序列时为 kNoSequence
    std::vector<int32_t> pos;
    std::vector<int32_t> member;
    std::vector<uint32_t> length;  // 每序列的位置数量
    std::vector<std::vector<std::string>> members;

    bool empty() const { return length.empty(); }
    bool has(uint32_t img) const { return img < seq.size() && seq[img] != kNoSequence; }
    int32_t sequenceOf(uint32_t img) const { return has(img) ? seq[img] : kNoSequence; }

    // 同序列中的位置距离，不同序列返回 INT_MAX。
    int distance(uint32_t a, uint32_t b) const {
        if (!has(a) || !has(b) || seq[a] != seq[b]) return INT_MAX;
        return std::abs(pos[a] - pos[b]);
    }
    bool nearby(uint32_t a, uint32_t b, int window) const { return distance(a, b) <= window; }

    // 映射到原子子数据库时保留原位置，使时间窗口含义不变。
    SequenceTable subset(const std::vector<uint32_t>& local, size_t num_local) const {
        SequenceTable out;
        out.length = length;
        out.members = members;
        out.seq.assign(num_local, kNoSequence);
        out.pos.assign(num_local, 0);
        out.member.assign(num_local, 0);
        for (uint32_t g = 0; g < seq.size() && g < local.size(); g++) {
            if (local[g] == kNoImage || local[g] >= num_local) continue;
            out.seq[local[g]] = seq[g];
            out.pos[local[g]] = pos[g];
            out.member[local[g]] = member[g];
        }
        return out;
    }
};

// 按图像名解析序列，最长成员前缀优先；图像重复归属或成员无匹配图像时报错。
inline SequenceTable buildSequenceTable(const std::vector<std::string>& names,
                                        const std::vector<SequenceDef>& defs) {
    SequenceTable out;
    out.seq.assign(names.size(), kNoSequence);
    out.pos.assign(names.size(), 0);
    out.member.assign(names.size(), 0);
    for (const SequenceDef& d : defs) {
        const int32_t id = (int32_t)out.length.size();
        if (d.members.empty()) throw std::runtime_error("sequence: no members");
        std::vector<std::pair<std::string, uint32_t>> keyed;
        std::vector<size_t> used(d.members.size(), 0);
        for (uint32_t img = 0; img < names.size(); img++) {
            int best = -1;
            size_t best_len = 0;
            for (size_t m = 0; m < d.members.size(); m++) {
                if (!rig_detail::prefixMatches(names[img], d.members[m])) continue;
                if (best < 0 || d.members[m].size() > best_len) {
                    best = (int)m;
                    best_len = d.members[m].size();
                }
            }
            if (best < 0) continue;
            if (out.seq[img] != kNoSequence)
                throw std::runtime_error("image " + names[img] + " is in two sequences");
            out.seq[img] = id;
            out.member[img] = best;
            used[best]++;
            keyed.emplace_back(rig_detail::frameKey(names[img], d.members[best]), img);
        }
        for (size_t m = 0; m < d.members.size(); m++)
            if (!used[m])
                throw std::runtime_error("sequence: no image matches member '" + d.members[m] +
                                         "'");
        std::sort(keyed.begin(), keyed.end());
        int32_t p = -1;
        for (size_t k = 0; k < keyed.size(); k++) {
            if (k == 0 || keyed[k].first != keyed[k - 1].first) p++;
            out.pos[keyed[k].second] = p;
        }
        out.length.push_back((uint32_t)(p + 1));
        out.members.push_back(d.members);
    }
    return out;
}

}  // 命名空间 sfm
