// 对已验证的 a/i 与 b/j，利用成员已知旋转推导其他朝向重叠的镜头对并补充匹配，避免帧间连接只依赖筛选恰好选中的镜头。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "sfm/core/Rig.h"

namespace sfm {

// 对种子成员不足四分之一的弱帧间连接，选择背离 a 的 a'，并使 a'/i 与 b'/j 在假定种子朝向一致时夹角不超过 max_angle_deg。
inline std::vector<std::pair<uint32_t, uint32_t>> rigMatePairs(
    const RigTable& rigs, const std::vector<std::pair<uint32_t, uint32_t>>& seeds,
    double max_angle_deg) {
    const double cos_max = std::cos(max_angle_deg * M_PI / 180.0);
    // implied[r][a*n+b] 保存成员对 (a,b) 推导出的伙伴成员对。
    std::vector<std::vector<std::vector<std::pair<uint32_t, uint32_t>>>> implied(rigs.rigs.size());
    for (size_t r = 0; r < rigs.rigs.size(); r++) {
        const std::vector<RigMemberDef>& mem = rigs.rigs[r].members;
        const size_t n = mem.size();
        implied[r].resize(n * n);
        // 约 190 度双鱼眼的种子可近似视为同向；90 度视图可在相差 60 度时重叠，.360 的伙伴仅 6% 验证成功。
        if (rigs.rigs[r].kind != "dual-fisheye") continue;
        for (size_t a = 0; a < n; a++)
            for (size_t b = 0; b < n; b++) {
                if (!mem[a].has_ext || !mem[b].has_ext) continue;
                // 假定 a/i 与 b/j 同向时，帧 j 的 rig 坐标在帧 i 中的旋转。
                const Mat3 H = mul(transpose(mem[a].ext.R), mem[b].ext.R);
                for (size_t a2 = 0; a2 < n; a2++)
                    for (size_t b2 = 0; b2 < n; b2++) {
                        if ((a2 == a && b2 == b) || !mem[a2].has_ext || !mem[b2].has_ext) continue;
                        const Mat3& Ra = mem[a2].ext.R;
                        // 仅补充背离种子镜头的方向；相邻镜头共享内容，通常已被原筛选找到。
                        const Mat3& Rs = mem[a].ext.R;
                        if (Ra[6] * Rs[6] + Ra[7] * Rs[7] + Ra[8] * Rs[8] > 0) continue;
                        const Mat3 Rb = mul(H, transpose(mem[b2].ext.R));
                        // 光轴为 cam_from_rig 的一行，或 rig_from_cam 的一列。
                        const double c = Ra[6] * Rb[2] + Ra[7] * Rb[5] + Ra[8] * Rb[8];
                        if (c >= cos_max) implied[r][a * n + b].push_back({(uint32_t)a2, (uint32_t)b2});
                    }
            }
    }
    auto frameKey = [](const RigSlot& a, const RigSlot& b) {
        const uint64_t lo = std::min(a.frame, b.frame), hi = std::max(a.frame, b.frame);
        return std::make_pair(a.rig, (lo << 32) | hi);
    };
    std::map<std::pair<uint32_t, uint64_t>, size_t> links;
    for (const auto& s : seeds) {
        const RigSlot si = rigs.slot(s.first), sj = rigs.slot(s.second);
        if (si.valid() && sj.valid() && si.rig == sj.rig && si.frame != sj.frame)
            links[frameKey(si, sj)]++;
    }
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (const auto& s : seeds) {
        const RigSlot si = rigs.slot(s.first), sj = rigs.slot(s.second);
        if (!si.valid() || !sj.valid() || si.rig != sj.rig || si.frame == sj.frame) continue;
        const RigSpec& spec = rigs.rigs[si.rig];
        const size_t n = spec.members.size();
        if (links[frameKey(si, sj)] > std::max<size_t>(1, n / 4)) continue;
        for (const auto& mm : implied[si.rig][si.member * n + sj.member]) {
            const uint32_t p = spec.frames[si.frame][mm.first];
            const uint32_t q = spec.frames[sj.frame][mm.second];
            if (p != kNoImage && q != kNoImage) out.emplace_back(std::min(p, q), std::max(p, q));
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    std::vector<std::pair<uint32_t, uint32_t>> sorted_seeds = seeds;
    std::sort(sorted_seeds.begin(), sorted_seeds.end());
    out.erase(std::remove_if(out.begin(), out.end(),
                             [&](const std::pair<uint32_t, uint32_t>& q) {
                                 return std::binary_search(sorted_seeds.begin(),
                                                           sorted_seeds.end(), q);
                             }),
              out.end());
    return out;
}

}  // 命名空间 sfm
