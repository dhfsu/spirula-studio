// bottom-up 将视图图切成带重叠的原子，独立并行增量重建，再进入共享装配调度。
// 全程按相机组共享内参，避免小原子自行估焦距产生几何不一致；相邻原子的共同图像为初次 Sim(3) 合并提供依据。
#pragma once

#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstdio>
#include <map>
#include <string>
#include <set>
#include <vector>

#include "sfm/core/Features.h"
#include "sfm/core/Model.h"
#include "sfm/map/Assemble.h"
#include "sfm/map/Atoms.h"
#include "sfm/map/Mapper.h"
#include "sfm/map/Merge.h"
#include "sfm/map/ModelOps.h"
#include "sfm/map/Partition.h"
#include "sfm/core/Log.h"

namespace sfm {

struct BottomUpOptions {
    // 默认原子 48 图并保留重叠；重叠 12->8 虽省约五分之一时间，却损失覆盖，且须小于 min_part 以保证递归缩小。
    // 原子增至 96 可将重复覆盖从 2.1–2.4 倍降到约 1.4 倍并快 1.2–2.1 倍，但 798 图数据出现半场景错位，1322 图由一个主模型变六片。
    // 提高共同图阈值、收紧 BA 或提前拆分均未解决，接缝检测也未可靠发现，因此保留 48，允许显式选项权衡。
    PartitionOptions partition{48, 12, 16};
    // 原子构建配置与并发线程数。
    AtomOptions atom;
    // 足够多原子时，合并前用一次共享内参联合 BA 替代大量低效小求解；原子过少时额外求解不划算，480 图测量增加约 19% 开销。
    bool joint_after_atoms = true;
    size_t joint_min_models = 32;
    bool verbose = true;
};

struct BottomUpStats {
    size_t atoms = 0;
    size_t atom_images = 0;        // 按原子累加，重叠重复计数
    size_t models_from_atoms = 0;
    int atom_threads = 1;
    double t_atoms = 0;
    // 原子之后的共享装配统计。
    AssembleStats assemble;
};

// mapper 已面向完整数据库初始化，负责原子以上处理，各原子使用独立局部 Mapper；合并和清理阈值与 flat 共用。
inline std::vector<Reconstruction> bottomUpReconstruct(Mapper& mapper, const MatchesDatabase& db,
                                                       const std::vector<FeatureSet>& feats,
                                                       const BottomUpOptions& opt,
                                                       const ManagerOptions& mopt,
                                                       const AssembleOptions& aso,
                                                       BottomUpStats& st) {
    auto clk = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

    // 在完整数据库上一次性确定初始内参，再交给每个原子，避免首个小原子决定全局焦距（D48）。
    mapper.bootstrapCameras();

    ViewGraph g = buildViewGraph(db);
    std::vector<std::vector<uint32_t>> atoms = partitionViewGraph(g, opt.partition);
    st.atoms = atoms.size();
    for (const std::vector<uint32_t>& a : atoms) st.atom_images += a.size();
    if (opt.verbose) {
        size_t smallest = SIZE_MAX, biggest = 0;
        for (const std::vector<uint32_t>& a : atoms) {
            smallest = std::min(smallest, a.size());
            biggest = std::max(biggest, a.size());
        }
        slog::diag(slog::Tag::Map,
                   "[bup] %zu image(s) -> %zu atom(s) of %zu..%zu images (%.2fx cover)",
                   db.images.size(), atoms.size(), atoms.empty() ? 0 : smallest, biggest,
                   db.images.empty() ? 0.0 : (double)st.atom_images / (double)db.images.size());
    }

    // 不足两个原子时无需分层合并，直接运行 flat。
    if (atoms.size() < 2) {
        if (opt.verbose) slog::diag(slog::Tag::Map, "[bup] one atom: reconstructing it flat");
        return mapper.run();
    }

    // ---------------- 原子重建 ----------------
    AtomStats as;
    AtomOptions ao = opt.atom;
    ao.verbose = opt.verbose;
    std::vector<Reconstruction> models =
        reconstructAtoms(db, feats, mapper.options(), mapper.cameraIds(),
                         mapper.startingCameras(), atoms, ao, as, mapper.rigs(),
                         mapper.sequences(), mapper.priors());
    st.t_atoms = as.secs;
    st.models_from_atoms = models.size();
    st.atom_threads = as.threads;
    if (opt.verbose)
        slog::diag(slog::Tag::Map,
                   "[bup] %zu atom(s) on %d thread(s) -> %zu model(s), %zu registrations, "
                   "%zu empty: %.1f s", as.atoms, as.threads, as.models, as.registered, as.empty,
                   as.secs);
    mapper.claimAll(models);
    // 全部原子失败时回到 flat，快速产生可供调用方报告失败的模型。
    if (models.empty()) return mapper.run();

    // 按相机组共享内参，将各原子放入一个 BA 问题，以充分利用设备。
    if (opt.joint_after_atoms && models.size() >= opt.joint_min_models) {
        auto t0 = clk();
        mapper.jointRefine(models, aso.coarse_joint_ba);
        st.assemble.joint_ba++;
        st.assemble.t_ba += secs(t0, clk());
        if (opt.verbose)
            slog::diag(slog::Tag::Map, "[bup] joint refinement over %zu atom model(s): %.1f s",
                       models.size(), st.assemble.t_ba);
    }

    // ---------------- 向上合并与收尾 ----------------
    AssembleOptions aopt = aso;
    aopt.verbose = opt.verbose;
    aopt.tag = "bup";
    return assembleModels(mapper, std::move(models), mopt, aopt, st.assemble);
}

}  // 命名空间 sfm
