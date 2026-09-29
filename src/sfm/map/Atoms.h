// 并行重建互相独立的小原子分组，每原子独享局部编号子数据库与 Mapper，减少记录分配和重置范围。
// 每工作线程复用自己的 Vulkan 上下文，不跨线程共享；相机 ID 和全局初始化内参保持一致，供后续联合优化。
// 5402 图测量中串行原子阶段占建图 1137 s 中的 721 s，主要为主机工作，小 BA 不能充分利用 GPU。
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "sfm/core/Events.h"
#include "sfm/core/Features.h"
#include "sfm/core/Matches.h"
#include "sfm/core/Model.h"
#include "sfm/core/PriorSource.h"
#include "sfm/map/Mapper.h"
#include "sfm/vk/VkContext.h"
#include "sfm/core/Log.h"

namespace sfm {

struct AtomOptions {
    // 每工作线程独享上下文，默认核心数受 max_threads 限制；每上下文两个 64 MB 暂存缓冲，过多线程还会争用设备。
    int threads = 0;
    int max_threads = 8;
    // 原子内部放宽 BA 调用频率，小问题无法填满设备，后续联合优化可吸收小模型误差。
    double ba_growth = 2.0;
    // 控制原子末次精化是否严格；增长容差不能随意放宽，379 图上 rtol=1e-3、patience=2 仅省 2 s，却使旋转误差 0.286->0.672 度、AUC@10 96.1->92.5。
    bool tight_final_ba = false;
    // 原子内部寻找附加分量的种子预算；零散残余通常由上层增长处理。
    int model_trials = 4;
    // 原子主模型的种子预算与覆盖比例，0 沿用 Mapper 默认；原子部分结果还会合并增长，无需为覆盖半个原子反复试探。
    int init_trials = 8;
    double min_model_fraction = 0;
    bool verbose = true;
};

struct AtomStats {
    size_t atoms = 0;      // 输入簇数量
    size_t models = 0;     // 产生的重建模型数
    size_t registered = 0; // 按模型累加，重叠图像重复计数
    size_t empty = 0;      // 未产生模型的原子数
    int threads = 1;
    double secs = 0;
};

namespace detail {

// 原子图像局部编号 0..n-1，以及组内匹配。
struct SubDatabase {
    MatchesDatabase db;
    std::vector<FeatureSet> feats;
    std::vector<uint32_t> cam_ids;   // 逐局部图像的全局相机 ID
    std::vector<uint32_t> to_global; // 局部 ID 到数据库 ID 的映射
    RigTable rigs;                   // 局部 ID 上的 rig 表
    SequenceTable seqs;              // 局部序列表
    std::unique_ptr<RemappedPriorSource> priors;   // 局部传感器先验
};

// 预先构建逐图像匹配邻接表，避免每原子扫描全部图像对；300 原子、70 万匹配时 O(原子数×匹配数) 会成为主开销。
inline std::vector<std::vector<uint32_t>> pairAdjacency(const MatchesDatabase& db) {
    std::vector<std::vector<uint32_t>> adj(db.images.size());
    std::vector<uint32_t> deg(db.images.size(), 0);
    for (const TwoViewMatches& p : db.pairs) {
        if (p.image1 < deg.size()) deg[p.image1]++;
        if (p.image2 < deg.size()) deg[p.image2]++;
    }
    for (size_t i = 0; i < adj.size(); i++) adj[i].reserve(deg[i]);
    for (uint32_t k = 0; k < db.pairs.size(); k++) {
        const TwoViewMatches& p = db.pairs[k];
        if (p.image1 < adj.size()) adj[p.image1].push_back(k);
        if (p.image2 < adj.size()) adj[p.image2].push_back(k);
    }
    return adj;
}

// local 作为可复用全数据库临时映射，原子外为 UINT32_MAX，返回前恢复为空，使提取子数据库保持 O(原子大小)。
inline SubDatabase carveAtom(const MatchesDatabase& db, const std::vector<FeatureSet>& feats,
                             const std::vector<uint32_t>& cam_ids,
                             const std::vector<std::vector<uint32_t>>& adj,
                             const std::vector<uint32_t>& images,
                             std::vector<uint32_t>& local, const RigTable* rigs = nullptr,
                             const SequenceTable* seqs = nullptr, PriorSource* priors = nullptr) {
    SubDatabase s;
    s.to_global = images;
    std::sort(s.to_global.begin(), s.to_global.end());
    s.to_global.erase(std::unique(s.to_global.begin(), s.to_global.end()), s.to_global.end());
    for (uint32_t i = 0; i < s.to_global.size(); i++) local[s.to_global[i]] = i;

    s.db.images.resize(s.to_global.size());
    s.feats.resize(s.to_global.size());
    s.cam_ids.resize(s.to_global.size());
    for (uint32_t i = 0; i < s.to_global.size(); i++) {
        const uint32_t g = s.to_global[i];
        s.db.images[i] = db.images[g];
        // 仅复制建图所需关键点与颜色，不复制每图约半 MB、建图完全不使用的描述子。
        const FeatureSet& f = feats[g];
        FeatureSet& sf = s.feats[i];
        sf.width = f.width;
        sf.height = f.height;
        sf.extract_width = f.extract_width;
        sf.extract_height = f.extract_height;
        sf.exif_focal = f.exif_focal;
        sf.exif_camera = f.exif_camera;
        sf.keypoints = f.keypoints;
        sf.colors = f.colors;
        s.cam_ids[i] = cam_ids.empty() ? 1 : cam_ids[g];
    }
    // 仅从较小端点收集图像对，保证每对一次。
    for (uint32_t i = 0; i < s.to_global.size(); i++) {
        const uint32_t g = s.to_global[i];
        for (uint32_t k : adj[g]) {
            const TwoViewMatches& p = db.pairs[k];
            if (p.image1 != g) continue;
            if (p.image2 >= local.size() || local[p.image2] == UINT32_MAX) continue;
            s.db.pairs.push_back(p);
            s.db.pairs.back().image1 = i;
            s.db.pairs.back().image2 = local[p.image2];
        }
    }
    if (rigs) s.rigs = rigs->subset(local, s.to_global.size());
    if (seqs) s.seqs = seqs->subset(local, s.to_global.size());
    if (priors) s.priors = std::make_unique<RemappedPriorSource>(*priors, s.to_global);
    for (uint32_t g : s.to_global) local[g] = UINT32_MAX;
    return s;
}

// 将局部重建图像 ID 映回数据库 ID，相机 ID 已全局统一。
inline void toGlobalIds(Reconstruction& m, const std::vector<uint32_t>& to_global) {
    std::map<uint32_t, Image> images;
    for (auto& kv : m.images) {
        if (kv.first >= to_global.size()) continue;
        Image im = std::move(kv.second);
        im.id = to_global[kv.first];
        images[im.id] = std::move(im);
    }
    m.images = std::move(images);
    for (auto& kv : m.points3D)
        for (TrackElement& e : kv.second.track)
            if (e.image_id < to_global.size()) e.image_id = to_global[e.image_id];
}

}  // 命名空间 detail

// 按原子顺序返回模型；base 提供运行配置，seed_mapper 仅提供全局初始化内参。
inline std::vector<Reconstruction> reconstructAtoms(
    const MatchesDatabase& db, const std::vector<FeatureSet>& feats,
    const MapperOptions& base, const std::vector<uint32_t>& cam_ids,
    const std::map<uint32_t, Camera>& start_cams,
    const std::vector<std::vector<uint32_t>>& atoms, const AtomOptions& opt, AtomStats& st,
    const RigTable* rigs = nullptr, const SequenceTable* seqs = nullptr,
    PriorSource* priors = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    st.atoms = atoms.size();

    MapperOptions mo = base;
    mo.verbose = false;
    mo.threads = 1;  // 原子层已并行，禁用嵌套并行
    // 局部原子快照不能冒充全局模型，完成后统一映回数据库 ID 报告。
    mo.report_progress = false;
    mo.ba_growth_ratio = std::max(1.0 + 1e-9, opt.ba_growth);
    mo.ba_final_tight = opt.tight_final_ba;
    // 原子求解均为中间结果，每工作线程共用一个粗阶段标量配置与上下文。
    if (!opt.tight_final_ba) mo.ba_real = mo.ba_real_coarse;
    mo.max_model_trials = opt.model_trials;
    // 焦距已在全数据库确定，禁止原子用少量图像重搜并重复试探重建（D48）。
    mo.focal_trials = 0;
    if (opt.init_trials > 0) mo.max_init_trials = opt.init_trials;
    if (opt.min_model_fraction > 0) mo.min_model_fraction = opt.min_model_fraction;
    mo.initial_cameras = start_cams;
    for (const auto& kv : start_cams) mo.known_focal_cameras.insert(kv.first);

    const std::vector<std::vector<uint32_t>> adj = detail::pairAdjacency(db);

    // 显式线程数照用，默认线程数设上限以控制上下文暂存内存和设备争用。
    unsigned hc = std::thread::hardware_concurrency();
    int nt = opt.threads > 0 ? opt.threads
                             : std::min(hc > 0 ? (int)hc : 1, std::max(1, opt.max_threads));
    nt = std::max(1, std::min<int>(nt, (int)std::max<size_t>(atoms.size(), 1)));
    st.threads = nt;

    std::vector<std::vector<Reconstruction>> per_atom(atoms.size());
    std::atomic<size_t> next{0};
    std::mutex log_mu;
    std::atomic<size_t> done{0};
    // 工作线程必须捕获并报告设备失败，不能让异常跨线程触发 terminate。
    std::mutex err_mu;
    std::exception_ptr first_error;

    auto worker = [&] {
        try {
            // 每线程创建一次 Vulkan 上下文并复用于所有原子，设备初始化比单原子重建更贵，且不能跨线程共享。
            VkContext ctx;
            std::vector<uint32_t> local(db.images.size(), UINT32_MAX);
            for (size_t i = next++; i < atoms.size(); i = next++) {
                detail::SubDatabase sub =
                    detail::carveAtom(db, feats, cam_ids, adj, atoms[i], local, rigs, seqs, priors);
                Mapper m(sub.db, sub.feats, mo, sub.cam_ids, rigs ? &sub.rigs : nullptr,
                         seqs ? &sub.seqs : nullptr, sub.priors.get());
                m.useBaContext(&ctx);
                uint32_t reg = 0;
                for (Reconstruction& r : m.run()) {
                    if (r.numRegistered() < 2) continue;
                    detail::toGlobalIds(r, sub.to_global);
                    reg += r.numRegistered();
                    for (const auto& kv : r.images)
                        if (kv.second.registered) events::map_placed(kv.first);
                    per_atom[i].push_back(std::move(r));
                }
                if (opt.verbose) {
                    const size_t n = ++done;
                    std::lock_guard<std::mutex> lk(log_mu);
                    slog::diag(slog::Tag::Map,
                               "[bup] atom %zu/%zu: %zu images -> %zu model(s), %u registered",
                               n, atoms.size(), atoms[i].size(), per_atom[i].size(), reg);
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(err_mu);
            if (!first_error) first_error = std::current_exception();
            next = atoms.size();  // 阻止其他线程继续领取任务
        }
    };
    if (nt == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nt);
        for (int t = 0; t < nt; t++) pool.emplace_back(worker);
        for (std::thread& t : pool) t.join();
    }
    if (first_error) std::rethrow_exception(first_error);

    std::vector<Reconstruction> models;
    for (std::vector<Reconstruction>& v : per_atom) {
        if (v.empty()) st.empty++;
        for (Reconstruction& r : v) {
            st.registered += r.numRegistered();
            models.push_back(std::move(r));
        }
    }
    st.models = models.size();
    st.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return models;
}

}  // 命名空间 sfm
