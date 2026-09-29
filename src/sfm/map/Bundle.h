// 将 Reconstruction 装配为 BAProblem 并调用统一求解器；相机约定与求解器一致，无需额外坐标转换。
// 坐标规范保持自由，由 LM 阻尼正则化；设备上下文可由调用方持久复用。
#pragma once

#include <algorithm>
#include <atomic>
#include <map>
#include <set>
#include <vector>

#include "sfm/ba/Priors.h"
#include "sfm/ba/Problem.h"
#include "sfm/ba/Solver.h"
#include "sfm/core/Model.h"
#include "sfm/map/Profile.h"
#include "sfm/core/Log.h"

namespace sfm {

struct BundleOptions {
    RealCfg real = RealCfg::F64;
    int max_iters = 25;
    bool verbose = false;
    // 规范 uuid:<hex> 指定求解设备，空值沿用共享优先级；整数仅保留 CLI/API 输入形式。
    std::string device_selector;
    int device = -1;
    // 建图 BA 默认使用 Huber：缺少局部 BA 时，错误配准可能在过滤前扭曲小模型；阈值内二次、阈值外线性（D36）。
    std::string loss = "huber";
    float loss_param = 2.0f;
    // 增长阶段可覆盖为较松收敛容差，0 沿用求解器默认，最终精化使用默认严格设置（D38）。
    double rtol = 0;
    int patience = 0;
    // 建图期间默认固定主点；主点偏移 d 近似等效旋转 d/f，多相机组独立漂移会改变相对朝向，双鱼眼实测误差 1.0–1.8 度（D50）。
    // 完成后可在有足够共享图像时单独优化主点，以改善可观性（D51）。
    bool refine_principal_point = false;
    // 畸变默认可优化；固定畸变也会固定其后主点，以保持自由参数为前缀（D72）。
    bool refine_extra_params = true;
    // 可拆分问题的调用方可要求超设备预算时立即拒绝，随后分批重试。
    bool over_budget_throws = false;
    // 仅为图像数量达到阈值的相机组优化主点；单图组无法区分主点变化与相机旋转，0 表示全部组。
    size_t pp_min_images = 20;
    // auto 按系统维度选路径，dense 使用 Cholesky，cg 使用隐式 Schur；硬件改变交叉点，因此提供 ba-solver 显式选择。
    std::string solver = "auto";
    // 调用方持久上下文复用设备、流水线和描述符，同一上下文的 real/loss 必须固定；空值则每次创建局部上下文。
    VkContext* shared_ctx = nullptr;
    // CPU 主机线程数，0 使用 hardware_concurrency
    int threads = 0;
    // 已标定成员同帧共享一个位姿块，成员外参可精化；use_rigs 关闭时每图独占一帧。
    const RigTable* rigs = nullptr;
    bool use_rigs = true;
    bool refine_rigs = true;
    // 优化成员外参前，需足够共同观测其他成员的帧数，避免位姿与外参相互补偿。
    int rig_min_frames = 3;
    // 还需足够成员图像观测数
    int rig_min_obs = 100;
    // 先验使用重建图像 ID，问题中缺失图像的因子将被丢弃。
    const PosePriors* priors = nullptr;
};

// BA 问题及索引到原重建对象的映射，与 runGlobalBA 分离，便于独立 ba 命令复用装配。
struct BundleLayout {
    BAProblem P;
    std::vector<Image*> imgOf;      // 按 BA 图像索引
    std::vector<Point3D*> ptOf;     // 按 BA 三维点索引
    std::vector<uint32_t> camIds;   // 按相机组
    std::vector<std::pair<uint32_t, uint32_t>> memberOf;  // 每 BA 成员对应（rig，成员）
    // P.priors 须在返回值地址稳定后由 attachPriors 指向本对象内先验。
    PosePriors priors;
    void attachPriors() { P.priors = priors.empty() ? nullptr : &priors; }
};

namespace bundle_detail {

// 已标定 rig 图像使用（rig，frame）位姿块，其余使用（kNoRig，图像 ID）。
struct FrameKey {
    uint32_t rig, frame;
    bool operator<(const FrameKey& o) const {
        return rig != o.rig ? rig < o.rig : frame < o.frame;
    }
    bool operator==(const FrameKey& o) const { return rig == o.rig && frame == o.frame; }
};

inline FrameKey frameKeyOf(const Reconstruction& rec, const RigTable* rigs, bool use,
                           uint32_t image_id) {
    if (rigs && use && !rec.rig_detached.count(image_id)) {
        const RigSlot sl = rigs->slot(image_id);
        if (sl.valid() && sl.rig < rec.rigs.size() && rec.rigs[sl.rig].usable(sl.member))
            return {sl.rig, sl.frame};
    }
    return {kNoRig, image_id};
}

inline void packPose(const Pose& p, double* out) {
    const Vec3 aa = rotationToAngleAxis(p.R);
    out[0] = aa.x; out[1] = aa.y; out[2] = aa.z;
    out[3] = p.t.x; out[4] = p.t.y; out[5] = p.t.z;
}

inline Pose unpackPose(const double* v) {
    return {angleAxisToRotation({v[0], v[1], v[2]}), {v[3], v[4], v[5]}};
}

}  // 命名空间 bundle_detail

// 将重建装配为 BAProblem，不足两图像时返回空布局。
inline BundleLayout buildBundle(Reconstruction& rec, const BundleOptions& bopt) {
    // 使用平铺 ID 到稠密 BA 索引映射，避免数百万观测逐项树查找使问题构建成为主要开销。
    BundleLayout L;
    std::vector<uint32_t> imgIds;
    std::vector<Image*>& imgOf = L.imgOf;  // 按 BA 索引
    uint32_t max_img_id = 0;
    for (auto& kv : rec.images) max_img_id = std::max(max_img_id, kv.first);
    std::vector<uint32_t> imgBA(max_img_id + 1, UINT32_MAX);
    // 图像按帧排序，使 rig 帧连续，CPU 求解器依赖此约束。
    using bundle_detail::FrameKey;
    const RigTable* rigs = bopt.use_rigs ? bopt.rigs : nullptr;
    std::vector<std::pair<FrameKey, uint32_t>> order;
    for (auto& kv : rec.images)
        if (kv.second.registered)
            order.push_back({bundle_detail::frameKeyOf(rec, rigs, true, kv.first), kv.first});
    std::stable_sort(order.begin(), order.end(),
                     [](const std::pair<FrameKey, uint32_t>& a,
                        const std::pair<FrameKey, uint32_t>& b) { return a.first < b.first; });
    for (const auto& o : order) {
        imgBA[o.second] = (uint32_t)imgIds.size();
        imgIds.push_back(o.second);
        imgOf.push_back(&rec.images.at(o.second));
    }
    std::vector<uint64_t> ptIds;
    std::vector<Point3D*>& ptOf = L.ptOf;  // 按 BA 索引
    for (auto& kv : rec.points3D) {
        if (kv.second.track.size() < 2) continue;
        ptIds.push_back(kv.first);
        ptOf.push_back(&kv.second);
    }
    if (imgIds.size() < 2 || ptIds.empty()) return BundleLayout{};

    // 每个实际使用的相机 ID 对应一个内参组。
    std::vector<uint32_t>& camIds = L.camIds;
    std::map<uint32_t, uint32_t> camGroup;
    for (Image* im : imgOf) {
        uint32_t cid = im->camera_id;
        if (!camGroup.count(cid)) {
            camGroup[cid] = (uint32_t)camIds.size();
            camIds.push_back(cid);
        }
    }

    BAProblem& P = L.P;
    P.num_images = (uint32_t)imgIds.size();
    P.num_points = (uint32_t)ptIds.size();

    // 按点顺序输出观测，仅需在各短轨迹内按图像排序，避免对全部观测做全局排序。
    struct Obs { uint32_t img, pt; double x, y; };
    std::vector<Obs> obs;
    obs.reserve((size_t)P.num_points * 3);
    P.obs_ranges.assign(P.num_points + 1, 0);
    for (uint32_t p = 0; p < P.num_points; p++) {
        const size_t start = obs.size();
        for (const TrackElement& e : ptOf[p]->track) {
            if (e.image_id > max_img_id) continue;
            const uint32_t bi = imgBA[e.image_id];
            if (bi == UINT32_MAX) continue;
            const Vec2& xy = imgOf[bi]->points2D[e.point2D_idx];
            obs.push_back({bi, p, xy.x, xy.y});
        }
        std::sort(obs.begin() + start, obs.end(),
                  [](const Obs& a, const Obs& b) { return a.img < b.img; });
        P.obs_ranges[p + 1] = (uint32_t)obs.size();
    }
    P.num_obs = (uint32_t)obs.size();
    P.obs_image.resize(P.num_obs);
    P.obs_point.resize(P.num_obs);
    P.obs_xy.resize(2 * P.num_obs);
    for (uint32_t i = 0; i < P.num_obs; i++) {
        P.obs_image[i] = obs[i].img;
        P.obs_point[i] = obs[i].pt;
        P.obs_xy[2 * i] = obs[i].x;
        P.obs_xy[2 * i + 1] = obs[i].y;
    }

    // rig 帧初始位姿取观测最多的成员图像，其余按标定对齐并由 BA 协调；普通图像独占帧。
    P.image_frame.assign(P.num_images, 0);
    P.image_member.assign(P.num_images, kNoMember);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> memberBA;  // （rig，成员）到索引的映射
    std::vector<uint32_t> memberCo;  // 每 BA 成员与其他成员共同出现的帧数
    std::vector<std::vector<uint32_t>> frameImgs;  // 每 BA 帧包含的图像
    for (uint32_t i = 0; i < P.num_images; i++) {
        const FrameKey key = order[i].first;
        if (i == 0 || !(order[i - 1].first == key)) frameImgs.emplace_back();
        P.image_frame[i] = (uint32_t)frameImgs.size() - 1;
        frameImgs.back().push_back(i);
        if (key.rig == kNoRig) continue;
        const RigSlot sl = rigs->slot(imgIds[i]);
        auto it = memberBA.find({sl.rig, sl.member});
        if (it == memberBA.end()) {
            it = memberBA.emplace(std::make_pair(sl.rig, sl.member), (uint32_t)L.memberOf.size()).first;
            L.memberOf.push_back({sl.rig, sl.member});
            memberCo.push_back(0);
        }
        P.image_member[i] = it->second;
    }
    P.num_frames = (uint32_t)frameImgs.size();
    for (const std::vector<uint32_t>& fi : frameImgs)
        if (fi.size() > 1)
            for (uint32_t i : fi) memberCo[P.image_member[i]]++;

    P.poses.resize(6 * P.num_frames);
    for (uint32_t f = 0; f < P.num_frames; f++) {
        uint32_t best = frameImgs[f][0];
        for (uint32_t i : frameImgs[f])
            if (imgOf[i]->numPoint3D() > imgOf[best]->numPoint3D()) best = i;
        const Image& im = *imgOf[best];
        Pose fp = im.pose;
        if (P.image_member[best] != kNoMember) {
            const auto& rm = L.memberOf[P.image_member[best]];
            fp = rec.rigs[rm.first].rigFromWorld(rm.second, im.pose);
        }
        bundle_detail::packPose(fp, &P.poses[6 * f]);
    }
    // 只有显式要求、共同帧数及观测数足够时才优化成员外参；已知外参的镜头可能尚无观测，如朝天镜头。
    std::vector<uint32_t> memberObs(L.memberOf.size(), 0);
    for (uint32_t o = 0; o < P.num_obs; o++) {
        const uint32_t m = P.image_member[P.obs_image[o]];
        if (m != kNoMember) memberObs[m]++;
    }
    P.members.resize(L.memberOf.size());
    P.exts.resize(6 * L.memberOf.size());
    P.ext_dim = 0;
    for (uint32_t m = 0; m < L.memberOf.size(); m++) {
        const RigCalib& c = rec.rigs[L.memberOf[m].first];
        const uint32_t member = L.memberOf[m].second;
        bundle_detail::packPose(c.cam_from_rig[member], &P.exts[6 * m]);
        const uint32_t mask = rigs->rigs[L.memberOf[m].first].members[member].dof;
        const bool held = !bopt.refine_rigs || (int)member == c.ref ||
                          (member < c.fixed.size() && c.fixed[member]) || mask == 0 ||
                          (int)memberCo[m] < bopt.rig_min_frames ||
                          (int)memberObs[m] < bopt.rig_min_obs;
        const uint32_t nf = held ? 0u : extFreeCount(mask);
        P.members[m] = {6 * m, P.ext_dim, nf, mask};
        P.ext_dim += nf;
    }

    // 相机模型、参数数量与布局统一来自 Camera.h，按组偏移打包，兼容混合镜头。
    // 存储全部参数，但仅 n_intr 个自由参数占约化列；例如存八参、固定主点时仅占六列，更新必须按组表映射。
    std::vector<size_t> group_images(camIds.size(), 0);
    std::vector<uint32_t> img_group(P.num_images);
    for (uint32_t i = 0; i < P.num_images; i++) {
        img_group[i] = camGroup[imgOf[i]->camera_id];
        group_images[img_group[i]]++;
    }
    P.groups.resize(camIds.size());
    P.intr.clear();
    for (size_t g = 0; g < camIds.size(); g++) {
        const Camera& c = rec.cameras[camIds[g]];
        const bool pp = bopt.refine_principal_point && bopt.refine_extra_params &&
                        group_images[g] >= bopt.pp_min_images;
        uint32_t nf = (uint32_t)camNumFreeParams(c.model, pp, bopt.refine_extra_params);
        uint32_t off = (uint32_t)P.intr.size();
        uint32_t ni = (uint32_t)camNumParams(c.model);
        double ps[12];
        packIntrinsics(c, ps);
        for (uint32_t i = 0; i < ni; i++) P.intr.push_back(ps[i]);
        P.groups[g] = {off, P.free_intr, nf, (uint32_t)camBaModel(c.model)};
        P.free_intr += nf;
    }
    P.image_group = std::move(img_group);

    // 三维点。
    P.points.resize(3 * P.num_points);
    for (uint32_t i = 0; i < P.num_points; i++) {
        const Vec3& X = ptOf[i]->xyz;
        P.points[3 * i] = X.x; P.points[3 * i + 1] = X.y; P.points[3 * i + 2] = X.z;
    }

    P.pose_dim = 6 * P.num_frames;
    P.total_intr = (uint32_t)P.intr.size();
    P.n_dim = P.pose_dim + P.ext_dim + P.free_intr;
    for (auto& m : P.members) m.ext_col += P.pose_dim;
    for (auto& g : P.groups) g.intr_col += P.pose_dim + P.ext_dim;
    finalizeTables(P);

    // 将先验映射到 BA 索引，同一帧对只保留一个旋转因子，避免 rig 多镜头重复约束同一位姿块。
    if (bopt.priors && !bopt.priors->empty()) {
        const PosePriors& in = *bopt.priors;
        PosePriors& out = L.priors;
        out.up_w = in.up_w;
        out.huber = in.huber;
        auto ba = [&](uint32_t id, uint32_t& idx) {
            if (id > max_img_id || imgBA[id] == UINT32_MAX) return false;
            idx = imgBA[id];
            return true;
        };
        std::set<std::pair<uint32_t, uint32_t>> seen;
        for (PriorRotation r : in.rotations) {
            if (!ba(r.i, r.i) || !ba(r.j, r.j)) continue;
            const uint32_t fi = P.image_frame[r.i], fj = P.image_frame[r.j];
            if (fi == fj || !seen.insert({std::min(fi, fj), std::max(fi, fj)}).second) continue;
            out.rotations.push_back(r);
        }
        for (PriorUp u : in.ups)
            if (ba(u.i, u.i)) out.ups.push_back(u);
        for (PriorCentre c : in.centres) {
            bool ok = true;
            for (int k = 0; k < c.n; k++) ok = ok && ba(c.img[k], c.img[k]);
            if (ok) out.centres.push_back(c);
        }
    }
    return L;
}

// 建图器使用的求解选项。
inline SolverOptions bundleSolverOptions(const BundleOptions& bopt) {
    SolverOptions sopt;
    sopt.real = bopt.real;
    sopt.max_iters = bopt.max_iters;
    sopt.verbose = bopt.verbose;
    sopt.device_selector = bopt.device_selector;
    sopt.device = bopt.device;
    sopt.loss = bopt.loss;
    sopt.loss_param = bopt.loss_param;
    if (bopt.rtol > 0) sopt.rtol = bopt.rtol;
    if (bopt.patience > 0) sopt.patience = bopt.patience;
    if (bopt.solver == "dense") sopt.solver = SolverSel::Dense;
    else if (bopt.solver == "cg") sopt.solver = SolverSel::CG;
    sopt.over_budget_throws = bopt.over_budget_throws;
    sopt.threads = bopt.threads;
    return sopt;
}

// 将求解参数写回原重建；P 通常为布局内部问题，也允许调用方移出后传给求解器。
inline void writeBundle(Reconstruction& rec, const BundleLayout& L, const BAProblem& P) {
    for (uint32_t m = 0; m < P.members.size(); m++) {
        const auto& rm = L.memberOf[m];
        rec.rigs[rm.first].cam_from_rig[rm.second] = bundle_detail::unpackPose(&P.exts[6 * m]);
    }
    for (uint32_t i = 0; i < P.num_images; i++) {
        Image& im = *L.imgOf[i];
        const Pose fp = bundle_detail::unpackPose(&P.poses[6 * P.image_frame[i]]);
        const uint32_t m = P.image_member[i];
        im.pose = m == kNoMember ? fp
                                 : rec.rigs[L.memberOf[m].first].camFromWorld(L.memberOf[m].second, fp);
    }
    for (size_t g = 0; g < L.camIds.size(); g++) {
        Camera& c = rec.cameras[L.camIds[g]];
        unpackIntrinsics(c, &P.intr[P.groups[g].intr_offset]);
    }
    for (uint32_t i = 0; i < P.num_points; i++)
        L.ptOf[i]->xyz = {P.points[3 * i], P.points[3 * i + 1], P.points[3 * i + 2]};
}

// ---------------- 设备失败后的主机回退 ----------------

// 设备丢失或分配失败后改用 CPU 重算，后续达到同等规模的问题直接走主机，避免重复失败。
inline std::atomic<uint64_t>& baHostObsThreshold() {
    static std::atomic<uint64_t> n{UINT64_MAX};
    return n;
}

inline void noteBaDeviceFailure(const VkError& e, uint64_t num_obs) {
    uint64_t from = e.result == VK_ERROR_DEVICE_LOST ? 0 : num_obs;
    uint64_t was = baHostObsThreshold().load();
    while (from < was && !baHostObsThreshold().compare_exchange_weak(was, from)) {}
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
        slog::warn(slog::Tag::Map, spirula::i18n::msg::sfm::ba_host_fallback, {e.what()});
}

struct BundleRun {
    SolverStats stats;
    RealCfg real = RealCfg::F64;  // 最终成功求解使用的算术配置
    double t_init = 0, t_solve = 0;
};

// 原地求解 P；设备每数秒保存已接受参数，失败后主机从检查点继续而非从头开始。
inline BundleRun solveBundle(BAProblem& P, SolverOptions sopt, VkContext* shared) {
    BundleRun r;
    if (P.num_obs >= baHostObsThreshold().load()) sopt.real = RealCfg::CPU;
    SolverCheckpoint ck;
    sopt.checkpoint = &ck;
    auto attempt = [&] {
        auto t0 = std::chrono::steady_clock::now();
        BundleSolver solver(P, sopt, shared);
        solver.init();
        auto t1 = std::chrono::steady_clock::now();
        solver.solve();
        auto t2 = std::chrono::steady_clock::now();
        solver.downloadParams();
        r.stats = solver.stats();
        r.real = solver.real();
        r.t_init = std::chrono::duration<double>(t1 - t0).count();
        r.t_solve = std::chrono::duration<double>(t2 - t1).count();
    };
    try {
        attempt();
    } catch (const VkError& e) {
        if (!vkErrorIsResourceFailure(e.result)) throw;
        noteBaDeviceFailure(e, P.num_obs);
        sopt.real = RealCfg::CPU;
        sopt.checkpoint = nullptr;
        if (ck.iterations > 0) {
            sopt.init_damping = ck.damping;
            sopt.max_iters = std::max(1, sopt.max_iters - ck.iterations);
            slog::diag(slog::Tag::Map, "[ba] resuming on the host at iteration %d, cost %.6e",
                       ck.iterations, ck.cost);
        }
        attempt();
        r.stats.iterations += ck.iterations;
    }
    return r;
}

// 全局 BA 更新全部已配准位姿、三维点与内参，返回求解器最终 RMS 代价，无工作时为 0。
inline double runGlobalBA(Reconstruction& rec, const BundleOptions& bopt) {
    auto prof_t0 = std::chrono::steady_clock::now();
    auto prof_lap = [&prof_t0] {
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - prof_t0).count();
        prof_t0 = t1;
        return dt;
    };
    BundleLayout L = buildBundle(rec, bopt);
    L.attachPriors();
    BAProblem& P = L.P;
    if (P.num_images < 2) return 0;
    double t_build = prof_lap();

    const BundleRun run = solveBundle(P, bundleSolverOptions(bopt), bopt.shared_ctx);
    const SolverStats& stats = run.stats;
    const double t_init = run.t_init, t_solve = run.t_solve;
    prof_lap();
    writeBundle(rec, L, P);

    double t_write = prof_lap();
    g_map_prof.ba_build += t_build;
    g_map_prof.ba_init += t_init;
    g_map_prof.ba_solve += t_solve;
    g_map_prof.ba_write += t_write;
    g_map_prof.n_ba++;
    g_map_prof.n_ba_iters += stats.iterations;
    if (MapProf::enabled())
        slog::diag(slog::Tag::Map,
                   "[prof] BA #%ld: %u img %u pt %u obs | build %.3f init %.3f solve %.3f "
                   "write %.3f s | %d LM iters, %s%s",
                   (long)g_map_prof.n_ba, P.num_images, P.num_points, P.num_obs, t_build, t_init,
                   t_solve, t_write, stats.iterations, stats.solver,
                   stats.cg_solves ? (" " + std::to_string((int)std::lround(
                                                 stats.cg_iters_total / stats.cg_solves)) +
                                      " its/solve").c_str()
                                   : "");
    return stats.final_cost;
}

// ---------------- 多分量联合精化（D45）----------------

// 各模型使用独立图像 ID 区间，但共享同相机 ID 的内参；可选 priors[k] 使用模型自身图像 ID。
inline double runJointBA(std::vector<Reconstruction*> models, const BundleOptions& bopt,
                         const std::vector<const PosePriors*>* priors = nullptr) {
    if (models.empty()) return 0;
    size_t live = 0;
    for (const Reconstruction* m : models)
        if (m->numRegistered() >= 2) live++;
    if (live == 0) return 0;
    if (live == 1 && models.size() == 1) {
        BundleOptions one = bopt;
        one.priors = priors && !priors->empty() ? (*priors)[0] : nullptr;
        return runGlobalBA(*models[0], one);
    }

    // 使用独立 ID 步长，使联合视图可无歧义拆回。
    uint32_t img_stride = 0;
    uint64_t pt_stride = 0;
    for (const Reconstruction* m : models) {
        for (const auto& kv : m->images) img_stride = std::max(img_stride, kv.first + 1);
        for (const auto& kv : m->points3D) pt_stride = std::max(pt_stride, kv.first + 1);
    }
    if (img_stride == 0 || pt_stride == 0) return 0;
    // rig 表可引用当前分量未包含的图像，步长须覆盖表能产生的所有 ID。
    if (bopt.rigs && bopt.use_rigs)
        img_stride = std::max(img_stride, (uint32_t)bopt.rigs->of_image.size());

    Reconstruction all;
    // 先验随图像 ID 平移；各模型向上轴不同，联合问题只采用首个含向上因子模型的轴，其余向上因子忽略。
    PosePriors joint_priors;
    bool joint_up = false;
    // 各模型尺度不同，保留独立 rig 标定与表副本，并按模型平移图像 ID。
    RigTable joint_rigs;
    const bool rigs = bopt.rigs && bopt.use_rigs && !bopt.rigs->empty();
    std::vector<uint32_t> rig_base(models.size(), 0);
    // 相机按 ID 共享，初值取图像最多的模型。
    std::map<uint32_t, double> cam_weight;
    for (const Reconstruction* m : models) {
        std::map<uint32_t, double> w;
        for (const auto& kv : m->images)
            if (kv.second.registered) w[kv.second.camera_id] += 1.0;
        for (const auto& kv : w) {
            auto it = m->cameras.find(kv.first);
            if (it == m->cameras.end()) continue;
            if (!cam_weight.count(kv.first) || kv.second > cam_weight[kv.first]) {
                cam_weight[kv.first] = kv.second;
                all.cameras[kv.first] = it->second;
            }
        }
    }
    for (size_t mi = 0; mi < models.size(); mi++) {
        const Reconstruction& m = *models[mi];
        if (m.numRegistered() < 2) continue;
        const uint32_t io = (uint32_t)mi * img_stride;
        const uint64_t po = (uint64_t)mi * pt_stride;
        if (rigs) {
            rig_base[mi] = (uint32_t)joint_rigs.rigs.size();
            for (const RigSpec& r : bopt.rigs->rigs) {
                RigSpec c = r;
                for (auto& fr : c.frames)
                    for (uint32_t& img : fr) {
                        if (img == kNoImage) continue;
                        auto it = m.images.find(img);
                        img = it != m.images.end() && it->second.registered ? img + io : kNoImage;
                    }
                joint_rigs.rigs.push_back(std::move(c));
            }
            std::vector<RigCalib> calib = m.rigs;
            calib.resize(bopt.rigs->rigs.size());
            all.rigs.insert(all.rigs.end(), calib.begin(), calib.end());
        }
        for (const auto& kv : m.images) {
            if (!kv.second.registered) continue;
            Image im = kv.second;
            im.id = kv.first + io;
            for (uint64_t& p : im.point3D_ids)
                if (p != kInvalidPoint3D) p += po;
            all.images[im.id] = std::move(im);
        }
        for (const auto& kv : m.points3D) {
            Point3D pt = kv.second;
            for (TrackElement& e : pt.track) e.image_id += io;
            all.points3D[kv.first + po] = std::move(pt);
        }
        if (priors && mi < priors->size() && (*priors)[mi] && !(*priors)[mi]->empty()) {
            const PosePriors& pr = *(*priors)[mi];
            joint_priors.huber = pr.huber;
            for (PriorRotation r : pr.rotations) {
                r.i += io;
                r.j += io;
                joint_priors.rotations.push_back(r);
            }
            if (!pr.ups.empty() && (!joint_up || pr.up_w.dot(joint_priors.up_w) > 0.9999)) {
                if (!joint_up) joint_priors.up_w = pr.up_w;
                joint_up = true;
                for (PriorUp u : pr.ups) {
                    u.i += io;
                    joint_priors.ups.push_back(u);
                }
            }
            for (PriorCentre c : pr.centres) {
                for (int k = 0; k < c.n; k++) c.img[k] += io;
                joint_priors.centres.push_back(c);
            }
        }
    }
    if (all.images.size() < 2 || all.points3D.empty()) return 0;

    BundleOptions jopt = bopt;
    jopt.priors = joint_priors.empty() ? nullptr : &joint_priors;
    if (rigs) {
        joint_rigs.index((size_t)models.size() * img_stride);
        jopt.rigs = &joint_rigs;
    }
    double cost = runGlobalBA(all, jopt);

    // 拆回模型时将共享内参写入所有分量。
    for (size_t mi = 0; mi < models.size(); mi++) {
        Reconstruction& m = *models[mi];
        if (m.numRegistered() < 2) {
            for (auto& kv : m.cameras) {
                auto it = all.cameras.find(kv.first);
                if (it != all.cameras.end()) kv.second = it->second;
            }
            continue;
        }
        const uint32_t io = (uint32_t)mi * img_stride;
        const uint64_t po = (uint64_t)mi * pt_stride;
        for (auto& kv : m.images) {
            if (!kv.second.registered) continue;
            auto it = all.images.find(kv.first + io);
            if (it != all.images.end()) kv.second.pose = it->second.pose;
        }
        for (auto& kv : m.points3D) {
            auto it = all.points3D.find(kv.first + po);
            if (it != all.points3D.end()) kv.second.xyz = it->second.xyz;
        }
        for (auto& kv : m.cameras) {
            auto it = all.cameras.find(kv.first);
            if (it != all.cameras.end()) kv.second = it->second;
        }
        if (rigs)
            for (size_t r = 0; r < bopt.rigs->rigs.size() && r < m.rigs.size(); r++)
                m.rigs[r] = all.rigs[rig_base[mi] + r];
    }
    return cost;
}

inline double runJointBA(std::vector<Reconstruction>& models, const BundleOptions& bopt,
                         const std::vector<const PosePriors*>* priors = nullptr) {
    std::vector<Reconstruction*> p;
    p.reserve(models.size());
    for (Reconstruction& m : models) p.push_back(&m);
    return runJointBA(std::move(p), bopt, priors);
}

}  // 命名空间 sfm
