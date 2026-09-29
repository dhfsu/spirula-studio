// 增量 SfM：从种子对恢复相对位姿并三角化，反复以对应图构造二维、三维匹配，通过 PnP 配准，再补轨迹和三角化。
// 种子阈值逐步放宽，配准结合内点数、比例和非线性精化；周期全局 BA 与过滤会撤销失去支持的图像（D36）。
// 点颜色由提取阶段保存的关键点颜色沿轨迹平均。
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "sfm/core/Progress.h"
#include "sfm/core/Features.h"
#include "sfm/core/Model.h"
#include "sfm/core/PriorSource.h"
#include "sfm/core/Sequence.h"
#include "sfm/geometry/AbsolutePose.h"
#include "sfm/geometry/KnownRotation.h"
#include "sfm/geometry/Triangulation.h"
#include "sfm/geometry/TwoView.h"
#include "sfm/core/Cancel.h"
#include "sfm/core/Events.h"
#include "sfm/core/Log.h"
#include "sfm/map/Bundle.h"
#include "i18n/catalog/Sfm.h"
#include "sfm/map/CorrespondenceGraph.h"
// 结构对齐复用 Merge.h 的相似变换与像素评分；Merge.h 不包含本文件，因此无循环依赖。
#include "sfm/map/Merge.h"
#include "sfm/map/Profile.h"
#include "core/Env.h"

namespace sfm {

struct MapperOptions {
    double focal = 0;                  // 0 使用 COLMAP 的 1.2*最大尺寸猜测
    // 下列像素阈值均以提取分辨率定义，由 Camera::pixel_scale 换算，保证质量预设和混合分辨率下含义一致（D47）。
    double max_reproj_error = 3.0;     // 提取像素单位（D47）
    double min_tri_angle_deg = 1.5;
    // 种子接受阈值参考 COLMAP；无候选通过时 initialize 逐步放宽。
    double init_min_tri_angle_deg = 16.0;   // 种子点三角化角度的中位数
    int init_min_inliers = 100;
    double init_max_forward_motion = 0.95;  // |基线·视线方向| 的上限
    // PnP 同时要求内点比例，最少内点保留 15，以支持只有 15–25 内点的稀疏匹配，后续事务式精化负责拒绝有害配准（D36）。
    int min_num_pnp_inliers = 15;
    double min_pnp_inlier_ratio = 0.25;
    // 内点比例分母仅统计位姿可能解释的对应：相机后方或视野外的点不构成反证，重复房间的竞争点仍在前方、视野内（D69）。
    // 仅靠绝对数量在 1146 图房间多配 48 图、提升 8 分 AUC，却使 7620 图室内混合数据损失 20 分，因此必须保留有效比例判据。
    bool pnp_ratio_visible_only = true;
    // 默认禁用以绝对支持替代内点比例；7620 图模型曾从 2816 增至 5379 图，却使绝对旋转中位误差从 0.6 增至 4.1 度、AUC 降 20 分。
    // 竞争位姿检查触发 91 次仍未修复，真正区分冗余与歧义的是可解释对应的分母。
    int strong_pnp_inliers = 0;
    // 绝对支持还需排除竞争位姿：对最佳解拒绝的对应寻找第二解，若其支持相对最佳解过大，则保留比例限制。
    // 0 跳过；该检查不足以单独补救绝对支持放行，相关失败测量见上方。
    double strong_pnp_max_rival = 0.5;
    int max_reg_trials = 3;            // 逐图像上限，参考 COLMAP
    // 下一图按支持特征的金字塔空间分布排序，关闭时仅按数量排序。
    bool rank_by_visibility = true;
    // 种子重试避开先前尝试已覆盖区域，减少重复重建；改变种子会影响整条重建轨迹，提供开关便于归因（D58）。
    bool seed_blocking = true;
    // 相机组首次配准且焦距仍为几何猜测时，按对数比例扫描 P3P 焦距假设，近似替代 P4Pf，比例范围参考 COLMAP（D18）。
    int min_model_size = 10;           // 小于此规模时重试种子
    // 在种子自身内点上重判平面/全景虽占搜索约 90%，但禁用后五组平均损失 2.2 分 AUC@10，1322 图下降 9.3 分，省时被下游抵消，因此保留。
    bool seed_homography = true;
    // 模型同时达到最小图像数和覆盖比例时停止种子重试，否则其他种子通常值得尝试。
    double min_model_fraction = 0.5;
    int max_init_trials = 8;           // 保留最佳模型前的种子尝试上限
    // 随后仅在未覆盖图像中播种子模型，允许合并所需重叠但必须带来新增覆盖；各模型独立输出，max_num_models=1 限制单模型（D41）。
    int max_num_models = 50;           // COLMAP 默认模型数上限
    // 共享图像预算以绝对数量为下限，新增覆盖可按 model_overlap_ratio 换取额外重叠；7620 图数据纯绝对限制曾停止 17 次，子模型平均仅 100 图（D66）。
    int max_model_overlap = 20;        // COLMAP 默认值
    double model_overlap_ratio = 1.0;
    // 后续模型的总种子预算包含因太小而丢弃的尝试；否则残余噪声会不断生成双图模型，千图数据可尝试数万次。
    int max_model_trials = 20;  // 200
    int focal_search_samples = 15;     // 0 禁用搜索
    double min_focal_ratio = 0.1, max_focal_ratio = 10.0;
    // 焦距初始化用于旋转退化或真实广角数据，避免 BA 无法摆脱 1.2*max_dim 猜测；0 禁用，通常下降一至两级即停止（D48）。
    int focal_trials = 5;
    // 只有共同一致观测的相对增益足够大才改变焦距；年轻模型的观测数噪声较大，几百分点改善不足以支持替换一致初值。
    double focal_min_gain = 0.15;
    size_t focal_model_size = 20;      // 每个试探模型的图像预算
    // 姿态离散度仅报告，不作门限；114 帧直行 KITTI 跨 8.5 度，普通 garden 前 20 图仍跨 32 度，体现可观性差别。
    double focal_max_rot_spread_deg = 15.0;
    // 沿用 COLMAP 的全局精化触发频率；增长比例放宽至 1.25 会减少重三角化并损害困难数据覆盖，速度应由自适应收敛与持久上下文获得（D36/D38）。
    double ba_growth_ratio = 1.1;      // COLMAP 的 ba_global_images_ratio
    int ba_max_refinements = 5;        // 最终精化轮数，增长阶段使用两轮
    double ba_refine_change = 0.0005;  // 观测变化比例低于此值时停止精化
    // 增长 BA 在相对收益连续 ba_growth_patience 次低于 ba_growth_rtol 时停止，最终阶段保持严格，0 表示求解器默认（D38）。
    double ba_growth_rtol = 1e-4;
    int ba_growth_patience = 5;
    // 最终结果使用严格精化；原子结束后还会联合及逐层重算，不必在小模型上反复收敛到最高精度。
    bool ba_final_tight = true;
    // 重三角化与轨迹补全的误差阈值为 max_reproj_error 的比例，形成滞回；0 完全禁用。
    double retri_scale = 0.75;
    // 合并对应关系指向同一特征的重复三维点；1146 图测量多恢复 13 图、提升 2.4 分 AUC@5，轨迹变长的求解成本由 kMergeMaxTrack 限制。
    bool merge_tracks = true;
    // 审查只在图像未参与建立的结构支持明显更强且位置不同的替代位姿时撤销原配准；证据下限防止少量随机对应误移正确相机（D44）。
    int audit_min_evidence = 40;
    int audit_min_alternative = 25;
    double audit_alternative_factor = 3.0;
    double audit_min_rotation_deg = 5.0;
    // 相机中心位移超过模型 RMS 尺度的此比例也算位姿变化，用于捕获方向相同但位置错误的图像。
    double audit_min_shift_frac = 0.01;
    int audit_ransac_trials = 1000;
    int min_image_points = 5;          // 图像支持低于此值时撤销配准
    double max_extra_param = 1.0;      // 畸变参数绝对值超过此值视为异常
    CamModel camera_model = CamModel::Radial;  // 新相机的畸变模型（D29）
    // 逐 ID 初始内参来自 CameraSetup，缺项回退 defaultFor，使测试与库调用方保持默认行为（D46）。
    std::map<uint32_t, Camera> initial_cameras;
    // 组级 EXIF 或显式组焦距先验禁止逐图焦距扫描；全局焦距不归入此集合，因为它未指明适用镜头（D45/D46）。
    std::set<uint32_t> known_focal_cameras;
    // 所有给定焦距，包括组级、EXIF 和全局值，均跳过初始化替换；可优化给定值，但不能改成几何猜测（D48）。
    std::set<uint32_t> given_focal_cameras;
    // 双视图测量值仍用试探模型的 BA 精化，但跳过为摆脱成倍错误猜测设计的减半阶梯（D53）。
    std::set<uint32_t> measured_focal_cameras;
    std::string ba_loss = "huber";     // 建图 BA 的稳健损失（D36）
    // 约化相机系统线性求解器参见 BundleOptions。
    std::string ba_solver = "auto";
    double ba_loss_param = 2.0;        // Huber delta/Cauchy c，提取像素单位
    // 默认固定主点，避免各相机组主点漂移表现为 rig 镜头相对旋转误差（D50）。
    bool refine_principal_point = false;
    // 最终完整模型可单独释放主点做一次全局 BA，由 CLI 调用 polish，而非增长循环执行（D51）。
    size_t pp_min_images = 20;   // 释放主点所需的组内最少图像数
    // 建图默认优化畸变，关闭则固定到初始值并留待收尾（D72）。
    bool refine_extra_params = true;
    // 求解器算术类型；df 用两个 fp32 提供约 49 位有效精度，在 fp64 吞吐较低的硬件上可能更快。
    std::string ba_real = "double";
    // 中间求解也默认 double，虽然 fp32 可节省约 25–35% 建图时间，但原子累加约 1e-7 的扰动会越过决策阈值。
    // 379 图三次 fp64 的 AUC@10 均为 96.1，fp32 为 96.5/92.3/91.5；896 图为 88.3/87.7/88.0 对 85.2/87.5/87.1，平均分别损失 2.7/1.4 分。
    std::string ba_real_coarse = "double";
    int device = -1;
    // 各次 BA 共用规范 uuid:<hex>，空值沿用设备选择优先级。
    std::string device_selector;
    // 逐点处理与 CPU BA 的主机线程数，0 使用 hardware_concurrency。
    int threads = 0;
    bool verbose = true;
    // 是否代表整次运行输出快照与累计进度；原子私有 Mapper 应关闭。
    bool report_progress = true;
    // use_rigs 关闭时忽略装置表，refine_rigs 关闭时固定已标定外参。
    bool use_rigs = true;
    bool refine_rigs = true;
    // 成员对应不足时允许仅由 rig 推断位姿，覆盖朝天或朝向操作者等缺少场景特征的镜头。
    bool rig_complete_blind = true;
    RigCalibOptions rig_calib;
    // 序列位置距离在此窗口内的图像视为邻居，其对应优先可信；对应匹配 overlap（D79）。
    int sequence_window = 2;
    // 配准旋转偏离已放置邻居与陀螺预测超过阈值或三倍 sigma 时，固定旋转重新求解或拒绝。
    bool use_priors = true;
    double prior_rot_tol_deg = 2.0;
};

class Mapper {
public:
    // camera_ids 为从 1 开始的相机组 ID，空值表示共享相机；rigs/seqs 使用数据库图像编号且生命周期长于 Mapper。
    Mapper(const MatchesDatabase& db, const std::vector<FeatureSet>& feats, MapperOptions opt,
           std::vector<uint32_t> camera_ids = {}, const RigTable* rigs = nullptr,
           const SequenceTable* seqs = nullptr, PriorSource* priors = nullptr)
        : db_(db), feats_(feats), opt_(opt), cam_ids_(std::move(camera_ids)),
          rigs_(rigs && !rigs->empty() && opt.use_rigs ? rigs : nullptr),
          seq_(seqs && !seqs->empty() ? seqs : nullptr),
          priors_(priors && opt.use_priors ? priors : nullptr) {}

    const RigTable* rigs() const { return rigs_; }
    const SequenceTable* sequences() const { return seq_; }
    PriorSource* priors() const { return priors_; }

    // 传感器作用统计，供摘要显示。
    struct PriorStats {
        uint32_t corrected = 0;   // 使用陀螺旋转重解的配准数
        uint32_t refused = 0;     // 重解无有效结果而拒绝的数量
        uint32_t vouched = 0;     // 由邻居旋转决定的审查次数
        uint32_t seeds = 0;       // 使用陀螺旋转定位的种子对数
        size_t rotations = 0, ups = 0, centres = 0;   // 最近一次求解的因子数
    };
    PriorStats priorStats() const {
        PriorStats st = prior_stats_;
        st.vouched = prior_vouched_.load();
        st.seeds = prior_seeds_.load();
        return st;
    }

    // 输出全部模型并按三维点数降序，首项写 sparse/0；即使完全无法播种也返回一个空模型供错误报告。
    std::vector<Reconstruction> run() {
        auto prof_start = std::chrono::steady_clock::now();
        ensureSetup();
        std::vector<Reconstruction> models;

        // ---------------- 阶段一：多种子尝试中的主模型 ----------------
        // 先检查焦距可观性并初始化，再尝试不同种子，保留可带来新增覆盖的结果，避免从头重建已发现的分量（D19/D41/D48）。
        {
            ProfTimer pt(g_map_prof.init_seed);
            ProfTimer pb(g_map_prof.bootstrap);
            bootstrapFocalLength();
        }

        std::vector<Reconstruction> attempts;
        size_t seed_from = 0;
        seeded_.clear();  // 种子屏蔽仅在此重试循环内有效（D58）
        for (int attempt = 0; attempt < std::max(1, opt_.max_init_trials); attempt++) {
            cancel::check();
            resetModel();
            bool seeded;
            {
                ProfTimer pt(g_map_prof.init_seed);
                seeded = initialize(seed_from);
            }
            if (!seeded) {
                if (opt_.verbose && attempt == 0) reportInitFailure();
                break;
            }
            globalRefine(false);  // 增长前执行双视图 BA，与 COLMAP 一致
            grow();
            uint32_t reg = rec_.numRegistered();
            if (reg) {
                attempts.push_back(snapshotModel());
                if (opt_.seed_blocking) blockSeeds(attempts.back());
            }
            const uint32_t enough = (uint32_t)std::max(
                (double)opt_.min_model_size, opt_.min_model_fraction * allowedCount());
            if (reg >= enough) break;
            if (opt_.verbose)
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_model_too_small,
                         {(long long)reg, (long long)enough, (long long)seededImages()});
        }
        seeded_.clear();
        if (attempts.empty()) {
            // 无法播种时返回空重建，供调用方报告失败。
            resetModel();
            models.push_back(snapshotModel());
            return finishRun(models, prof_start);
        }

        // 最大的尝试作为主模型，其余仅在新增覆盖足够时接纳；允许有助于后续合并的重叠，拒绝纯副本（D66）。
        std::stable_sort(attempts.begin(), attempts.end(),
                         [](const Reconstruction& a, const Reconstruction& b) {
                             return a.numRegistered() > b.numRegistered();
                         });
        for (size_t i = 0; i < attempts.size(); i++) {
            if ((int)models.size() >= std::max(1, opt_.max_num_models)) break;
            std::string why;
            if (i > 0 && !admitModel(attempts[i], why)) {
                if (opt_.verbose)
                    slog::diag(slog::Tag::Map, "[map] seed attempt discarded: %s", why.c_str());
                continue;
            }
            claimImages(attempts[i]);
            recordCameras(attempts[i]);
            models.push_back(std::move(attempts[i]));
        }

        // ---------------- 阶段二：未覆盖区域的附加模型 ----------------
        // 仅从尚未被保留模型配准的图像播种，全部覆盖时不执行。
        seedFurtherModels(models);
        return finishRun(models, prof_start);
    }

    // ---------------- 外部调度接口（D44）----------------
    // 对已有模型提供增长、精化与继续播种，先将其接入 rec_；模型可来自磁盘、合并或其他调度器。

    struct GrowStats {
        uint32_t before = 0, after = 0;
        uint32_t registered = 0;   // 本轮新配准图像数
        bool refined = false;      // 是否运行最终精化
    };

    // 接入模型后继续增长，others 提供已有覆盖以限制重复配准，同时允许建立合并所需重叠；为空则不设重叠边界。
    // max_reg 另限模型总大小，0 不限制；没有新增配准时原样返回，不执行无收益 BA。
    Reconstruction continueFrom(const Reconstruction& m, GrowStats* out = nullptr,
                                const std::vector<const Reconstruction*>& others = {},
                                uint32_t max_reg = 0) {
        ensureSetup();
        resetModel();
        adopt(m);
        model_count_.clear();
        for (const Reconstruction* o : others)
            if (o != &m) claimImages(*o);
        rebuildScores();
        GrowStats st;
        st.before = rec_.numRegistered();
        st.registered = growLoop(max_reg);
        st.after = rec_.numRegistered();
        if (out) *out = st;
        if (!st.registered) return m;
        st.refined = true;
        checkedRefine(true);
        if (out) {
            out->refined = true;
            out->after = rec_.numRegistered();
        }
        return snapshotModel();
    }

    // ---------------- 无共同图像的模型对齐（D70）----------------
    // 两个模型可从不同图像看到同一结构，如 7620 图数据的 5474/474 图模型共同图像不足三张；对应图中的匹配三维点可提供跨规范的相似变换。

    // 按连接模型的特征对应数排序候选，而非按图像对数量，反映对齐实际使用的证据。
    struct StructureLink {
        size_t a = 0, b = 0;
        size_t matches = 0;
    };
    std::vector<StructureLink> structureLinks(const std::vector<Reconstruction>& models,
                                              size_t min_matches) const {
        std::unordered_map<uint32_t, std::vector<uint32_t>> in_model;
        for (size_t i = 0; i < models.size(); i++)
            for (const auto& kv : models[i].images)
                if (kv.second.registered) in_model[kv.first].push_back((uint32_t)i);
        std::unordered_map<uint64_t, size_t> w;
        for (const TwoViewMatches& p : db_.pairs) {
            auto ia = in_model.find(p.image1), ib = in_model.find(p.image2);
            if (ia == in_model.end() || ib == in_model.end()) continue;
            for (uint32_t x : ia->second)
                for (uint32_t y : ib->second) {
                    if (x == y) continue;
                    const uint32_t lo = std::min(x, y), hi = std::max(x, y);
                    w[((uint64_t)lo << 32) | hi] += p.matches.size();
                }
        }
        std::vector<StructureLink> out;
        for (const auto& kv : w) {
            if (kv.second < min_matches) continue;
            out.push_back({(size_t)(kv.first >> 32), (size_t)(kv.first & 0xffffffffu), kv.second});
        }
        // 先构造全序，消除哈希遍历顺序的不确定性。
        std::sort(out.begin(), out.end(), [](const StructureLink& x, const StructureLink& y) {
            return x.a != y.a ? x.a < y.a : x.b < y.b;
        });
        std::stable_sort(out.begin(), out.end(),
                         [](const StructureLink& x, const StructureLink& y) {
                             return x.matches > y.matches;
                         });
        return out;
    }

    // 由两模型三角化的对应点拟合 src 到 dst 相似变换，再按目标观测重投影像素评分。
    AlignmentResult alignByStructure(const Reconstruction& dst, const Reconstruction& src,
                                     const MergeOptions& opt, size_t max_corr = 6000) const {
        AlignmentResult r;
        struct Corr {
            Vec3 d, s;             // 各模型自身规范中的点
            uint32_t img, feat;    // 用于评分的目标观测
        };
        std::vector<Corr> corr;
        std::set<std::pair<uint64_t, uint64_t>> seen;  // 每个点对仅投一票
        for (const TwoViewMatches& p : db_.pairs) {
            for (int flip = 0; flip < 2; flip++) {
                const uint32_t ia = flip ? p.image2 : p.image1;
                const uint32_t ib = flip ? p.image1 : p.image2;
                auto da = dst.images.find(ia);
                auto sb = src.images.find(ib);
                if (da == dst.images.end() || sb == src.images.end()) continue;
                if (!da->second.registered || !sb->second.registered) continue;
                for (const FeatureMatch& fm : p.matches) {
                    const uint32_t fa = flip ? fm.idx2 : fm.idx1;
                    const uint32_t fb = flip ? fm.idx1 : fm.idx2;
                    if (fa >= da->second.point3D_ids.size() ||
                        fb >= sb->second.point3D_ids.size())
                        continue;
                    const uint64_t pd = da->second.point3D_ids[fa];
                    const uint64_t ps = sb->second.point3D_ids[fb];
                    if (pd == kInvalidPoint3D || ps == kInvalidPoint3D) continue;
                    auto itd = dst.points3D.find(pd);
                    auto its = src.points3D.find(ps);
                    if (itd == dst.points3D.end() || its == src.points3D.end()) continue;
                    if (!seen.insert({pd, ps}).second) continue;
                    corr.push_back({itd->second.xyz, its->second.xyz, ia, fa});
                }
            }
        }
        r.structure_pairs = corr.size();
        r.from_structure = true;
        const size_t need = (size_t)std::max(3, opt.min_common_images);
        if (corr.size() < need) {
            r.reason = "only " + std::to_string(corr.size()) +
                       " point(s) triangulated by both models";
            return r;
        }
        // 候选可达数十万，RANSAC 用跨列表等间隔采样降低评分成本，不能只截取接缝一端。
        if (corr.size() > max_corr) {
            std::vector<Corr> thin;
            thin.reserve(max_corr);
            const double step = (double)corr.size() / (double)max_corr;
            for (size_t k = 0; k < max_corr; k++) thin.push_back(corr[(size_t)(k * step)]);
            corr.swap(thin);
        }

        const int n = (int)corr.size();
        auto fit = [&](const std::vector<int>& idx) {
            std::vector<Sim3> out;
            std::vector<Vec3> a, b;
            for (int i : idx) { a.push_back(corr[i].s); b.push_back(corr[i].d); }
            Sim3 t;
            if (estimateSim3(a, b, t)) out.push_back(t);
            return out;
        };
        auto res = [&](const Sim3& t, int i) {
            const Corr& c = corr[i];
            const Image& im = dst.images.at(c.img);
            auto cam = dst.cameras.find(im.camera_id);
            if (cam == dst.cameras.end()) return 1e30;
            const double e = reprojErrorAt(cam->second, im.pose, kp(c.img, c.feat),
                                           transformPoint(t, c.s));
            return e * e;
        };
        RansacOptions ro;
        ro.max_error = opt.max_reproj_error;
        ro.seed = opt.seed;
        ro.max_num_trials = opt.ransac_max_trials;
        RansacReport<Sim3> rep = loransac<Sim3>(n, 3, fit, fit, res, ro);
        if (!rep.success || rep.num_inliers < (int)need) {
            r.reason = "alignment on shared structure found only " +
                       std::to_string(rep.success ? rep.num_inliers : 0) + "/" +
                       std::to_string(n) + " consistent point(s)";
            return r;
        }
        // 采用与位姿对齐相同的内点比例要求，避免从大量匹配中少量偶合误认相似场景。
        if ((double)rep.num_inliers < opt.min_inlier_ratio * (double)n) {
            r.reason = "only " + std::to_string(rep.num_inliers) + "/" + std::to_string(n) +
                       " shared points agree";
            return r;
        }
        double sum = 0;
        for (int i = 0; i < n; i++)
            if (rep.inlier_mask[i]) sum += std::sqrt(res(rep.model, i));
        r.transform = rep.model;
        r.inliers = (size_t)rep.num_inliers;
        r.mean_error = sum / (double)rep.num_inliers;
        r.common_images = sharedImages(src, dst).size();
        r.success = true;
        return r;
    }

    // 仅用 PnP 增长，到需要 BA 时立即停止，由调用方对整轮模型执行一次联合优化；37 模型分别完整增长曾耗费 8000 次求解。
    // 仅检查本轮新增位姿，矛盾者撤销而不现场修复，避免引入本接口要省去的额外精化（D57）。
    Reconstruction growByPnP(const Reconstruction& m, GrowStats* out,
                             const std::vector<const Reconstruction*>& others, uint32_t max_reg,
                             uint32_t* rejected = nullptr) {
        ensureSetup();
        resetModel();
        adopt(m);
        model_count_.clear();
        for (const Reconstruction* o : others)
            if (o != &m) claimImages(*o);
        rebuildScores();
        GrowStats st;
        st.before = rec_.numRegistered();
        st.registered = growLoop(max_reg, /*stop_at_ba=*/true);
        if (st.registered) {
            modelScale();  // 预热 poseContradicted 使用的延迟缓存
            const std::vector<uint32_t> fresh = recent_regs_;
            uint32_t bad = 0;
            for (uint32_t img : fresh) {
                Pose alt;
                if (!rec_.images.at(img).registered || !poseContradicted(img, alt)) continue;
                deregisterImage(img);
                bad++;
            }
            if (rejected) *rejected = bad;
            st.registered -= std::min(st.registered, bad);
        }
        st.after = rec_.numRegistered();
        if (out) *out = st;
        if (!st.registered) return m;  // 没有变化，返回原模型
        return snapshotModel();
    }

    // 建模前统一确定初始内参，公开供 bottom-up 在完整数据库上执行，避免由首个小原子决定全局焦距（D48）。
    void bootstrapCameras() {
        ensureSetup();
        bootstrapFocalLength();
    }

    // 接入模型后执行 BA、过滤与撤销配准，以共同优化合并接缝；coarse 使用增长阶段两轮宽松精化，供可修复性判断，最终结果另行严格求解。
    Reconstruction refine(const Reconstruction& m, bool coarse = false) {
        ensureSetup();
        resetModel();
        adopt(m);
        rebuildScores();
        const bool tight = opt_.ba_final_tight;
        opt_.ba_final_tight = tight && !coarse;
        globalRefine(true);
        opt_.ba_final_tight = tight;
        return snapshotModel();
    }

    // 原地试探精化，超设备预算则返回 false 并保持 m 不变；单模型共享点无法像联合问题那样分批。
    bool refineIfItFits(Reconstruction& m, bool coarse = false) {
        ba_over_budget_throws_ = true;
        try {
            m = refine(m, coarse);
        } catch (const BAOverBudget& e) {
            ba_over_budget_throws_ = false;
            if (opt_.verbose)
                slog::diag(slog::Tag::Map,
                           "[map] a %u-image refinement needs %.0f MB against a %.0f MB "
                           "budget; declining it", m.numRegistered(), e.need_mb, e.budget_mb);
            return false;
        }
        ba_over_budget_throws_ = false;
        return true;
    }

    // 完整模型的额外全局 BA，释放建图期间固定的主点与畸变，见 README 的收尾说明（D51/D72）。
    Reconstruction polish(const Reconstruction& m, bool free_pp = true, bool free_extra = false) {
        // 使用本次运行的相机分组；磁盘模型因按尺寸拆分，不能直接用其相机 ID 判断组数。
        std::set<uint32_t> groups;
        for (const auto& kv : m.images)
            if (kv.second.registered)
                groups.insert(kv.first < cam_ids_.size() ? cam_ids_[kv.first]
                                                        : kv.second.camera_id);
        // 多组主点各自漂移会形成真实相对旋转误差，双鱼眼测量曾因此损失 21 分 AUC（D51）。
        const bool pp = free_pp && groups.size() == 1;
        if (free_pp && !pp && opt_.verbose)
            slog::err(slog::Tag::Map, spirula::i18n::msg::sfm::map_pp_skipped,
                      {(long long)groups.size()});
        // 已在建图中优化的参数无需再次释放。
        const bool extra = free_extra && !opt_.refine_extra_params;
        if (!pp && !extra) return m;
        ensureSetup();
        resetModel();
        adopt(m);
        rebuildScores();
        std::map<uint32_t, Vec2> before;
        for (const auto& kv : rec_.cameras) before[kv.first] = {kv.second.cx, kv.second.cy};
        final_.pp = pp;
        final_.extra = extra;
        globalRefine(true);
        final_ = FinalRelease{};
        if (opt_.verbose)
            for (const auto& kv : rec_.cameras) {
                const Vec2& b = before[kv.first];
                const double d = std::hypot(kv.second.cx - b.x, kv.second.cy - b.y);
                if (d > 1e-9)
                    slog::diag(slog::Tag::Map,
                               "[map] camera %u principal point %.1f,%.1f -> %.1f,%.1f (%.1f px)",
                               kv.first, b.x, b.y, kv.second.cx, kv.second.cy, d);
            }
        return snapshotModel();
    }

    // 仅完整模型才允许逐图像独立内参；增长阶段共享相机是焦距可观性的来源（D73）。
    Reconstruction perImageIntrinsics(const Reconstruction& m, bool free_extra = true) {
        if (m.numRegistered() < 2) return m;
        ensureSetup();
        resetModel();
        adopt(m);
        rebuildScores();
        if (!splitCamerasPerImage()) return m;
        final_.extra = free_extra;
        final_.no_sanitize = true;
        globalRefine(true);
        final_ = FinalRelease{};
        return snapshotModel();
    }

    // 最终严格精化可解除 rig 约束，让各图像按自身观测调整，以吸收曝光不同步或支架形变。
    Reconstruction releaseRigs(const Reconstruction& m) {
        if (!rigs_ || m.numRegistered() < 2) return m;
        ensureSetup();
        resetModel();
        adopt(m);
        rebuildScores();
        final_.no_rig = true;
        globalRefine(true);
        final_ = FinalRelease{};
        return snapshotModel();
    }

    struct AuditStats {
        uint32_t checked = 0, deregistered = 0, unsupported = 0, reregistered = 0;
    };

    // 根据图像未参与建立的外部结构寻找明显更好的替代位姿，修复合并或重复结构导致的错位；仅检查自身轨迹无法揭示这种错误（D44）。
    // 直接移动到胜出位姿并重新关联、三角化，再按普通 BA 和过滤判定；先删除再完整重注册曾损失 60 图，因巨大噪声池无法满足比例门限。
    // 无审查时某长序列约 7% rig 帧相差数十度；max_reg 限制修复额外增长，避免曾出现的 1752->3933 图隐藏重建。
    Reconstruction audit(const Reconstruction& m, AuditStats* out = nullptr,
                         uint32_t max_reg = 0) {
        ensureSetup();
        resetModel();
        adopt(m);
        // 这是已有模型修复，不应被子模型已认领集合的重叠限制阻断。
        model_count_.clear();
        rebuildScores();
        AuditStats st;
        std::vector<std::pair<uint32_t, Pose>> repairs;
        // 逐图审查只读，可并行运行 RANSAC，再按图像顺序应用修复以保持确定性；逐图诊断输出路径仍串行。
        std::vector<uint32_t> ids;
        for (const auto& kv : rec_.images)
            if (kv.second.registered) ids.push_back(kv.first);
        st.checked = (uint32_t)ids.size();
        auto audit_t0 = std::chrono::steady_clock::now();
        modelScale();  // 工作线程读取前预热延迟缓存
        std::vector<char> hit(ids.size(), 0);
        std::vector<Pose> alts(ids.size());
        const unsigned hc = std::thread::hardware_concurrency();
        int nt = opt_.threads > 0 ? opt_.threads : (hc > 0 ? (int)hc : 1);
        if (audit_dump_) nt = 1;
        nt = std::max(1, std::min<int>(nt, (int)std::max<size_t>(ids.size(), 1)));
        std::atomic<size_t> next{0};
        auto worker = [&] {
            for (size_t i = next++; i < ids.size(); i = next++)
                hit[i] = poseContradicted(ids[i], alts[i]) ? 1 : 0;
        };
        if (nt == 1) {
            worker();
        } else {
            std::vector<std::thread> pool;
            pool.reserve(nt);
            for (int t = 0; t < nt; t++) pool.emplace_back(worker);
            for (std::thread& t : pool) t.join();
        }
        for (size_t i = 0; i < ids.size(); i++)
            if (hit[i]) repairs.emplace_back(ids[i], alts[i]);
        st.unsupported = (uint32_t)repairs.size();
        auto audit_t1 = std::chrono::steady_clock::now();
        g_map_prof.audit_check += std::chrono::duration<double>(audit_t1 - audit_t0).count();
        ProfTimer audit_pt(g_map_prof.audit_fix);
        if (!repairs.empty()) {
            if (opt_.verbose)
                slog::diag(slog::Tag::Map,
                           "[map] audit: %u/%u image(s) sit where the rest of the model "
                           "contradicts them; moving", st.unsupported, st.checked);
            // 先全部解除旧观测，避免它们继续支持已判错的位姿。
            for (const auto& r : repairs) deregisterImage(r.first);
            for (const auto& r : repairs) {
                Image& im = rec_.images[r.first];
                im.pose = r.second;
                im.registered = true;
            }
            rebuildScores();
            // 新位姿先连接已有三维点，再创建新点；若仅三角化，已有结构会导致修复图像没有观测而被过滤为无效。
            for (const auto& r : repairs) attachExisting(r.first);
            for (const auto& r : repairs) triangulateForImage(r.first);
            rebuildScores();
            // 模型变化并重置试验预算后，先前失败的图像可能获得配准机会。
            reg_trials_.assign(db_.images.size(), 0);
            growLoop(max_reg);
        }
        checkedRefine(true);
        for (const auto& r : repairs)
            if (rec_.images.at(r.first).registered) st.reregistered++;
        st.deregistered = st.unsupported - st.reregistered;
        if (out) *out = st;
        return snapshotModel();
    }

    // 在未覆盖图像中继续播种；restart_relaxation 重置种子阈值阶梯，因为已认领集合变化可能使旧无资格图像对重新可用。
    void seedFurtherModels(std::vector<Reconstruction>& models, bool restart_relaxation = false) {
        ensureSetup();
        if (restart_relaxation) init_relax_ = seed_phase_ = 0;
        int trials = 0;
        // 未覆盖图像少于 min_model_size 时不再尝试，每个结果都必定因新增覆盖不足而丢弃，继续播种只会浪费完整增长与优化。
        const size_t need = (size_t)std::max(1, opt_.min_model_size);
        while ((int)models.size() < std::max(1, opt_.max_num_models) &&
               unclaimedImages() >= need) {
            if (++trials > std::max(1, opt_.max_model_trials)) {
                if (opt_.verbose)
                    slog::diag(slog::Tag::Map,
                               "[map] sub-model search hit its %d-attempt budget with %zu "
                               "image(s) unaccounted for", opt_.max_model_trials,
                               unclaimedImages());
                break;
            }
            resetModel();
            size_t from = 0;
            bool seeded;
            {
                ProfTimer pt(g_map_prof.init_seed);
                seeded = initialize(from);
            }
            if (!seeded) break;  // 没有可建立模型的剩余种子
            globalRefine(false);
            grow();
            Reconstruction sub = snapshotModel();
            std::string why;
            if (!admitModel(sub, why)) {
                // 过小模型不认领图像，留给后续种子；used_seeds_ 防止重复使用同一对。
                if (opt_.verbose)
                    slog::diag(slog::Tag::Map, "[map] sub-model discarded: %s", why.c_str());
                continue;
            }
            const uint32_t reg = sub.numRegistered();
            models.push_back(std::move(sub));
            claimImages(models.back());
            recordCameras(models.back());
            if (opt_.verbose)
                slog::diag(slog::Tag::Map,
                           "[map] sub-model %zu: %u images, %zu points (%zu images left)",
                           models.size() - 1, reg, models.back().points3D.size(),
                           unclaimedImages());
        }
    }

    // 用当前模型集合替换已认领图像记录。
    void claimAll(const std::vector<Reconstruction>& models) {
        model_count_.assign(db_.images.size(), 0);
        for (const Reconstruction& m : models) claimImages(m);
    }

    // 种子屏蔽与已认领集合分开：重试不能在原区域重新起步，但仍须允许穿过相同图像增长到更远区域（D58）。
    void blockSeeds(const Reconstruction& m) {
        if (seeded_.size() != db_.images.size()) seeded_.assign(db_.images.size(), 0);
        for (const auto& kv : m.images)
            if (kv.second.registered && kv.first < seeded_.size()) seeded_[kv.first] = 1;
    }
    bool seedBlocked(uint32_t img) const {
        return img < seeded_.size() && seeded_[img];
    }
    size_t seededImages() const {
        size_t n = 0;
        for (uint32_t i = 0; i < seeded_.size(); i++) n += (seeded_[i] && allowed(i));
        return n;
    }

    // ---------------- 模型与独立双视图几何的一致性 ----------------
    // 合并自身使用的共享位姿、三维点可能因重复结构而错误自洽；检查模型相对位姿能解释多少预先验证匹配，提供独立证据。
    // 全程使用单位视线，兼容鱼眼（D45）。

    // 返回模型相对位姿解释的验证匹配比例，缺图像或相机等无法判断时为 -1。
    double pairAgreement(const Reconstruction& m, const TwoViewMatches& p, double max_error_px,
                         double model_scale) const {
        auto ia = m.images.find(p.image1);
        auto ib = m.images.find(p.image2);
        if (ia == m.images.end() || ib == m.images.end()) return -1;
        if (!ia->second.registered || !ib->second.registered || p.matches.empty()) return -1;
        auto ca = m.cameras.find(ia->second.camera_id);
        auto cb = m.cameras.find(ib->second.camera_id);
        if (ca == m.cameras.end() || cb == m.cameras.end()) return -1;
        Mat3 R = mul(ib->second.pose.R, transpose(ia->second.pose.R));
        Vec3 t = ib->second.pose.t - mul(R, ia->second.pose.t);
        // 提取像素阈值换算为弧度（D47）。
        const double thr =
            0.5 * (ca->second.errRad(max_error_px) + cb->second.errRad(max_error_px));
        const bool wide_baseline = model_scale > 0 && t.norm() > 1e-3 * model_scale;
        Mat3 E = mul(crossMatrix(t.normalized()), R);
        size_t ok = 0;
        for (const FeatureMatch& fm : p.matches) {
            Vec3 b1 = ca->second.bearing(kp(p.image1, fm.idx1));
            Vec3 b2 = cb->second.bearing(kp(p.image2, fm.idx2));
            double err2;
            if (wide_baseline) {
                err2 = sampsonSqBearing(E, b1, b2);
            } else {
                // 基线近零时极线约束无信息，直接用 R 比较视线。
                Vec3 pred = mul(R, b1);
                Vec3 c = pred.cross(b2);
                double ang = std::atan2(c.norm(), pred.dot(b2));
                err2 = ang * ang;
            }
            if (err2 < thr * thr) ok++;
        }
        return (double)ok / (double)p.matches.size();
    }

    struct SeamCheck {
        size_t cross_pairs = 0;    // 跨接缝的已验证图像对数
        size_t tested = 0;         // 实际参与评估的图像对数
        size_t agree = 0;
        double median_frac = 0;    // 逐对匹配解释比例的中位数
    };

    // crossing=false 评估接缝内部图像对，作为该数据自身可达到的一致性基线。
    // 纹理良好室外可达 0.95，重复室内可能较低；固定高门限会用数据自身也达不到的标准拒绝正确合并（D68）。
    SeamCheck checkSeam(const Reconstruction& m, const std::set<uint32_t>& src_side,
                        double max_error_px = 8.0, double min_pair_frac = 0.5,
                        size_t max_pairs = 600, bool crossing = true) const {
        SeamCheck sc;
        std::vector<const TwoViewMatches*> cross;
        for (const TwoViewMatches& p : db_.pairs) {
            auto ia = m.images.find(p.image1);
            auto ib = m.images.find(p.image2);
            if (ia == m.images.end() || ib == m.images.end()) continue;
            if (!ia->second.registered || !ib->second.registered) continue;
            if ((src_side.count(p.image1) == src_side.count(p.image2)) == crossing) continue;
            cross.push_back(&p);
        }
        sc.cross_pairs = cross.size();
        if (cross.empty()) return sc;
        // 优先证据最多的图像对，再分散取样，数百个强连接足以判断接缝。
        std::stable_sort(cross.begin(), cross.end(), [](const TwoViewMatches* a,
                                                        const TwoViewMatches* b) {
            return a->matches.size() > b->matches.size();
        });
        if (cross.size() > max_pairs) cross.resize(max_pairs);

        const double scale = modelScaleOf(m);
        std::vector<double> fracs;
        for (const TwoViewMatches* p : cross) {
            double frac = pairAgreement(m, *p, max_error_px, scale);
            if (frac < 0) continue;
            fracs.push_back(frac);
            sc.tested++;
            if (frac >= min_pair_frac) sc.agree++;
        }
        if (!fracs.empty()) {
            std::sort(fracs.begin(), fracs.end());
            sc.median_frac = fracs[fracs.size() / 2];
        }
        return sc;
    }

    // ---------------- 沿违反的几何关系拆分模型（D45）----------------
    // 只保留模型能重现的验证边；若形成两个大连通组，则可能以错误位姿拼接，即使各自内部完美，BA 也无法分离。
    // 拆分后可重新合并或独立增长，过小组撤销配准留待后续处理。
    struct SplitStats {
        size_t pairs_tested = 0, pairs_agree = 0;
        size_t groups = 0;          // 相互一致图像的连通组
        size_t largest = 0;
        size_t dropped_images = 0;  // 过小组中被丢弃的图像数
        // 排序后的逐对解释比例可区分真实冲突的双峰分布与门限偏紧的平滑离散。
        std::vector<double> fractions;
        double percentile(double q) const {
            if (fractions.empty()) return 0;
            size_t i = (size_t)(q * (double)(fractions.size() - 1));
            return fractions[i];
        }
    };

    std::vector<Reconstruction> splitInconsistent(const Reconstruction& m, double max_error_px,
                                                  double min_pair_frac, int min_matches,
                                                  size_t min_group, SplitStats* out = nullptr) const {
        SplitStats st;
        std::vector<uint32_t> ids;
        std::map<uint32_t, size_t> pos;
        for (const auto& kv : m.images)
            if (kv.second.registered) { pos[kv.first] = ids.size(); ids.push_back(kv.first); }
        if (ids.size() < 2) return {m};

        std::vector<size_t> parent(ids.size());
        for (size_t i = 0; i < parent.size(); i++) parent[i] = i;
        std::function<size_t(size_t)> find = [&](size_t x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        const double scale = modelScaleOf(m);
        for (const TwoViewMatches& p : db_.pairs) {
            if ((int)p.matches.size() < min_matches) continue;
            auto a = pos.find(p.image1);
            auto b = pos.find(p.image2);
            if (a == pos.end() || b == pos.end()) continue;
            double frac = pairAgreement(m, p, max_error_px, scale);
            if (frac < 0) continue;
            st.pairs_tested++;
            st.fractions.push_back(frac);
            if (frac < min_pair_frac) continue;
            st.pairs_agree++;
            size_t ra = find(a->second), rb = find(b->second);
            if (ra != rb) parent[ra] = rb;
        }
        // 已标定 rig 同帧图像由固定外参连接，即使背靠背鱼眼无共同匹配也不能拆开；4000 图数据曾因此按镜头分裂后又合并。
        if (rigs_) {
            std::map<std::pair<uint32_t, uint32_t>, size_t> frame_of;
            for (size_t i = 0; i < ids.size(); i++) {
                const RigSlot sl = rigs_->slot(ids[i]);
                if (!sl.valid() || m.rig_detached.count(ids[i]) || sl.rig >= m.rigs.size() ||
                    !m.rigs[sl.rig].usable(sl.member))
                    continue;
                auto it = frame_of.emplace(std::make_pair(sl.rig, sl.frame), i).first;
                size_t ra = find(i), rb = find(it->second);
                if (ra != rb) parent[ra] = rb;
            }
        }

        std::sort(st.fractions.begin(), st.fractions.end());
        std::map<size_t, std::vector<uint32_t>> groups;
        for (size_t i = 0; i < ids.size(); i++) groups[find(i)].push_back(ids[i]);
        std::vector<std::vector<uint32_t>> gs;
        for (auto& kv : groups) gs.push_back(std::move(kv.second));
        std::sort(gs.begin(), gs.end(),
                  [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
                      return a.size() > b.size();
                  });
        st.groups = gs.size();
        st.largest = gs.empty() ? 0 : gs[0].size();
        if (out) *out = st;
        if (gs.size() <= 1) return {m};

        std::vector<Reconstruction> parts;
        for (const std::vector<uint32_t>& g : gs) {
            if (g.size() < min_group) { st.dropped_images += g.size(); continue; }
            std::set<uint32_t> keep(g.begin(), g.end());
            parts.push_back(subsetModel(m, keep));
        }
        if (out) *out = st;
        if (parts.empty()) return {m};
        return parts;
    }

    // 为重复结构检查提供有足够匹配支持的图像对查询，每次调用从验证列表构建一次。
    MatchedFn matchedPredicate(int min_matches = 15) const {
        auto index = std::make_shared<std::set<std::pair<uint32_t, uint32_t>>>();
        for (const TwoViewMatches& p : db_.pairs) {
            if ((int)p.matches.size() < min_matches) continue;
            uint32_t a = p.image1, b = p.image2;
            if (a > b) std::swap(a, b);
            index->insert({a, b});
        }
        return [index](uint32_t a, uint32_t b) {
            if (a > b) std::swap(a, b);
            return index->count({a, b}) > 0;
        };
    }

    // 跨模型联合 BA 按相机组共享内参，使小分量获得大分量约束；中间层可用 coarse 容差，最终审查与精化保持严格。
    void jointRefine(std::vector<Reconstruction>& models, bool coarse = false) {
        ensureSetup();
        size_t live = 0;
        for (const Reconstruction& m : models)
            if (m.numRegistered() >= 2) live++;
        if (live < 2) return;
        BundleOptions bo;
        bo.real = baReal(coarse);
        bo.device = opt_.device;
        bo.device_selector = opt_.device_selector;
        bo.threads = opt_.threads;
        bo.verbose = false;
        bo.loss = opt_.ba_loss;
        bo.loss_param = (float)(opt_.ba_loss_param * medianPixelScale());
        bo.refine_principal_point = opt_.refine_principal_point || final_.pp;
        bo.refine_extra_params = opt_.refine_extra_params || final_.extra;
        bo.pp_min_images = opt_.pp_min_images;
        bo.solver = opt_.ba_solver;
        if (coarse && opt_.ba_growth_rtol > 0) {
            bo.rtol = opt_.ba_growth_rtol;
            bo.patience = opt_.ba_growth_patience;
        }
        bo.shared_ctx = &baContext(coarse);
        bo.over_budget_throws = true;
        bo.rigs = rigs_;
        bo.refine_rigs = opt_.refine_rigs;
        for (Reconstruction& m : models) calibrateRigs(m);
        std::vector<PosePriors> pfs(models.size());
        std::vector<const PosePriors*> pp(models.size(), nullptr);
        if (priors_)
            for (size_t i = 0; i < models.size(); i++) {
                pfs[i] = priorFactors(models[i]);
                pp[i] = &pfs[i];
            }
        // 先尝试整问题，超预算则按模型分批；5356 图在 2.2 倍重叠覆盖下形成 11564 图像实例，可超过 8 GB 显存。
        // 模型间只共享内参、不共享三维点，因此可以分批，但各批必须统一最终相机参数（D65）。
        for (int batches = 1;; ) {
            try {
                jointRefineBatched(models, bo, batches, pp);
                break;
            } catch (const BAOverBudget& e) {
                const int want =
                    std::max(batches + 1, (int)std::ceil(e.need_mb / e.budget_mb * batches));
                if (want > 64) throw;
                if (opt_.verbose)
                    slog::diag(slog::Tag::Map,
                               "[map] the joint solve needs %.0f MB against a %.0f MB budget: "
                               "splitting it %d ways", e.need_mb, e.budget_mb, want);
                batches = want;
            }
        }
        // 联合求解后不立即逐模型重新拟合内参，避免抵消共享效果；不再解释的观测由后续审查或增长过滤。
        clearCameraConsensus();
        for (const Reconstruction& m : models) recordCameras(m);
    }

private:
    // 按模型从大到小轮流分批，使各批具有代表性；首批含最大模型，其内参作为全部批次最终答案，后续批以此初始化位姿和点优化。
    void jointRefineBatched(std::vector<Reconstruction>& models, const BundleOptions& bo,
                            int batches, const std::vector<const PosePriors*>& pp) {
        if (batches <= 1) {
            runJointBA(models, bo, &pp);
            return;
        }
        std::vector<size_t> order(models.size());
        std::iota(order.begin(), order.end(), (size_t)0);
        std::stable_sort(order.begin(), order.end(), [&models](size_t a, size_t b) {
            return models[a].numRegistered() > models[b].numRegistered();
        });
        std::vector<std::vector<Reconstruction*>> group((size_t)batches);
        std::vector<std::vector<const PosePriors*>> gpp((size_t)batches);
        for (size_t k = 0; k < order.size(); k++) {
            group[k % (size_t)batches].push_back(&models[order[k]]);
            gpp[k % (size_t)batches].push_back(pp[order[k]]);
        }
        std::map<uint32_t, Camera> shared;
        for (size_t b = 0; b < group.size(); b++) {
            if (group[b].size() < 2) continue;
            // 用首批确定的内参初始化当前批，避免从各原子旧值重新分歧。
            if (b)
                for (Reconstruction* m : group[b])
                    for (auto& kv : m->cameras) {
                        auto it = shared.find(kv.first);
                        if (it != shared.end()) kv.second = it->second;
                    }
            runJointBA(group[b], bo, &gpp[b]);
            if (!b)
                for (const Reconstruction* m : group[b])
                    for (const auto& kv : m->cameras) shared.emplace(kv.first, kv.second);
        }
        for (Reconstruction& m : models)
            for (auto& kv : m.cameras) {
                auto it = shared.find(kv.first);
                if (it != shared.end()) kv.second = it->second;
            }
    }

public:

    // ---------------- 子模型间相机共识（D45）----------------
    // 已接纳模型按支持图像数量发布内参，后续子模型继承最佳约束结果；不能让最新的小模型覆盖大模型的可靠参数。
    void recordCameras(const Reconstruction& m) {
        std::map<uint32_t, double> weight;
        for (const auto& kv : m.images)
            if (kv.second.registered) weight[kv.second.camera_id] += 1.0;
        for (const auto& kv : weight) {
            auto it = m.cameras.find(kv.first);
            if (it == m.cameras.end()) continue;
            auto cur = cam_consensus_.find(kv.first);
            if (cur == cam_consensus_.end() || kv.second > cur->second.second)
                cam_consensus_[kv.first] = {it->second, kv.second};
        }
    }

    // 清空旧发布结果，再由当前模型集合重建，避免已拆分或修复模型继续投票。
    void clearCameraConsensus() { cam_consensus_.clear(); }

    // 新模型将采用的初始内参，供报告使用。
    const std::map<uint32_t, std::pair<Camera, double>>& cameraConsensus() const {
        return cam_consensus_;
    }

    // 用图像未参与构造的三维结构寻找明显更强且位置不同的替代位姿；现有轨迹会随错误图像一起移动，不能作为独立证据。
    // 噪声池无法产生可靠替代时保留原位姿；找到后返回 alternative，由调用方移动并通过精化过滤，而非直接删除。
    bool poseContradicted(uint32_t img, Pose& alternative) const {
        std::vector<Vec3> X, br;
        const Image& im = rec_.images.at(img);
        for (uint32_t f = 0; f < feats_[img].count(); f++) {
            if (im.point3D_ids[f] != kInvalidPoint3D) continue;  // 图像自身带来的证据
            for (const Correspondence& c : graph_.at(img, f)) {
                if (c.image_id == img) continue;
                const Image& oi = rec_.images.at(c.image_id);
                if (!oi.registered) continue;
                uint64_t pid = oi.point3D_ids[c.feature_idx];
                if (pid == kInvalidPoint3D) continue;
                auto pt = rec_.points3D.find(pid);
                if (pt == rec_.points3D.end()) continue;
                X.push_back(pt->second.xyz);
                br.push_back(bearing(img, f));
                break;
            }
        }
        const int n = (int)X.size();
        if (n < opt_.audit_min_evidence) return false;  // 没有足够外部证据反驳

        // 统计当前位姿对同一证据池的解释能力。
        const double thr = camOf(img).errRad(opt_.max_reproj_error);
        const double thr2 = thr * thr;
        int cur = 0;
        for (int k = 0; k < n; k++) cur += pnpResidualSq(im.pose, X[k], br[k]) < thr2 ? 1 : 0;

        // 若当前支持已使任何替代解都不可能达到优势比例，则精确跳过 RANSAC，不改变判定。
        // 7620 图审查曾对 5107 图运行 RANSAC 却只移动 6 图，此提前退出可省去大部分稳定模型审查。
        if (n <= (int)(opt_.audit_alternative_factor * cur) || n < opt_.audit_min_alternative) {
            if (audit_dump_)
                slog::diag(slog::Tag::Map,
                           "[audit] %s: pool %d, current %d -> ok (no alternative can win)",
                           db_.images[img].name.c_str(), n, cur);
            return false;
        }

        PnPResult r = ransacPnP(X, br, camOf(img).focal(), errPx(img), 0,
                                opt_.audit_ransac_trials);
        bool contradicted = false;
        double rot_deg = 0, shift = 0;
        // 审查不要求整个池的内点比例；错误图像的未拼接噪声池很大，应比较绝对支持及相对当前位姿的优势。
        if (r.success && r.num_inliers >= opt_.audit_min_alternative &&
            r.num_inliers > (int)(opt_.audit_alternative_factor * cur)) {
            // 替代位姿必须真的不同，同时检查旋转与中心位移；重复立面或楼层可能仅位置错而朝向相同。
            Mat3 D = mul(r.pose.R, transpose(im.pose.R));
            double tr = std::max(-1.0, std::min(1.0, (D[0] + D[4] + D[8] - 1) * 0.5));
            rot_deg = std::acos(tr) * 180.0 / M_PI;
            shift = (cameraCenter(r.pose) - cameraCenter(im.pose)).norm() / modelScale();
            contradicted =
                rot_deg > opt_.audit_min_rotation_deg || shift > opt_.audit_min_shift_frac;
            // 序列邻居更支持当前位姿时，其他位置的重复结构不能将其替换（D79）。
            if (contradicted && seq_ && nearVouches(img, im.pose, r.pose)) contradicted = false;
            // 陀螺同样可支持符合实测转动的当前位姿。
            if (contradicted && priors_ && priorVouches(img, im.pose, r.pose)) {
                contradicted = false;
                prior_vouched_++;
            }
            if (contradicted) alternative = r.pose;
        }
        if (audit_dump_)
            slog::diag(slog::Tag::Map,
                       "[audit] %s: pool %d, current %d, alternative %d, rot %.1f deg, shift %.4f "
                       "-> %s", db_.images[img].name.c_str(), n, cur,
                       r.success ? r.num_inliers : 0, rot_deg, shift,
                       contradicted ? "CONTRADICTED" : "ok");
        return contradicted;
    }

    // 检查当前位姿对序列邻居三维点的支持是否至少达到最少 PnP 内点且多于替代位姿。
    bool nearVouches(uint32_t img, const Pose& cur, const Pose& alt) const {
        const double thr = camOf(img).errRad(opt_.max_reproj_error);
        const double thr2 = thr * thr;
        int n_cur = 0, n_alt = 0;
        for (uint32_t f = 0; f < feats_[img].count(); f++)
            for (const Correspondence& c : graph_.at(img, f)) {
                if (!nearby(img, c.image_id)) continue;
                const Image& oi = rec_.images.at(c.image_id);
                if (!oi.registered) continue;
                const uint64_t pid = oi.point3D_ids[c.feature_idx];
                if (pid == kInvalidPoint3D) continue;
                auto pt = rec_.points3D.find(pid);
                if (pt == rec_.points3D.end()) continue;
                const Vec3 b = bearing(img, f);
                n_cur += pnpResidualSq(cur, pt->second.xyz, b) < thr2 ? 1 : 0;
                n_alt += pnpResidualSq(alt, pt->second.xyz, b) < thr2 ? 1 : 0;
                break;
            }
        return n_cur >= opt_.min_num_pnp_inliers && n_cur > n_alt;
    }

    // ---------------- 传感器先验 ----------------

    std::vector<PosedImage> posedImages(const Reconstruction& rec) const {
        std::vector<PosedImage> out;
        for (const auto& kv : rec.images) {
            if (!kv.second.registered) continue;
            PosedImage p;
            p.image = kv.first;
            p.camera = kv.first < cam_ids_.size() ? cam_ids_[kv.first] : kv.second.camera_id;
            p.pose = kv.second.pose;
            out.push_back(p);
        }
        return out;
    }

    // 在 rec 自身坐标规范中构建求解因子。
    PosePriors priorFactors(const Reconstruction& rec) {
        if (!priors_) return PosePriors{};
        PosePriors pf = priors_->factors(posedImages(rec));
        prior_stats_.rotations = pf.rotations.size();
        prior_stats_.ups = pf.ups.size();
        prior_stats_.centres = pf.centres.size();
        return pf;
    }

    // 选择最紧旋转先验的已放置邻居，预测 img 的旋转；无可用邻居时失败。
    bool priorRotation(uint32_t img, Mat3& R, double& sigma_deg) const {
        if (!priors_) return false;
        bool have = false;
        for (uint32_t j : priors_->neighbours(img)) {
            auto it = rec_.images.find(j);
            if (it == rec_.images.end() || !it->second.registered) continue;
            Mat3 Rji;
            double sig;
            if (!priors_->relativeRotation(j, img, Rji, sig)) continue;
            const double deg = sig * 180.0 / M_PI;
            if (have && deg >= sigma_deg) continue;
            R = mul(Rji, it->second.pose.R);
            sigma_deg = deg;
            have = true;
        }
        return have;
    }

    double priorTolDeg(double sigma_deg) const {
        return std::max(opt_.prior_rot_tol_deg, 3.0 * sigma_deg);
    }

    // 检查当前旋转符合陀螺而替代旋转不符合。
    bool priorVouches(uint32_t img, const Pose& cur, const Pose& alt) const {
        Mat3 Rp;
        double sig;
        if (!priorRotation(img, Rp, sig)) return false;
        const double tol = priorTolDeg(sig);
        return rotationAngleDeg(mul(cur.R, transpose(Rp))) <= tol &&
               rotationAngleDeg(mul(alt.R, transpose(Rp))) > tol;
    }

    // PnP 旋转偏离陀螺时固定旋转重解，替换 r；支持不足返回 false 并拒绝配准。
    bool priorCheckPose(uint32_t img, const std::vector<Vec3>& X, const std::vector<Vec3>& br,
                        PnPResult& r, bool& constrained) {
        constrained = false;
        Mat3 Rp;
        double sig;
        if (!priorRotation(img, Rp, sig)) return true;
        const double tol = priorTolDeg(sig);
        if (rotationAngleDeg(mul(r.pose.R, transpose(Rp))) <= tol) return true;
        // 先验仅角度级精度，先放宽半径求平移，再在先验容差内自由精化，最终仍按图像自身半径判定。
        const Camera& cam = camOf(img);
        const double loose_px = std::max(errPx(img), tol * M_PI / 180.0 * cam.focal());
        PnPResult k = ransacPnPKnownRotation(X, br, Rp, cam.focal(), loose_px);
        if (k.success && k.num_inliers >= opt_.min_num_pnp_inliers) {
            Pose refined = k.pose;
            if (refinePose(X, br, k.inlier_mask, refined) &&
                rotationAngleDeg(mul(refined.R, transpose(Rp))) <= tol)
                k.pose = refined;
            classify(img, X, br, k);
        }
        if (prior_dump_)
            slog::diag(slog::Tag::Map,
                       "[prior] %s: PnP rotation %.1f deg off the gyro's (tol %.1f); held pose "
                       "%d/%zu inliers against %d", db_.images[img].name.c_str(),
                       rotationAngleDeg(mul(r.pose.R, transpose(Rp))), tol,
                       k.success ? k.num_inliers : 0, X.size(), r.num_inliers);
        if (!k.success || k.num_inliers < opt_.min_num_pnp_inliers) {
            prior_stats_.refused++;
            return false;
        }
        r = k;
        constrained = true;
        prior_stats_.corrected++;
        return true;
    }

    // 以已配准相机中心到均值的 RMS 作为无单位模型尺度；按接入模型缓存，避免审查逐图重复计算。
    double modelScale() const {
        if (scale_cache_ > 0) return scale_cache_;
        Vec3 c{0, 0, 0};
        size_t n = 0;
        for (const auto& kv : rec_.images)
            if (kv.second.registered) { c = c + cameraCenter(kv.second.pose); n++; }
        if (!n) return scale_cache_ = 1.0;
        c = c * (1.0 / (double)n);
        double s = 0;
        for (const auto& kv : rec_.images)
            if (kv.second.registered) {
                Vec3 d = cameraCenter(kv.second.pose) - c;
                s += d.dot(d);
            }
        return scale_cache_ = std::max(1e-12, std::sqrt(s / (double)n));
    }

    // 为尚未接入 rec_ 的候选合并模型计算相同尺度。
    static double modelScaleOf(const Reconstruction& m) {
        Vec3 c{0, 0, 0};
        size_t n = 0;
        for (const auto& kv : m.images)
            if (kv.second.registered) { c = c + cameraCenter(kv.second.pose); n++; }
        if (!n) return 1.0;
        c = c * (1.0 / (double)n);
        double s = 0;
        for (const auto& kv : m.images)
            if (kv.second.registered) {
                Vec3 d = cameraCenter(kv.second.pose) - c;
                s += d.dot(d);
            }
        return std::max(1e-12, std::sqrt(s / (double)n));
    }

    // 将种子与配准限制到数据库子集，空值表示全部；复用对应图、特征和双视图缓存。
    void restrictTo(const std::vector<uint32_t>& images) {
        ensureSetup();
        // 切换子集时重置种子阶梯与已用种子，避免继承前一问题的放宽程度。
        used_seeds_.clear();
        init_relax_ = seed_phase_ = 0;
        seed_pair_ = nullptr;
        seeded_.clear();
        seed_cand_valid_ = false;  // 由 allowed 过滤
        if (images.empty()) {
            allow_.clear();
            allow_count_ = db_.images.size();
            return;
        }
        allow_.assign(db_.images.size(), 0);
        for (uint32_t i : images)
            if (i < allow_.size()) allow_[i] = 1;
        allow_count_ = images.size();
    }
    bool allowed(uint32_t img) const { return allow_.empty() || allow_[img]; }
    // 当前允许处理的图像集合，未限制时为整个数据库。
    size_t allowedCount() const { return allow_.empty() ? db_.images.size() : allow_count_; }

    // 首次 BA 前可指定调用方上下文；原子工作线程复用各自设备，不能跨线程共享，避免每原子重复昂贵设备创建。
    void useBaContext(VkContext* ctx) { ext_ba_ctx_ = ctx; }

    // 返回全局焦距初始化后的组内参，原子直接继承并禁止自行搜索，避免少量图像决定焦距（D48）。
    const std::map<uint32_t, Camera>& startingCameras() const { return default_cams_; }

    // setup 后解析出的逐图像相机组，比可能为空的构造参数更明确。
    const std::vector<uint32_t>& cameraIds() const { return cam_ids_; }

    size_t unclaimed() const { return unclaimedImages(); }
    size_t numImages() const { return db_.images.size(); }
    const MapperOptions& options() const { return opt_; }
    MapperOptions& options() { return opt_; }

private:
    // 接入已有模型位姿、相机与三维点，重新建立轨迹；未包含图像保持 resetModel 的未配准状态。
    void adopt(const Reconstruction& m) {
        size_t missing = 0, name_mismatch = 0, count_mismatch = 0;
        if (rigs_) {
            rec_.rigs = m.rigs;
            rec_.rig_detached = m.rig_detached;
            initRigCalib(rec_);
        }
        // point2D_idx 必须对应本次特征数组，关键点数量不同意味着压缩策略或输入不一致，不能直接恢复。
        std::set<uint32_t> reindexed;
        // 按本次分组接收磁盘相机参数，保留当前图像尺寸与 pixel_scale；后者不存于 cameras.bin（D47）。
        std::set<uint32_t> adopted;
        for (const auto& kv : m.images) {
            if (!kv.second.registered) continue;
            auto im = rec_.images.find(kv.first);
            if (im == rec_.images.end()) continue;
            auto mc = m.cameras.find(kv.second.camera_id);
            auto cur = rec_.cameras.find(im->second.camera_id);
            if (mc == m.cameras.end() || cur == rec_.cameras.end()) continue;
            if (!adopted.insert(cur->first).second) continue;
            const int w = cur->second.width, h = cur->second.height;
            const double ps = cur->second.pixel_scale;
            cur->second = mc->second;
            cur->second.id = cur->first;
            cur->second.width = w;
            cur->second.height = h;
            cur->second.pixel_scale = ps;
            focal_known_.insert(cur->first);
        }
        for (const auto& kv : m.images) {
            if (!kv.second.registered) continue;
            auto it = rec_.images.find(kv.first);
            if (it == rec_.images.end()) { missing++; continue; }
            // 图像 ID 是当前数据库位置，必须核对名称以拒绝其他数据库模型；磁盘名带扩展名，内部名不带。
            if (!kv.second.name.empty()) {
                const std::string& want = it->second.name;
                if (kv.second.name.compare(0, want.size(), want) != 0 ||
                    (kv.second.name.size() > want.size() && kv.second.name[want.size()] != '.'))
                    name_mismatch++;
            }
            if (kv.second.points2D.size() != it->second.points2D.size()) {
                count_mismatch++;
                reindexed.insert(kv.first);
            }
            it->second.pose = kv.second.pose;
            it->second.registered = true;
        }
        rec_.points3D.clear();
        rec_.next_point3D_id = 1;
        for (const auto& kv : m.points3D) {
            std::vector<TrackElement> tr;
            for (const TrackElement& e : kv.second.track) {
                auto it = rec_.images.find(e.image_id);
                if (it == rec_.images.end() || !it->second.registered) continue;
                if (reindexed.count(e.image_id)) continue;
                if (e.point2D_idx >= it->second.point3D_ids.size()) continue;
                if (it->second.point3D_ids[e.point2D_idx] != kInvalidPoint3D) continue;
                tr.push_back(e);
            }
            if (tr.size() >= 2) {
                uint64_t id = rec_.addPoint3D(kv.second.xyz, tr);
                rec_.points3D[id].rgb[0] = kv.second.rgb[0];
                rec_.points3D[id].rgb[1] = kv.second.rgb[1];
                rec_.points3D[id].rgb[2] = kv.second.rgb[2];
            }
        }
        if (name_mismatch)
            slog::diag(slog::Tag::Map,
                       "[map] WARNING: %zu adopted image(s) have a different name than "
                       "the database entry with the same id -- the model was probably "
                       "built from other matches, "
                       "and adopting it will produce nonsense", name_mismatch);
        if (count_mismatch)
            slog::diag(slog::Tag::Map,
                       "[map] WARNING: %zu adopted image(s) hold a different keypoint count than "
                       "this run's features -- their observations index other keypoints and were "
                       "dropped, poses kept. Feature compaction on one side only does this; "
                       "--compact-unused-features has to match the run that wrote the model",
                       count_mismatch);
        if (missing && opt_.verbose)
            slog::diag(slog::Tag::Map, "[map] adopted model: %zu image(s) not in this database",
                       missing);
    }


    // 统一排序、报告与返回，供 run 的多个出口使用。
    std::vector<Reconstruction> finishRun(std::vector<Reconstruction>& models,
                                          std::chrono::steady_clock::time_point prof_start) {
        // 按三维点数降序写出，sparse/0 为结构最多的模型，而非最先发现者。
        std::stable_sort(models.begin(), models.end(),
                         [](const Reconstruction& a, const Reconstruction& b) {
                             return a.points3D.size() > b.points3D.size();
                         });
        if (opt_.verbose) {
            // 按不同图像统计，模型间重叠不能重复计数。
            std::set<uint32_t> covered;
            for (const Reconstruction& m : models)
                for (const auto& kv : m.images)
                    if (kv.second.registered) covered.insert(kv.first);
            slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_done,
                     {(long long)models.size(), (long long)covered.size(),
                      (long long)db_.images.size()});
            for (size_t i = 0; i < models.size(); i++)
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_model_line,
                         {(long long)i, (long long)models[i].numRegistered(),
                          (long long)models[i].points3D.size()});
            // 分别报告配准失败类型：候选不足通常是匹配或覆盖问题，比例低常表示位置歧义，所需修复不同。
            if (rigs_)
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_rig_summary,
                         {(long long)reg_by_rig_, (long long)reg_rig_word_});
            if (seq_)
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_sequence_summary,
                         {(long long)reg_near_won_, (long long)reg_vouched_});
            if (priors_)
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_prior_summary,
                         {(long long)prior_stats_.corrected, (long long)prior_stats_.refused,
                          (long long)prior_stats_.rotations, (long long)prior_stats_.ups,
                          (long long)prior_stats_.centres});
            if (covered.size() < db_.images.size())
                slog::diag(slog::Tag::Map,
                           "[map] registration attempts that failed: %u too few candidates, "
                           "%u too few PnP inliers, %u inlier ratio below %.2f, %u lost on refit "
                           "(%u came in on absolute support with the ratio failed, %u more were "
                           "refused for a rival pose fitting the leftovers; %u correspondence(s) "
                           "were left out of the ratio as unseeable)",
                           reg_fail_.few_corr, reg_fail_.few_inliers, reg_fail_.low_ratio,
                           opt_.min_pnp_inlier_ratio, reg_fail_.refined_out, reg_fail_.strong,
                           reg_fail_.ambiguous, reg_fail_.occluded);
        }
        g_map_prof.report(std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - prof_start).count());
        return models;
    }

private:
    // ---------------- 多模型记录（D41）----------------
    // 快照仅保留已配准图像、所用相机和着色点；rec_ 本身复用全数据库记录。
    // 每图 8192 特征约 200 KB，未裁剪的 2000 图模型即使只配准 20 图也占约 400 MB。
    Reconstruction snapshotModel() const {
        Reconstruction m = rec_;
        for (auto it = m.images.begin(); it != m.images.end();)
            it = it->second.registered ? std::next(it) : m.images.erase(it);
        // 删除没有已配准图像使用的相机，避免把无依据的默认内参写入 cameras.bin。
        std::set<uint32_t> used;
        for (const auto& kv : m.images) used.insert(kv.second.camera_id);
        for (auto it = m.cameras.begin(); it != m.cameras.end();)
            it = used.count(it->first) ? std::next(it) : m.cameras.erase(it);
        assignColors(m);
        return m;
    }

    // 记录保留模型已配准图像以阻止重复播种，但允许在重叠预算内重新配准，供后续合并。
    void claimImages(const Reconstruction& m) {
        if (model_count_.size() != db_.images.size()) model_count_.assign(db_.images.size(), 0);
        for (const auto& kv : m.images)
            if (kv.second.registered) model_count_[kv.first]++;
    }

    size_t unclaimedImages() const {
        size_t n = 0;
        for (uint32_t i = 0; i < model_count_.size(); i++) n += (model_count_[i] == 0 && allowed(i));
        return n;
    }

    bool claimed(uint32_t img) const {
        return img < model_count_.size() && model_count_[img] > 0;
    }

    // 共享图像预算由绝对下限加新增覆盖收益决定；仍发现新区域时允许继续，耗尽新区域后仅保留足够 Sim(3) 对齐的重叠。
    size_t overlapBudget(size_t fresh) const {
        const double floor_v = (double)std::max(1, opt_.max_model_overlap);
        return (size_t)std::max(floor_v, opt_.model_overlap_ratio * (double)fresh);
    }

    // 统计 m 中已被其他保留模型覆盖的图像数。
    size_t overlapWithKept(const Reconstruction& m) const {
        if (model_count_.empty()) return 0;
        size_t n = 0;
        for (const auto& kv : m.images)
            if (kv.second.registered && model_count_[kv.first] > 0) n++;
        return n;
    }

    // 只按新增覆盖判断子模型是否值得保留，不因有用重叠惩罚它。
    // 7620 图数据按绝对重叠拒绝曾丢弃 2751/1751 图模型，仅因共享 212/27 图；后续仅恢复其中 1188 图（D66）。
    bool admitModel(const Reconstruction& m, std::string& why) const {
        const uint32_t reg = m.numRegistered();
        const size_t fresh = reg - overlapWithKept(m);
        if (fresh < (size_t)opt_.min_model_size) {
            char buf[160];
            snprintf(buf, sizeof buf, "%u images but only %zu not already covered", reg, fresh);
            why = buf;
            return false;
        }
        return true;
    }

    // 对正在构建的模型做相同判断；主模型尚无认领记录，检查近乎零成本。
    size_t sharedRegistered() const {
        if (model_count_.empty()) return 0;
        size_t n = 0;
        for (const auto& kv : rec_.images)
            if (kv.second.registered && model_count_[kv.first] > 0) n++;
        return n;
    }

    // 按轨迹平均提取阶段保存的颜色，无需重新解码图像；无有效颜色时保持默认中性灰。
    void pointColor(const Point3D& p, uint8_t rgb[3]) const {
        uint32_t acc[3] = {0, 0, 0}, n = 0;
        for (const TrackElement& e : p.track) {
            const FeatureSet& fs = feats_[e.image_id];
            if (!fs.hasColors()) continue;
            const uint8_t* c = &fs.colors[(size_t)e.point2D_idx * 3];
            acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]; n++;
        }
        if (!n) return;
        for (int k = 0; k < 3; k++) rgb[k] = (uint8_t)((acc[k] + n / 2) / n);
    }

    void assignColors(Reconstruction& rec) const {
        for (auto& kv : rec.points3D) pointColor(kv.second, kv.second.rgb);
    }

    // 持续增长直到无图可配准或耗尽本轮共享图像预算；主模型阶段没有认领限制（D41/D66）。
    void grow() {
        growLoop();
        checkedRefine(true);
    }

    // 单独增长循环返回新增数量，max_reg 限制总规模，stop_at_ba 在需要优化时返回交由调用方联合求解。
    // 无新增时跳过最终精化，避免无谓开销和数值扰动。
    uint32_t growLoop(uint32_t max_reg = 0, bool stop_at_ba = false) {
        uint32_t registered_here = 0;
        // BA 触发数量相对当前模型规模设置，不能使用固定 3 图门槛，否则接入大模型后会每新增一图就精化或停止。
        double next_ba = std::max(3.0, std::ceil(rec_.numRegistered() * opt_.ba_growth_ratio));
        recent_regs_.clear();
        rebuildScores();
        // 重叠预算只计算本轮新增共享，不能把模型原本重叠计入。
        const size_t shared_at_entry = sharedRegistered();
        while (true) {
            cancel::check();
            if (max_reg && rec_.numRegistered() >= max_reg) break;
            // 两项均为本轮增量，途中撤销配准可能减少计数，减法须防止无符号回绕。
            const size_t shared_now = sharedRegistered();
            const size_t shared_here = shared_now > shared_at_entry ? shared_now - shared_at_entry : 0;
            const size_t fresh_here = registered_here > shared_here ? registered_here - shared_here : 0;
            if (shared_here > overlapBudget(fresh_here)) {
                if (opt_.verbose)
                    slog::diag(slog::Tag::Map,
                               "[map] growth took %zu image(s) an earlier model holds "
                               "against %zu of its own; stopping it", shared_here, fresh_here);
                break;
            }
            // 按候选排序逐个尝试，成功配准一图后再重排；一次失败不永久排除图像，后续三维结构可能提供更多支持（D15）。
            std::vector<uint32_t> cands;
            {
                ProfTimer pt(g_map_prof.choose);
                g_map_prof.n_choose++;
                cands = chooseNextImages();
            }
            if (cands.empty()) break;
            bool registered = false;
            for (uint32_t img : cands) {
                reg_trials_[img]++;
                g_map_prof.n_reg_try++;
                bool ok;
                {
                    ProfTimer pt(g_map_prof.reg);
                    ok = registerImage(img);
                }
                if (ok) {
                    g_map_prof.n_reg_ok++;
                    registered = true;
                    registered_here++;
                    {
                        ProfTimer pt(g_map_prof.tri);
                        triangulateForImage(img);
                    }
                    recent_regs_.push_back(img);
                    registered_here += completeFrameOf(img) + frame_regs_;
                    frame_regs_ = 0;
                    break;
                }
            }
            if (!registered) break;  // 排序中的所有候选均无法配准
            if (rec_.numRegistered() >= next_ba) {
                if (stop_at_ba) break;
                checkedRefine(false);
                // 精化可能整体改变观测、轨迹或恢复快照，须重建增量评分缓存。
                rebuildScores();
                registered_here += completeRigFrames();
                // 撤销配准可能缩小模型，下次 BA 门槛按实际保留规模更新。
                next_ba = std::ceil(rec_.numRegistered() * opt_.ba_growth_ratio);
            }
        }
        return registered_here;
    }

    // 事务式全局精化先备份，若模型崩塌则恢复并撤销上次精化后新增的可疑图像，保留剩余重试预算（D36）。
    // 仅备份已配准图像，未配准记录按不变量全为空；5400 图中的 40 图原子可减少约 135 倍复制。
    struct Snapshot {
        std::map<uint32_t, Camera> cameras;
        std::map<uint64_t, Point3D> points3D;
        uint64_t next_point3D_id = 1;
        std::vector<std::pair<uint32_t, Image>> images;  // 仅已配准图像
        std::set<uint32_t> focal_known;
        std::vector<RigCalib> rigs;
        std::set<uint32_t> rig_detached;
    };

    Snapshot takeSnapshot() const {
        Snapshot s;
        s.cameras = rec_.cameras;
        s.points3D = rec_.points3D;
        s.next_point3D_id = rec_.next_point3D_id;
        s.focal_known = focal_known_;
        s.rigs = rec_.rigs;
        s.rig_detached = rec_.rig_detached;
        for (const auto& kv : rec_.images)
            if (kv.second.registered) s.images.emplace_back(kv.first, kv.second);
        return s;
    }

    void restoreSnapshot(Snapshot& s) {
        rec_.cameras = std::move(s.cameras);
        rec_.points3D = std::move(s.points3D);
        rec_.next_point3D_id = s.next_point3D_id;
        focal_known_ = std::move(s.focal_known);
        rec_.rigs = std::move(s.rigs);
        rec_.rig_detached = std::move(s.rig_detached);
        std::set<uint32_t> was;
        for (const auto& kv : s.images) was.insert(kv.first);
        // 通常精化只会撤销图像，此循环为空；仍保留处理以免恢复依赖该假设。
        for (auto& kv : rec_.images) {
            if (!kv.second.registered || was.count(kv.first)) continue;
            kv.second.registered = false;
            kv.second.pose = {mat3Identity(), {0, 0, 0}};
            std::fill(kv.second.point3D_ids.begin(), kv.second.point3D_ids.end(),
                      kInvalidPoint3D);
        }
        for (auto& kv : s.images) rec_.images[kv.first] = std::move(kv.second);
    }

    void checkedRefine(bool final_pass) {
        uint32_t reg_before = rec_.numRegistered();
        Snapshot snap;
        {
            ProfTimer pt(g_map_prof.snapshot);
            snap = takeSnapshot();
        }
        globalRefine(final_pass);
        // 丢失一半已配准图像才算崩塌；仅丢大量噪声观测不算，植被初始模型正常过滤可丢三分之二观测。
        bool collapsed = 2 * rec_.numRegistered() < reg_before;
        if (collapsed && !recent_regs_.empty()) {
            if (opt_.verbose)
                slog::diag(slog::Tag::Map,
                           "[map] refinement collapsed the model (%u -> %u images); undoing %zu "
                           "recent registration(s)", reg_before, rec_.numRegistered(),
                           recent_regs_.size());
            restoreSnapshot(snap);
            for (uint32_t img : recent_regs_) deregisterImage(img);
            resetOrphanCameras();
        }
        recent_regs_.clear();
    }

    // ---------------- 辅助函数 ----------------
    Vec2 kp(uint32_t img, uint32_t f) const {
        const Keypoint& k = feats_[img].keypoints[f];
        return {k.x, k.y};
    }
    // 图像内参与尺寸可不同，每次投影须使用该图像自己的相机。
    const Camera& camOf(uint32_t img) const {
        return rec_.cameras.at(rec_.images.at(img).camera_id);
    }
    // 统一把提取像素阈值换为 img 的源图像素，使用 Camera::pixel_scale（D47）。
    double errPx(uint32_t img) const { return camOf(img).errPx(opt_.max_reproj_error); }
    // GPU 全问题只接受单个尺度时，取已配准相机尺度中位数；同提取尺度时精确，混合分辨率时为折中。
    double medianPixelScale() const {
        std::vector<double> v;
        for (const auto& kv : rec_.images)
            if (kv.second.registered) v.push_back(camOf(kv.first).pixel_scale);
        if (v.empty()) {
            for (const auto& kv : rec_.cameras) v.push_back(kv.second.pixel_scale);
            if (v.empty()) return 1.0;
        }
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }
    // 图像 img 特征 f 的单位视线，几何核心统一使用，兼容鱼眼（D31）。
    Vec3 bearing(uint32_t img, uint32_t f) const { return camOf(img).bearing(kp(img, f)); }
    Mat34 Pmat(uint32_t img) const {
        const Pose& p = rec_.images.at(img).pose;
        return {p.R[0], p.R[1], p.R[2], p.t.x, p.R[3], p.R[4], p.R[5], p.t.y,
                p.R[6], p.R[7], p.R[8], p.t.z};
    }
    // 普通针孔沿用 p.z>0，宽角使用 p·bearing>0 判断点是否沿观测射线前向，避免拒绝合法的负 z 鱼眼点（D33）。
    static bool inFront(const Camera& cam, const Vec3& pc, const Vec3& b, double zmin) {
        return cam.wideFov() ? pc.dot(b) > 0 : pc.z >= zmin;
    }
    double reprojErr(uint32_t img, uint32_t f, const Vec3& X) const {
        const Pose& p = rec_.images.at(img).pose;
        const Camera& cam = camOf(img);
        Vec3 pc = mul(p.R, X) + p.t;
        // 针孔保持 pc.z<1e-8 的原判据；宽角沿视线判断正深度，球面任意方向均可见，复用已有视线即可（D33）。
        if (cam.wideFov()) {
            if (pc.dot(cam.bearing(kp(img, f))) <= 0) return 1e30;
        } else if (pc.z < 1e-8) {
            return 1e30;
        }
        Vec2 px = cam.project(pc);
        Vec2 o = kp(img, f);
        return std::hypot(px.x - o.x, px.y - o.y);
    }

    // ---------------- 模型平铺索引 ----------------
    // 将稠密图像 ID 映射为指针数组，避免每观测多次 std::map 查找；仅在图像、相机集合不增删时有效，修改已有节点字段无妨。
    // 保留 pixel_scale 而非预乘阈值，使调用处浮点乘法结合顺序不变，避免末位差异改变边界判定。
    struct ModelIndex {
        std::vector<Image*> img;          // 按图像 ID，缺失时为空
        std::vector<Camera*> cam;         // 该图像的相机
        std::vector<double> pixel_scale;  // Camera::pixel_scale，提取到源图比例（D47）
    };

    ModelIndex indexModel() {
        ModelIndex mi;
        const size_t n = db_.images.size();
        mi.img.assign(n, nullptr);
        mi.cam.assign(n, nullptr);
        mi.pixel_scale.assign(n, 1.0);
        for (size_t i = 0; i < n; i++) {
            auto it = rec_.images.find((uint32_t)i);
            if (it == rec_.images.end()) continue;
            mi.img[i] = &it->second;
            auto ct = rec_.cameras.find(it->second.camera_id);
            if (ct == rec_.cameras.end()) continue;
            mi.cam[i] = &ct->second;
            mi.pixel_scale[i] = ct->second.pixel_scale;
        }
        return mi;
    }

    // 通过索引计算重投影误差，保持算术与正深度规则一致。
    double reprojErrAt(const ModelIndex& mi, uint32_t img, uint32_t f, const Vec3& X) const {
        const Camera& cam = *mi.cam[img];
        return reprojErrAt(mi, img, f, X, cam.wideFov() ? cam.bearing(kp(img, f)) : Vec3{});
    }
    // 允许传入已算视线，避免鱼眼重三角化中数百万次重复 Newton 反解。
    double reprojErrAt(const ModelIndex& mi, uint32_t img, uint32_t f, const Vec3& X,
                       const Vec3& bearing) const {
        const Pose& p = mi.img[img]->pose;
        const Camera& cam = *mi.cam[img];
        Vec3 pc = mul(p.R, X) + p.t;
        if (cam.wideFov()) {
            if (pc.dot(bearing) <= 0) return 1e30;
        } else if (pc.z < 1e-8) {
            return 1e30;
        }
        Vec2 px = cam.project(pc);
        Vec2 o = kp(img, f);
        return std::hypot(px.x - o.x, px.y - o.y);
    }

    // 通过平铺索引执行 triangulatePair。
    bool triangulatePairAt(const ModelIndex& mi, uint32_t a, uint32_t fa, uint32_t b, uint32_t fb,
                           Vec3& X, double err_scale = 1.0) const {
        return triangulatePairAt(mi, a, fa, mi.cam[a]->bearing(kp(a, fa)), b, fb,
                                 mi.cam[b]->bearing(kp(b, fb)), X, err_scale);
    }
    bool triangulatePairAt(const ModelIndex& mi, uint32_t a, uint32_t fa, const Vec3& ba,
                           uint32_t b, uint32_t fb, const Vec3& bb, Vec3& X,
                           double err_scale = 1.0) const {
        const Pose& pa_ = mi.img[a]->pose;
        const Pose& pb_ = mi.img[b]->pose;
        const Camera& ca = *mi.cam[a];
        const Camera& cb = *mi.cam[b];
        Mat34 Pa{pa_.R[0], pa_.R[1], pa_.R[2], pa_.t.x, pa_.R[3], pa_.R[4], pa_.R[5], pa_.t.y,
                 pa_.R[6], pa_.R[7], pa_.R[8], pa_.t.z};
        Mat34 Pb{pb_.R[0], pb_.R[1], pb_.R[2], pb_.t.x, pb_.R[3], pb_.R[4], pb_.R[5], pb_.t.y,
                 pb_.R[6], pb_.R[7], pb_.R[8], pb_.t.z};
        X = triangulateDLT(Pa, Pb, ba, bb);
        Vec3 pa = mul(pa_.R, X) + pa_.t;
        Vec3 pb = mul(pb_.R, X) + pb_.t;
        if (!inFront(ca, pa, ba, 1e-6) || !inFront(cb, pb, bb, 1e-6)) return false;
        double ang = triangulationAngle(X, cameraCenter(pa_), cameraCenter(pb_));
        if (ang * 180.0 / M_PI < opt_.min_tri_angle_deg) return false;
        if (reprojErrAt(mi, a, fa, X, ba) > err_scale * (opt_.max_reproj_error * mi.pixel_scale[a]))
            return false;
        if (reprojErrAt(mi, b, fb, X, bb) > err_scale * (opt_.max_reproj_error * mi.pixel_scale[b]))
            return false;
        return true;
    }

    // 三角化 (a,fa)<->(b,fb)，检查正深度、角度和重投影；重三角化用 err_scale<1 收紧接受阈值，避免删除后立即重建的反复震荡（D36）。
    bool triangulatePair(uint32_t a, uint32_t fa, uint32_t b, uint32_t fb, Vec3& X,
                         double err_scale = 1.0) const {
        Vec3 ba = bearing(a, fa), bb = bearing(b, fb);
        X = triangulateDLT(Pmat(a), Pmat(b), ba, bb);
        Vec3 pa = mul(rec_.images.at(a).pose.R, X) + rec_.images.at(a).pose.t;
        Vec3 pb = mul(rec_.images.at(b).pose.R, X) + rec_.images.at(b).pose.t;
        if (!inFront(camOf(a), pa, ba, 1e-6) || !inFront(camOf(b), pb, bb, 1e-6)) return false;
        double ang = triangulationAngle(X, cameraCenter(rec_.images.at(a).pose),
                                        cameraCenter(rec_.images.at(b).pose));
        if (ang * 180.0 / M_PI < opt_.min_tri_angle_deg) return false;
        if (reprojErr(a, fa, X) > err_scale * errPx(a)) return false;
        if (reprojErr(b, fb, X) > err_scale * errPx(b)) return false;
        return true;
    }

    // 初始化幂等，图与逐图记录属于数据库而非具体模型，所有公共入口均可调用。
    void ensureSetup() {
        if (!setup_done_) {
            setup();
            setup_done_ = true;
        }
    }

    void setup() {
        if (cam_ids_.size() != db_.images.size()) cam_ids_.assign(db_.images.size(), 1);
        // 每相机 ID 从首张使用图像确定尺寸，保留原始默认值，便于全部配准被撤销后摆脱错误内参重新开始。
        for (uint32_t i = 0; i < db_.images.size(); i++) {
            uint32_t cid = cam_ids_[i];
            if (rec_.cameras.count(cid)) continue;
            auto proto = opt_.initial_cameras.find(cid);
            rec_.cameras[cid] = proto != opt_.initial_cameras.end()
                                    ? proto->second
                                    : Camera::defaultFor(cid, feats_[i].width, feats_[i].height,
                                                         opt_.focal, opt_.camera_model);
            rec_.cameras[cid].id = cid;
            default_cams_[cid] = rec_.cameras[cid];
        }
        // 所有重置均保留焦距先验集合。
        for (uint32_t cid : opt_.known_focal_cameras)
            if (rec_.cameras.count(cid)) focal_known_.insert(cid);
        for (uint32_t i = 0; i < db_.images.size(); i++) {
            Image im;
            im.id = i;
            im.camera_id = cam_ids_[i];
            im.name = db_.images[i].name;  // 特征主干名，真实文件名由 CLI 恢复
            im.exif_orientation = feats_[i].exif_orientation;
            im.points2D.resize(feats_[i].count());
            im.point3D_ids.assign(feats_[i].count(), kInvalidPoint3D);
            for (uint32_t f = 0; f < feats_[i].count(); f++) im.points2D[f] = kp(i, f);
            rec_.images[i] = im;
        }
        graph_.build(db_, [&] {
            std::vector<uint32_t> nf(db_.images.size());
            for (size_t i = 0; i < db_.images.size(); i++) nf[i] = feats_[i].count();
            return nf;
        }());
        reg_trials_.assign(db_.images.size(), 0);
        if (rigs_) initRigCalib(rec_);
        // 首次 grow 前就可能由种子 BA 的重三角化添加观测，因此先分配评分记录；具体值由首次排序前重建。
        rebuildScores();
    }

    // 清空前一种子尝试的模型状态，使下一尝试从初始化状态开始。
    void resetModel() {
        scale_cache_ = 0;
        rig_refined_at_ = 0;
        rec_.points3D.clear();
        rec_.cameras.clear();
        if (rigs_) {
            rec_.rigs.clear();
            rec_.rig_detached.clear();
            initRigCalib(rec_);
        }
        focal_known_ = opt_.known_focal_cameras;
        for (uint32_t i = 0; i < db_.images.size(); i++) {
            uint32_t cid = cam_ids_[i];
            if (!rec_.cameras.count(cid)) {
                // 优先继承先前模型充分约束的内参并关闭焦距搜索，禁止新小模型的单图匹配覆盖大模型结果（D45）。
                auto cons = cam_consensus_.find(cid);
                if (cons != cam_consensus_.end()) {
                    rec_.cameras[cid] = cons->second.first;
                    focal_known_.insert(cid);
                } else {
                    rec_.cameras[cid] = default_cams_.at(cid);
                }
            }
            Image& im = rec_.images[i];
            im.registered = false;
            im.pose = {mat3Identity(), {0, 0, 0}};
            im.point3D_ids.assign(feats_[i].count(), kInvalidPoint3D);
        }
        reg_trials_.assign(db_.images.size(), 0);
    }

    // ---------------- 初始化 ----------------
    // 分别报告内点不足、低视差与前向运动等种子拒绝原因，便于判断应改善匹配、基线还是运动方式。
    struct InitTally {
        size_t candidates = 0, no_pose = 0, config = 0, few_inliers = 0, forward = 0,
               few_points = 0, low_angle = 0;
        double best_angle = 0;      // 候选达到的最大角度中位数
        double best_forward = 2.0;  // 候选达到的最大侧向运动量
    };

    void reportInitFailure() const {
        const InitTally& t = init_tally_;
        slog::diag(slog::Tag::Map, "[map] initialization failed: %zu candidate pair(s) tried; "
                   "rejected %zu no pose, %zu wrong config, %zu too few inliers, "
                   "%zu forward motion, %zu too few points, %zu low angle",
                   t.candidates, t.no_pose, t.config, t.few_inliers, t.forward, t.few_points,
                   t.low_angle);
        if (t.candidates) {
            slog::diag(slog::Tag::Map, "[map]   best median triangulation angle %.2f deg, "
                       "most sideways baseline %.3f (cap %.2f)",
                       t.best_angle, t.best_forward, opt_.init_max_forward_motion);
            if (t.best_angle < opt_.init_min_tri_angle_deg / 8)
                slog::diag(slog::Tag::Map, "[map]   no pair has enough parallax to triangulate on: "
                           "the capture may be a pure rotation, or too small a sweep");
        }
    }

    // ---------------- 旋转退化数据的焦距初始化（D48）----------------
    // 所有相机近同向时，沿视轴拉伸场景并调整焦距可重现相同图像，纯平移使焦距不可观，畸变仅弱打破退化。
    // 实测直行 KITTI 姿态跨度 8.5 度，而 18 个基准场景均约 180 度。
    double rotationSpreadDeg() const {
        std::vector<const Mat3*> Rs;
        for (const auto& kv : rec_.images)
            if (kv.second.registered) Rs.push_back(&kv.second.pose.R);
        // 最大姿态差本为 O(n²)，采样约 60 相机即可在现有数据上接近真实最大值一度以内，足以判定退化。
        const size_t kMax = 60;
        const size_t stride = std::max<size_t>(1, Rs.size() / kMax);
        double worst = 0;
        for (size_t i = 0; i < Rs.size(); i += stride)
            for (size_t j = i + stride; j < Rs.size(); j += stride) {
                const Mat3& A = *Rs[i];
                const Mat3& B = *Rs[j];
                // trace(A^T B)，等于对应列点积之和。
                double tr = 0;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++) tr += A[r * 3 + c] * B[r * 3 + c];
                double cs = std::min(1.0, std::max(-1.0, 0.5 * (tr - 1.0)));
                worst = std::max(worst, std::acos(cs) * 180.0 / M_PI);
            }
        return worst;
    }

    // 焦距试探恢复用户初始畸变，不能继承其他试探拟合系数，也不能丢弃用户标定。
    void resetExtraParams(Camera& c) const {
        Camera src;
        auto it = opt_.initial_cameras.find(c.id);
        if (it != opt_.initial_cameras.end()) src = it->second;
        c.k1 = src.k1; c.k2 = src.k2; c.p1 = src.p1; c.p2 = src.p2;
        c.k3 = src.k3; c.k4 = src.k4; c.k5 = src.k5; c.k6 = src.k6;
        c.sx1 = src.sx1; c.sy1 = src.sy1;
    }

    struct FocalTrial {
        bool ok = false;
        double focal = 0;
        uint32_t registered = 0;
        size_t observations = 0;
        double rot_spread = 0;
    };

    // 同种子构建小模型，以共同一致观测数而非重投影残差评分焦距；退化场景错误焦距也可拟合自身模型。
    // 25 帧 KITTI 正确焦距得到 7.35 万观测，1.9 倍焦距为 5.2 万，而两者残差均约 0.29 px。
    FocalTrial focalTrial(const TwoViewMatches& pm, const std::vector<uint32_t>& cams,
                          double focal, uint32_t cap) {
        for (uint32_t cid : cams) {
            default_cams_[cid].setFocal(focal);
            resetExtraParams(default_cams_[cid]);
        }
        resetModel();
        FocalTrial t;
        t.focal = focal;
        if (!trySeedPair(pm, 0.0, 0, true, false)) return t;
        globalRefine(false);
        growLoop(cap);
        globalRefine(false);
        t.ok = rec_.numRegistered() >= 3;
        t.registered = rec_.numRegistered();
        t.observations = countObservations();
        t.rot_spread = rotationSpreadDeg();
        // 试探已通过 BA 精化焦距，返回实际收敛值。
        if (!cams.empty() && rec_.cameras.count(cams[0])) t.focal = rec_.cameras[cams[0]].focal();
        return t;
    }

    // 仅搜索仍为几何猜测且未获相机共识的焦距；先前模型发布的内参视为测量，不应由新试探替换。
    std::vector<uint32_t> guessedFocalCameras() const {
        std::vector<uint32_t> out;
        for (const auto& kv : rec_.cameras)
            if (!opt_.given_focal_cameras.count(kv.first) && !cam_consensus_.count(kv.first) &&
                !kv.second.isSpherical())  // 球面模型没有可搜索焦距（D49）
                out.push_back(kv.first);
        return out;
    }

    // 无测量时从同一种子按焦距减半下降，只有一致观测显著增加才继续；已在正确收敛区的初值仅多付出一次小模型试探。
    void bootstrapFocalLength() {
        if (opt_.focal_trials <= 0) return;
        std::vector<uint32_t> cams = guessedFocalCameras();
        if (cams.empty()) return;
        // 试探结果会丢弃，不能计入正式配准进度；单独命名阶段，避免小数据进度提前跳高后停滞。
        struct SeedPhase {
            bool& report;
            const bool was;
            explicit SeedPhase(bool& f) : report(f), was(f) {
                if (was) events::stage_begin(Stage::Seed);
                f = false;
            }
            ~SeedPhase() {
                report = was;
                if (was) events::stage_end(Stage::Seed);
            }
        } phase(opt_.report_progress);
        // 仅对单相机组执行全局比例试探，多组镜头需要不同答案，单一得分无法定位出错组，应依赖组级测量或标定（D46）。
        if (cams.size() > 1) return;
        const double f0 = rec_.cameras.at(cams[0]).focal();

        // 用普通种子增长到足以评分的小模型。
        const uint32_t cap = (uint32_t)std::max<size_t>(
            8, std::min<size_t>(opt_.focal_model_size, db_.images.size()));
        size_t from = 0;
        resetModel();
        if (!initialize(from) || !seed_pair_) { restoreAfterBootstrap(f0, cams, f0); return; }
        const TwoViewMatches& pm = *seed_pair_;
        globalRefine(false);
        growLoop(cap);
        globalRefine(false);
        FocalTrial base;
        base.ok = true;
        base.focal = rec_.cameras.at(cams[0]).focal();
        base.registered = rec_.numRegistered();
        base.observations = countObservations();
        base.rot_spread = rotationSpreadDeg();
        if (opt_.verbose)
            slog::diag(slog::Tag::Map,
                       "[map] focal probe at %.0f: %u image(s), %zu observation(s), "
                       "orientations span %.1f deg%s", f0, base.registered, base.observations,
                       base.rot_spread,
                       base.rot_spread <= opt_.focal_max_rot_spread_deg
                       ? " (too little to determine the focal)" : "");

        FocalTrial best = base;
        bool moved = false;
        int used = 0;
        // 测量焦距已在合理范围，仅接受试探 BA 精化，不执行按配准数择优的减半搜索。
        if (opt_.measured_focal_cameras.count(cams[0])) {
            if (opt_.verbose)
                slog::diag(slog::Tag::Map,
                           "[map] focal %.0f -> %.0f (probe refinement of a measured "
                           "focal)", f0, best.focal);
            restoreAfterBootstrap(best.focal, cams, f0);
            return;
        }
        // 优先向下搜索：过短焦距扩散视线，BA 可回升，KITTI 从真实值 0.36–1.0 倍均能收敛；过长焦距易形成自洽扁平模型而无法摆脱。
        double f = f0;
        while (used < opt_.focal_trials) {
            f *= 0.5;
            used++;
            FocalTrial t = focalTrial(pm, cams, f, cap);
            const bool better = t.ok && (double)t.observations >
                                        (1.0 + opt_.focal_min_gain) * (double)best.observations;
            if (opt_.verbose)
                slog::diag(slog::Tag::Map,
                           "[map]   focal %.0f -> %.0f: %u image(s), %zu observation(s)%s",
                           f, t.focal, t.registered, t.observations,
                           !t.ok ? " (failed)" : better ? " (better)" : " (no better, stopping)");
            if (!better) break;
            best = t;
            moved = true;
        }
        // 向下无改善时向上试一次，覆盖初始猜测低于真实值 2–4 倍的长焦数据。
        if (!moved && used < opt_.focal_trials) {
            FocalTrial t = focalTrial(pm, cams, f0 * 1.6, cap);
            const bool better = t.ok && (double)t.observations >
                                        (1.0 + opt_.focal_min_gain) * (double)base.observations;
            if (opt_.verbose)
                slog::diag(slog::Tag::Map,
                           "[map]   focal %.0f -> %.0f: %u image(s), %zu observation(s)%s",
                           f0 * 1.6, t.focal, t.registered, t.observations,
                           !t.ok ? " (failed)" : better ? " (better)" : " (no better)");
            if (better) { best = t; moved = true; }
        }
        // 无论搜索是否有显著提升，均保留试探 BA 精化后的焦距，而非原始猜测。
        if (opt_.verbose)
            slog::diag(slog::Tag::Map,
                       "[map] focal %.0f -> %.0f (%s; %zu observations vs %zu at the "
                       "guess)", f0, best.focal,
                       moved ? "trial reconstruction" : "probe refinement only",
                       best.observations, base.observations);
        restoreAfterBootstrap(best.focal, cams, f0);
    }

    // 恢复试探扰动的种子、放宽阶梯和模型状态，仅保留焦距；只有最终焦距等于缓存生成时值，才保留可复用双视图几何。
    // 否则缓存失效，避免试探结果泄漏；完全无法播种且焦距未变时保留缓存可省去整套重复 RANSAC。
    void restoreAfterBootstrap(double focal, const std::vector<uint32_t>& cams,
                               double probe_focal) {
        for (uint32_t cid : cams) {
            Camera fresh = default_cams_.at(cid);
            fresh.setFocal(focal);
            resetExtraParams(fresh);
            default_cams_[cid] = fresh;
        }
        if (std::fabs(focal - probe_focal) > 1e-9) seed_geom_.clear();
        used_seeds_.clear();
        seed_pair_ = nullptr;
        init_relax_ = seed_phase_ = 0;
        resetModel();
    }

    // 种子游标继续上次尝试之后的位置；先四级逐步放宽 16 度中位视差和 100 内点要求，再四级解除前向运动限制。
    // 始终优先侧向种子，直行数据最终也可尝试；各级仍检查真实三角化角度，不能把运动方向当作唯一质量判据。
    struct InitLevel { double angle; int inliers; bool allow_forward; };

    InitLevel initLevel(int level) const {
        InitLevel l{opt_.init_min_tri_angle_deg, opt_.init_min_inliers, level >= 4};
        for (int r = level % 4; r > 0; r--) {
            l.angle = std::max(l.angle * 0.5, 2.0);
            l.inliers = std::max(l.inliers / 2, 2 * TwoViewOptions().min_num_inliers);
        }
        return l;
    }

    bool initialize(size_t& from) {
        init_tally_ = InitTally();
        // 种子重试间保留放宽等级及序列阶段，先邻居后全部图像对，保证游标索引含义稳定（D19/D79）。
        const int levels = opt_.init_max_forward_motion >= 1.0 ? 4 : 8;
        const int phases = seq_ ? 2 : 1;
        for (; seed_phase_ < phases; seed_phase_++, init_relax_ = 0, from = 0) {
            for (; init_relax_ < levels; init_relax_++, from = 0) {
                const InitLevel l = initLevel(init_relax_);
                if (init_relax_ && from == 0 && opt_.verbose)
                    slog::out(slog::Tag::Map,
                             l.allow_forward ? spirula::i18n::msg::sfm::map_seed_relax_forward
                                             : spirula::i18n::msg::sfm::map_seed_relax,
                             {(long long)l.inliers, slog::num(l.angle, 0)});
                if (initializeAttempt(from, l.angle, l.inliers, l.allow_forward)) return true;
            }
            if (seed_phase_ == 0 && seq_ && opt_.verbose)
                slog::diag(slog::Tag::Map,
                           "[map] no seed among sequence neighbours; trying every pair");
        }
        return false;
    }

    bool initializeAttempt(size_t& from, double min_ang_deg, int min_inliers,
                           bool allow_forward) {
        // 已验证非平面候选按内点数排序，每个限制子集只构建一次；放宽仅降低阈值，无需每级重复扫描排序整个匹配列表。
        if (!seed_cand_valid_) {
            seed_cand_.clear();
            seed_cand_far_.clear();
            for (const TwoViewMatches& p : db_.pairs) {
                if (p.config != (int)TwoViewConfig::Uncalibrated || !allowed(p.image1) ||
                    !allowed(p.image2))
                    continue;
                // 优先序列邻居，其他候选等到第二阶段。
                (!seq_ || nearby(p.image1, p.image2) ? seed_cand_ : seed_cand_far_).push_back(&p);
            }
            auto by_inliers = [](auto* a, auto* b) {
                if (a->matches.size() != b->matches.size())
                    return a->matches.size() > b->matches.size();
                return a->image1 != b->image1 ? a->image1 < b->image1 : a->image2 < b->image2;
            };
            std::sort(seed_cand_.begin(), seed_cand_.end(), by_inliers);
            std::sort(seed_cand_far_.begin(), seed_cand_far_.end(), by_inliers);
            seed_cand_valid_ = true;
        }
        const std::vector<const TwoViewMatches*>& cand = seed_phase_ ? seed_cand_far_ : seed_cand_;

        // 候选接受顺序仍串行，但可并行预计算无状态 seedGeometry，按原顺序读取缓存；预取块逐步加倍，避免小原子首候选即成功时浪费计算。
        // 单线程完全禁用预取，34 原子只需 84 次 RANSAC，启用预取曾执行 2578 次。
        const unsigned hc = std::thread::hardware_concurrency();
        const size_t max_block =
            std::max<size_t>(1, opt_.threads > 0 ? (size_t)opt_.threads : (hc ? hc : 1));
        size_t block = 1;
        size_t prefetched = max_block > 1 ? from : cand.size();

        for (size_t ci = from; ci < cand.size(); ci++) {
            const TwoViewMatches* p = cand[ci];
            // 候选已按内点数降序，首个不足当前阈值的候选即可结束本级。
            if ((int)p->matches.size() < min_inliers) break;
            uint32_t a = p->image1, b = p->image2;
            // 放宽级别会重新扫描，须额外记录已用种子，防止重复消耗重试预算。
            if (used_seeds_.count({a, b})) continue;
            // 子模型仅从未覆盖处起步；主模型重试虽尚未认领，也须避开前一次已到达区域（D41/D58）。
            if (claimed(a) || claimed(b) || seedBlocked(a) || seedBlocked(b)) continue;
            if (ci >= prefetched) {
                prefetched = std::min(cand.size(), ci + block);
                prefetchSeedGeometry(cand, ci, prefetched);
                block = std::min(max_block, block * 2);
            }
            if (!trySeedPair(*p, min_ang_deg, min_inliers, allow_forward, true)) continue;
            used_seeds_.insert({a, b});
            from = ci + 1;
            return true;
        }
        return false;
    }

    // 种子几何只依赖原始相机与特征，不修改重建，可缓存并在线程外预计算。
    TwoViewGeometry seedGeometry(const TwoViewMatches& pm) const {
        ProfTimer pt(g_map_prof.seed_geom);
        g_map_prof.n_seed_geom++;
        const uint32_t a = pm.image1, b = pm.image2;
        TwoViewOptions tvo;
        tvo.recover_pose = true;
        // 可在验证保留的内点上重判平面/全景；非平面 H 的内点率低，RANSAC 试验多，550 图测量中约占种子搜索 90%。
        tvo.estimate_homography = opt_.seed_homography;
        const Camera& ca = camOf(a);
        const Camera& cb = camOf(b);
        Mat3 Rp;
        double sig;
        if (priors_ && priors_->relativeRotation(a, b, Rp, sig)) {
            // 陀螺旋转配合两点平移若可解释自由估计的支持，则使用测量旋转建立种子。
            std::vector<Vec3> b1(pm.matches.size()), b2(pm.matches.size());
            for (size_t k = 0; k < pm.matches.size(); k++) {
                b1[k] = ca.bearing(kp(a, pm.matches[k].idx1));
                b2[k] = cb.bearing(kp(b, pm.matches[k].idx2));
            }
            tvo.ransac.max_error =
                0.5 * (ca.errRad(tvo.ransac.max_error) + cb.errRad(tvo.ransac.max_error));
            TwoViewGeometry g = estimateTwoViewBearing(b1, b2, tvo);
            KnownRotationOptions ko;
            ko.ransac = tvo.ransac;
            ko.min_num_inliers = tvo.min_num_inliers;
            ko.max_rotation_only_ratio = tvo.max_H_inlier_ratio;
            ko.rot_sigma = sig;
            ko.start = g.has_pose ? &g.pose : nullptr;
            const KnownRotationGeometry k = estimateTwoViewKnownRotation(b1, b2, Rp, ko);
            const int free_inl = g.config == TwoViewConfig::Uncalibrated ? g.num_inliers : 0;
            if (k.ok && !k.panoramic && k.num_inliers >= 0.7 * (double)std::max(free_inl, 1)) {
                prior_seeds_++;
                g.config = TwoViewConfig::Uncalibrated;
                g.inlier_mask = k.inlier_mask;
                g.num_inliers = k.num_inliers;
                g.pose = k.pose;
                g.has_pose = true;
            }
            return g;
        }
        if (ca.wideFov() || cb.wideFov()) {
            // 鱼眼种子使用单位视线，避免丢弃宽角对应；像素阈值按焦距换成弧度（D45）。
            std::vector<Vec3> b1(pm.matches.size()), b2(pm.matches.size());
            for (size_t k = 0; k < pm.matches.size(); k++) {
                b1[k] = ca.bearing(kp(a, pm.matches[k].idx1));
                b2[k] = cb.bearing(kp(b, pm.matches[k].idx2));
            }
            tvo.ransac.max_error =
                0.5 * (ca.errRad(tvo.ransac.max_error) + cb.errRad(tvo.ransac.max_error));
            return estimateTwoViewBearing(b1, b2, tvo);
        }
        std::vector<Vec2> q1(pm.matches.size()), q2(pm.matches.size());
        for (size_t k = 0; k < pm.matches.size(); k++) {
            q1[k] = kp(a, pm.matches[k].idx1);
            q2[k] = kp(b, pm.matches[k].idx2);
        }
        tvo.K1 = ca.K();
        tvo.K2 = cb.K();
        return estimateTwoView(q1, q2, tvo);
    }

    // 线程池预计算候选几何，跳过缓存项，缓存 map 本身仅由调用线程写入。
    void prefetchSeedGeometry(const std::vector<const TwoViewMatches*>& cand, size_t lo,
                              size_t hi) {
        std::vector<const TwoViewMatches*> todo;
        for (size_t i = lo; i < hi; i++) {
            const TwoViewMatches* p = cand[i];
            std::pair<uint32_t, uint32_t> key{p->image1, p->image2};
            if (used_seeds_.count(key) || claimed(key.first) || claimed(key.second)) continue;
            if (seed_geom_.count(key)) continue;
            todo.push_back(p);
        }
        if (todo.size() < 2) return;  // 单个图像对不值得启动线程池
        std::vector<TwoViewGeometry> out(todo.size());
        std::atomic<size_t> next{0};
        const unsigned hc = std::thread::hardware_concurrency();
        const size_t want = opt_.threads > 0 ? (size_t)opt_.threads : std::max(1u, hc);
        const size_t nt = std::min<size_t>(todo.size(), want);
        std::vector<std::thread> pool;
        pool.reserve(nt);
        for (size_t t = 0; t < nt; t++)
            pool.emplace_back([&] {
                for (size_t i = next++; i < todo.size(); i = next++)
                    out[i] = seedGeometry(*todo[i]);
            });
        for (std::thread& t : pool) t.join();
        for (size_t i = 0; i < todo.size(); i++)
            seed_geom_[{todo[i]->image1, todo[i]->image2}] = std::move(out[i]);
    }

    // 构建双相机种子，通过当前阈值则保留，否则回滚为空；焦距试探复用同一候选对以保证得分可比。
    bool trySeedPair(const TwoViewMatches& pm, double min_ang_deg, int min_inliers,
                     bool allow_forward, bool memoize) {
        const TwoViewMatches* p = &pm;
        const uint32_t a = p->image1, b = p->image2;
        // 固定原始相机与确定性估计器下，重扫种子的 RANSAC 结果可缓存；焦距试探改变相机，必须关闭 memoize。
        TwoViewGeometry g;
        auto cached = memoize ? seed_geom_.find({a, b}) : seed_geom_.end();
        if (cached != seed_geom_.end()) {
            g = cached->second;
        } else {
            g = seedGeometry(pm);
            if (memoize) seed_geom_[{a, b}] = g;
        }
        init_tally_.candidates++;
        if (!g.has_pose) { init_tally_.no_pose++; return false; }
        if (g.config != TwoViewConfig::Uncalibrated) { init_tally_.config++; return false; }
        if (g.num_inliers < min_inliers) { init_tally_.few_inliers++; return false; }

        // 近纯前向运动使窄视场射线围绕极点退化；球面相机侧向仍有充分视差，不应应用此上限（D49）。
        double fwd;
        {
            Vec3 c2 = cameraCenter(g.pose).normalized();  // 相机 1 位于原点
            fwd = std::max(std::fabs(c2.z),
                           std::fabs(c2.x * g.pose.R[6] + c2.y * g.pose.R[7] +
                                     c2.z * g.pose.R[8]));
            init_tally_.best_forward = std::min(init_tally_.best_forward, fwd);
            const bool spherical = camOf(a).isSpherical() && camOf(b).isSpherical();
            if (!allow_forward && !spherical && fwd > opt_.init_max_forward_motion) {
                init_tally_.forward++;
                return false;
            }
        }

        // 建立两个相机并三角化内点。
        rec_.images[a].pose = {mat3Identity(), {0, 0, 0}};
        rec_.images[a].registered = true;
        rec_.images[b].pose = g.pose;
        rec_.images[b].registered = true;

        std::vector<double> angles;
        int created = 0;
        for (size_t k = 0; k < p->matches.size(); k++) {
            if (!g.inlier_mask[k]) continue;
            uint32_t fa = p->matches[k].idx1, fb = p->matches[k].idx2;
            Vec3 X;
            if (!triangulatePair(a, fa, b, fb, X)) continue;
            double ang = triangulationAngle(X, cameraCenter(rec_.images[a].pose),
                                            cameraCenter(rec_.images[b].pose));
            angles.push_back(ang * 180.0 / M_PI);
            rec_.addPoint3D(X, {{a, fa}, {b, fb}});
            created++;
        }
        double medAng = 0;
        if (!angles.empty()) {
            std::sort(angles.begin(), angles.end());
            medAng = angles[angles.size() / 2];
        }
        init_tally_.best_angle = std::max(init_tally_.best_angle, medAng);
        if (created < std::max(30, min_inliers / 2)) init_tally_.few_points++;
        else if (medAng < min_ang_deg) init_tally_.low_angle++;
        if (created >= std::max(30, min_inliers / 2) && medAng >= min_ang_deg) {
            // 种子相机即将由首次全局 BA 精化，禁止后续焦距扫描覆盖。
            focal_known_.insert(rec_.images[a].camera_id);
            focal_known_.insert(rec_.images[b].camera_id);
            // 种子正式接受后才计入进度，回滚候选不能虚增配准数量。
            if (opt_.report_progress) {
                events::map_placed(a);
                events::map_placed(b);
            }
            seed_pair_ = &pm;
            seed_forward_ = fwd;
            completeFrameOf(a);
            completeFrameOf(b);
            if (opt_.verbose)
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_init_pair,
                         {(long long)a, (long long)b, (long long)created, slog::num(medAng, 1),
                          (fwd > 0.5 ? spirula::i18n::msg::sfm::baseline_forward
                                     : spirula::i18n::msg::sfm::baseline_sideways).get()});
            return true;
        }
        // 回滚并尝试下一候选。
        rollbackInit(a, b);
        return false;
    }

    // 种子候选只修改两张图，回滚不扫描全数据库；3000 帧视频逐候选清理全部图像曾产生约 80 MB 写入。
    void rollbackInit(uint32_t a, uint32_t b) {
        rec_.points3D.clear();
        rec_.next_point3D_id = 1;
        for (uint32_t i : {a, b}) {
            Image& im = rec_.images[i];
            im.registered = false;
            std::fill(im.point3D_ids.begin(), im.point3D_ids.end(), kInvalidPoint3D);
        }
    }

    // ---------------- 图像配准 ----------------
    // 此函数作为二维、三维支持计数的参考实现，增量 score_cache 保持同值，SS_SFM_SCORE_CHECK 可逐次核对。
    int score(uint32_t img) const {
        if (rec_.images.at(img).registered) return -1;
        int n = 0;
        for (uint32_t f = 0; f < feats_[img].count(); f++)
            for (const Correspondence& c : graph_.at(img, f))
                if (rec_.images.at(c.image_id).registered &&
                    rec_.images.at(c.image_id).point3D_ids[c.feature_idx] != kInvalidPoint3D) {
                    n++;
                    break;  // 每个特征只计一次
                }
        return n;
    }

    // support_[i][f] 记录通向已配准三维特征的对应数，score_cache 计支持非零的特征；添加观测增量更新，精化、撤销、重播种后整体重建。
    // 避免每次配准重复评分全部未配准图像，此开销曾占 279 图数据建图时间三分之一（D37）。
    void rebuildScores() {
        ProfTimer pt(g_map_prof.choose);
        if (support_.size() != db_.images.size()) {
            support_.resize(db_.images.size());
            for (size_t i = 0; i < db_.images.size(); i++)
                support_[i].assign(feats_[i].count(), 0);
            score_cache_.assign(db_.images.size(), 0);
            pyramid_.assign(db_.images.size(), std::vector<uint16_t>(kPyrCells, 0));
            pyramid_score_.assign(db_.images.size(), 0);
            if (seq_) {
                near_support_.resize(db_.images.size());
                for (size_t i = 0; i < db_.images.size(); i++)
                    near_support_[i].assign(feats_[i].count(), 0);
                near_score_.assign(db_.images.size(), 0);
            }
        } else {
            for (auto& s : support_) std::fill(s.begin(), s.end(), 0);
            std::fill(score_cache_.begin(), score_cache_.end(), 0);
            for (auto& p : pyramid_) std::fill(p.begin(), p.end(), 0);
            std::fill(pyramid_score_.begin(), pyramid_score_.end(), 0);
            for (auto& s : near_support_) std::fill(s.begin(), s.end(), 0);
            std::fill(near_score_.begin(), near_score_.end(), 0);
        }
        for (const auto& kv : rec_.images) {
            const Image& im = kv.second;
            if (!im.registered) continue;
            for (uint32_t f = 0; f < (uint32_t)im.point3D_ids.size(); f++)
                if (im.point3D_ids[f] != kInvalidPoint3D) attachObservation(kv.first, f);
        }
    }

    // 已配准特征关联三维点后，其所有对应各增加一单位支持。
    void attachObservation(uint32_t img, uint32_t f) {
        for (const Correspondence& c : graph_.at(img, f)) {
            if (++support_[c.image_id][c.feature_idx] == 1) {
                score_cache_[c.image_id]++;
                pyramidSet(c.image_id, c.feature_idx);
            }
            if (seq_ && nearby(img, c.image_id) && ++near_support_[c.image_id][c.feature_idx] == 1)
                near_score_[c.image_id]++;
        }
    }

    // ---------------- 序列（D79）----------------
    // 重复结构可能错误匹配到另一副本，优先采用时序邻居提供的对应。
    bool nearby(uint32_t a, uint32_t b) const {
        return seq_ && seq_->nearby(a, b, opt_.sequence_window);
    }

    // 检查仅邻居支持是否足以配准，rig 可整帧定位时使用帧级支持。
    bool nearReady(uint32_t img) const {
        if (!seq_) return false;
        return frameScoreOf(img, near_score_) >= opt_.min_num_pnp_inliers;
    }

    // ---------------- 可见性金字塔（D52）----------------
    // 按支持特征覆盖范围而非数量评分，避免大量角落特征产生病态位姿；采用 2×2 至 32×32 金字塔，与 COLMAP 的 MIN_UNCERTAINTY 一致。
    // 单元首次占用按层权重加分，与 support 的 0->1 事件一起增量维护。
    static constexpr int kPyrLevels = 5;
    static constexpr int kPyrDim = 1 << kPyrLevels;             // 最细网格
    static constexpr int kPyrCells = (4 * ((1 << (2 * kPyrLevels)) - 1)) / 3;  // 4+16+..+1024

    void pyramidSet(uint32_t img, uint32_t f) {
        const FeatureSet& fs = feats_[img];
        if (fs.width <= 0 || fs.height <= 0) return;
        const Keypoint& k = fs.keypoints[f];
        int cx = (int)(kPyrDim * (double)k.x / fs.width);
        int cy = (int)(kPyrDim * (double)k.y / fs.height);
        cx = std::min(std::max(cx, 0), kPyrDim - 1);
        cy = std::min(std::max(cy, 0), kPyrDim - 1);
        uint16_t* level = pyramid_[img].data() + kPyrCells;  // 从最细层开始遍历
        uint32_t score = 0;
        for (int i = kPyrLevels - 1; i >= 0; i--) {
            const int dim = 2 << i;
            level -= (size_t)dim * dim;
            if (++level[(size_t)cy * dim + cx] == 1) score += (uint32_t)dim * dim;
            cx >>= 1;
            cy >>= 1;
        }
        pyramid_score_[img] += score;
    }

    // 候选须未配准、未耗尽重试且对应数达标；按空间覆盖排序，失败过的图像排在所有未尝试图像之后。
    std::vector<uint32_t> chooseNextImages() const {
        static const bool score_check = spirula::env("SFM_SCORE_CHECK") != nullptr;
        // 逐序列位置的已配准数量，用于计算前沿距离。
        std::vector<std::vector<uint16_t>> at_pos;
        if (seq_) {
            at_pos.resize(seq_->length.size());
            for (size_t k = 0; k < at_pos.size(); k++) at_pos[k].assign(seq_->length[k], 0);
            for (const auto& kv : rec_.images)
                if (kv.second.registered && seq_->has(kv.first))
                    at_pos[seq_->seq[kv.first]][seq_->pos[kv.first]]++;
        }
        auto frontier = [&](uint32_t i) {
            if (!seq_->has(i)) return opt_.sequence_window + 1;  // 位于范围外的 rig 伙伴
            const int32_t sq = seq_->seq[i], p = seq_->pos[i];
            for (int d = 0; d <= opt_.sequence_window; d++) {
                if (p - d >= 0 && at_pos[sq][p - d]) return d;
                if (p + d < (int32_t)at_pos[sq].size() && at_pos[sq][p + d]) return d;
            }
            return opt_.sequence_window + 1;
        };
        std::vector<std::pair<uint64_t, uint32_t>> ranked;
        for (uint32_t i = 0; i < db_.images.size(); i++) {
            if (rec_.images.at(i).registered) continue;
            if (reg_trials_[i] >= opt_.max_reg_trials || !allowed(i)) continue;
            int s = score_cache_[i];
            if (score_check && s != score(i)) {
                slog::diag(slog::Tag::Map, "[map] SCORE MISMATCH image %u: cache %d, reference %d",
                           i, s, score(i));
                abort();
            }
            s = frameScore(i);
            if (s < opt_.min_num_pnp_inliers) continue;
            // 关闭时统一按对应数量排序，开启时按空间分布并优先未尝试候选。
            uint64_t rank = opt_.rank_by_visibility ? (uint64_t)pyramid_score_[i] : (uint64_t)s;
            if (opt_.rank_by_visibility && !reg_trials_[i]) rank |= 1ull << 48;
            // 仅靠邻居即可定位的序列前沿优先，且越近现有模型越先，使增长沿时序推进并获得完整支持（D79）。
            if (nearReady(i)) {
                const uint64_t closeness = (uint64_t)std::min(
                    65535, std::max(0, opt_.sequence_window + 1 - frontier(i)));
                rank |= (1ull << 49) | (closeness << 32);
            }
            ranked.emplace_back(rank, i);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;  // 平局按索引稳定排序
        });
        std::vector<uint32_t> out;
        out.reserve(ranked.size());
        for (const auto& r : ranked) out.push_back(r.second);
        return out;
    }

    // 每特征选择一个三维对应，优先序列邻居，否则首个可见点；nearf 标记邻居来源。
    void gatherCorrespondences(uint32_t img, std::vector<Vec3>& X, std::vector<Vec3>& br,
                               std::vector<uint32_t>& feat, std::vector<uint64_t>& pid,
                               std::vector<char>* nearf = nullptr) const {
        for (uint32_t f = 0; f < feats_[img].count(); f++) {
            uint64_t chosen = kInvalidPoint3D;
            bool from_near = false;
            for (const Correspondence& c : graph_.at(img, f)) {
                const Image& oi = rec_.images.at(c.image_id);
                if (!oi.registered || oi.point3D_ids[c.feature_idx] == kInvalidPoint3D) continue;
                if (chosen == kInvalidPoint3D) chosen = oi.point3D_ids[c.feature_idx];
                if (!seq_) break;
                if (nearby(img, c.image_id)) {
                    chosen = oi.point3D_ids[c.feature_idx];
                    from_near = true;
                    break;
                }
            }
            if (chosen == kInvalidPoint3D) continue;
            X.push_back(rec_.points3D.at(chosen).xyz);
            br.push_back(bearing(img, f));
            feat.push_back(f);
            pid.push_back(chosen);
            if (nearf) nearf->push_back(from_near ? 1 : 0);
        }
    }

    // 比较全池位姿与仅邻居支持位姿，优先邻居内点数；r 写回胜者，返回是否改变（D79）。
    bool preferNearPose(uint32_t img, const std::vector<Vec3>& X, const std::vector<Vec3>& br,
                        const std::vector<char>& nearf, PnPResult& r) {
        std::vector<Vec3> Xn, bn;
        for (size_t k = 0; k < X.size(); k++)
            if (nearf[k]) { Xn.push_back(X[k]); bn.push_back(br[k]); }
        if ((int)Xn.size() < opt_.min_num_pnp_inliers) return false;
        PnPResult n = ransacPnP(Xn, bn, camOf(img).focal(), errPx(img));
        if (!n.success || n.num_inliers < opt_.min_num_pnp_inliers) return false;
        int r_near = 0;
        if (r.success)
            for (size_t k = 0; k < X.size(); k++) r_near += (nearf[k] && r.inlier_mask[k]) ? 1 : 0;
        if (n.num_inliers <= r_near) return false;
        r.pose = n.pose;
        r.success = true;
        classify(img, X, br, r);
        reg_near_won_++;
        return true;
    }

    // 按图像自身半径统计 r.pose 的全池内点。
    void classify(uint32_t img, const std::vector<Vec3>& X, const std::vector<Vec3>& br,
                  PnPResult& r) const {
        double thr = camOf(img).errRad(opt_.max_reproj_error);
        thr *= thr;
        r.inlier_mask.assign(X.size(), 0);
        r.num_inliers = 0;
        for (size_t k = 0; k < X.size(); k++) {
            r.inlier_mask[k] = pnpResidualSq(r.pose, X[k], br[k]) < thr;
            r.num_inliers += r.inlier_mask[k] ? 1 : 0;
        }
    }

    // SS_SFM_SEQ_DUMP=1 逐次输出序列配准诊断。
    void seqDump(uint32_t img, const std::vector<Vec3>& X, const std::vector<char>& nearf,
                 const PnPResult& r, const PnPResult& rival, const char* verdict) const {
        if (!seq_dump_ || !seq_) return;
        int near_pool = 0, near_inl = 0, rival_inl = 0;
        for (size_t k = 0; k < X.size(); k++) {
            near_pool += nearf[k] ? 1 : 0;
            near_inl += (nearf[k] && r.inlier_mask[k]) ? 1 : 0;
            rival_inl += (rival.success && rival.inlier_mask[k]) ? 1 : 0;
        }
        slog::diag(slog::Tag::Map, "[seq] %s: near %d/%d, whole %d/%zu, rival %d -> %s",
                   db_.images[img].name.c_str(), near_inl, near_pool, r.num_inliers, X.size(),
                   rival_inl, verdict);
    }

    // 邻居推翻全池位姿后，将竞争解解释而当前解不解释的重复结构证据移出比例分母，普通噪声保留（D79）。
    bool ratioOkRival(uint32_t img, const std::vector<Vec3>& X, const std::vector<Vec3>& br,
                      const PnPResult& r, const PnPResult& rival, bool count = true) {
        std::vector<char> vis;
        const size_t pool = visibleMask(img, X, br, r.pose, vis);
        if (!rival.success) return ratioOk(r.num_inliers, pool);
        size_t excluded = 0;
        for (size_t k = 0; k < X.size(); k++)
            excluded += (vis[k] && rival.inlier_mask[k] && !r.inlier_mask[k]) ? 1 : 0;
        if (!ratioOk(r.num_inliers, pool - excluded)) return false;
        if (count && !ratioOk(r.num_inliers, pool)) reg_vouched_++;
        return true;
    }

    // 统一提交位姿、延续内点轨迹并报告，供 PnP 与 rig 补全共用。
    void commitPose(uint32_t img, const Pose& pose, const std::vector<uint32_t>& feat,
                    const std::vector<uint64_t>& pid, const std::vector<char>& inlier,
                    int num_inliers, size_t pool) {
        rec_.images[img].pose = pose;
        rec_.images[img].registered = true;
        for (size_t k = 0; k < feat.size(); k++) {
            if (!inlier[k]) continue;
            uint32_t f = feat[k];
            if (rec_.images[img].point3D_ids[f] != kInvalidPoint3D) continue;
            Point3D& pt = rec_.points3D[pid[k]];
            if (reprojErr(img, f, pt.xyz) > errPx(img)) continue;
            pt.track.push_back({img, f});
            rec_.images[img].point3D_ids[f] = pid[k];
            attachObservation(img, f);
        }
        if (opt_.verbose)
            slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_registered,
                     {(long long)img, (long long)num_inliers, (long long)pool,
                      (long long)rec_.numRegistered()});
        // 模型增长处更新限流快照；颜色通过回调仅为实际输出点计算。
        if (opt_.report_progress) {
            progress::model(rec_, false,
                            [this](const Point3D& p, uint8_t rgb[3]) {
                                pointColor(p, rgb);
                            });
            Event ev;
            ev.kind = Event::Kind::ModelUpdated;
            ev.stage = Stage::Map;
            ev.registered = rec_.numRegistered();
            ev.images = (int64_t)db_.images.size();
            ev.points = (int64_t)rec_.points3D.size();
            events::emit(ev);
            // 进度按整次采集累计，不能使用种子重试时会归零的当前模型数量。
            events::map_placed(img);
        }
    }

    bool registerImage(uint32_t img) {
        // 已有成员放置的帧可预测当前镜头，否则整帧联合注册。
        Pose rig_pose;
        if (rigPredictedPose(img, rig_pose)) {
            // 其余成员使用同一帧位姿一起放置。
            const uint32_t placed = completeFrame(rigs_->slot(img), img);
            const bool self = rec_.images.at(img).registered;
            frame_regs_ += placed - (self ? 1 : 0);
            return self;
        }
        if (registerFrame(img)) return true;
        std::vector<Vec3> X;
        std::vector<Vec3> br;   // 观测单位视线
        std::vector<uint32_t> feat;
        std::vector<uint64_t> pid;
        std::vector<char> nearf;
        gatherCorrespondences(img, X, br, feat, pid, seq_ ? &nearf : nullptr);
        if ((int)X.size() < opt_.min_num_pnp_inliers) { reg_fail_.few_corr++; return false; }

        const uint32_t cid = rec_.images[img].camera_id;
        PnPResult r;
        double swept_f = 0;  // 搜索选出的焦距，0 表示未搜索
        // 全局给定焦距仍允许其他组扫描，完全禁用曾少配主模型 75 图、总计 39 图；扫描发散由相机检查与联合优化限制（D45）。
        PnPResult rival;  // 邻居推翻后保留的全池竞争位姿
        if (focal_known_.count(cid) || opt_.focal_search_samples <= 0) {
            r = ransacPnP(X, br, camOf(img).focal(), errPx(img));
            if (seq_) {
                const PnPResult whole = r;
                if (preferNearPose(img, X, br, nearf, r)) rival = whole;
            }
        } else {
            // 组内首次图像且仍为无 EXIF 猜测时，按对数网格搜索焦距，取内点最多者；PnP 依赖标定视线，错误焦距可能直接失败。
            const Camera& c0 = camOf(img);
            const int N = opt_.focal_search_samples;
            for (int s = 0; s < N; s++) {
                double t = N > 1 ? (double)s / (N - 1) : 0.0;
                double ratio = opt_.min_focal_ratio *
                               std::pow(opt_.max_focal_ratio / opt_.min_focal_ratio, t);
                Camera trial = c0;
                trial.setFocal(c0.focal() * ratio);
                trial.k1 = trial.k2 = trial.p1 = trial.p2 = 0;
                std::vector<Vec3> bt(feat.size());
                for (size_t k = 0; k < feat.size(); k++) bt[k] = trial.bearing(kp(img, feat[k]));
                PnPResult t_r = ransacPnP(X, bt, trial.focal(), errPx(img));
                if (t_r.success && t_r.num_inliers > r.num_inliers) { r = t_r; swept_f = trial.focal(); }
            }
        }
        // 同时要求足够内点数量和比例，避免数百候选中偶然最小共识扭曲模型（D36）。
        if (!r.success || r.num_inliers < opt_.min_num_pnp_inliers) {
            reg_fail_.few_inliers++;
            return false;
        }
        bool prior_held = false;
        if (priors_ && !priorCheckPose(img, X, br, r, prior_held)) return false;
        if (!ratioOkRival(img, X, br, r, rival, false) && !strongUnambiguous(img, X, br, r)) {
            reg_fail_.low_ratio++;
            seqDump(img, X, nearf, r, rival, "refused (ratio)");
            return false;
        }

        // 门限通过后才提交搜索焦距，并在内点上联合精化位姿与焦距；网格残余 20% 焦距误差也可能通过 RANSAC，不能留给畸变补偿。
        if (swept_f > 0) {
            const double f0 = camOf(img).focal();
            rec_.cameras[cid].setFocal(swept_f);
            for (size_t k = 0; k < feat.size(); k++) br[k] = bearing(img, feat[k]);
            double fs = 1.0;
            refinePose(X, br, r.inlier_mask, r.pose, &fs);
            double refined = swept_f * fs;
            double ratio = refined / default_cams_.at(cid).focal();
            if (fs != 1.0 && ratio > opt_.min_focal_ratio && ratio < opt_.max_focal_ratio) {
                rec_.cameras[cid].setFocal(refined);
                for (size_t k = 0; k < feat.size(); k++) br[k] = bearing(img, feat[k]);
            }
            if (opt_.verbose && std::fabs(camOf(img).focal() - f0) > 1e-6)
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_camera_focal,
                         {(long long)cid, slog::num(f0, 0),
                          slog::num(camOf(img).focal(), 0),
                          (long long)r.num_inliers});
        } else {
            const Pose held = r.pose;
            refinePose(X, br, r.inlier_mask, r.pose);
            // 自由精化再次偏离陀螺时，保留受约束位姿。
            Mat3 Rp;
            double sig;
            if (prior_held && priorRotation(img, Rp, sig) &&
                rotationAngleDeg(mul(r.pose.R, transpose(Rp))) > priorTolDeg(sig))
                r.pose = held;
        }
        // 按精化后位姿重判内点，接受门限作用于最终共识。
        classify(img, X, br, r);
        if (r.num_inliers < opt_.min_num_pnp_inliers) { reg_fail_.refined_out++; return false; }
        const size_t pool = visiblePool(img, X, br, r.pose);
        const uint32_t vouched_before = reg_vouched_;
        if (!ratioOkRival(img, X, br, r, rival)) {
            reg_fail_.refined_out++;
            seqDump(img, X, nearf, r, rival, "refused (ratio after refinement)");
            return false;
        }
        if (!ratioOk(r.num_inliers, pool) && reg_vouched_ == vouched_before) reg_fail_.strong++;
        seqDump(img, X, nearf, r, rival, reg_vouched_ > vouched_before ? "placed (rival excluded)"
                                                                        : "placed");
        if (pool < X.size()) reg_fail_.occluded += (uint32_t)(X.size() - pool);
        focal_known_.insert(cid);

        commitPose(img, r.pose, feat, pid, r.inlier_mask, r.num_inliers, X.size());
        return true;
    }

    // ---------------- 相机装置 ----------------

    // 模型优先使用用户外参；零平移直接保留，部分平移自由时从零开始，其余由共同帧估计。
    void initRigCalib(Reconstruction& rec) const {
        rec.rigs.resize(rigs_->rigs.size());
        for (size_t r = 0; r < rigs_->rigs.size(); r++) {
            const RigSpec& spec = rigs_->rigs[r];
            RigCalib& c = rec.rigs[r];
            if (c.cam_from_rig.size() == spec.members.size()) continue;
            c = RigCalib{};
            c.resize(spec.members.size());
            if (!spec.anyKnownExt()) continue;
            bool zero_t = true;
            for (size_t m = 0; m < spec.members.size(); m++) {
                const RigMemberDef& md = spec.members[m];
                if (!md.has_ext) continue;
                if (c.ref < 0) c.ref = (int)m;
                zero_t = zero_t && md.ext.t.norm() == 0.0;
            }
            const Pose base = invertPose(spec.members[(size_t)c.ref].ext);
            for (size_t m = 0; m < spec.members.size(); m++) {
                const RigMemberDef& md = spec.members[m];
                if (!md.has_ext) continue;
                c.cam_from_rig[m] = composePose(md.ext, base);
                const bool partial = md.dof != kRigDofAll && (md.dof & kRigDofTranslation);
                if (!zero_t) c.cam_from_rig[m].t = {0, 0, 0};
                if (zero_t || partial || (int)m == c.ref) {
                    c.established[m] = 1;
                    c.fixed[m] = md.ext_fixed || md.dof == kRigDofNone ? 1 : 0;
                }
            }
        }
    }

    // 利用独立注册的共同帧估计尚未建立的成员外参，返回新增标定数。
    size_t calibrateRigs(Reconstruction& rec) const {
        if (!rigs_) return 0;
        initRigCalib(rec);
        size_t newly = 0;
        for (size_t r = 0; r < rigs_->rigs.size(); r++) {
            const RigSpec& spec = rigs_->rigs[r];
            RigCalib& c = rec.rigs[r];
            const size_t nm = spec.members.size();
            auto regd = [&](uint32_t img) {
                if (img == kNoImage || rec.rig_detached.count(img)) return false;
                auto it = rec.images.find(img);
                return it != rec.images.end() && it->second.registered;
            };
            if (c.ref < 0) {
                // 选择与其他成员共同配准帧最多的成员为参考，最大化其余外参的估计支持。
                std::vector<size_t> co(nm, 0);
                for (const auto& fr : spec.frames) {
                    size_t n = 0;
                    for (uint32_t img : fr) n += regd(img) ? 1 : 0;
                    if (n < 2) continue;
                    for (size_t m = 0; m < nm; m++) co[m] += regd(fr[m]) ? 1 : 0;
                }
                size_t best = 0;
                for (size_t m = 1; m < nm; m++)
                    if (co[m] > co[best]) best = m;
                if ((int)co[best] < opt_.rig_calib.min_frames) continue;
                c.ref = (int)best;
                c.cam_from_rig[best] = {mat3Identity(), {0, 0, 0}};
                c.established[best] = 1;
                c.support[best] = (uint32_t)co[best];
                newly++;
            }
            const uint32_t ref = (uint32_t)c.ref;
            for (uint32_t m = 0; m < nm; m++) {
                if (m == ref || c.established[m]) continue;
                std::vector<Pose> rel;
                for (const auto& fr : spec.frames)
                    if (regd(fr[ref]) && regd(fr[m]))
                        rel.push_back(relativePose(rec.images.at(fr[ref]).pose,
                                                   rec.images.at(fr[m]).pose));
                if ((int)rel.size() < opt_.rig_calib.min_frames) continue;
                Pose avg;
                double spread = 0;
                const Mat3* fixed_R = spec.members[m].has_ext ? &c.cam_from_rig[m].R : nullptr;
                const int inl = averageRelativePoses(rel, opt_.rig_calib, avg, spread, fixed_R);
                c.support[m] = (uint32_t)inl;
                c.spread_deg[m] = spread;
                if (inl < opt_.rig_calib.min_frames ||
                    (double)inl < opt_.rig_calib.min_inlier_frac * (double)rel.size() ||
                    spread > opt_.rig_calib.max_spread_deg) {
                    // 首次报告后，仅证据量翻倍时再次报告。
                    if (opt_.verbose && (int)rel.size() >= opt_.rig_calib.min_frames &&
                        rel.size() >= 2 * (size_t)c.declined_at[m]) {
                        c.declined_at[m] = (uint32_t)rel.size();
                        slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_rig_declined,
                                 {spec.name, spec.members[m].prefix, (long long)inl,
                                  (long long)rel.size(), slog::num(spread, 2)});
                    }
                    continue;
                }
                c.cam_from_rig[m] = avg;
                c.established[m] = 1;
                newly++;
                if (opt_.verbose)
                    slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_rig_calibrated,
                             {spec.name, spec.members[m].prefix, spec.members[ref].prefix,
                              (long long)inl, (long long)rel.size(), slog::num(spread, 2)});
            }
        }
        return newly;
    }

    // 从已标定、已配准且观测最多的成员推算整帧位姿。
    bool rigPredictedFrame(const RigSlot& sl, Pose& out,
                           uint32_t exclude = UINT32_MAX) const {
        if (!rigs_ || !sl.valid() || sl.rig >= rec_.rigs.size()) return false;
        const RigCalib& c = rec_.rigs[sl.rig];
        const std::vector<uint32_t>& fr = rigs_->frameOf(sl);
        int best = -1;
        uint32_t best_pts = 0;
        for (uint32_t m = 0; m < fr.size(); m++) {
            if (m == exclude || fr[m] == kNoImage || !c.usable(m)) continue;
            if (rec_.rig_detached.count(fr[m])) continue;
            auto it = rec_.images.find(fr[m]);
            if (it == rec_.images.end() || !it->second.registered) continue;
            const uint32_t n = it->second.numPoint3D();
            if (best < 0 || n > best_pts) { best = (int)m; best_pts = n; }
        }
        if (best < 0) return false;
        out = c.rigFromWorld((uint32_t)best, rec_.images.at(fr[best]).pose);
        return true;
    }

    // 再由帧位姿推算单镜头位置。
    bool rigPredictedPose(uint32_t img, Pose& out) const {
        if (!rigs_ || rec_.rig_detached.count(img)) return false;
        const RigSlot sl = rigs_->slot(img);
        if (!sl.valid() || sl.rig >= rec_.rigs.size()) return false;
        const RigCalib& c = rec_.rigs[sl.rig];
        if (!c.usable(sl.member)) return false;
        Pose frame;
        if (!rigPredictedFrame(sl, frame, sl.member)) return false;
        out = c.camFromWorld(sl.member, frame);
        return true;
    }

    // 尚无成员放置的帧联合全部镜头做 ransacRigPnP，以所有对应评分，使单镜头不足的支持也可共同定位（D78）。
    bool registerFrame(uint32_t img) {
        if (!rigs_ || rec_.rig_detached.count(img)) return false;
        const RigSlot sl = rigs_->slot(img);
        if (!sl.valid() || sl.rig >= rec_.rigs.size()) return false;
        const RigCalib& c = rec_.rigs[sl.rig];
        if (!c.usable(sl.member)) return false;
        struct Member {
            uint32_t img = 0, m = 0;
            std::vector<Vec3> X, br;
            std::vector<uint32_t> feat;
            std::vector<uint64_t> pid;
            std::vector<char> inl, nearf;
            std::vector<Vec3> Xn, bn;   // X 与 br 中来自序列邻居的部分
            int n = 0;
            size_t pool = 0;
        };
        std::vector<Member> ms;
        size_t total = 0, total_near = 0;
        for (uint32_t m = 0; m < rigs_->frameOf(sl).size(); m++) {
            const uint32_t j = rigs_->frameOf(sl)[m];
            if (j == kNoImage || !c.usable(m) || rec_.rig_detached.count(j) || !allowed(j))
                continue;
            if (rec_.images.at(j).registered) return false;
            Member e;
            e.img = j;
            e.m = m;
            gatherCorrespondences(j, e.X, e.br, e.feat, e.pid, seq_ ? &e.nearf : nullptr);
            e.inl.assign(e.X.size(), 0);
            for (size_t k = 0; k < e.nearf.size(); k++)
                if (e.nearf[k]) { e.Xn.push_back(e.X[k]); e.bn.push_back(e.br[k]); }
            total += e.X.size();
            total_near += e.Xn.size();
            ms.push_back(std::move(e));
        }
        if (ms.size() < 2 || (int)total < opt_.min_num_pnp_inliers) return false;
        auto consensus = [&](const Pose& F) {
            int n = 0;
            for (Member& e : ms) {
                const Pose p = c.camFromWorld(e.m, F);
                const double t = camOf(e.img).errRad(opt_.max_reproj_error);
                e.n = 0;
                for (size_t k = 0; k < e.X.size(); k++)
                    e.n += (e.inl[k] = pnpResidualSq(p, e.X[k], e.br[k]) < t * t) ? 1 : 0;
                n += e.n;
            }
            return n;
        };
        auto nearInliers = [&](const Pose& F) {
            int n = 0;
            for (const Member& e : ms) {
                const Pose p = c.camFromWorld(e.m, F);
                const double t = camOf(e.img).errRad(opt_.max_reproj_error);
                for (size_t k = 0; k < e.Xn.size(); k++)
                    n += pnpResidualSq(p, e.Xn[k], e.bn[k]) < t * t ? 1 : 0;
            }
            return n;
        };
        std::vector<RigPnPMember> gm, gm_near;
        gm.reserve(ms.size());
        for (Member& e : ms) {
            const double t = camOf(e.img).errRad(opt_.max_reproj_error);
            gm.push_back({&e.X, &e.br, c.cam_from_rig[e.m], t});
            gm_near.push_back({&e.Xn, &e.bn, c.cam_from_rig[e.m], t});
        }
        const RigPnPResult r = ransacRigPnP(gm);
        bool have = r.success && r.num_inliers >= opt_.min_num_pnp_inliers;
        Pose best = r.rig_from_world;
        // 比较邻居位姿与全池位姿，保留全池结果作为竞争解，规则与 preferNearPose/ratioOkRival 一致。
        std::vector<std::vector<char>> rival;
        if (seq_ && (int)total_near >= opt_.min_num_pnp_inliers) {
            const RigPnPResult rn = ransacRigPnP(gm_near);
            if (rn.success && rn.num_inliers >= opt_.min_num_pnp_inliers &&
                rn.num_inliers > (have ? nearInliers(best) : -1)) {
                if (have) {
                    consensus(best);
                    for (const Member& e : ms) rival.push_back(e.inl);
                }
                best = rn.rig_from_world;
                have = true;
                reg_near_won_++;
            }
        }
        if (!have) return false;
        // 由任一镜头已放置邻居预测帧旋转，并用最紧镜头先验检查。
        if (priors_) {
            Mat3 Rp;
            double sig = 0;
            bool got = false;
            for (const Member& e : ms) {
                Mat3 Rc;
                double s2;
                if (!priorRotation(e.img, Rc, s2)) continue;
                if (got && s2 >= sig) continue;
                Rp = mul(transpose(c.cam_from_rig[e.m].R), Rc);
                sig = s2;
                got = true;
            }
            if (got && rotationAngleDeg(mul(best.R, transpose(Rp))) > priorTolDeg(sig)) {
                // 固定旋转时放宽半径求解，再在容差内自由精化，最终按严格共识判断。
                const double tol = priorTolDeg(sig);
                std::vector<RigPnPMember> gl = gm;
                for (RigPnPMember& e : gl) e.max_error += tol * M_PI / 180.0;
                RigPnPResult k = ransacRigPnPKnownRotation(gl, Rp);
                int strict = 0;
                if (k.success && k.num_inliers >= opt_.min_num_pnp_inliers) {
                    for (Member& e : ms) {
                        const Pose p = c.camFromWorld(e.m, k.rig_from_world);
                        const double t = camOf(e.img).errRad(opt_.max_reproj_error) + tol * M_PI / 180.0;
                        for (size_t q = 0; q < e.X.size(); q++)
                            e.inl[q] = pnpResidualSq(p, e.X[q], e.br[q]) < t * t;
                    }
                    std::vector<FrameMember> fm;
                    for (Member& e : ms)
                        fm.push_back({&e.X, &e.br, &e.inl, c.cam_from_rig[e.m],
                                      1.0 / camOf(e.img).errRad(opt_.max_reproj_error)});
                    Pose refined = k.rig_from_world;
                    if (refineFramePose(fm, refined) &&
                        rotationAngleDeg(mul(refined.R, transpose(Rp))) <= tol)
                        k.rig_from_world = refined;
                    strict = consensus(k.rig_from_world);
                }
                if (prior_dump_)
                    slog::diag(slog::Tag::Map,
                               "[prior] frame of %s: rig PnP rotation %.1f deg off the gyro's "
                               "(tol %.1f); held frame %d/%zu inliers against %d",
                               db_.images[img].name.c_str(),
                               rotationAngleDeg(mul(best.R, transpose(Rp))), tol, strict, total,
                               r.num_inliers);
                if (strict < opt_.min_num_pnp_inliers) {
                    prior_stats_.refused++;
                    return false;
                }
                best = k.rig_from_world;
                rival.clear();
                prior_stats_.corrected++;
            }
        }
        const int n = consensus(best);
        size_t pool = 0, excluded = 0;
        int near_pool = 0, near_inl = 0, rival_inl = 0;
        for (size_t mi = 0; mi < ms.size(); mi++) {
            Member& e = ms[mi];
            std::vector<char> vis;
            e.pool = visibleMask(e.img, e.X, e.br, c.camFromWorld(e.m, best), vis);
            pool += e.pool;
            for (size_t k = 0; k < e.X.size(); k++) {
                if (!rival.empty() && vis[k] && rival[mi][k] && !e.inl[k]) excluded++;
                if (!rival.empty()) rival_inl += rival[mi][k] ? 1 : 0;
                if (!e.nearf.empty()) {
                    near_pool += e.nearf[k] ? 1 : 0;
                    near_inl += (e.nearf[k] && e.inl[k]) ? 1 : 0;
                }
            }
        }
        const bool ok = n >= opt_.min_num_pnp_inliers && ratioOk(n, pool - excluded);
        if (ok && !ratioOk(n, pool)) reg_vouched_++;
        if (seq_dump_ && seq_)
            slog::diag(slog::Tag::Map, "[seq] frame of %s: near %d/%d, whole %d/%zu, rival %d -> %s",
                       db_.images[img].name.c_str(), near_inl, near_pool, n, total, rival_inl,
                       ok ? (ratioOk(n, pool) ? "placed" : "placed (rival excluded)")
                          : "refused (ratio)");
        if (rig_dump_) {
            std::string per;
            for (Member& e : ms)
                per += (per.empty() ? "" : " ") + std::to_string(e.n) + "/" +
                       std::to_string(e.X.size());
            slog::diag(slog::Tag::Map,
                       "[rig] frame of %s: %zu members, %d/%zu inliers (%zu visible) [%s] -> %s",
                       db_.images[img].name.c_str(), ms.size(), n, total, pool, per.c_str(),
                       ok ? "placed together" : "REFUSED");
        }
        if (!ok) return false;
        for (Member& e : ms) {
            if (e.X.empty() && !opt_.rig_complete_blind) continue;
            focal_known_.insert(rec_.images[e.img].camera_id);
            commitPose(e.img, c.camFromWorld(e.m, best), e.feat, e.pid, e.inl, e.n, e.pool);
            reg_by_rig_++;
            if (e.n == 0) reg_rig_word_++;
            if (e.img == img) continue;
            {
                ProfTimer pt(g_map_prof.tri);
                triangulateForImage(e.img);
            }
            recent_regs_.push_back(e.img);
            frame_regs_++;
        }
        return true;
    }

    // 候选尝试价值取自身支持，或可联合定位时的整帧支持。
    int frameScore(uint32_t img) const { return frameScoreOf(img, score_cache_); }

    int frameScoreOf(uint32_t img, const std::vector<int>& scores) const {
        const int s = scores[img];
        if (!rigs_ || rec_.rig_detached.count(img)) return s;
        const RigSlot sl = rigs_->slot(img);
        if (!sl.valid() || sl.rig >= rec_.rigs.size()) return s;
        const RigCalib& c = rec_.rigs[sl.rig];
        if (!c.usable(sl.member)) return s;
        int sum = 0;
        for (uint32_t m = 0; m < rigs_->frameOf(sl).size(); m++) {
            const uint32_t j = rigs_->frameOf(sl)[m];
            if (j == kNoImage || !c.usable(m) || rec_.rig_detached.count(j)) continue;
            if (rec_.images.at(j).registered) return s;
            sum += scores[j];
        }
        return std::max(s, sum);
    }

    // 将 rig 可推断但尚未放置的所有成员一起补齐，先用全部对应精化一个帧位姿，再允许有充分观测的镜头做受限修正（D78）。
    uint32_t completeFrame(const RigSlot& sl, uint32_t caller_triangulates) {
        if (!rigs_ || !sl.valid() || sl.rig >= rec_.rigs.size()) return 0;
        const RigCalib& c = rec_.rigs[sl.rig];
        Pose pred;
        if (!rigPredictedFrame(sl, pred)) return 0;

        struct Pending {
            uint32_t img = 0, m = 0;
            std::vector<Vec3> X, br;
            std::vector<uint32_t> feat;
            std::vector<uint64_t> pid;
            std::vector<char> inl;
            double thr = 0;
            int n = 0;
            size_t pool = 0;
        };
        std::vector<Pending> ps;
        size_t total = 0;
        for (uint32_t m = 0; m < rigs_->frameOf(sl).size(); m++) {
            const uint32_t j = rigs_->frameOf(sl)[m];
            if (j == kNoImage || !c.usable(m) || rec_.rig_detached.count(j) || !allowed(j))
                continue;
            if (rec_.images.at(j).registered) continue;
            Pending e;
            e.img = j;
            e.m = m;
            e.thr = camOf(j).errRad(opt_.max_reproj_error);
            gatherCorrespondences(j, e.X, e.br, e.feat, e.pid);
            if ((int)e.X.size() < opt_.min_num_pnp_inliers && !opt_.rig_complete_blind) {
                reg_fail_.few_corr++;
                continue;
            }
            e.inl.assign(e.X.size(), 0);
            total += e.X.size();
            ps.push_back(std::move(e));
        }
        if (ps.empty()) return 0;

        // scale 统一放宽各镜头半径，因预测同时包含已放置镜头误差与外参标定误差。
        auto consensus = [&](const Pose& F, double scale) {
            int sum = 0;
            for (Pending& e : ps) {
                const Pose p = c.camFromWorld(e.m, F);
                const double t = scale * e.thr;
                e.n = 0;
                for (size_t k = 0; k < e.X.size(); k++)
                    e.n += (e.inl[k] = pnpResidualSq(p, e.X[k], e.br[k]) < t * t) ? 1 : 0;
                sum += e.n;
            }
            return sum;
        };
        auto visible = [&](const Pose& F) {
            size_t sum = 0;
            for (Pending& e : ps) {
                e.pool = visiblePool(e.img, e.X, e.br, c.camFromWorld(e.m, F));
                sum += e.pool;
            }
            return sum;
        };

        Pose frame = pred;
        bool own = false;
        int inl = 0;
        size_t vis = 0;
        double moved = 0;
        if (consensus(pred, 3.0) >= opt_.min_num_pnp_inliers) {
            Pose refined = pred;
            std::vector<FrameMember> fm;
            for (Pending& e : ps)
                fm.push_back({&e.X, &e.br, &e.inl, c.cam_from_rig[e.m], 1.0 / e.thr});
            if (refineFramePose(fm, refined)) {
                inl = consensus(refined, 1.0);
                vis = visible(refined);
                moved = rotationAngleDeg(mul(refined.R, transpose(pred.R)));
                double tol = 0;
                for (const Pending& e : ps) tol = std::max(tol, rigMoveTolDeg(e.img));
                own = inl >= opt_.min_num_pnp_inliers && ratioOk(inl, vis) && moved <= tol;
                if (own) frame = refined;
            }
        }
        if (!own) {
            inl = consensus(pred, 1.0);
            vis = visible(pred);
        }
        if (rig_dump_)
            slog::diag(slog::Tag::Map,
                       "[rig] frame of %s: %zu lens(es) to place, predicted pose refined by "
                       "%.2f deg, %d/%zu inliers (%zu visible) -> %s",
                       db_.images[ps.front().img].name.c_str(), ps.size(), moved, inl, total,
                       vis, own ? "refined" : "the rig's word");

        uint32_t placed = 0;
        for (Pending& e : ps) {
            Pose pose = c.camFromWorld(e.m, frame);
            // 支持充分的镜头可在帧位姿上精化，但修正受标定不确定度限制。
            std::vector<char> mask(e.X.size(), 0);
            int wide = 0;
            for (size_t k = 0; k < e.X.size(); k++)
                wide += (mask[k] = pnpResidualSq(pose, e.X[k], e.br[k]) <
                                   9.0 * e.thr * e.thr) ? 1 : 0;
            if (wide >= opt_.min_num_pnp_inliers) {
                Pose alone = pose;
                if (refinePose(e.X, e.br, mask, alone)) {
                    int n = 0;
                    for (size_t k = 0; k < e.X.size(); k++)
                        n += (mask[k] = pnpResidualSq(alone, e.X[k], e.br[k]) <
                                        e.thr * e.thr) ? 1 : 0;
                    const size_t v = visiblePool(e.img, e.X, e.br, alone);
                    const double mv = rotationAngleDeg(mul(alone.R, transpose(pose.R)));
                    if (n > e.n && ratioOk(n, v) && mv <= rigMoveTolDeg(e.img)) {
                        pose = alone;
                        e.n = n;
                        e.pool = v;
                        e.inl.swap(mask);
                    }
                }
            }
            if (e.n == 0) reg_rig_word_++;
            focal_known_.insert(rec_.images[e.img].camera_id);
            commitPose(e.img, pose, e.feat, e.pid, e.inl, e.n, e.pool);
            reg_by_rig_++;
            placed++;
            if (e.img == caller_triangulates) continue;
            {
                ProfTimer pt(g_map_prof.tri);
                triangulateForImage(e.img);
            }
            recent_regs_.push_back(e.img);
        }
        return placed;
    }

    // 精化可偏离 rig 预测的角度为标定离散度与下限的较大者，单位度。
    double rigMoveTolDeg(uint32_t img) const {
        const RigSlot sl = rigs_->slot(img);
        const RigCalib& c = rec_.rigs[sl.rig];
        double spread = 0;
        if (sl.member < c.spread_deg.size()) spread = c.spread_deg[sl.member];
        return std::max(1.0, 3.0 * spread);
    }

    // 补齐 img 同帧未配准成员，返回新增数量。
    uint32_t completeFrameOf(uint32_t img) {
        return rigs_ ? completeFrame(rigs_->slot(img), kNoImage) : 0;
    }

    // 标定建立后补齐所有已配准帧的成员。
    uint32_t completeRigFrames() {
        if (!rigs_) return 0;
        uint32_t n = 0;
        std::vector<uint32_t> regd;
        for (const auto& kv : rec_.images)
            if (kv.second.registered && rigs_->slot(kv.first).valid()) regd.push_back(kv.first);
        for (uint32_t img : regd) n += completeFrameOf(img);
        return n;
    }

    // 返回 rig 约束绑定的已配准同帧图像，包含 img；无绑定时仅返回自身。
    std::vector<uint32_t> frameMates(uint32_t img) const {
        std::vector<uint32_t> out{img};
        if (!rigs_ || rec_.rig_detached.count(img)) return out;
        const RigSlot sl = rigs_->slot(img);
        if (!sl.valid() || sl.rig >= rec_.rigs.size() || !rec_.rigs[sl.rig].usable(sl.member))
            return out;
        for (uint32_t m = 0; m < rigs_->frameOf(sl).size(); m++) {
            const uint32_t j = rigs_->frameOf(sl)[m];
            if (m == sl.member || j == kNoImage || !rec_.rigs[sl.rig].usable(m)) continue;
            if (rec_.rig_detached.count(j)) continue;
            auto it = rec_.images.find(j);
            if (it != rec_.images.end() && it->second.registered) out.push_back(j);
        }
        return out;
    }

    // 对因无观测而未进入 BA 的 rig 图像，使用已求解同帧成员恢复位姿。
    void snapRigFrames() {
        if (!rigs_) return;
        for (auto& kv : rec_.images) {
            if (!kv.second.registered || kv.second.numPoint3D() > 0) continue;
            Pose pose;
            if (rigPredictedPose(kv.first, pose)) kv.second.pose = pose;
        }
    }

    // 将图像特征连接到位姿可解释的现有三维点，供普通配准与外部审查修复共用，返回新增观测数。
    uint32_t attachExisting(uint32_t img) {
        uint32_t attached = 0;
        Image& im = rec_.images[img];
        for (uint32_t f = 0; f < feats_[img].count(); f++) {
            if (im.point3D_ids[f] != kInvalidPoint3D) continue;
            for (const Correspondence& c : graph_.at(img, f)) {
                if (c.image_id == img) continue;
                const Image& oi = rec_.images.at(c.image_id);
                if (!oi.registered) continue;
                uint64_t pid = oi.point3D_ids[c.feature_idx];
                if (pid == kInvalidPoint3D) continue;
                auto pt = rec_.points3D.find(pid);
                if (pt == rec_.points3D.end()) continue;
                if (reprojErr(img, f, pt->second.xyz) > errPx(img)) continue;
                bool dup = false;  // 每条轨迹每张图像最多一个观测
                for (const TrackElement& e : pt->second.track)
                    if (e.image_id == img) { dup = true; break; }
                if (dup) continue;
                pt->second.track.push_back({img, f});
                im.point3D_ids[f] = pid;
                attachObservation(img, f);
                attached++;
                break;
            }
        }
        return attached;
    }

    // ---------------- 新配准图像的三维点三角化 ----------------
    void triangulateForImage(uint32_t img, double err_scale = 1.0) {
        ModelIndex mi = indexModel();
        triangulateForImageAt(mi, img, err_scale);
    }

    using WorldRays = std::vector<std::vector<float>>;  // 按图像 ID，每特征三个分量

    struct NewTrack {
        Vec3 X;
        uint32_t f = 0;  // 候选所属图像的特征索引
        std::vector<TrackElement> track;
    };

    // 索引构建需扫描全模型，逐图重三角化时只构建一次并共用。
    void triangulateForImageAt(const ModelIndex& mi, uint32_t img, double err_scale) {
        const uint32_t n = feats_[img].count();
        const size_t kBlock = 256;
        std::vector<std::vector<NewTrack>> made((n + kBlock - 1) / kBlock);
        parallelFor(n, kBlock, [&](size_t lo, size_t hi, std::vector<uint8_t>&) {
            std::vector<Correspondence> obs;
            NewTrack t;
            for (size_t f = lo; f < hi; f++)
                if (featureTrack(mi, img, (uint32_t)f, err_scale, obs, t))
                    made[lo / kBlock].push_back(t);
        });
        commitCollected(mi, img, made, err_scale);
    }

    // 按顺序提交先前状态收集的轨迹；若元素已被更早提交占用，则按当前状态重算。
    void commitCollected(const ModelIndex& mi, uint32_t img,
                         const std::vector<std::vector<NewTrack>>& made, double err_scale) {
        std::vector<Correspondence> obs;
        NewTrack again;
        for (const std::vector<NewTrack>& block : made)
            for (const NewTrack& t : block) {
                bool clash = false;
                for (const TrackElement& e : t.track)
                    clash = clash || mi.img[e.image_id]->point3D_ids[e.point2D_idx] != kInvalidPoint3D;
                if (!clash) commitNewTrack(t);
                else if (featureTrack(mi, img, t.f, err_scale, obs, again)) commitNewTrack(again);
            }
    }

    // 只读计算图像特征可能生成的点；后续变化只会占用候选元素，包含最佳图像对，因此可检查快照答案是否仍有效。
    bool featureTrack(const ModelIndex& mi, uint32_t img, uint32_t f, double err_scale,
                      std::vector<Correspondence>& obs, NewTrack& out,
                      const WorldRays* rays = nullptr) const {
        const Image& me = *mi.img[img];
        if (me.point3D_ids[f] != kInvalidPoint3D) return false;
        // 候选观测须已配准且尚未关联三维点
        obs.clear();
        for (const Correspondence& c : graph_.at(img, f))
            if (mi.img[c.image_id]->registered &&
                mi.img[c.image_id]->point3D_ids[c.feature_idx] == kInvalidPoint3D)
                obs.push_back(c);
        if (obs.empty()) return false;
        const Vec3 bf = mi.cam[img]->bearing(kp(img, f));
        std::vector<Vec3>& bo = obs_bearing_scratch();
        bo.assign(obs.size(), Vec3{0, 0, 0});
        auto bearingOf = [&](size_t k) -> const Vec3& {
            if (bo[k].x == 0 && bo[k].y == 0 && bo[k].z == 0)
                bo[k] = mi.cam[obs[k].image_id]->bearing(kp(obs[k].image_id, obs[k].feature_idx));
            return bo[k];
        };

        // 选择最大视差对应三角化，先按最小角度减两侧误差容差排除不可能候选；30 Hz 双鱼眼视频可提前排除约 94%。
        Vec3 ra = mul(transpose(me.pose.R), bf);
        ra = ra * (1.0 / ra.norm());
        const double tol_a = rayTolerance(mi, img, err_scale);
        Vec3 bestX;
        double bestAng = -1;
        const Correspondence* bestC = nullptr;
        for (size_t k = 0; k < obs.size(); k++) {
            const Correspondence& c = obs[k];
            const double lim = opt_.min_tri_angle_deg * M_PI / 180.0 - tol_a -
                               rayTolerance(mi, c.image_id, err_scale);
            if (lim > 0) {
                Vec3 rb;
                if (rays) {
                    const float* r = &(*rays)[c.image_id][3 * (size_t)c.feature_idx];
                    rb = {r[0], r[1], r[2]};
                } else {
                    rb = mul(transpose(mi.img[c.image_id]->pose.R), bearingOf(k));
                    rb = rb * (1.0 / rb.norm());
                }
                if (ra.dot(rb) > std::cos(lim)) continue;
            }
            Vec3 X;
            if (!triangulatePairAt(mi, img, f, bf, c.image_id, c.feature_idx, bearingOf(k), X,
                                   err_scale))
                continue;
            double ang = triangulationAngle(X, cameraCenter(me.pose),
                                            cameraCenter(mi.img[c.image_id]->pose));
            if (ang > bestAng) { bestAng = ang; bestX = X; bestC = &c; }
        }
        if (!bestC) return false;

        // 轨迹包含当前观测及所有重投影合格候选。
        std::vector<TrackElement>& track = out.track;
        track.clear();
        track.push_back({img, f});
        for (size_t k = 0; k < obs.size(); k++) {
            const Correspondence& c = obs[k];
            if (reprojErrAt(mi, c.image_id, c.feature_idx, bestX, bearingOf(k)) <=
                err_scale * (opt_.max_reproj_error * mi.pixel_scale[c.image_id]))
                track.push_back({c.image_id, c.feature_idx});
        }
        if (track.size() < 2) return false;
        // 禁止同一图像的两个特征进入同一轨迹。
        std::sort(track.begin(), track.end(),
                  [](const TrackElement& a, const TrackElement& b) {
                      return a.image_id < b.image_id;
                  });
        track.erase(std::unique(track.begin(), track.end(),
                                [](const TrackElement& a, const TrackElement& b) {
                                    return a.image_id == b.image_id;
                                }),
                    track.end());
        if (track.size() < 2) return false;
        out.X = bestX;
        out.f = f;
        return true;
    }

    // 缓存已配准特征的世界单位视线为 float，避免过滤每轮约 1.66 亿次鱼眼反解；粗筛误差包含于余量，最终接受仍用精确视线。
    WorldRays worldRays(const ModelIndex& mi) {
        WorldRays out(db_.images.size());
        std::vector<uint32_t> imgs;
        for (uint32_t i = 0; i < db_.images.size(); i++)
            if (mi.img[i] && mi.img[i]->registered) imgs.push_back(i);
        parallelFor(imgs.size(), 4, [&](size_t lo, size_t hi, std::vector<uint8_t>&) {
            for (size_t j = lo; j < hi; j++) {
                const uint32_t i = imgs[j];
                const Mat3 Rt = transpose(mi.img[i]->pose.R);
                const uint32_t n = feats_[i].count();
                std::vector<float>& r = out[i];
                r.resize(3 * (size_t)n);
                for (uint32_t f = 0; f < n; f++) {
                    Vec3 w = mul(Rt, mi.cam[i]->bearing(kp(i, f)));
                    w = w * (1.0 / w.norm());
                    r[3 * (size_t)f] = (float)w.x;
                    r[3 * (size_t)f + 1] = (float)w.y;
                    r[3 * (size_t)f + 2] = (float)w.z;
                }
            }
        });
        return out;
    }

    // 视线角度余量取两倍像素容差除焦距，覆盖镜头边缘压缩像素尺度的情况。
    double rayTolerance(const ModelIndex& mi, uint32_t img, double err_scale) const {
        const Camera& c = *mi.cam[img];
        return 2.0 * err_scale * opt_.max_reproj_error * mi.pixel_scale[img] / std::min(c.fx, c.fy);
    }

    // featureTrack 候选视线临时区，每线程独立定尺寸并使用。
    static std::vector<Vec3>& obs_bearing_scratch() {
        static thread_local std::vector<Vec3> v;
        return v;
    }

    void commitNewTrack(const NewTrack& t) {
        rec_.addPoint3D(t.X, t.track);
        // 添加观测更新支持计数；轨迹补全的直接修改可暂不更新，因为下次排序前会 rebuildScores。
        for (const TrackElement& e : t.track) attachObservation(e.image_id, e.point2D_idx);
    }


    // ---------------- 全局精化：BA、过滤与撤销配准 ----------------
    // 迭代求解和过滤直到模型稳定，再撤销失去支持的图像，允许后续重新配准，避免坏位姿持续拉偏模型（D36）。
    void globalRefine(bool final_pass) {
        if (rec_.numRegistered() < 2 || rec_.points3D.size() < 10) return;
        const bool tight = final_pass && opt_.ba_final_tight;
        int rounds = tight ? opt_.ba_max_refinements : 2;
        for (int i = 0; i < rounds; i++) {
            // 精化后补全轨迹并重三角化，为先前被过滤或位姿太粗的观测重新提供机会，避免稀疏匹配模型只能不断失去结构。
            if (i > 0 && opt_.retri_scale > 0) {
                ProfTimer pt(g_map_prof.retri);
                completeAndRetriangulate();
            }
            size_t before = countObservations();
            BundleOptions bo;
            bo.real = realCfgFromName(opt_.ba_real);
            bo.device = opt_.device;
            bo.device_selector = opt_.device_selector;
            bo.threads = opt_.threads;
            bo.verbose = false;
            bo.loss = opt_.ba_loss;
            bo.loss_param = (float)(opt_.ba_loss_param * medianPixelScale());
            bo.refine_principal_point = opt_.refine_principal_point || final_.pp;
            bo.refine_extra_params = opt_.refine_extra_params || final_.extra;
            bo.pp_min_images = opt_.pp_min_images;
            // 增长精化首轮采用宽松容差，观测变化较大才追加严格轮次，最终精化始终严格。
            // 标量配置须匹配容差；fp32 无法达到低于噪声底的要求，1194 图数据错误配对曾使收尾从 48 s 增至 260 s（D38）。
            const bool loose = !tight && i == 0 && opt_.ba_growth_rtol > 0;
            if (loose) {
                bo.rtol = opt_.ba_growth_rtol;
                bo.patience = opt_.ba_growth_patience;
            }
            bo.solver = opt_.ba_solver;
            bo.real = baReal(loose);
            bo.shared_ctx = &baContext(loose);
            bo.over_budget_throws = ba_over_budget_throws_;
            bo.rigs = rigs_;
            bo.use_rigs = !final_.no_rig;
            bo.refine_rigs = opt_.refine_rigs && rigRefineDue(tight && i == 0);
            if (rigs_ && !final_.no_rig) calibrateRigs(rec_);
            PosePriors pf;
            if (priors_) {
                pf = priorFactors(rec_);
                bo.priors = &pf;
            }
            double cost = runGlobalBA(rec_, bo);
            if (rigs_ && !final_.no_rig) snapRigFrames();
            ProfTimer pt(g_map_prof.filter);
            // 每次建图 BA 后约束相机到物理合理范围；811 图数据关闭后 AUC@10 从 84.3 降至 0，焦距由默认 2813 发散至 25219。
            // 1363 图网络集合开启为 83.2，关闭为 68.7，因此同样保留。
            if (!final_.no_sanitize) sanitizeCameras();
            int removedObs = 0, removedPts = 0;
            filterPoints(removedObs, removedPts);
            if (opt_.verbose) {
                char cost_s[32];
                std::snprintf(cost_s, sizeof cost_s, "%.3e", cost);
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_global_ba,
                         {cost_s, (long long)removedObs, (long long)removedPts,
                          (long long)rec_.points3D.size()});
            }
            if (!before || (double)removedObs / (double)before <= opt_.ba_refine_change) break;
        }
        int dropped;
        {
            ProfTimer pt(g_map_prof.filter);
            dropped = filterImages();
        }
        if (dropped && opt_.verbose)
            slog::diag(slog::Tag::Map,
                       "[map] de-registered %d image(s) (few points or bogus camera), "
                       "%u remain", dropped, rec_.numRegistered());
        if (rigs_ && !final_.no_rig) completeRigFrames();
    }

    // 合并对应图指向同一点的重复轨迹，使回访形成真正闭环；仅创建自由特征无法连接已分别三角化的两半。
    // 联合轨迹须每图最多一个观测，所有观测均能解释按轨迹长度加权的位置，且保留最小三角化角度（D54）。
    size_t mergeTracks(const ModelIndex& mi, double err_scale) {
        const double err = err_scale * opt_.max_reproj_error;
        size_t merged = 0;
        std::vector<uint64_t> ids;
        ids.reserve(rec_.points3D.size());
        for (const auto& kv : rec_.points3D) ids.push_back(kv.first);
        // 用打包的两点 ID 记录已判定点对，避免从两端重复测试；哈希替代树节点降低数百万次查询开销。
        std::unordered_set<uint64_t> tried;
        tried.reserve(4 * rec_.points3D.size());
        std::vector<uint8_t> on_track(db_.images.size(), 0);
        std::vector<Vec3> centers(db_.images.size());
        for (uint32_t i = 0; i < db_.images.size(); i++)
            if (mi.img[i] && mi.img[i]->registered) centers[i] = cameraCenter(mi.img[i]->pose);
        for (uint64_t pid : ids) {
            // 合并可能删除当前点，重新检查 ID 并从幸存点继续遍历。
            for (int hop = 0; hop < 8; hop++) {
                auto pit = rec_.points3D.find(pid);
                if (pit == rec_.points3D.end()) break;
                uint64_t absorbed = mergeOne(mi, pit->first, err, tried, on_track, centers);
                if (!absorbed) break;
                merged += absorbed;
            }
        }
        return merged;
    }

    // 对 pid 轨迹的全部对应尝试一次合并，返回吸收观测数，0 表示没有成功合并。
    static constexpr size_t kMergeMaxTrack = 20;

    size_t mergeOne(const ModelIndex& mi, uint64_t pid, double err,
                    std::unordered_set<uint64_t>& tried, std::vector<uint8_t>& on_track,
                    const std::vector<Vec3>& centers) {
        Point3D& pt = rec_.points3D.at(pid);
        for (size_t ti = 0; ti < pt.track.size(); ti++) {
            const TrackElement e = pt.track[ti];
            for (const Correspondence& c : graph_.at(e.image_id, e.point2D_idx)) {
                const Image* oi = mi.img[c.image_id];
                if (!oi || !oi->registered) continue;
                const uint64_t qid = oi->point3D_ids[c.feature_idx];
                if (qid == kInvalidPoint3D || qid == pid) continue;
                const uint64_t lo = std::min(pid, qid), hi = std::max(pid, qid);
                if (!tried.insert((lo << 32) ^ hi).second) continue;
                auto qit = rec_.points3D.find(qid);
                if (qit == rec_.points3D.end()) continue;
                Point3D& q = qit->second;

                // 轨迹长 t 时 Schur 图像对数为 t(t+1)/2，合并长轨迹收益递减而成本平方增长。
                // 1194 图鱼眼中无界融合使平均轨迹 3.5->7.2，BA 每轮 0.03->0.36 s，因此限制长度。
                if (pt.track.size() + q.track.size() > kMergeMaxTrack) continue;
                // 每图最多一个观测，否则不是合法轨迹。
                bool clash = false;
                for (const TrackElement& a : pt.track) on_track[a.image_id] = 1;
                for (const TrackElement& b : q.track)
                    if (on_track[b.image_id]) { clash = true; break; }
                for (const TrackElement& a : pt.track) on_track[a.image_id] = 0;
                if (clash) continue;

                const double wa = (double)pt.track.size(), wb = (double)q.track.size();
                const Vec3 x = (pt.xyz * wa + q.xyz * wb) * (1.0 / (wa + wb));
                bool ok = true;
                for (const std::vector<TrackElement>* tr : {&pt.track, &q.track}) {
                    for (const TrackElement& el : *tr)
                        if (reprojErrAt(mi, el.image_id, el.point2D_idx, x) >
                            err * mi.pixel_scale[el.image_id]) { ok = false; break; }
                    if (!ok) break;
                }
                if (!ok) continue;
                // 联合轨迹必须保持足够视差；仅重投影合格仍可能变成共线退化点，被下一轮删除后又重建，引发无收益 BA 震荡。
                if (!wellTriangulated(pt.track, q.track, x, centers)) continue;

                const size_t absorbed = q.track.size();
                pt.xyz = x;
                for (const TrackElement& el : q.track) {
                    mi.img[el.image_id]->point3D_ids[el.point2D_idx] = pid;
                    pt.track.push_back(el);
                }
                rec_.points3D.erase(qit);
                return absorbed;
            }
        }
        return 0;
    }

    // 检查联合轨迹是否有足够基线约束深度，与 filterPoints 使用相同判据。
    bool wellTriangulated(const std::vector<TrackElement>& a, const std::vector<TrackElement>& b,
                          const Vec3& x, const std::vector<Vec3>& centers) const {
        const double min_ang = opt_.min_tri_angle_deg * M_PI / 180.0;
        // 长轨迹按步长分散采样，避免二次扫描且覆盖首尾最大基线，不能仅截取前缀。
        constexpr size_t kMaxScan = 12;
        std::vector<uint32_t> imgs;
        imgs.reserve(2 * kMaxScan);
        for (const std::vector<TrackElement>* t : {&a, &b}) {
            const size_t stride = std::max<size_t>(1, t->size() / kMaxScan);
            for (size_t i = 0; i < t->size(); i += stride) imgs.push_back((*t)[i].image_id);
        }
        for (size_t i = 0; i + 1 < imgs.size(); i++)
            for (size_t j = i + 1; j < imgs.size(); j++)
                if (triangulationAngle(x, centers[imgs[i]], centers[imgs[j]]) >= min_ang)
                    return true;
        return false;
    }

    // 先传递补全现有轨迹，再重三角化各已配准图像，最后融合重复点。
    // 新加观测使用比删除阈值更严格的 0.75 倍门限，防止边界噪声反复删除、重建并拉偏 BA（D36）。
    void completeAndRetriangulate() {
        const double err = opt_.retri_scale * opt_.max_reproj_error;  // 下方再乘 pixel_scale
        ModelIndex mi = indexModel();
        std::vector<std::pair<uint64_t, Point3D*>> pts;
        pts.reserve(rec_.points3D.size());
        for (auto& kv : rec_.points3D) pts.emplace_back(kv.first, &kv.second);
        std::vector<uint32_t> imgs;
        for (const auto& kv : rec_.images)
            if (kv.second.registered) imgs.push_back(kv.first);

        // 按初始状态并行收集，再按串行顺序提交；被更早提交占用的候选重算，保证与串行结果一致。
        std::vector<std::vector<TrackElement>> added(pts.size());
        parallelFor(pts.size(), 256, [&](size_t lo, size_t hi, std::vector<uint8_t>& on_track) {
            for (size_t i = lo; i < hi; i++) collectCompletion(mi, *pts[i].second, err, on_track,
                                                                added[i]);
        });
        std::vector<uint8_t> on_track(db_.images.size(), 0);
        std::vector<TrackElement> redo;
        for (size_t i = 0; i < pts.size(); i++) {
            Point3D& pt = *pts[i].second;
            bool clash = false;
            for (const TrackElement& e : added[i])
                clash = clash || mi.img[e.image_id]->point3D_ids[e.point2D_idx] != kInvalidPoint3D;
            if (clash) collectCompletion(mi, pt, err, on_track, redo);
            for (const TrackElement& e : clash ? redo : added[i]) {
                pt.track.push_back(e);
                mi.img[e.image_id]->point3D_ids[e.point2D_idx] = pts[i].first;
            }
        }

        std::vector<std::vector<std::vector<NewTrack>>> made(imgs.size());
        {
            const WorldRays rays = worldRays(mi);
            parallelFor(imgs.size(), 4, [&](size_t lo, size_t hi, std::vector<uint8_t>&) {
                std::vector<Correspondence> obs;
                NewTrack t;
                for (size_t i = lo; i < hi; i++) {
                    made[i].resize(1);
                    for (uint32_t f = 0; f < feats_[imgs[i]].count(); f++)
                        if (featureTrack(mi, imgs[i], f, opt_.retri_scale, obs, t, &rays))
                            made[i][0].push_back(t);
                }
            });
        }
        for (size_t i = 0; i < imgs.size(); i++)
            commitCollected(mi, imgs[i], made[i], opt_.retri_scale);
        if (opt_.merge_tracks) {
            ProfTimer pt(g_map_prof.merge);
            g_map_prof.n_merged += mergeTracks(mi, opt_.retri_scale);
        }
    }

    // 递归收集自由、尚未出现在轨迹且重投影误差小于 err 的对应，on_track 进入与返回时均清零。
    void collectCompletion(const ModelIndex& mi, const Point3D& pt, double err,
                           std::vector<uint8_t>& on_track, std::vector<TrackElement>& add) const {
        add.clear();
        for (const TrackElement& e : pt.track) on_track[e.image_id] = 1;
        const size_t n0 = pt.track.size();
        for (size_t ti = 0; ti < n0 + add.size(); ti++) {
            const TrackElement e = ti < n0 ? pt.track[ti] : add[ti - n0];
            for (const Correspondence& c : graph_.at(e.image_id, e.point2D_idx)) {
                const Image& oi = *mi.img[c.image_id];
                if (!oi.registered || on_track[c.image_id] ||
                    oi.point3D_ids[c.feature_idx] != kInvalidPoint3D)
                    continue;
                if (reprojErrAt(mi, c.image_id, c.feature_idx, pt.xyz) <=
                    err * mi.pixel_scale[c.image_id]) {
                    add.push_back({c.image_id, c.feature_idx});
                    on_track[c.image_id] = 1;
                }
            }
        }
        for (const TrackElement& e : pt.track) on_track[e.image_id] = 0;
        for (const TrackElement& e : add) on_track[e.image_id] = 0;
    }

    // 按块在工作线程调用 fn，每线程获得与图像表同大小的独立零标记缓冲。
    template <class F>
    void parallelFor(size_t n, size_t block, F&& fn) {
        const unsigned hc = std::thread::hardware_concurrency();
        int nt = opt_.threads > 0 ? opt_.threads : (hc > 0 ? (int)hc : 1);
        nt = std::max(1, std::min<int>(nt, (int)((n + block - 1) / block)));
        std::atomic<size_t> next{0};
        auto worker = [&] {
            std::vector<uint8_t> scratch(db_.images.size(), 0);
            for (size_t b; (b = next.fetch_add(block)) < n;) fn(b, std::min(b + block, n), scratch);
        };
        if (nt == 1) return worker();
        std::vector<std::thread> pool;
        for (int t = 0; t < nt; t++) pool.emplace_back(worker);
        for (std::thread& t : pool) t.join();
    }

    size_t countObservations() const {
        size_t n = 0;
        for (const auto& kv : rec_.points3D) n += kv.second.track.size();
        return n;
    }

    // 删除重投影不合格观测和失去视差的点，避免近退化轨迹向 PnP 提供不稳定三维结构。
    void filterPoints(int& removedObs, int& removedPts) {
        // 逐轮遍历全部观测，通过平铺索引避免每观测约五次 map 查找。
        ModelIndex mi = indexModel();
        std::vector<Vec3> centers(db_.images.size());
        for (uint32_t i = 0; i < db_.images.size(); i++)
            if (mi.img[i] && mi.img[i]->registered) centers[i] = cameraCenter(mi.img[i]->pose);
        const double min_ang = opt_.min_tri_angle_deg * M_PI / 180.0;

        // 各点独立读轨迹与共享相机表，唯一共享写入是该点独占的图像特征关联，因此可按点并行且结果不依赖任务划分。
        std::vector<std::pair<uint64_t, Point3D*>> pts;
        pts.reserve(rec_.points3D.size());
        for (auto& kv : rec_.points3D) pts.emplace_back(kv.first, &kv.second);

        std::vector<uint64_t> drop;
        std::atomic<int> removed_obs_atomic{0};
        const unsigned hc = std::thread::hardware_concurrency();
        int nt = opt_.threads > 0 ? opt_.threads : (hc > 0 ? (int)hc : 1);
        nt = std::max(1, std::min<int>(nt, (int)std::max<size_t>(pts.size() / 512, 1)));
        std::vector<std::vector<uint64_t>> drop_per_thread(nt);
        std::atomic<size_t> next{0};
        const size_t kBlock = 256;

        auto worker = [&](int t) {
            int local_removed = 0;
            std::vector<uint64_t>& local_drop = drop_per_thread[t];
            for (;;) {
                const size_t b = next.fetch_add(kBlock);
                if (b >= pts.size()) break;
                const size_t e = std::min(b + kBlock, pts.size());
                for (size_t pi = b; pi < e; pi++) {
                    Point3D& pt = *pts[pi].second;
                    size_t keep = 0;  // 原地压缩，保持幸存顺序
                    for (size_t r = 0; r < pt.track.size(); r++) {
                        const TrackElement el = pt.track[r];
                        if (reprojErrAt(mi, el.image_id, el.point2D_idx, pt.xyz) <=
                            opt_.max_reproj_error * mi.pixel_scale[el.image_id]) {
                            pt.track[keep++] = el;
                        } else {
                            mi.img[el.image_id]->point3D_ids[el.point2D_idx] = kInvalidPoint3D;
                            local_removed++;
                        }
                    }
                    pt.track.resize(keep);
                    bool degenerate = pt.track.size() < 2;
                    if (!degenerate) {
                        double best = 0;
                        for (size_t i = 0; i + 1 < pt.track.size() && best < min_ang; i++)
                            for (size_t j = i + 1; j < pt.track.size() && best < min_ang; j++)
                                best = std::max(best,
                                                triangulationAngle(pt.xyz,
                                                                   centers[pt.track[i].image_id],
                                                                   centers[pt.track[j].image_id]));
                        degenerate = best < min_ang;
                    }
                    if (degenerate) {
                        for (const TrackElement& el : pt.track) {
                            mi.img[el.image_id]->point3D_ids[el.point2D_idx] = kInvalidPoint3D;
                            local_removed++;
                        }
                        local_drop.push_back(pts[pi].first);
                    }
                }
            }
            removed_obs_atomic += local_removed;
        };
        if (nt == 1) {
            worker(0);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(nt);
            for (int t = 0; t < nt; t++) pool.emplace_back(worker, t);
            for (std::thread& t : pool) t.join();
        }
        removedObs += removed_obs_atomic.load();
        for (const std::vector<uint64_t>& d : drop_per_thread)
            drop.insert(drop.end(), d.begin(), d.end());
        for (uint64_t id : drop) { rec_.points3D.erase(id); removedPts++; }
    }

    // 撤销配准时解除全部观测，删除少于两视图的轨迹；已消耗试验计数保留，剩余预算允许后续重试。
    void deregisterImage(uint32_t img) {
        Image& im = rec_.images[img];
        for (uint32_t f = 0; f < (uint32_t)im.point3D_ids.size(); f++) {
            uint64_t id = im.point3D_ids[f];
            if (id == kInvalidPoint3D) continue;
            im.point3D_ids[f] = kInvalidPoint3D;
            auto it = rec_.points3D.find(id);
            if (it == rec_.points3D.end()) continue;
            auto& tr = it->second.track;
            tr.erase(std::remove_if(tr.begin(), tr.end(),
                                    [&](const TrackElement& e) { return e.image_id == img; }),
                     tr.end());
            if (tr.size() < 2) {
                for (const TrackElement& e : tr)
                    rec_.images[e.image_id].point3D_ids[e.point2D_idx] = kInvalidPoint3D;
                rec_.points3D.erase(it);
            }
        }
        im.registered = false;
    }

    // 相机参数超出合理范围时先拉回，不直接撤销整个组；若异常系数只反映未约束区域则残差变化小，若在掩盖坏几何，后续过滤会暴露对应问题（D36）。
    void sanitizeCameras() {
        for (auto& kv : rec_.cameras) {
            Camera& c = kv.second;
            const Camera& d = default_cams_.at(kv.first);
            int fixed = 0;
            double ratio = c.focal() / d.focal();
            if (!(ratio > opt_.min_focal_ratio && ratio < opt_.max_focal_ratio)) {
                c.setFocal(d.focal());
                fixed++;
            }
            // FullOpenCV 分母 k4..k6 与分子 k1..k3 使用相同归一化半径单位，共用阈值。
            for (double* k : {&c.k1, &c.k2, &c.k3, &c.k4, &c.k5, &c.k6, &c.p1, &c.p2,
                              &c.sx1, &c.sy1})
                if (std::fabs(*k) > opt_.max_extra_param) { *k = 0; fixed++; }
            // 建图中主点不应远离中心，大幅偏移通常是 BA 在补偿其他错误。
            if (std::fabs(c.cx - d.cx) > 0.2 * c.width) { c.cx = d.cx; fixed++; }
            if (std::fabs(c.cy - d.cy) > 0.2 * c.height) { c.cy = d.cy; fixed++; }
            if (fixed && opt_.verbose)
                slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_runaway_params,
                         {(long long)kv.first, (long long)fixed});
        }
    }

    // 撤销过滤后观测不足的图像；异常相机已原地修正，不按相机异常直接删除整组。
    int filterImages() {
        if (rec_.numRegistered() <= 2) return 0;
        // rig 帧作为整体判断与撤销。
        std::vector<uint32_t> drop;
        std::set<uint32_t> seen;
        for (auto& kv : rec_.images) {
            if (!kv.second.registered || seen.count(kv.first)) continue;
            const std::vector<uint32_t> frame = frameMates(kv.first);
            size_t points = 0;
            for (uint32_t j : frame) {
                seen.insert(j);
                points += rec_.images.at(j).numPoint3D();
            }
            if ((int)points < opt_.min_image_points)
                drop.insert(drop.end(), frame.begin(), frame.end());
        }
        for (uint32_t id : drop) deregisterImage(id);
        if (!drop.empty()) resetOrphanCameras();
        return (int)drop.size();
    }

    // 为每已配准图像复制独立相机，返回创建数；已逐图独立时为 0。
    size_t splitCamerasPerImage() {
        std::set<uint32_t> used;
        size_t registered = 0;
        uint32_t next = 0;
        for (const auto& kv : rec_.cameras) next = std::max(next, kv.first);
        for (const auto& kv : rec_.images)
            if (kv.second.registered) {
                used.insert(kv.second.camera_id);
                registered++;
            }
        if (used.size() >= registered) return 0;
        size_t made = 0;
        for (auto& kv : rec_.images) {
            if (!kv.second.registered) continue;
            Camera c = rec_.cameras.at(kv.second.camera_id);
            c.id = ++next;
            rec_.cameras[c.id] = c;
            default_cams_[c.id] = c;
            kv.second.camera_id = c.id;
            made++;
        }
        return made;
    }

    // 相机组全部图像被撤销后恢复原始默认，避免重试继承已被否定配准拟合出的坏内参。
    void resetOrphanCameras() {
        std::set<uint32_t> used;
        for (const auto& kv : rec_.images)
            if (kv.second.registered) used.insert(kv.second.camera_id);
        for (auto& kv : rec_.cameras)
            if (!used.count(kv.first) && focal_known_.count(kv.first)) {
                kv.second = default_cams_.at(kv.first);
                // 镜头先验独立于失败配准，保持已知，default_cams_ 中已保存。
                if (!opt_.known_focal_cameras.count(kv.first)) focal_known_.erase(kv.first);
            }
    }

    const MatchesDatabase& db_;
    const std::vector<FeatureSet>& feats_;
    MapperOptions opt_;
    std::vector<uint32_t> cam_ids_;   // 逐图像相机 ID，从 1 开始
    std::set<uint32_t> focal_known_;  // 焦距不再是猜测的相机集合
    // 收尾释放或跳过的临时配置仅在单次调用期间生效。
    struct FinalRelease {
        bool pp = false;           // 释放主点（D51）
        bool extra = false;        // 释放畸变系数（D72）
        bool no_sanitize = false;  // 逐图独立内参，无共享组可用于钳位（D73）
        bool no_rig = false;       // 各图像独立位姿，解除 rig 约束
    };
    FinalRelease final_;
    std::map<uint32_t, Camera> default_cams_;  // 原始组默认内参
    // 下项保存已接纳模型发布的最佳内参与支持图像数（D45）。
    std::map<uint32_t, std::pair<Camera, double>> cam_consensus_;
    bool setup_done_ = false;
    // SS_SFM_AUDIT_DUMP=1 输出逐图审查支持比例，用于在 rig 数据上选择阈值。
    const bool audit_dump_ = spirula::env("SFM_AUDIT_DUMP") != nullptr;
    // SS_SFM_RIG_DUMP=1 输出各 rig 放置判定和精化偏移，用于选择容差。
    const bool rig_dump_ = spirula::env("SFM_RIG_DUMP") != nullptr;
    const bool seq_dump_ = spirula::env("SFM_SEQ_DUMP") != nullptr;  // seqDump 使用的开关
    mutable double scale_cache_ = 0;  // modelScale 缓存，由 resetModel 重置
    int init_relax_ = 0;              // 已达到的种子阈值放宽等级
    InitTally init_tally_;            // 最近 initialize 未找到种子的原因
    const TwoViewMatches* seed_pair_ = nullptr;  // 最近成功种子的图像对
    double seed_forward_ = 0;         // 对应的 |基线·视线方向|
    std::set<std::pair<uint32_t, uint32_t>> used_seeds_;  // 已增长过的种子
    std::map<std::pair<uint32_t, uint32_t>, TwoViewGeometry> seed_geom_;  // 双视图几何缓存（D38）
    Reconstruction rec_;
    CorrespondenceGraph graph_;
    std::vector<std::vector<uint16_t>> support_;  // 逐（图像，特征），见 rebuildScores
    std::vector<int> score_cache_;                // 每图像均等于 score(i)
    std::vector<std::vector<uint16_t>> pyramid_;  // 逐图像 kPyrCells 个占用计数
    std::vector<uint32_t> pyramid_score_;         // 逐图像金字塔分数，见 pyramidSet
    std::vector<int> reg_trials_;
    // 逐图像被保留模型配准的次数，首模型保留前为空；另累计本次运行的配准拒绝原因。
    // 精化后的第二次比例检查沿用已做过的歧义判定，绝对支持策略见 strong_pnp_inliers。
    bool ratioOk(int inliers, size_t pool) const {
        if ((double)inliers >= opt_.min_pnp_inlier_ratio * (double)pool) return true;
        return opt_.strong_pnp_inliers > 0 && inliers >= opt_.strong_pnp_inliers;
    }

    // 统计位于相机前方且投影在图内的可解释对应，作为 visible-only 内点比例分母。
    size_t visibleMask(uint32_t img, const std::vector<Vec3>& X, const std::vector<Vec3>& br,
                       const Pose& pose, std::vector<char>& vis) const {
        vis.assign(X.size(), 1);
        if (!opt_.pnp_ratio_visible_only) return X.size();
        const Camera& cam = camOf(img);
        const double w = cam.width > 0 ? (double)cam.width : 1e9;
        const double h = cam.height > 0 ? (double)cam.height : 1e9;
        const double mx = 0.05 * w, my = 0.05 * h;
        size_t n = 0;
        for (size_t k = 0; k < X.size(); k++) {
            vis[k] = 0;
            const Vec3 pc = mul(pose.R, X[k]) + pose.t;
            if (cam.wideFov()) {
                if (k < br.size() && pc.dot(br[k]) <= 0) continue;
            } else if (pc.z < 1e-8) {
                continue;
            }
            const Vec2 px = cam.project(pc);
            if (!std::isfinite(px.x) || !std::isfinite(px.y)) continue;
            if (px.x < -mx || px.y < -my || px.x > w + mx || px.y > h + my) continue;
            vis[k] = 1;
            n++;
        }
        return n;
    }

    size_t visiblePool(uint32_t img, const std::vector<Vec3>& X,
                       const std::vector<Vec3>& br, const Pose& pose) const {
        if (!opt_.pnp_ratio_visible_only) return X.size();
        const Camera& cam = camOf(img);
        const double w = cam.width > 0 ? (double)cam.width : 1e9;
        const double h = cam.height > 0 ? (double)cam.height : 1e9;
        // 图像边界留余量，避免一像素级位姿差改变刚出界点的门限判定。
        const double mx = 0.05 * w, my = 0.05 * h;
        size_t n = 0;
        for (size_t k = 0; k < X.size(); k++) {
            const Vec3 pc = mul(pose.R, X[k]) + pose.t;
            // 针孔按 z，宽角按关键点实际射线方向判断正深度，与其他建图路径一致（D33）。
            if (cam.wideFov()) {
                if (k < br.size() && pc.dot(br[k]) <= 0) continue;
            } else if (pc.z < 1e-8) {
                continue;
            }
            const Vec2 px = cam.project(pc);
            if (!std::isfinite(px.x) || !std::isfinite(px.y)) continue;
            if (px.x < -mx || px.y < -my || px.x > w + mx || px.y > h + my) continue;
            n++;
        }
        return n;
    }

    // 比例未通过时检查绝对支持是否充分且无竞争位姿；在最佳解拒绝的对应中搜索其他位置。
    bool strongUnambiguous(uint32_t img, const std::vector<Vec3>& X,
                           const std::vector<Vec3>& br, const PnPResult& r) {
        if (opt_.strong_pnp_inliers <= 0 || r.num_inliers < opt_.strong_pnp_inliers) return false;
        if (opt_.strong_pnp_max_rival <= 0) return true;
        std::vector<Vec3> X2, b2;
        X2.reserve(X.size());
        b2.reserve(X.size());
        for (size_t k = 0; k < X.size(); k++)
            if (!r.inlier_mask[k]) { X2.push_back(X[k]); b2.push_back(br[k]); }
        const int need = (int)std::ceil(opt_.strong_pnp_max_rival * (double)r.num_inliers);
        if ((int)X2.size() < need) return true;  // 剩余对应不足以形成竞争解
        PnPResult alt = ransacPnP(X2, b2, camOf(img).focal(), errPx(img), 0,
                                  opt_.audit_ransac_trials);
        if (!alt.success || alt.num_inliers < need) return true;
        // 竞争解若与原位姿相同，仅表示内点阈值差异，不算另一位置；中心差按相机到可见结构距离归一化。
        Mat3 D = mul(alt.pose.R, transpose(r.pose.R));
        const double tr = std::max(-1.0, std::min(1.0, (D[0] + D[4] + D[8] - 1) * 0.5));
        if (std::acos(tr) * 180.0 / M_PI > opt_.audit_min_rotation_deg) {
            reg_fail_.ambiguous++;
            return false;
        }
        const Vec3 c = cameraCenter(r.pose);
        double depth = 0;
        size_t nd = 0;
        for (size_t k = 0; k < X.size(); k++)
            if (r.inlier_mask[k]) { depth += (X[k] - c).norm(); nd++; }
        if (nd && (cameraCenter(alt.pose) - c).norm() > 0.1 * depth / (double)nd) {
            reg_fail_.ambiguous++;
            return false;
        }
        return true;
    }

    struct RegFail {
        uint32_t few_corr = 0, few_inliers = 0, low_ratio = 0, refined_out = 0;
        uint32_t strong = 0;     // 比例失败但按绝对支持接纳的次数（D69）
        uint32_t ambiguous = 0;  // 因剩余对应存在竞争位姿而拒绝的次数
        uint32_t occluded = 0;   // 接受位姿完全不可见的候选对应数
    } reg_fail_;
    const RigTable* rigs_ = nullptr;  // 空值表示没有 rig 或已禁用
    const SequenceTable* seq_ = nullptr;  // 空值表示没有序列
    PriorSource* priors_ = nullptr;   // 空值表示无传感器先验或禁用 sensor-map
    PriorStats prior_stats_;
    // 审查和种子预取等 const 并行阶段的原子统计。
    mutable std::atomic<uint32_t> prior_vouched_{0}, prior_seeds_{0};
    // SS_SFM_PRIOR_DUMP=1 输出每次被陀螺推翻的配准。
    const bool prior_dump_ = spirula::env("SFM_PRIOR_DUMP") != nullptr;
    std::vector<std::vector<uint16_t>> near_support_;  // 仅包含序列邻居的 support_
    std::vector<int> near_score_;                      // 逐图像具有邻居支持的特征数
    uint32_t reg_vouched_ = 0;   // 依靠邻居支持越过全池比例门限的配准数
    uint32_t reg_near_won_ = 0;  // 其中邻居位姿胜过全池位姿的数量
    int seed_phase_ = 0;         // 0 仅从邻居对播种，1 使用全部候选
    uint32_t reg_by_rig_ = 0;         // 整次运行由 rig 放置的图像数
    uint32_t reg_rig_word_ = 0;       // 其中自身没有内点的图像数
    uint32_t frame_regs_ = 0;         // 随候选共同放置的 rig 伙伴数
    uint32_t rig_refined_at_ = 0;     // 上次外参精化时的模型规模

    // 每个可优化成员为各观测增加六列，因此增长期仅在模型规模翻倍时释放外参；严格精化首轮则允许优化。
    bool rigRefineDue(bool tight) {
        const uint32_t n = rec_.numRegistered();
        if (!tight && n < 2 * rig_refined_at_) return false;
        rig_refined_at_ = n;
        return true;
    }
    std::vector<uint8_t> allow_;      // restrictTo 的限制集合，空值为全部图像
    size_t allow_count_ = 0;          // 限制集合中选中的图像数
    std::vector<uint32_t> model_count_;
    std::vector<uint8_t> seeded_;     // 种子尝试已到达的图像
    // 当前限制内候选按内点数排序，有序列时先邻居对再其他对。
    std::vector<const TwoViewMatches*> seed_cand_, seed_cand_far_;
    bool seed_cand_valid_ = false;
    // 持久 BA 上下文复用设备与流水线，每次求解仅分配问题缓冲。
    // 每种标量配置独立上下文，float 与 double 模块不能混用；两阶段配置相同时可共用调用方上下文（D38）。
    VkContext ba_ctx_[2];
    VkContext* ext_ba_ctx_ = nullptr;  // 外部上下文，空值使用内部 ba_ctx_
    bool ba_over_budget_throws_ = false;  // 仅 refineIfItFits 使用
    RealCfg baReal(bool coarse) const {
        return realCfgFromName(coarse ? opt_.ba_real_coarse : opt_.ba_real);
    }
    VkContext& baContext(bool coarse) {
        const bool second = coarse && baReal(true) != baReal(false);
        if (ext_ba_ctx_ && !second) return *ext_ba_ctx_;
        return ba_ctx_[second];
    }
    std::vector<uint32_t> recent_regs_;  // 上次精化后新增的配准图像
};

}  // 命名空间 sfm
