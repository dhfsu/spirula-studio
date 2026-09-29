// 模型集合的共享操作：合并、拆分、补配、去重与切开折叠，由 Assemble 调度，两种建图器使用相同阈值。
// 通过模型状态摘要跳过已处理且未变化的结果，避免重复支付昂贵检查成本。
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "sfm/core/Model.h"
#include "sfm/map/Mapper.h"
#include "sfm/map/Merge.h"
#include "sfm/core/Log.h"

namespace sfm {

// 统一的模型管理策略阈值，规定合并、拆分及所需证据，两种建图器共用。
struct ManagerOptions {
    MergeOptions merge;
    bool do_merge = true;
    bool do_grow = true;      // 向已有模型补配未注册图像
    bool do_reseed = true;    // 在未覆盖区域寻找新模型
    // 审查修复默认关闭，但精化保留；5356 图修复 77 图使 AUC@10 93.5->65.0，耗时 421/2076 s，7620 图修复 558 图损失 7 分，耗时 242/2005 s。
    // 弱于正常配准的修复证据在大规模上不可靠，应先用接缝与拼接检查拒绝坏合并（D71）。
    bool do_audit = false;
    // 审查修复后的最大增长比例，0 不限制；装配器会按自身增长计划覆盖，避免隐藏完整增量重建。
    double audit_growth_frac = 0;
    // 模型大部分图像已被更大结果覆盖时视为重复，不单独写出。
    double redundant_ratio = 0.9;
    // 组焦距偏离总体共识超过此倍数时，合并前重新拟合，避免错误焦距使模型无法对齐。
    double focal_consensus_tol = 0.15;  // 0 禁用焦距共识重拟合
    // 接缝验证要求模型解释未参与对齐的跨缝双视图证据，阈值 0 可禁用检查。
    double seam_min_agreement = 0.6;   // 通过检查的跨缝图像对比例
    double seam_min_pair_fraction = 0.5;  // 单对匹配至少解释的比例
    int seam_min_pairs = 10;           // 少于此图像对数无法判断
    // 实际接缝门限取固定上限与内部一致性乘比例的较小者（D68）；7620 图数据固定门限曾拒绝 43 次中的 41 次，禁用后恢复 268 图及 3 分 AUC。
    double seam_relative_bar = 0.6;    // 0 表示直接使用固定 seam_min_pair_fraction
    int seam_reference_pairs = 400;    // 建立基线所用的非跨缝采样对数
    // 失败接缝若仍能解释典型对约五分之一匹配，可尝试一次粗精化复判；成功结果直接保留，仅失败救援消耗预算（D64）。
    double seam_rescue_frac = 0.2;     // 0 禁用接缝救援
    int seam_max_rescues = 8;          // 放弃前允许的失败救援精化次数
    // 拆分仅采用匹配数达标的验证边，过小组撤销配准而不单独输出。
    bool do_split = true;
    int split_min_matches = 30;
    size_t split_min_group = 10;
    // 默认检查折叠，通过缺失共同结构发现错误重叠；冲突率不能区分密集正确采集，实际切割损失才是判据，真伪折叠可相差百倍（D46）。
    bool do_duplicate_split = true;
    DuplicateOptions duplicate;
    bool verbose = true;
};

struct ManagerStats {
    size_t rounds = 0;
    size_t merges = 0, merges_refused = 0;
    size_t grown_images = 0;       // 增长阶段新增配准图像数
    size_t reseeded_models = 0;
    size_t dropped_redundant = 0;
    size_t audited_out = 0, audited_repaired = 0;
    size_t cameras_refit = 0;
    size_t seam_checked = 0, seam_refused = 0, seam_skipped = 0;
    size_t seam_rescued = 0, seam_rescue_failed = 0;
    // 分别累计接受与拒绝合并的接缝解释比例及证据数量，便于评估门限区分度；rescue_gain 记录失败救援对中位解释比例的改善。
    double seam_refused_median = 0, seam_passed_median = 0, seam_rescue_gain = 0;
    // 累计实际门限和内部非跨缝一致性，辅助解释低拒绝分数是否真低于该数据可达到的基线（D68）。
    double seam_bar_sum = 0, seam_reference_sum = 0;
    size_t seam_refused_pairs = 0, seam_passed = 0;
    size_t splits = 0, duplicate_splits = 0, split_dropped = 0;
    size_t models_before = 0, models_after = 0;
    size_t covered_before = 0, covered_after = 0;
};

// 模型索引会重排，以图像和点数量描述状态以跳过重复工作；偶然摘要碰撞最多漏掉一次通常无效的尝试。
struct ModelMemo {
    using Signature = std::pair<uint32_t, size_t>;
    static Signature of(const Reconstruction& m) {
        return {m.numRegistered(), m.points3D.size()};
    }
    std::set<Signature> barren;   // 此前增长未能扩展的模型状态
    std::set<Signature> audited;  // 已检查过的模型状态
    std::set<Signature> split;    // 已审查接缝的模型状态
    void clear() {
        barren.clear();
        audited.clear();
        split.clear();
    }
};

// 以图像角点像素变化判断联合精化是否产生可影响后续决策的改变，门限远小于配准、过滤容差。
inline constexpr double kJointBaMovedPx = 1.0;

// 各分量内参在图像角点造成的最大像素差；已达亚像素一致时无需为消除内参分歧再做整场景 BA。
inline double focalSpreadPx(const std::vector<Reconstruction>& models) {
    std::map<uint32_t, std::pair<double, double>> range;  // 相机 ID ->（最小焦距，最大焦距）
    std::map<uint32_t, double> radius;                    // 以及对应图像角点
    for (const Reconstruction& m : models)
        for (const auto& kv : m.cameras) {
            auto it = range.find(kv.first);
            if (it == range.end()) {
                range[kv.first] = {kv.second.focal(), kv.second.focal()};
                radius[kv.first] = std::hypot(kv.second.cx, kv.second.cy);
            } else {
                it->second.first = std::min(it->second.first, kv.second.focal());
                it->second.second = std::max(it->second.second, kv.second.focal());
            }
        }
    double px = 0;
    for (const auto& kv : range) {
        const double f = 0.5 * (kv.second.first + kv.second.second);
        if (f > 0)
            px = std::max(px, (kv.second.second - kv.second.first) / f * radius[kv.first]);
    }
    return px;
}

// 接缝验证查询预先验证的跨模型双视图几何，避免仅用对齐自身证据认可重复结构（D45）。
// 边缘失败可粗精化后再判，通过则保留修复结果；flat 与 bottom-up 均共用此验证。
inline std::function<std::string(Reconstruction&, const Reconstruction&, const Sim3&,
                                 const MergeCounts&)>
seamValidator(Mapper& mapper, const ManagerOptions& opt, ManagerStats* st = nullptr) {
    if (opt.seam_min_agreement <= 0) return {};
    Mapper* mp = &mapper;
    const double max_err = opt.merge.max_reproj_error;
    const double pair_frac = opt.seam_min_pair_fraction;
    const double min_agree = opt.seam_min_agreement;
    const size_t min_pairs = (size_t)std::max(0, opt.seam_min_pairs);
    const double rescue_frac = opt.seam_rescue_frac;
    const double rel_bar = opt.seam_relative_bar;
    const size_t ref_pairs = (size_t)std::max(0, opt.seam_reference_pairs);
    // 全部合并尝试共享救援预算，不能每次重新获得预算。
    auto budget = std::make_shared<int>(std::max(0, opt.seam_max_rescues));
    // 内部一致性基线在证据足够后测量并复用，-1 表示尚未建立，小模型首次合并可能证据不足。
    auto reference = std::make_shared<double>(-1.0);
    const double max_splice = opt.merge.max_splice_conflict_ratio;
    return [mp, max_err, pair_frac, min_agree, min_pairs, rescue_frac, rel_bar, ref_pairs,
            max_splice, budget, reference, st](Reconstruction& merged, const Reconstruction& src,
                                               const Sim3&,
                                               const MergeCounts& counts) -> std::string {
        std::set<uint32_t> src_side;
        for (const auto& kv : src.images)
            if (kv.second.registered) src_side.insert(kv.first);
        // 共享位姿非常可靠但共享点不一致时，先精化协调两侧漂移，再进行独立验证，避免重复惩罚同一未优化形状误差（D64）。
        const bool contested =
            counts.points_spliced &&
            counts.splice_conflicts > max_splice * (double)counts.points_spliced;
        // 设备容不下精化时仍按原几何评估，不能仅因机器内存较小直接拒绝合并。
        if (contested && !mp->refineIfItFits(merged, /*coarse=*/true))
            if (st) st->seam_rescue_failed++;
        // 在当前合并模型中测量非跨缝图像对一致性，作为其接缝的比较基线（D68）。
        double bar = pair_frac;
        if (rel_bar > 0 && ref_pairs) {
            if (*reference < 0) {
                Mapper::SeamCheck ref = mp->checkSeam(merged, src_side, max_err, pair_frac,
                                                      ref_pairs, /*crossing=*/false);
                if (ref.tested >= ref_pairs / 2) *reference = ref.median_frac;
                else if (ref.tested >= min_pairs) bar = std::min(bar, rel_bar * ref.median_frac);
            }
            if (*reference >= 0) bar = std::min(bar, rel_bar * *reference);
            if (st) st->seam_reference_sum += *reference >= 0 ? *reference : bar / rel_bar;
        }
        Mapper::SeamCheck sc = mp->checkSeam(merged, src_side, max_err, bar);
        // 单独统计因证据不足而跳过的检查，不能将其混为验证通过。
        if (sc.tested < min_pairs) {
            if (st) st->seam_skipped++;
            return "";
        }
        if (st) { st->seam_checked++; st->seam_bar_sum += bar; }
        if ((double)sc.agree / (double)sc.tested >= min_agree) {
            if (st) {
                st->seam_passed++;
                st->seam_passed_median += sc.median_frac;
                // 原本会因拼接冲突拒绝的模型，经精化后由独立接缝证据支持，属于仲裁成功。
                if (contested) st->seam_rescued++;
            }
            return "";
        }

        // 接近通过但尚未收敛时粗精化并复判，成功保留精化结果；已为拼接仲裁精化过的候选不能重复救援。
        if (!contested && rescue_frac > 0 && sc.median_frac >= rescue_frac && *budget > 0) {
            Reconstruction fixed = merged;
            Mapper::SeamCheck s2;
            if (mp->refineIfItFits(fixed, /*coarse=*/true))
                s2 = mp->checkSeam(fixed, src_side, max_err, bar);
            if (s2.tested >= min_pairs && (double)s2.agree / (double)s2.tested >= min_agree) {
                if (st) st->seam_rescued++;
                merged = std::move(fixed);
                return "";
            }
            --*budget;  // 仅失败救援计入预算
            if (st) {
                st->seam_rescue_failed++;
                if (s2.tested) st->seam_rescue_gain += s2.median_frac - sc.median_frac;
            }
            if (s2.tested) sc = s2;
        }
        if (st) {
            st->seam_refused++;
            st->seam_refused_median += sc.median_frac;
            st->seam_refused_pairs += sc.tested;
        }
        char buf[224];
        snprintf(buf, sizeof buf,
                 "only %zu of the %zu verified pairs that cross the seam still hold "
                 "(median %.0f%% of their matches explained, against a %.0f%% bar)",
                 sc.agree, sc.tested, 100.0 * sc.median_frac, 100.0 * bar);
        return std::string(buf);
    };
}

// 所有模型配准的不同图像集合。
inline std::set<uint32_t> coveredImages(const std::vector<Reconstruction>& models) {
    std::set<uint32_t> ids;
    for (const Reconstruction& m : models)
        for (const auto& kv : m.images)
            if (kv.second.registered) ids.insert(kv.first);
    return ids;
}

// 所有模型排序均按三维点数降序，保持 sparse/0 为结构最多者。
inline void sortModels(std::vector<Reconstruction>& models) {
    std::stable_sort(models.begin(), models.end(),
                     [](const Reconstruction& a, const Reconstruction& b) {
                         return a.points3D.size() > b.points3D.size();
                     });
}

// ---------------- 模型处理步骤 ----------------

// 按大模型优先，将每个模型与此前保留模型的图像并集比较，不能仅逐一比较。
// 5356 图结果中的多个模型虽不被任一单模型覆盖 90%，却被其他模型并集覆盖 98–100%；follow 标记须同步重排和过滤。
inline std::vector<Reconstruction> dropRedundantModels(std::vector<Reconstruction> models,
                                                       const ManagerOptions& opt,
                                                       ManagerStats& st,
                                                       std::vector<std::vector<char>*> follow = {}) {
    std::vector<size_t> order(models.size());
    std::iota(order.begin(), order.end(), (size_t)0);
    std::stable_sort(order.begin(), order.end(), [&models](size_t a, size_t b) {
        return models[a].points3D.size() > models[b].points3D.size();
    });
    std::vector<size_t> keep;
    std::set<uint32_t> covered;
    for (size_t rank = 0; rank < order.size(); rank++) {
        const Reconstruction& m = models[order[rank]];
        const uint32_t n = m.numRegistered();
        size_t shared = 0;
        for (const auto& kv : m.images)
            if (kv.second.registered && covered.count(kv.first)) shared++;
        if (rank && n && (double)shared >= opt.redundant_ratio * (double)n) {
            st.dropped_redundant++;
            if (opt.verbose)
                slog::diag(slog::Tag::Map,
                           "[mgr] dropped a %u-image model: %zu of its images are already in "
                           "larger ones", n, shared);
            continue;
        }
        for (const auto& kv : m.images)
            if (kv.second.registered) covered.insert(kv.first);
        keep.push_back(order[rank]);
    }
    for (std::vector<char>* f : follow) {
        std::vector<char> next;
        next.reserve(keep.size());
        for (size_t i : keep) next.push_back(i < f->size() ? (*f)[i] : 0);
        *f = std::move(next);
    }
    std::vector<Reconstruction> out;
    out.reserve(keep.size());
    for (size_t i : keep) out.push_back(std::move(models[i]));
    return out;
}

// 按对应图拆开内部几何矛盾的模型，即使它从未经历合并也可能因重复结构配准出错；仅重查变化后的状态，避免拆分、合并循环。
inline std::vector<Reconstruction> splitInconsistentModels(Mapper& mapper,
                                                           std::vector<Reconstruction> models,
                                                           const ManagerOptions& opt,
                                                           ModelMemo& memo, ManagerStats& st) {
    std::vector<Reconstruction> out;
    for (Reconstruction& m : models) {
        const ModelMemo::Signature sig = ModelMemo::of(m);
        if (m.numRegistered() < 2 * opt.split_min_group || !memo.split.insert(sig).second) {
            out.push_back(std::move(m));
            continue;
        }
        Mapper::SplitStats ss;
        std::vector<Reconstruction> parts = mapper.splitInconsistent(
            m, opt.merge.max_reproj_error, opt.seam_min_pair_fraction, opt.split_min_matches,
            opt.split_min_group, &ss);
        if (parts.size() <= 1) {
            out.push_back(std::move(m));
            continue;
        }
        st.splits++;
        st.split_dropped += ss.dropped_images;
        if (opt.verbose)
            slog::diag(slog::Tag::Map,
                       "[mgr] split a %u-image model into %zu (%zu of %zu inner pairs hold; "
                       "largest group %zu, %zu images dropped)",
                       m.numRegistered(), parts.size(), ss.pairs_agree, ss.pairs_tested, ss.largest,
                       ss.dropped_images);
        for (Reconstruction& p : parts) {
            memo.split.insert(ModelMemo::of(p));
            out.push_back(std::move(p));
        }
    }
    return out;
}

// 检查同位同向图像缺少共同结构的折叠，普通一致性检查无法发现内部自洽的错误叠合。
inline std::vector<Reconstruction> splitFoldedModels(Mapper& mapper,
                                                     std::vector<Reconstruction> models,
                                                     const ManagerOptions& opt, ModelMemo& memo,
                                                     ManagerStats& st) {
    std::vector<Reconstruction> out;
    MatchedFn matched = mapper.matchedPredicate();
    for (Reconstruction& m : models) {
        if (m.numRegistered() < 2 * opt.split_min_group) {
            out.push_back(std::move(m));
            continue;
        }
        DuplicateReport dr = findDuplicateStructure(m, opt.duplicate, matched);
        if (!dr.duplicated(opt.duplicate)) {
            out.push_back(std::move(m));
            continue;
        }
        size_t dropped = 0;
        DuplicateCut cut;
        std::vector<Reconstruction> parts =
            splitDuplicateStructure(m, dr, opt.split_min_group, &dropped, &cut,
                                    opt.duplicate.min_fold_overlap);
        // 冲突说明可能折叠，低切割代价说明可分离，两者同时满足才接受（D46）。
        if (parts.size() <= 1 || !foldSplitAccepted(dr, cut, opt.duplicate)) {
            if (opt.verbose && parts.size() > 1)
                slog::diag(slog::Tag::Map,
                           "[mgr] a %u-image model has %zu of %zu co-located pairs with nothing "
                           "in common, but splitting it would sever %.1f%% of its co-visibility "
                           "(>%.1f%%): keeping it whole",
                           m.numRegistered(), dr.conflicts, dr.colocated, 100.0 * cut.fraction(),
                           100.0 * opt.duplicate.max_cut_fraction);
            else if (opt.verbose && cut.reattached)
                slog::diag(slog::Tag::Map,
                           "[mgr] a %u-image model has %zu of %zu co-located pairs with nothing "
                           "in common, but the %zu piece(s) they would cut off stand where nothing "
                           "else does (<%.0f%%): keeping it whole",
                           m.numRegistered(), dr.conflicts, dr.colocated, cut.reattached,
                           100.0 * opt.duplicate.min_fold_overlap);
            out.push_back(std::move(m));
            continue;
        }
        st.duplicate_splits++;
        st.split_dropped += dropped;
        if (opt.verbose) {
            char head[512];
            snprintf(head, sizeof head,
                     "[mgr] a %u-image model has %zu of %zu co-located image pairs with no "
                     "structure in common and no match either (%zu more share nothing but were "
                     "matched), a cut that severs %.2f%% of its co-visibility, and every piece "
                     "standing where another one does (%.0f%% at worst): two places written on "
                     "top of each other. Splitting into",
                     m.numRegistered(), dr.conflicts, dr.colocated, dr.unmatched_but_seen,
                     100.0 * cut.fraction(), 100.0 * cut.min_overlap);
            std::string line = head;
            for (const Reconstruction& p : parts)
                line += " " + std::to_string(p.numRegistered());
            if (dropped) line += " (" + std::to_string(dropped) + " images dropped)";
            slog::diag(slog::Tag::Map, "%s", line.c_str());
        }
        for (Reconstruction& p : parts) {
            memo.split.insert(ModelMemo::of(p));
            out.push_back(std::move(p));
        }
    }
    return out;
}

// ---------------- 相机参数共识 ----------------
// 按支持图像数求共享镜头内参共识，替换偏离严重模型的整套相机参数，再精化位姿与结构；仍有错误者会被过滤或合并检查拒绝。
inline void refitOutlierCameras(Mapper& mapper, std::vector<Reconstruction>& models,
                                const ManagerOptions& opt, ManagerStats& st) {
    if (models.size() < 3) return;  // 没有足够总体数据求共识
    std::map<uint32_t, std::vector<std::pair<double, uint32_t>>> focals;  // 相机 ID ->（焦距，权重）
    std::map<uint32_t, uint32_t> widest;  // 相机 ID -> 任一模型提供的最大支持图像数
    for (const Reconstruction& m : models) {
        std::map<uint32_t, uint32_t> used;
        for (const auto& kv : m.images)
            if (kv.second.registered) used[kv.second.camera_id]++;
        for (const auto& kv : m.cameras) {
            auto u = used.find(kv.first);
            if (u == used.end()) continue;
            focals[kv.first].push_back({kv.second.focal(), u->second});
            widest[kv.first] = std::max(widest[kv.first], u->second);
        }
    }
    std::map<uint32_t, double> consensus;
    for (const auto& kv : focals) {
        // 单图相机组没有更可靠群体共识，禁止为替换同样无依据的猜测执行整模型 BA；网络照片集合中这曾成为最慢阶段。
        if (widest[kv.first] < 2 || kv.second.size() < 3) continue;
        // 按图像数加权中位数，平局由最大模型决定，避免少量小模型发散值拖动共识。
        std::vector<std::pair<double, uint32_t>> v = kv.second;
        std::sort(v.begin(), v.end());
        uint64_t total = 0;
        for (const auto& p : v) total += p.second;
        uint64_t acc = 0;
        double med = v.empty() ? 0 : v.front().first;
        for (const auto& p : v) {
            acc += p.second;
            med = p.first;
            if (acc * 2 >= total) break;
        }
        consensus[kv.first] = med;
    }
    for (Reconstruction& m : models) {
        std::vector<uint32_t> bad;
        for (const auto& kv : m.cameras) {
            auto c = consensus.find(kv.first);
            if (c == consensus.end()) continue;
            const double f = kv.second.focal(), ref = c->second;
            if (ref > 0 && std::fabs(f - ref) > opt.focal_consensus_tol * ref)
                bad.push_back(kv.first);
        }
        if (bad.empty()) continue;
        // 复制共识来源的整套相机而非仅焦距，因为发散畸变也可能在补偿同一错误。
        for (uint32_t id : bad) {
            const Camera* donor = nullptr;
            for (const Reconstruction& other : models) {
                auto it = other.cameras.find(id);
                if (it == other.cameras.end()) continue;
                if (std::fabs(it->second.focal() - consensus[id]) < 1e-6 * consensus[id]) {
                    donor = &it->second;
                    break;
                }
            }
            if (!donor) continue;
            if (opt.verbose)
                slog::diag(slog::Tag::Map,
                           "[mgr] camera %u of a %u-image model: focal %.0f vs the %.0f the other "
                           "models agree on; refitting", id, m.numRegistered(),
                           m.cameras[id].focal(), donor->focal());
            m.cameras[id] = *donor;
            st.cameras_refit++;
        }
        const uint32_t before = m.numRegistered();
        m = mapper.refine(m);
        if (opt.verbose)
            slog::diag(slog::Tag::Map, "[mgr] refit model: %u -> %u images, %zu points", before,
                       m.numRegistered(), m.points3D.size());
    }
}

// 仅对变化模型执行配置启用的审查与精化，未变化者保持原值；单一相似变换不能消除两侧漂移，接缝需联合几何处理。
inline std::vector<Reconstruction> auditModels(Mapper& mapper, std::vector<Reconstruction> models,
                                               const ManagerOptions& opt, ModelMemo& memo,
                                               ManagerStats& st) {
    for (Reconstruction& m : models) {
        if (!memo.audited.insert(ModelMemo::of(m)).second) continue;
        const size_t pts = m.points3D.size();
        const uint32_t imgs = m.numRegistered();
        Mapper::AuditStats as;
        const uint32_t cap =
            opt.audit_growth_frac > 0
                ? (uint32_t)std::ceil((1.0 + opt.audit_growth_frac) * (double)imgs)
                : 0;
        m = opt.do_audit ? mapper.audit(m, &as, cap) : mapper.refine(m);
        st.audited_out += as.deregistered;
        st.audited_repaired += as.reregistered;
        memo.audited.insert(ModelMemo::of(m));
        if (opt.verbose && (as.unsupported || m.numRegistered() != imgs))
            slog::diag(slog::Tag::Map, "[mgr] audited a model: %u -> %u images, %zu -> %zu points "
                       "(%u unsupported, %u re-registered, %u dropped)",
                       imgs, m.numRegistered(), pts, m.points3D.size(), as.unsupported,
                       as.reregistered, as.deregistered);
    }
    return models;
}

}  // 命名空间 sfm
