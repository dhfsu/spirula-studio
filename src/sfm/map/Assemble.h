// flat 与 bottom-up 共用的模型装配（D57/D63）：逐层合并有可靠重叠的模型、扩展未合并模型、优化变更，再重复到稳定。
// 最终仅一次执行审查、冲突拆分、折叠检查、尾部配准与重新播种；合并验证使用未参与对齐的独立证据。
// 限制重复修复与扩展，避免小模型长成主模型副本；5356 图数据中无界管理曾耗时 65 分钟仅合并三个模型并新增 119 图。
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "sfm/core/Model.h"
#include "sfm/map/Mapper.h"
#include "sfm/map/Merge.h"
#include "sfm/core/Log.h"
#include "sfm/map/ModelOps.h"
#include "i18n/TimeFormat.h"
#include "i18n/catalog/Sfm.h"

namespace sfm {

struct AssembleOptions {
    // 合并树层数上限，约为 log2(模型数) 加上精化后重新尝试拒绝合并所需的层数。
    int max_rounds = 16;
    // 有变化的层后执行跨模型联合 BA，按相机组共享内参。
    bool joint_intrinsics = true;
    // 中间联合求解可使用增长阶段容差以节省后续会被重算的收敛工作，但过松会增大像素残差，使接缝验证更严格。
    bool coarse_joint_ba = true;
    // 联合优化的层间隔，最终层始终求解；默认 1。间隔 2 在五组中四组快 5–56 s，另一组下降 6 分，而该组相同配置本身波动 9.3 分，因此不改默认。
    // 例如 windmill 树求解 152->93 s，vicon_room 31->25 s；可通过选项自行权衡。
    int joint_every = 1;
    // 未合并模型的增长层间隔，0 禁用；max_grow_passes 限制总轮数，避免大树只顾配准。
    int grow_every = 1;
    int max_grow_passes = 8;
    // 单次增长上限按模型规模比例且至少 grow_budget_min；目标仅为建立合并重叠，防止模型扩展到整个未覆盖区域。
    double grow_budget_frac = 0.25;
    size_t grow_budget_min = 25;
    // 无共同图像时可按共同三维结构尝试桥接，但默认关闭：用匹配对齐再用同一匹配验证，破坏证据独立性。
    // 7620 图室内混合数据七次测量，启用时 AUC@10 为 17.0–25.7，禁用为 44.7–46.3，而仅新增 9 或 65 图；成为默认前需要独立验证。
    int max_bridges = 0;
    size_t bridge_min_matches = 500;  // 连接两个模型的特征匹配数量
    // 收尾重新播种使用远小于主建图器的预算，残余弱连接连续三次失败通常已无值得寻找的模型。
    int reseed_trials = 3;
    // 审查修复后的模型大小按原规模比例限制，避免每次修复都变成完整增量重建并复制大模型。
    double audit_growth_frac = 0.25;
    bool verbose = true;
    const char* tag = "asm";  // 日志前缀，用于标明建图器来源
};

struct AssembleStats {
    size_t models_in = 0;
    size_t rounds = 0;
    size_t merges = 0, merges_refused = 0;
    size_t bridges = 0;            // 其中仅依靠共同结构完成的合并数
    size_t joint_ba = 0;
    size_t grown_images = 0;       // 逐层增长新增的配准图像数
    size_t grown_rejected = 0;     // 其中又被位姿检查剔除的数量
    ManagerStats finish;           // ModelOps 各处理步骤的统计
    double t_merge = 0, t_ba = 0, t_grow = 0;
    // 各收尾步骤分别计时，以评估它们处理不同问题的成本。
    double t_audit = 0, t_split = 0, t_grow_tail = 0, t_reseed = 0, t_final_merge = 0, t_fold = 0;
    double finishSecs() const {
        return t_audit + t_split + t_grow_tail + t_reseed + t_final_merge + t_fold;
    }
};

namespace detail {

// 每层每模型最多吸收一个伙伴，优先最大重叠、较小模型对，使规模同步增长。
// 连续吸收会反复复制不断变大的锚点，产生二次开销并累积未优化接缝；分层可逐轮解决。carry 随模型重编号，stalled 标记未合并者。
inline size_t mergeLevel(std::vector<Reconstruction>& models, const MergeOptions& opt,
                         std::vector<std::vector<char>*> carry, std::vector<char>& stalled,
                         size_t& refused, std::map<std::string, size_t>* why = nullptr) {
    MergeSession s(std::move(models), opt);
    std::vector<MergeCandidate> cands = s.candidates();
    std::stable_sort(cands.begin(), cands.end(),
                     [&s](const MergeCandidate& a, const MergeCandidate& b) {
                         if (a.common_images != b.common_images)
                             return a.common_images > b.common_images;
                         const uint32_t sa = s.model(a.dst).numRegistered() +
                                             s.model(a.src).numRegistered();
                         const uint32_t sb = s.model(b.dst).numRegistered() +
                                             s.model(b.src).numRegistered();
                         return sa < sb;
                     });
    std::vector<char> busy(s.numModels(), 0);
    size_t merges = 0;
    for (const MergeCandidate& c : cands) {
        if (busy[c.dst] || busy[c.src]) continue;
        // 合并被拒绝后双方仍可尝试本层其他伙伴。
        const MergeAttempt a = s.tryMerge(c.dst, c.src);
        if (a.merged) {
            merges++;
            busy[c.dst] = busy[c.src] = 1;
        } else {
            refused++;
            // 去除原因中的具体计数，按前导文本汇总拒绝类别。
            if (why) {
                std::string k = a.reason.substr(0, a.reason.find_first_of("0123456789"));
                while (!k.empty() && (k.back() == ' ' || k.back() == '(')) k.pop_back();
                (*why)[k.empty() ? a.reason : k]++;
            }
        }
    }
    std::vector<std::vector<char>> was;
    for (const std::vector<char>* c : carry) was.push_back(*c);
    models.clear();
    stalled.clear();
    for (std::vector<char>* c : carry) c->clear();
    for (size_t i = 0; i < s.numModels(); i++) {
        if (!s.alive(i)) continue;
        stalled.push_back(busy[i] ? 0 : 1);  // 本层保留且未发生合并
        models.push_back(std::move(s.modelMut(i)));
        for (size_t k = 0; k < carry.size(); k++)
            carry[k]->push_back(i < was[k].size() ? was[k][i] : 0);
    }
    return merges;
}

// 优先扩展大模型，通过 PnP 增加共享图像，为下次 Sim(3) 对齐提供重叠与更强证据。
// 增长受重叠和规模预算限制，并逐图检查新位姿；未经审查的增长曾损失 21 分 AUC@5，还会让接缝检查错误支持坏合并。
inline size_t growModels(Mapper& mapper, std::vector<Reconstruction>& models,
                         std::vector<char>& dirty, const std::vector<char>& which,
                         double budget_frac, size_t budget_min, size_t& rejected) {
    std::vector<size_t> order(models.size());
    std::iota(order.begin(), order.end(), (size_t)0);
    std::stable_sort(order.begin(), order.end(), [&models](size_t a, size_t b) {
        return models[a].numRegistered() > models[b].numRegistered();
    });
    size_t registered = 0;
    for (size_t i : order) {
        if (i < which.size() && !which[i]) continue;
        const uint32_t have = models[i].numRegistered();
        if (have < 2) continue;
        std::vector<const Reconstruction*> others;
        for (size_t j = 0; j < models.size(); j++)
            if (j != i) others.push_back(&models[j]);
        // 增长只需建立合并桥梁，预算随模型大小变化，并在吞并整个未覆盖区域前停止。
        const uint32_t budget =
            (uint32_t)std::max((double)budget_min, budget_frac * (double)have);
        Mapper::GrowStats gs;
        uint32_t bad = 0;
        Reconstruction grown =
            mapper.growByPnP(models[i], &gs, others, have + budget, &bad);
        rejected += bad;
        if (!gs.registered) continue;
        registered += gs.registered;
        models[i] = std::move(grown);
        dirty[i] = 1;
    }
    return registered;
}

// 层内没有成功合并时，可对对应图连接最强但无共同图像的少量模型对做三维 RANSAC 桥接，次数受 max_bridges 限制。
// 求得变换后仍执行普通拼接、折叠和接缝检查（D70）。
inline size_t bridgeModels(Mapper& mapper, std::vector<Reconstruction>& models,
                           const MergeOptions& opt, size_t min_matches, int max_bridges,
                           std::vector<std::vector<char>*> carry, size_t& refused,
                           std::map<std::string, size_t>* why = nullptr) {
    if (max_bridges <= 0 || models.size() < 2) return 0;
    std::vector<Mapper::StructureLink> links = mapper.structureLinks(models, min_matches);
    if (links.empty()) return 0;
    if (opt.verbose)
        slog::diag(slog::Tag::Merge,
                   "[merge] %zu model pair(s) the correspondence graph joins, strongest "
                   "%zu matched features", links.size(), links.front().matches);

    MergeSession s(std::move(models), opt);
    std::vector<char> busy(s.numModels(), 0);
    size_t merges = 0;
    int tried = 0;
    for (const Mapper::StructureLink& l : links) {
        if (tried >= max_bridges) break;
        if (busy[l.a] || busy[l.b]) continue;
        // 较大模型保留其坐标规范与内参。
        const bool a_first = s.model(l.a).numRegistered() >= s.model(l.b).numRegistered();
        const size_t dst = a_first ? l.a : l.b, src = a_first ? l.b : l.a;
        tried++;
        AlignmentResult al = mapper.alignByStructure(s.model(dst), s.model(src), opt);
        // 桥接尝试很少，完整打印原因，便于解释具体两模型仍未合并的原因。
        if (opt.verbose)
            slog::diag(slog::Tag::Merge,
                       "[merge] structure link %zu <- %zu (%zu matched features): %s", dst,
                       src, l.matches, al.success ? "aligned" : al.reason.c_str());
        if (!al.success) {
            refused++;
            if (why) (*why)["shared structure: " + al.reason.substr(0, al.reason.find_first_of(
                                                       "0123456789"))]++;
            continue;
        }
        const MergeAttempt a = s.tryMerge(dst, src, al);
        if (a.merged) {
            merges++;
            busy[dst] = busy[src] = 1;
        } else {
            refused++;
            if (why) {
                std::string k = a.reason.substr(0, a.reason.find_first_of("0123456789"));
                while (!k.empty() && (k.back() == ' ' || k.back() == '(')) k.pop_back();
                (*why)["shared structure: " + (k.empty() ? a.reason : k)]++;
            }
        }
    }
    std::vector<std::vector<char>> was;
    for (const std::vector<char>* c : carry) was.push_back(*c);
    models.clear();
    for (std::vector<char>* c : carry) c->clear();
    for (size_t i = 0; i < s.numModels(); i++) {
        if (!s.alive(i)) continue;
        models.push_back(std::move(s.modelMut(i)));
        for (size_t k = 0; k < carry.size(); k++)
            carry[k]->push_back(i < was[k].size() ? (busy[i] ? 1 : was[k][i]) : 0);
    }
    return merges;
}

}  // 命名空间 detail

// 逐层合并、增长、优化；dirty 标记变化，seamed 标记新接缝，均随模型重编号，供收尾跳过重复工作。
inline void mergeUpwards(Mapper& mapper, std::vector<Reconstruction>& models,
                         const MergeOptions& merge_opt, const ManagerOptions& mopt,
                         const AssembleOptions& opt, AssembleStats& st,
                         std::vector<char>& dirty, std::vector<char>& seamed) {
    auto clk = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

    dirty.resize(models.size(), 0);
    seamed.resize(models.size(), 0);
    std::vector<char> stalled(models.size(), 1);
    int grow_passes = 0;
    bool solve_pending = false;  // 已有合并但联合求解被延后
    if (!mopt.do_merge && !mopt.do_grow) return;
    for (int round = 0; round < std::max(1, opt.max_rounds) && models.size() > 1; round++) {
        st.rounds = (size_t)round + 1;
        size_t refused = 0;
        std::map<std::string, size_t> why;
        auto t0 = clk();
        const size_t merges =
            mopt.do_merge
                ? detail::mergeLevel(models, merge_opt, {&dirty, &seamed}, stalled, refused, &why)
                : 0;
        for (size_t i = 0; i < models.size(); i++)
            if (!stalled[i]) dirty[i] = seamed[i] = 1;
        st.t_merge += secs(t0, clk());
        st.merges += merges;
        st.merges_refused += refused;
        if (opt.verbose) {
            slog::diag(slog::Tag::Merge,
                       "[%s] level %zu: %zu merge(s), %zu refused, %zu model(s) left",
                       opt.tag, st.rounds, merges, refused, models.size());
            // 按类别汇总拒绝原因，使全部被拒绝与没有候选两种状态可区分。
            for (const auto& kv : why)
                slog::diag(slog::Tag::Merge, "[%s]   %4zu x %s", opt.tag, kv.second,
                           kv.first.c_str());
        }
        if (models.size() <= 1) break;

        // 联合优化前仅增长未合并模型；刚合并者先让新接缝稳定，下一层再处理。
        size_t reg = 0;
        if (mopt.do_grow && opt.grow_every > 0 && grow_passes < opt.max_grow_passes &&
            (round % opt.grow_every) == 0) {
            size_t want = 0;
            for (char c : stalled) want += c ? 1 : 0;
            if (want) {
                t0 = clk();
                size_t rejected = 0;
                reg = detail::growModels(mapper, models, dirty, stalled, opt.grow_budget_frac,
                                         opt.grow_budget_min, rejected);
                st.grown_rejected += rejected;
                st.t_grow += secs(t0, clk());
                st.grown_images += reg;
                if (reg) grow_passes++;
                if (opt.verbose)
                    slog::diag(slog::Tag::Merge,
                               "[%s]   growth: %zu image(s) into %zu of %zu model(s) that "
                               "did not merge (%zu rejected by the pose check)", opt.tag, reg, want,
                               models.size(), st.grown_rejected);
            }
        }
        // 本层无合并表示共同图像证据已耗尽，可选再尝试无共同帧但有结构对应的桥接（D70）。
        size_t bridged = 0;
        if (merges == 0 && mopt.do_merge && opt.max_bridges > 0) {
            t0 = clk();
            std::map<std::string, size_t> bwhy;
            bridged = detail::bridgeModels(mapper, models, merge_opt, opt.bridge_min_matches,
                                           opt.max_bridges, {&dirty, &seamed, &stalled}, refused,
                                           &bwhy);
            st.t_merge += secs(t0, clk());
            st.merges += bridged;
            st.bridges += bridged;
            if (opt.verbose && (bridged || !bwhy.empty())) {
                slog::diag(slog::Tag::Merge, "[%s]   %zu merge(s) on shared structure alone",
                           opt.tag,
                           bridged);
                for (const auto& kv : bwhy)
                    slog::diag(slog::Tag::Merge, "[%s]   %4zu x %s", opt.tag, kv.second,
                               kv.first.c_str());
            }
        }

        // 没有合并且增长不足以提供新重叠时停止，避免继续整数据集联合求解；7620 图测量的后续层仅增长 13/5/7/0 图却各付出一次求解成本。
        if (merges == 0 && bridged == 0 && reg < opt.grow_budget_min) break;

        // 增长可能使小模型变成大模型副本，立即丢弃以避免参与下一层候选与联合 BA。
        models = dropRedundantModels(std::move(models), mopt, st.finish,
                                     {&dirty, &seamed, &stalled});

        // 按相机组共享内参的一次联合 BA，实测比逐个精化变更模型更快，设备提交延迟是主要因素。
        // 允许按 joint_every 跳层，但仅剩两个模型或即将退出时必须补做求解。
        const bool solve_now =
            opt.joint_every <= 1 || models.size() <= 2 || (round % opt.joint_every) == 0;
        if (!solve_now) {
            solve_pending = true;
            continue;
        }
        solve_pending = false;
        t0 = clk();
        if (opt.joint_intrinsics && models.size() > 1) {
            mapper.jointRefine(models, opt.coarse_joint_ba);
            st.joint_ba++;
        } else {
            for (size_t i = 0; i < models.size(); i++)
                if (dirty[i] && models[i].numRegistered() >= 2)
                    models[i] = mapper.refine(models[i]);
        }
        st.t_ba += secs(t0, clk());
    }
    if (solve_pending && models.size() > 1) {
        auto t0 = clk();
        mapper.jointRefine(models, opt.coarse_joint_ba);
        st.joint_ba++;
        st.t_ba += secs(t0, clk());
    }

    if (opt.verbose) {
        const ManagerStats& f = st.finish;
        slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_assembled,
                 {spirula::i18n::format_duration(st.t_merge + st.t_grow + st.t_ba),
                  (long long)st.models_in, (long long)models.size(),
                  (long long)st.rounds, (long long)st.merges,
                  (long long)st.merges_refused, (long long)st.grown_images,
                  (long long)f.covered_before, (long long)coveredImages(models).size()});
        // 接缝诊断保持英文，分别报告少量证据和大量但不完全一致证据；前者需要更多重叠，后者可能由精化修复。
        if (f.seam_refused)
            slog::diag(slog::Tag::Merge,
                       "[%s]   a refused merge explained a median %.0f%% of its cross-seam "
                       "matches over %zu pair(s), an accepted one %.0f%%, against a %.0f%% bar "
                       "(the same pairs inside the model: %.0f%%); the rescue's refinement "
                       "moved a refusal by %+.0f points", opt.tag,
                       100.0 * f.seam_refused_median / (double)f.seam_refused,
                       f.seam_refused_pairs / f.seam_refused,
                       f.seam_passed ? 100.0 * f.seam_passed_median / (double)f.seam_passed : 0.0,
                       f.seam_checked ? 100.0 * f.seam_bar_sum / (double)f.seam_checked : 0.0,
                       f.seam_checked ? 100.0 * f.seam_reference_sum / (double)f.seam_checked : 0.0,
                       f.seam_rescue_failed
                       ? 100.0 * f.seam_rescue_gain / (double)f.seam_rescue_failed : 0.0);
    }
}

// 分层无法解决的故障按固定顺序收尾一次，复用 ModelOps 和相同阈值，不循环执行。
inline std::vector<Reconstruction> finishModels(Mapper& mapper,
                                                std::vector<Reconstruction> models,
                                                const MergeOptions& merge_opt,
                                                const ManagerOptions& mopt,
                                                const AssembleOptions& opt, AssembleStats& st,
                                                ModelMemo& memo, const std::vector<char>& dirty,
                                                const std::vector<char>& seamed) {
    auto clk = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    ManagerOptions fopt = mopt;
    fopt.audit_growth_frac = opt.audit_growth_frac;

    // 联合 BA 不执行重三角化、补轨迹或撤销配准；有新接缝的模型须按对应图审查后精化。
    // 仅增长者已逐图检查，未变化者已有精化结果，因此避免重复完整审查。
    for (size_t i = 0; i < models.size(); i++) {
        if (i < seamed.size() && seamed[i]) continue;
        if (i < dirty.size() && dirty[i] && models[i].numRegistered() >= 2)
            models[i] = mapper.refine(models[i]);
        memo.audited.insert(ModelMemo::of(models[i]));
    }
    auto t0 = clk();
    models = auditModels(mapper, std::move(models), fopt, memo, st.finish);
    st.t_audit = secs(t0, clk());

    // 模型自身也可能因重复结构链式配准而与验证匹配矛盾，不能只在合并时检查。
    size_t changed = 0;
    if (mopt.do_split) {
        t0 = clk();
        const size_t before = models.size();
        models = splitInconsistentModels(mapper, std::move(models), fopt, memo, st.finish);
        changed += models.size() - std::min(models.size(), before);
        st.t_split = secs(t0, clk());
    }

    // 最后向可接纳模型补配未覆盖图像，使用建图器完整增长与优化节奏，而非只做 PnP。
    // 预算仅覆盖缺失图像；550 图数据少配八图即可损失 2.6 分 AUC@10，即使已配准位姿误差几乎不变。
    if (mopt.do_grow) {
        t0 = clk();
        mapper.claimAll(models);
        const size_t missing = mapper.unclaimed();
        if (missing) {
            size_t reg = 0;
            for (size_t i = 0; i < models.size(); i++) {
                if (models[i].numRegistered() < 2) continue;
                std::vector<const Reconstruction*> others;
                for (size_t j = 0; j < models.size(); j++)
                    if (j != i) others.push_back(&models[j]);
                Mapper::GrowStats gs;
                Reconstruction grown = mapper.continueFrom(
                    models[i], &gs, others, (uint32_t)(models[i].numRegistered() + missing));
                if (!gs.registered) continue;
                reg += gs.registered;
                models[i] = std::move(grown);
                mapper.claimAll(models);
            }
            st.grown_images += reg;
            // 配准可创建原本不存在的重叠，必须触发新合并；5356 图数据的两个模型直到此步才共享 1291 张图。
            if (reg) changed++;
            if (opt.verbose)
                slog::diag(slog::Tag::Merge,
                           "[%s] tail growth: %zu of %zu uncovered image(s) registered",
                           opt.tag, reg, missing);
        }
        st.t_grow_tail = secs(t0, clk());
    }

    // 对完全未覆盖、被切得过小或弱连接区域重新播种，同时限制尝试与规模；1146 图数据无界播种曾耗时 21.6 s 却无结果。
    if (mopt.do_reseed) {
        t0 = clk();
        mapper.claimAll(models);
        if (mapper.unclaimed() >= (size_t)std::max(2, mapper.options().min_model_size)) {
            const size_t before = models.size();
            const int saved_trials = mapper.options().max_model_trials;
            mapper.options().max_model_trials = opt.reseed_trials;
            mapper.seedFurtherModels(models, true);
            mapper.options().max_model_trials = saved_trials;
            st.finish.reseeded_models += models.size() - before;
            changed += models.size() - before;
            if (opt.verbose && models.size() != before)
                slog::diag(slog::Tag::Merge,
                           "[%s] reseeded %zu model(s) among the images nothing reached",
                           opt.tag, models.size() - before);
        }
        st.t_reseed = secs(t0, clk());
    }

    // 将最后两步新产生的模型再尝试合并。
    if (changed && models.size() > 1 && mopt.do_merge) {
        t0 = clk();
        std::vector<char> s2(models.size(), 1);
        size_t refused = 0;
        const size_t merges = detail::mergeLevel(models, merge_opt, {}, s2, refused);
        st.merges += merges;
        st.merges_refused += refused;
        if (opt.verbose)
            slog::diag(slog::Tag::Merge,
                       "[%s] final level: %zu merge(s), %zu refused, %zu model(s) left",
                       opt.tag, merges, refused, models.size());
        if (merges)
            for (size_t i = 0; i < models.size(); i++)
                if (!s2[i] && models[i].numRegistered() >= 2)
                    models[i] = mapper.refine(models[i]);
        st.t_final_merge = secs(t0, clk());
    }

    models = dropRedundantModels(std::move(models), mopt, st.finish);

    // 折叠检查最后仅执行一次；重复场景重叠到一起可能内部自洽，须根据应有却缺失的共同结构判断（D46）。
    if (mopt.do_duplicate_split) {
        t0 = clk();
        const size_t before = models.size();
        models = splitFoldedModels(mapper, std::move(models), fopt, memo, st.finish);
        if (models.size() != before)
            for (Reconstruction& m : models)
                if (m.numRegistered() >= 2 && memo.audited.insert(ModelMemo::of(m)).second)
                    m = mapper.refine(m);
        st.t_fold = secs(t0, clk());
    }

    sortModels(models);
    st.finish.models_after = models.size();
    st.finish.covered_after = coveredImages(models).size();
    if (opt.verbose)
        slog::out(slog::Tag::Map, spirula::i18n::msg::sfm::map_finishing,
                 {spirula::i18n::format_duration(st.finishSecs()),
                  (long long)st.finish.splits,
                  (long long)st.finish.duplicate_splits,
                  (long long)st.finish.reseeded_models,
                  (long long)st.finish.dropped_redundant,
                  (long long)st.finish.audited_repaired,
                  (long long)st.finish.audited_out});
    mapper.claimAll(models);
    return models;
}

// 两种建图器得到模型后共用分层与收尾流程，mopt 统一提供合并和清理阈值。
inline std::vector<Reconstruction> assembleModels(Mapper& mapper,
                                                  std::vector<Reconstruction> models,
                                                  const ManagerOptions& mopt,
                                                  const AssembleOptions& opt, AssembleStats& st) {
    st.models_in = models.size();
    st.finish.models_before = models.size();
    st.finish.covered_before = coveredImages(models).size();

    // 昂贵处理前先按图像集合删除被大模型覆盖的重复种子结果，避免对每个副本重复优化、审查与增长。
    models = dropRedundantModels(std::move(models), mopt, st.finish);

    // 重复结构可能使模型内部全部检查自洽，必须借助合并器自身没有的对应图接缝验证（D45）。
    MergeOptions merge_opt = mopt.merge;
    merge_opt.duplicate = mopt.duplicate;
    merge_opt.validate = seamValidator(mapper, mopt, &st.finish);
    merge_opt.rigs = mapper.rigs();

    // 首轮合并前先替换明显偏离的组焦距，再联合优化共享内参；直接平均错误焦距会使像素级合并检查失去意义。
    if (mopt.focal_consensus_tol > 0) refitOutlierCameras(mapper, models, mopt, st.finish);
    if (opt.joint_intrinsics && models.size() > 1 && focalSpreadPx(models) > kJointBaMovedPx) {
        const auto t0 = std::chrono::steady_clock::now();
        mapper.jointRefine(models, opt.coarse_joint_ba);
        st.joint_ba++;
        st.t_ba += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (opt.verbose)
            slog::diag(slog::Tag::Merge,
                       "[%s] the components disagreed about the intrinsics: one joint "
                       "refinement before the first level", opt.tag);
    }

    std::vector<char> dirty(models.size(), 0), seamed(models.size(), 0);
    mergeUpwards(mapper, models, merge_opt, mopt, opt, st, dirty, seamed);
    ModelMemo memo;
    return finishModels(mapper, std::move(models), merge_opt, mopt, opt, st, memo, dirty, seamed);
}

}  // 命名空间 sfm
