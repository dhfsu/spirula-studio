// LM 主机驱动，管理 GPU 缓冲、装配、稠密 Cholesky 或隐式 Schur PCG，以及步长接受/拒绝。
// 按 SubmitBudget 拆分提交，避免慢 GPU 上的大问题触发看门狗；路径与内存预算见 sfm/ba/README.md。
// 设备不支持任何算术配置时转到主机双精度 bacpu::Solver，沿用相同 LM 与线性求解流程。
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include "core/SubmitBudget.h"
#include "sfm/ba/Options.h"
#include "sfm/ba/Priors.h"
#include "sfm/ba/Problem.h"
#include "sfm/ba/SolverCpu.h"
#include "sfm/vk/EmbeddedSpirv.h"
#include "sfm/vk/VkContext.h"
#include "core/Env.h"
#include "sfm/core/Cancel.h"
#include "sfm/core/Log.h"

// 设备算术能力：double 需要 fp64 运算与原子加，float 需要 fp32 原子加，df 使用双浮点模拟与 int64 原子操作，CPU 不需要设备能力。
inline bool realSupportedByDevice(RealCfg c, const VkDeviceCaps& caps) {
    switch (c) {
        case RealCfg::F64:
            return caps.float64 && caps.float32AtomicAdd && caps.float64AtomicAdd;
        case RealCfg::F32:
            return caps.float32AtomicAdd;
        case RealCfg::DF64:
            return caps.int64Atomics;
        case RealCfg::CPU:
            return true;
    }
    return false;
}

// 默认从 fp64 回退到主机双精度；float 正规方程精度约停在 1e-7，df 用 CAS 原子循环换取约 48 位精度，两者仅显式选择。
inline RealCfg pickRealForDevice(RealCfg want, const VkDeviceCaps& caps) {
    if (realSupportedByDevice(want, caps)) return want;
    if (realSupportedByDevice(RealCfg::F64, caps)) return RealCfg::F64;
    return RealCfg::CPU;
}

// 按规范 UUID 缓存设备能力，枚举序号可能变化。
inline const VkDeviceCaps& cachedDeviceCaps(const std::string& selector) {
    static std::mutex m;
    static std::map<std::string, VkDeviceCaps> cache;  // 节点式容器保证引用保持有效
    std::lock_guard<std::mutex> g(m);
    auto it = cache.find(selector);
    if (it == cache.end())
        it = cache.emplace(selector, VkContext::probeCaps(deviceOnlyOpt(-1, selector))).first;
    return it->second;
}

class BundleSolver {
    enum class LinSolve { DenseObs, DensePair, CG };
    // 预条件块步长必须与 cg.slang 的 kCamBlk 一致。
    static constexpr uint32_t kCamBlk = kMaxPlainDof * (kMaxPlainDof + 1) / 2;

public:
    // shared 指定调用方持有的持久上下文，跨 BA 复用设备、流水线和描述符设施；仅问题大小的缓冲由求解器分配与释放。
    // 未指定时求解器独占有作用域的上下文。
    BundleSolver(BAProblem& P, const SolverOptions& opt, VkContext* shared = nullptr)
        : P_(P), opt_(opt), owned_(shared ? nullptr : new VkContext),
          ctx_(shared ? *shared : *owned_) {}

    ~BundleSolver() {
        // 共享上下文下须立即归还本问题显存，不能等上下文销毁；独占上下文由析构统一释放。
        if (!owned_)
            for (GpuBuffer* b : ownBufs_) ctx_.destroyBuffer(*b);
    }

    void init() {
        auto prof_t0 = std::chrono::steady_clock::now();
        auto prof_lap = [&prof_t0] {
            auto t1 = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t1 - prof_t0).count();
            prof_t0 = t1;
            return dt;
        };
        // 显式设备请求即使已有上下文仍需解析，不能静默回退 CPU 或按能力改选设备。
        const bool explicit_req =
            !opt_.device_selector.empty() || opt_.device >= 0;
        const bool validate_request =
            explicit_req || (opt_.real != RealCfg::CPU && !ctx_.initialized());
        if (validate_request) {
            const spirula::vkselect::Resolution& res =
                VkContext::cachedResolution(
                    deviceOnlyOpt(opt_.device, opt_.device_selector));
            if (res.ok())
                selector_ = res.selector;
            else if (explicit_req ||
                     res.status != spirula::vkselect::ResolveStatus::NoDevice)
                throw std::runtime_error(res.error);
        }
        // 共享上下文必须比较稳定标识，不能跨实例比较序号；能力选择与分配前拒绝不匹配。
        if (ctx_.initialized() && !selector_.empty() &&
            selector_ != ctx_.selector())
            throw std::runtime_error("requested device " + selector_ +
                                     " is not the live device " + ctx_.selector());
        if (opt_.real != RealCfg::CPU) {
            static const VkDeviceCaps kNoDevice{};
            const VkDeviceCaps& caps =
                ctx_.initialized() ? ctx_.caps()
                : selector_.empty() ? kNoDevice
                                    : cachedDeviceCaps(selector_);
            RealCfg real = pickRealForDevice(opt_.real, caps);
            if (real != opt_.real) {
                // 每种请求、实际配置组合仅提示一次，避免周期 BA 重复输出相同消息。
                static std::mutex said_mu;
                static std::set<std::pair<int, int>> said;
                bool first;
                {
                    std::lock_guard<std::mutex> g(said_mu);
                    first = said.emplace((int)opt_.real, (int)real).second;
                }
                if (first)
                    sfm::slog::diag(sfm::slog::Tag::Map,
                                    "[ba] device does not support '%s' arithmetic; "
                               "falling back to '%s'",
                               realCfgName(opt_.real), realCfgName(real));
                opt_.real = real;
            }
        }
        if (opt_.real == RealCfg::CPU) {
            cpu_.reset(new bacpu::Solver(P_, opt_));
            cpu_->init();
            return;
        }
        VkContextOptions vopt;
        vopt.needFloat64 = opt_.real == RealCfg::F64;
        vopt.needFloatAtomics = opt_.real != RealCfg::DF64;
        vopt.needInt64Atomics = opt_.real == RealCfg::DF64;
        vopt.selector = selector_;      // 规范标识；空值沿用共享选择优先级
        vopt.deviceIndex = opt_.device; // 兼容序号，仅在未解析 UUID 时使用
        vopt.validate = opt_.validate;
        vopt.profile = opt_.profile;
        if (!ctx_.initialized()) ctx_.init(vopt);
        double t_ctx = prof_lap();

        decidePaths();
        hasPriors_ = P_.priors && !P_.priors->empty();
        if (hasPriors_) {
            prior_.init(P_);
            hasPriors_ = !prior_.empty();
        }
        hostPoses_ = P_.poses;
        hostExts_ = P_.exts;

        const size_t rs = realSize(opt_.real);
        const uint64_t packed = (uint64_t)P_.n_dim * (P_.n_dim + 1) / 2;
        const bool needS = !useCG_ || haveFallback_;
        const uint32_t npart = (P_.n_dim + 255) / 256;

        auto mkReal = [&](uint64_t n) { return ctx_.createBuffer(std::max<uint64_t>(n, 1) * rs); };
        auto mkUint = [&](uint64_t n) { return ctx_.createBuffer(std::max<uint64_t>(n, 1) * 4); };

        bObs_ = mkReal(2 * (uint64_t)P_.num_obs);
        bObsImage_ = mkUint(P_.num_obs);
        bObsPoint_ = mkUint(P_.num_obs);
        bImageInfo_ = ctx_.createBuffer(std::max<size_t>(P_.num_images, 1) * 16);
        bGroupInfo_ = ctx_.createBuffer(std::max<size_t>(P_.groups.size(), 1) * 16);
        bMemberInfo_ = ctx_.createBuffer(std::max<size_t>(P_.members.size(), 1) * 16);
        bPoses_ = mkReal(P_.pose_dim);
        bExts_ = mkReal(P_.exts.size());
        bIntr_ = mkReal(P_.total_intr);
        bPoints_ = mkReal(3 * (uint64_t)P_.num_points);
        bObsRanges_ = mkUint(P_.num_points + 1);
        bModelObs_ = mkUint(P_.model_obs.size());
        bJcOff_ = mkUint(P_.num_obs);
        bJp_ = mkReal(6 * (uint64_t)P_.num_obs);
        const uint64_t packedTc = (uint64_t)tcN_ * (tcN_ + 1) / 2;
        bS_ = mkReal(std::max<uint64_t>(needS ? packed : 1, packedTc + 42 * (uint64_t)P_.num_frames));
        bG_ = mkReal(P_.n_dim);
        bApp_ = mkReal(9 * (uint64_t)P_.num_points);
        bBp_ = mkReal(3 * (uint64_t)P_.num_points);
        bCost_ = mkReal(4);
        bPosesBak_ = mkReal(P_.pose_dim);
        bExtsBak_ = mkReal(P_.exts.size());
        bIntrBak_ = mkReal(P_.total_intr);
        bPointsBak_ = mkReal(3 * (uint64_t)P_.num_points);
        bPairEntries_ = mkUint(std::max<size_t>(P_.pair_entries.size() + tcEnt_.size(), 2));
        bPairChunks_ = mkUint(std::max<size_t>(P_.pair_chunks.size() + tcChk_.size(), 2));
        // 所有路径都从逐观测雅可比重建 S/g，拒绝步骤无需备份；仅 Bp 会被点回代原地覆盖，必须保存。
        bBp0_ = mkReal(3 * (uint64_t)P_.num_points);
        bJc_ = mkReal(P_.jc_total);
        bRes_ = mkReal(2 * (uint64_t)P_.num_obs);
        bW_ = mkReal(9 * (uint64_t)P_.num_points);
        bYp_ = mkReal(needS && P_.use_pair_schur ? 6 * (uint64_t)P_.num_obs : 1);
        bY_ = mkReal(std::max<uint64_t>(P_.n_dim, 34 * (uint64_t)tcN_));
        // CG 缓冲，未使用时分配单元素占位
        bCamRanges_ = mkUint(cgAllocated_ ? P_.num_images + 1 : 1);
        bCamObs_ = mkUint(cgAllocated_ ? P_.num_obs : 1);
        bCamChunks_ = mkUint(cgAllocated_ ? P_.cam_chunks.size() : 3);
        bCgR_ = mkReal(cgAllocated_ ? P_.n_dim : 1);
        bCgZ_ = mkReal(cgAllocated_ ? P_.n_dim : 1);
        bCgP_ = mkReal(cgAllocated_ ? P_.n_dim : 1);
        bCgSp_ = mkReal(cgAllocated_ ? P_.n_dim : 1);
        bCgV_ = mkReal(cgAllocated_ ? 3 * (uint64_t)P_.num_points : 1);
        bCgB_ = mkReal(cgAllocated_ ? (uint64_t)bBlk_ * P_.num_images : 1);
        bCgM_ = mkReal(cgAllocated_ ? (uint64_t)kCamBlk * P_.num_prec_blocks : 1);
        bCgScal_ = mkReal(16);
        bCgPart_ = mkReal(cgAllocated_ ? 2 * (uint64_t)npart : 1);
        bPrecBlocks_ = mkUint(cgAllocated_ ? P_.prec_blocks.size() : 4);
        // 无先验时各表保留一个元素，使共享上下文描述符绑定数量稳定。
        const uint32_t npe = hasPriors_ ? prior_.numEntries() : 0;
        bPriorRows_ = mkUint(hasPriors_ ? P_.num_frames + 1 : 1);
        bPriorCols_ = mkUint(npe);
        bPriorErow_ = mkUint(npe);
        bPriorBlk_ = mkReal(36 * (uint64_t)npe);
        bPriorG_ = mkReal(hasPriors_ ? P_.pose_dim : 1);
        if (hasPriors_ &&
            kDlPoses + (P_.pose_dim + P_.exts.size()) * rs > VkContext::stagingCapacity())
            throw std::runtime_error("pose priors: the parameter readback exceeds the staging buffer");

        // 绑定顺序须与 ba.slang 和 cg.slang 一致；奇数绑定的原子视图别名到同一个 VkBuffer。
        std::vector<VkBuffer> binds = {
            bObs_.buf, bObsImage_.buf, bObsPoint_.buf, bImageInfo_.buf, bGroupInfo_.buf,
            bPoses_.buf, bIntr_.buf, bPoints_.buf, bObsRanges_.buf, bModelObs_.buf, bJcOff_.buf,
            bJp_.buf,
            bS_.buf, bS_.buf, bG_.buf, bG_.buf, bApp_.buf, bApp_.buf, bBp_.buf, bBp_.buf,
            bCost_.buf, bCost_.buf,
            bPairEntries_.buf, bPairChunks_.buf, bW_.buf, bYp_.buf, bY_.buf,
            bJc_.buf, bRes_.buf,
            bCamRanges_.buf, bCamObs_.buf, bCgR_.buf, bCgZ_.buf, bCgP_.buf, bCgSp_.buf,
            bCgV_.buf, bCgB_.buf, bCgM_.buf, bCgScal_.buf, bCgPart_.buf,
            bCgSp_.buf, bCgB_.buf, bCgM_.buf, bCamChunks_.buf, bPrecBlocks_.buf,
            bMemberInfo_.buf, bExts_.buf,
            bPriorRows_.buf, bPriorCols_.buf, bPriorErow_.buf, bPriorBlk_.buf, bPriorG_.buf,
        };
        ownBufs_ = {&bObs_, &bObsImage_, &bObsPoint_, &bImageInfo_, &bGroupInfo_,
                    &bMemberInfo_, &bExts_, &bExtsBak_,
                    &bPoses_, &bIntr_, &bPoints_, &bObsRanges_, &bModelObs_, &bJcOff_,
                    &bJp_, &bS_, &bG_, &bApp_, &bBp_, &bCost_,
                    &bPosesBak_, &bIntrBak_, &bPointsBak_,
                    &bBp0_, &bJc_, &bRes_,
                    &bPairEntries_, &bPairChunks_, &bW_, &bYp_, &bY_,
                    &bCamRanges_, &bCamObs_, &bCamChunks_, &bCgR_, &bCgZ_, &bCgP_, &bCgSp_,
                    &bCgV_, &bCgB_, &bCgM_, &bCgScal_, &bCgPart_, &bPrecBlocks_,
                    &bPriorRows_, &bPriorCols_, &bPriorErow_, &bPriorBlk_, &bPriorG_};
        double t_buf = prof_lap();
        ctx_.createDescriptors(binds);
        // 持久上下文复用内存可能残留旧数据，所有累加缓冲必须显式清零；先更新描述符，再 begin，避免仍引用已释放缓冲。
        {
            VkCommandBuffer cb = ctx_.begin();
            for (GpuBuffer* b : ownBufs_) ctx_.fillZero(cb, *b);
            ctx_.submit(cb);
        }

        // 仅加载本问题实际使用的入口，冷驱动缓存每入口编译约 90 ms；共享上下文已创建的流水线可复用。
        std::vector<std::string> entries = {
            "point_prep", "point_update", "cam_update", "intr_update",
            std::string("dp_accum") + schurSuffix_,
        };
        if (!P_.members.empty()) entries.push_back("ext_update");
        if (hasPriors_) {
            entries.push_back("prior_add_g");
            if (!useCG_ || haveFallback_) entries.push_back("prior_add_s");
            if (useCG_) {
                entries.push_back("prior_add_m");
                entries.push_back("prior_matvec");
            }
        }
        if (useCG_) {
            const char* cg[] = {"cg_prec_fact", "cg_prec_apply", "cg_init", "cg_copy",
                                "cg_red2", "cg_fin", "cg_axpy", "cg_updp"};
            entries.insert(entries.end(), std::begin(cg), std::end(cg));
            for (const char* k : {"cg_cam_diag", "cg_gather", "cg_bmul", "cg_scatter"})
                entries.push_back(std::string(k) + cgSuffix_);
        }
        if (tcN_) {
            const char* tc[] = {"tc_basis", "tc_schur", "tc_reg", "tc_dinv", "tc_inv_row",
                                "tc_inv_copy", "tc_restrict", "tc_lmul", "tc_ltmul",
                                "tc_prolong", "chol_diag", "chol_panel", "chol_update"};
            entries.insert(entries.end(), std::begin(tc), std::end(tc));
            entries.push_back(std::string("tc_bpart") + cgSuffix_);
        }
        if (!useCG_ || haveFallback_) {
            const char* ch[] = {"chol_diag", "chol_panel", "chol_update", "tri_fwd", "tri_bwd"};
            entries.insert(entries.end(), std::begin(ch), std::end(ch));
            if (P_.use_pair_schur) {
                entries.push_back("y_prep");
                entries.push_back(std::string("schur_pair") + schurSuffix_);
            } else {
                entries.push_back(std::string("schur_obs") + schurSuffix_);
            }
        }
        for (const BAProblem::ModelRange& mr : P_.model_ranges) {
            entries.push_back(costEntry(mr));
            entries.push_back(jacEntry(mr));
        }
        // 共享上下文的 real/loss 必须保持一致，模块按该组合编译；runGlobalBA 的缓存确保此约束。
        if (!opt_.spv_path.empty()) {
            ctx_.loadPipelines(opt_.spv_path, entries);
        } else {
            std::string blob =
                std::string("ba_") + realCfgName(opt_.real) + "_" + opt_.loss;
            size_t words = 0;
            const uint32_t* code = sfm::findSpirv(blob.c_str(), &words);
            if (!code) {
                std::string have;
                for (size_t i = 0; const char* n = sfm::spirvBlobName(i); i++)
                    have += (i ? ", " : "") + std::string(n);
                throw std::runtime_error("shader variant '" + blob +
                                         "' is not built into this binary "
                                         "(configured with SS_SFM_REALS / "
                                         "SS_SFM_LOSSES); available: " + have);
            }
            ctx_.loadPipelines(code, words * 4, entries);
        }
        double t_pipe = prof_lap();

        // 将静态数据与初始参数合并到一次提交；十七次带栅栏复制曾占共享设备小问题的大部分开销。
        std::vector<uint8_t> obs, poses, exts, intr, points;
        packReals(obs, P_.obs_xy.data(), P_.obs_xy.size(), opt_.real);
        packReals(poses, P_.poses.data(), P_.poses.size(), opt_.real);
        packReals(exts, P_.exts.data(), P_.exts.size(), opt_.real);
        packReals(intr, P_.intr.data(), P_.intr.size(), opt_.real);
        packReals(points, P_.points.data(), P_.points.size(), opt_.real);
        std::vector<uint32_t> gi;
        for (auto& g : P_.groups) {
            gi.push_back(g.intr_offset);
            gi.push_back(g.intr_col);
            gi.push_back(g.n_intr);
            gi.push_back(g.model);
        }
        // 逐图像保存帧、成员、分组及帧是否共享，决定 cg_bmul 覆盖写入还是累加。
        std::vector<uint32_t> frame_images(P_.num_frames, 0);
        for (uint32_t i = 0; i < P_.num_images; i++) frame_images[P_.image_frame[i]]++;
        std::vector<uint32_t> ii;
        ii.reserve(4 * P_.num_images);
        for (uint32_t i = 0; i < P_.num_images; i++) {
            ii.push_back(P_.image_frame[i]);
            ii.push_back(P_.image_member[i]);
            ii.push_back(P_.image_group[i]);
            ii.push_back(frame_images[P_.image_frame[i]] > 1 ? 1u : 0u);
        }
        std::vector<uint32_t> mi;
        for (auto& m : P_.members) {
            mi.push_back(m.ext_offset);
            mi.push_back(m.ext_col);
            mi.push_back(m.n_free);
            mi.push_back(m.mask);
        }
        std::vector<VkContext::UploadItem> up = {
            {&bObs_, obs.data(), obs.size()},
            {&bObsImage_, P_.obs_image.data(), P_.obs_image.size() * 4},
            {&bObsPoint_, P_.obs_point.data(), P_.obs_point.size() * 4},
            {&bImageInfo_, ii.data(), ii.size() * 4},
            {&bGroupInfo_, gi.data(), gi.size() * 4},
            {&bObsRanges_, P_.obs_ranges.data(), P_.obs_ranges.size() * 4},
            {&bModelObs_, P_.model_obs.data(), P_.model_obs.size() * 4},
            {&bJcOff_, P_.jc_off.data(), P_.jc_off.size() * 4},
            {&bPoses_, poses.data(), poses.size()},
            {&bIntr_, intr.data(), intr.size()},
            {&bPoints_, points.data(), points.size()},
        };
        if (!P_.members.empty()) {
            up.push_back({&bMemberInfo_, mi.data(), mi.size() * 4});
            up.push_back({&bExts_, exts.data(), exts.size()});
        }
        if (P_.use_pair_schur || tcN_) {
            tcEnt_.insert(tcEnt_.begin(), P_.pair_entries.begin(), P_.pair_entries.end());
            tcChk_.insert(tcChk_.begin(), P_.pair_chunks.begin(), P_.pair_chunks.end());
            up.push_back({&bPairEntries_, tcEnt_.data(), tcEnt_.size() * 4});
            up.push_back({&bPairChunks_, tcChk_.data(), tcChk_.size() * 4});
        }
        struct Release {
            std::vector<uint32_t>& a;
            std::vector<uint32_t>& b;
            ~Release() { a = {}; b = {}; }
        } release{tcEnt_, tcChk_};
        if (hasPriors_) {
            up.push_back({&bPriorRows_, prior_.rows().data(), prior_.rows().size() * 4});
            up.push_back({&bPriorCols_, prior_.cols().data(), prior_.cols().size() * 4});
            up.push_back({&bPriorErow_, prior_.entryRow().data(), prior_.entryRow().size() * 4});
        }
        if (cgAllocated_) {
            up.push_back({&bCamRanges_, P_.cam_obs_ranges.data(), P_.cam_obs_ranges.size() * 4});
            up.push_back({&bCamObs_, P_.cam_obs.data(), P_.cam_obs.size() * 4});
            up.push_back({&bCamChunks_, P_.cam_chunks.data(), P_.cam_chunks.size() * 4});
            up.push_back({&bPrecBlocks_, P_.prec_blocks.data(), P_.prec_blocks.size() * 4});
        }
        ctx_.uploadMany(up.data(), up.size());

        double t_upload = prof_lap();
        if (spirula::env("SFM_MAP_PROF"))
            sfm::slog::diag(sfm::slog::Tag::Map,
                       "[prof]   solver init: ctx %.3f buf %.3f pipe %.3f upload %.3f s",
                       t_ctx, t_buf, t_pipe, t_upload);
        stats_.vram_mb = ctx_.totalAllocatedMB();
        stats_.solver = useCG_ ? (haveFallback_ ? "cg+fallback" : "cg") : "dense";
        if (opt_.verbose)
            sfm::slog::diag(sfm::slog::Tag::Map,
                            "[vk] n_dim = %u, solver = %s, VRAM allocated = %.1f MB",
                       P_.n_dim, stats_.solver, stats_.vram_mb);
    }

    // CPU 求解器直接操作问题的主机参数，上传下载均无需复制。
    void uploadParams() {
        if (cpu_) return;
        std::vector<uint8_t> poses, exts, intr, points;
        packReals(poses, P_.poses.data(), P_.poses.size(), opt_.real);
        packReals(exts, P_.exts.data(), P_.exts.size(), opt_.real);
        packReals(intr, P_.intr.data(), P_.intr.size(), opt_.real);
        packReals(points, P_.points.data(), P_.points.size(), opt_.real);
        std::vector<VkContext::UploadItem> up = {
            {&bPoses_, poses.data(), poses.size()},
            {&bIntr_, intr.data(), intr.size()},
            {&bPoints_, points.data(), points.size()},
        };
        if (!P_.exts.empty()) up.push_back({&bExts_, exts.data(), exts.size()});
        ctx_.uploadMany(up.data(), up.size());
    }

    void downloadParams() {
        if (cpu_) return;
        std::vector<uint8_t> tmp(std::max({bPoses_.size, bIntr_.size, bPoints_.size, bExts_.size}));
        ctx_.download(bPoses_, tmp.data(), P_.poses.size() * realSize(opt_.real));
        unpackReals(P_.poses, tmp.data(), P_.poses.size(), opt_.real);
        ctx_.download(bIntr_, tmp.data(), P_.intr.size() * realSize(opt_.real));
        unpackReals(P_.intr, tmp.data(), P_.intr.size(), opt_.real);
        ctx_.download(bPoints_, tmp.data(), P_.points.size() * realSize(opt_.real));
        unpackReals(P_.points, tmp.data(), P_.points.size(), opt_.real);
        if (!P_.exts.empty()) {
            ctx_.download(bExts_, tmp.data(), P_.exts.size() * realSize(opt_.real));
            unpackReals(P_.exts, tmp.data(), P_.exts.size(), opt_.real);
        }
    }

    double computeCost() {
        if (cpu_) return cpu_->computeCost();
        beginSeg();
        recordCost();
        ctx_.barrier(cb_);
        ctx_.recordDownload(cb_, bCost_, realSize(opt_.real), 0, kDlCost);
        endSeg();
        return readCost() + priorCost(hostPoses_, hostExts_);
    }

    // ---------------- 先验 ----------------
    // 主机以设备位姿镜像求值：在已接受参数处装配，在试探参数回读后计算代价。

    double priorCost(const std::vector<double>& poses, const std::vector<double>& exts) const {
        return hasPriors_ ? prior_.cost(P_, poses.data(), exts.data()) : 0.0;
    }

    void uploadPriors(double damping) {
        if (!hasPriors_) return;
        prior_.assemble(P_, hostPoses_.data(), hostExts_.data(), damping);
        std::vector<uint8_t> blk, g;
        packReals(blk, prior_.blocks().data(), prior_.blocks().size(), opt_.real);
        packReals(g, prior_.gradient().data(), prior_.gradient().size(), opt_.real);
        VkContext::UploadItem up[2] = {{&bPriorBlk_, blk.data(), blk.size()},
                                       {&bPriorG_, g.data(), g.size()}};
        ctx_.uploadMany(up, 2);
    }

    // 读取本轮设备代价，并用同一命令缓冲回读的参数计算先验代价。
    double readTotalCost() {
        double c = readCost();
        if (!hasPriors_) return c;
        const uint8_t* st = (const uint8_t*)ctx_.stagingDownloadPtr() + kDlPoses;
        unpackReals(trialPoses_, st, P_.poses.size(), opt_.real);
        unpackReals(trialExts_, st + P_.poses.size() * realSize(opt_.real), P_.exts.size(),
                    opt_.real);
        return c + priorCost(trialPoses_, trialExts_);
    }

    void acceptTrialParams() {
        if (!hasPriors_) return;
        hostPoses_.swap(trialPoses_);
        hostExts_.swap(trialExts_);
    }

    void solve() {
        if (cpu_) return cpu_->solve();
        auto t0 = std::chrono::high_resolution_clock::now();
        double damping = opt_.init_damping;
        double cost = computeCost();
        stats_.initial_cost = cost;
        int noimprov = 0;

        bool reuse = false;  // 拒绝步恢复参数后，已有装配仍与参数一致
        double reject_mult = 2.0;
        int consec_fallbacks = 0;
        auto last_ckpt = std::chrono::steady_clock::now();
        for (int it = 0; it < opt_.max_iters; it++) {
            sfm::cancel::check();
            if (opt_.verbose)
                sfm::slog::diag(sfm::slog::Tag::Map, "iter %3d: cost = %.9e, damping = %.3g%s", it,
                                cost,
                           damping,
                           reuse ? " (reuse)" : "");

            LinSolve path = useCG_ ? LinSolve::CG : densePath_;
            // CG 迭代少时不值得重建 A_c；每三次求解或阻尼变化十倍才重建，过期矩阵仅影响迭代数。
            tcUse_ = tcN_ && !tcOff_ && lastCg_ > kTcMinIters;
            const bool tcBuild = tcUse_ && (!tcHave_ || tcAge_ >= 2 ||
                                            std::fabs(std::log(damping / tcLambda_)) > std::log(10.0));
            tcBuild_ = tcBuild;
            if (tcBuild) {
                tcHave_ = true;
                tcAge_ = 0;
                tcLambda_ = damping;
            } else if (tcUse_) {
                tcAge_++;
            }
            uploadPriors(damping);
            beginSeg();
            recordIteration((float)damping, reuse, path);
            endSeg();
            double newCost = readTotalCost();
            stats_.iterations = it + 1;

            if (path == LinSolve::CG) {
                bool conv;
                double cg_iters;
                readCgStatus(conv, cg_iters);
                // 若 r.z 或 p.Sp 非正导致 CG 尚未迈步，先禁用 A_c 重试，再提高阻尼；1e23 量级 Gram 块曾在约 1e-9 相对舍入下失去正定性。
                if (cg_iters == 0 && !conv && tcUse_) {
                    tcUse_ = tcBuild_ = false;
                    restore_pending_ = true;
                    beginSeg();
                    recordIteration((float)damping, true, path);
                    endSeg();
                    newCost = readTotalCost();
                    readCgStatus(conv, cg_iters);
                    if (cg_iters > 0 || conv) {
                        tcOff_ = true;
                        sfm::slog::diag(sfm::slog::Tag::Map,
                                        "[vk] coarse correction broke down; continuing without it");
                    }
                }
                if (cg_iters == 0 && !conv) newCost = std::numeric_limits<double>::infinity();
                stats_.cg_solves++;
                stats_.cg_iters_total += cg_iters;
                lastCg_ = conv ? cg_iters : 1e9;
                if (conv) {
                    consec_fallbacks = 0;
                    // 根据实际迭代数调整记录的上限
                    cgMaxit_ = std::min<uint32_t>(
                        std::max<uint32_t>((uint32_t)(1.5 * cg_iters) + 8, 16),
                        (uint32_t)opt_.cg_max_iters);
                } else {
                    uint32_t usedCap = cgMaxit_;
                    cgMaxit_ = (uint32_t)opt_.cg_max_iters;
                    // 截断 CG 仍可给出阻尼下降步；改善代价则保留，仅原本会拒绝时才支付稠密重算成本。
                    bool stepOk = std::isfinite(newCost) && newCost <= cost * (1.0 + opt_.rtol);
                    if (stepOk) consec_fallbacks = 0;
                    if (haveFallback_ && !stepOk) {
                        // 丢弃当前步，复用装配并改用稠密求解器重算本轮
                        if (opt_.verbose)
                            sfm::slog::diag(sfm::slog::Tag::Map,
                                       "iter %3d: CG hit %u-iteration cap, dense fallback",
                                       it, usedCap);
                        restore_pending_ = true;
                        beginSeg();
                        recordIteration((float)damping, true, densePath_);
                        endSeg();
                        newCost = readTotalCost();
                        tcHave_ = false;  // 稠密求解复用了 u_S
                        stats_.cg_fallbacks++;
                        if (++consec_fallbacks >= 3) {
                            useCG_ = false;  // CG 效益不足，后续保持稠密求解
                            stats_.solver = "cg->dense";
                            if (opt_.verbose)
                                sfm::slog::diag(sfm::slog::Tag::Map,
                                           "[vk] repeated CG stalls, switching to dense");
                        }
                    }
                }
            }

            if (std::isfinite(newCost) && newCost <= cost * (1.0 + opt_.rtol)) {
                if (newCost / cost >= 1.0 - opt_.rtol) {
                    // 极小改善计入耐心阈值并提高 lambda，避免阻尼持续缩小使求解停滞于病态平台；df 的 871 图问题观察到此现象。
                    if (++noimprov >= opt_.patience) { cost = newCost; break; }
                } else {
                    noimprov = 0;
                    damping = std::max(damping / 3.0, kMinDamping);
                }
                cost = newCost;
                stats_.accepted++;
                acceptTrialParams();
                reuse = false;
                reject_mult = 2.0;
                // 每 5 s 保存一次参数；四百万点回读约 20 ms，小问题通常达不到该间隔。
                const auto now = std::chrono::steady_clock::now();
                if (opt_.checkpoint && now - last_ckpt > std::chrono::seconds(5)) {
                    downloadParams();
                    *opt_.checkpoint = {it + 1, damping, cost};
                    last_ckpt = now;
                }
            } else {
                // 拒绝时恢复参数，装配快照仍有效
                restore_pending_ = true;
                if (!std::isfinite(newCost)) {
                    if (++noimprov >= opt_.patience) break;
                } else
                    noimprov = 0;
                // 拒绝可复用装配，初期细调 lambda，连续拒绝时加快提升
                damping *= reject_mult;
                reject_mult = std::min(reject_mult * 2.0, 32.0);
                reuse = true;
            }
        }
        // 最后一轮可能被拒绝，最终下载必须返回最后一次接受的参数。
        flushRestore();
        stats_.final_cost = cost;
        auto t1 = std::chrono::high_resolution_clock::now();
        stats_.solve_seconds = std::chrono::duration<double>(t1 - t0).count();
        ctx_.printProfile();
    }

    const SolverStats& stats() const { return cpu_ ? cpu_->stats() : stats_; }
    // init 可能改变实际标量配置，外部打包必须查询实际类型；将 double 字节写入 df 缓冲会静默产生错误数据。
    RealCfg real() const { return opt_.real; }
    // 仅 GPU 路径有上下文与设备缓冲
    VkContext& ctx() { return ctx_; }
    GpuBuffer& bufS() { return bS_; }
    GpuBuffer& bufG() { return bG_; }

    // 调试：仅完整装配，不分解求解，以便导出 S 与 g
    void debugAssemble(float damping) {
        if (cpu_) return cpu_->assembleOnly(damping);
        uploadPriors(damping);
        beginSeg();
        recordAssembly(damping, false, densePath_);
        endSeg();
    }

    // 调试：将任一路径装配出的压缩下三角 S 和 g 转为 double
    std::vector<double> debugPackedS() {
        if (cpu_) return cpu_->packedS();
        std::vector<uint8_t> raw(bS_.size);
        ctx_.download(bS_, raw.data(), bS_.size);
        std::vector<double> v;
        unpackReals(v, raw.data(), (uint64_t)P_.n_dim * (P_.n_dim + 1) / 2, opt_.real);
        return v;
    }
    std::vector<double> debugG() { return cpu_ ? cpu_->gradient() : downloadG(); }

    // 调试：同一装配分别使用 CG 与稠密路径求解并比较步长，需要启用 CG 回退配置
    double debugCompareStep(float damping) {
        if (cpu_) return cpu_->compareStep(damping);
        if (!useCG_ || !haveFallback_)
            throw std::runtime_error("step comparison needs --solver cg + fallback on");
        beginSeg();
        recordAssembly(damping, false, LinSolve::CG);
        recordPCG((uint32_t)opt_.cg_max_iters);
        ctx_.barrier(cb_);
        ctx_.recordDownload(cb_, bCgScal_, 8 * realSize(opt_.real), 0, kDlCgScal);
        endSeg();
        std::vector<double> xcg = downloadG();
        bool conv;
        double cg_iters;
        readCgStatus(conv, cg_iters);
        beginSeg();
        recordAssembly(damping, true, densePath_);  // 复用同一次装配
        recordCholesky();
        endSeg();
        std::vector<double> xd = downloadG();
        double dmax = 0, xmax = 0;
        for (uint32_t i = 0; i < P_.n_dim; i++) {
            dmax = std::max(dmax, std::fabs(xcg[i] - xd[i]));
            xmax = std::max(xmax, std::fabs(xd[i]));
        }
        double rel = dmax / std::max(xmax, 1e-300);
        sfm::slog::diag(sfm::slog::Tag::Map,
                        "cmp-step lambda=%g: cg %s in %.0f iters, "
                        "|dx_cg - dx_dense|_inf/|dx|_inf = %.3e",
                        damping, conv ? "converged" : "hit cap", cg_iters, rel);
        return rel;
    }

    // 原地分解压缩 S 并对 g 求解，供 sfm_cholesky_test 使用。
    void cholesky() {
        beginSeg();
        recordCholesky();
        endSeg();
    }

private:
    // Schur 内核编译四档自由度，CG 编译两档；仅按本问题最宽相机选择，只有可优化成员外参需要 rig 档。
    void pickTiers() {
        uint32_t maxDof = 0;
        for (uint32_t i = 0; i < P_.num_images; i++)
            maxDof = std::max(maxDof, 6 + P_.memberFree(i) + P_.groups[P_.image_group[i]].n_intr);
        schurSuffix_ = maxDof <= 12 ? "_c" : maxDof <= 14 ? "_m" : maxDof <= 18 ? "_w" : "_x";
        cgSuffix_ = maxDof <= 18 ? "_w" : "_x";
        const uint32_t tier = maxDof <= 18 ? 18 : 24;
        bBlk_ = tier * (tier + 1) / 2;
        wide_ = std::max(maxDof, 6u) / 24.0;
        double t1 = 0, t2 = 0;
        for (uint32_t p = 0; p < P_.num_points; p++) {
            const double t = P_.obs_ranges[p + 1] - P_.obs_ranges[p];
            t1 += t;
            t2 += t * t;
        }
        if (t1 > 0) meanTrackT_ = t2 / t1;
    }

    static std::string costEntry(const BAProblem::ModelRange& mr) {
        return std::string(kModels[mr.model].cost_entry) + (mr.rig ? "_rig" : "");
    }
    static std::string jacEntry(const BAProblem::ModelRange& mr) {
        return std::string(kModels[mr.model].jac_entry) + (mr.rig ? "_rig" : "");
    }

    // 根据问题形状、选项与显存预算选择线性求解路径，并构建所需主机表。
    void decidePaths() {
        pickTiers();
        const bool exclusive = exclusiveGroups(P_);
        const uint64_t packed = (uint64_t)P_.n_dim * (P_.n_dim + 1) / 2;
        const bool denseOk = packed <= 0x7FFFFFFFull;  // 32 位压缩索引
        const uint64_t pairEntries = pairEntryCount(P_);
        const bool cgOk = P_.num_obs > 0;
        const double budget = opt_.vram_budget_mb > 0 ? opt_.vram_budget_mb
                                                      : 0.9 * ctx_.deviceLocalHeapMB();
        // 按图像对聚合的 Schur 仅为可选加速，会额外存储条目与 Y；显存不足时先退回原子稠密装配，而非直接切到 CG。
        const double denseObsMB = estimateMB(true, false, false, 0);
        const double densePairMB = estimateMB(true, false, true, pairEntries);
        const bool pairOk = exclusive && P_.num_obs > 0 && pairEntries <= kMaxPairEntries &&
                            densePairMB <= budget;
        const double denseMB = pairOk ? densePairMB : denseObsMB;
        double cgMB = estimateMB(false, true, false, 0);
        double bothMB = estimateMB(true, true, pairOk, pairEntries);

        const uint32_t kDenseMaxDim = 8192;

        switch (opt_.solver) {
            case SolverSel::Dense:
                useCG_ = false;
                if (!denseOk)
                    throw std::runtime_error(
                        "reduced system too large for the dense solver (n_dim*(n_dim+1)/2 "
                        "exceeds 32-bit packed indexing); use --solver cg");
                break;
            case SolverSel::CG:
                useCG_ = cgOk;
                if (!cgOk) {
                    sfm::slog::diag(sfm::slog::Tag::Map,
                               "[vk] warning: no observations, falling back to dense");
                    if (!denseOk) throw std::runtime_error("no usable solver path");
                }
                break;
            case SolverSel::Auto:
                useCG_ = cgOk && (P_.n_dim > kDenseMaxDim || !denseOk || denseMB > budget ||
                                  cgWork() < 0.5 * denseWork(pairOk, pairEntries));
                if (!useCG_ && !denseOk)
                    throw std::runtime_error("reduced system too large for the dense solver");
                break;
        }

        if (useCG_ && ::planCoarse(P_, kTcMaxDim, tcK_, tcN_, tcEntries_)) {
            cgMB = estimateMB(false, true, false, 0);
            bothMB = estimateMB(true, true, pairOk, pairEntries);
        }

        haveFallback_ = false;
        if (useCG_) {
            bool want = opt_.cg_fallback == CgFallback::On ||
                        (opt_.cg_fallback == CgFallback::Auto && bothMB <= 0.5 * budget);
            haveFallback_ = want && pairOk && denseOk;
            if (opt_.cg_fallback == CgFallback::On && !haveFallback_)
                sfm::slog::diag(sfm::slog::Tag::Map, "[vk] warning: dense fallback unavailable "
                           "(pair-Schur or packed-index limits)");
        }

        if (opt_.verbose)
            sfm::slog::diag(sfm::slog::Tag::Map,
                       "[vk] VRAM estimates: dense %.0f MB (%s Schur), cg %.0f MB (budget %.0f MB)",
                       denseMB, pairOk ? "pair" : "per-obs", cgMB, budget);
        // 在驱动分配失败前报告预算不足；CG 已无更省内存路径，只能缩小问题或增加内存。
        const double needMB = (useCG_ ? cgMB : denseMB) + (haveFallback_ ? denseMB : 0);
        if (needMB > budget) {
            if (opt_.over_budget_throws) throw BAOverBudget(needMB, budget);
            sfm::slog::diag(sfm::slog::Tag::Map,
                       "[vk] warning: the %s solver needs ~%.0f MB and the budget is %.0f MB; "
                       "this may run out of device memory",
                       useCG_ ? "cg" : "dense", needMB, budget);
        }

        // 所选路径的主机索引表
        P_.use_pair_schur = false;
        if ((!useCG_ || haveFallback_) && pairOk) buildPairTables(P_);
        densePath_ = P_.use_pair_schur ? LinSolve::DensePair : LinSolve::DenseObs;
        cgAllocated_ = useCG_;
        if (cgAllocated_) {
            buildCamTables(P_);
            buildPrecBlocks(P_, exclusive);
        }
        cgMaxit_ = (uint32_t)opt_.cg_max_iters;
        if (tcN_) {
            // 每簇对分为至多 128 条目的块，设备偏移位于 pair-Schur 条目之后。
            std::vector<uint32_t> key;
            buildCoarseEntries(P_, tcK_, tcEnt_, key);
            const uint32_t base = (uint32_t)(P_.pair_entries.size() / 2);
            tcChk_.clear();
            for (size_t kk = 0; kk + 1 < key.size(); kk++)
                for (uint32_t o = key[kk]; o < key[kk + 1]; o += 128) {
                    tcChk_.push_back(base + o);
                    tcChk_.push_back(std::min(128u, key[kk + 1] - o));
                }
            tcChunks_ = (uint32_t)(tcChk_.size() / 2);
        }
    }

    static constexpr uint32_t kTcMaxDim = 4096;

    // 估计单次 LM 迭代代价；稠密装配与轨迹长度平方相关，1068 图、平均 70 观测的轨迹数据每轮 3.7 s，CG 为 0.4 s。
    double denseWork(bool pair, uint64_t pairEntries) const {
        const double n = P_.n_dim;
        const double schur = pair ? kWPairEntry * wide_ * wide_ * (double)pairEntries
                                  : kWSchurObs * wide_ * wide_ * meanTrackT_ / kSchurObsT *
                                        P_.num_obs;
        return schur + kWFlop * 2 * n * n * n / 3 + kWJac * wide_ * P_.num_obs;
    }
    double cgWork() const {
        return kWJac * wide_ * P_.num_obs + kWCamDiag * wide_ * wide_ * P_.num_obs +
               kCgItersGuess * (kWGather + kWScatter) * wide_ * P_.num_obs;
    }
    static constexpr double kCgItersGuess = 40;

    // 路径组合的设备缓冲占用，单位 MB，与 init 分配一致
    double estimateMB(bool withDense, bool withCG, bool pairTables, uint64_t pairEntries) const {
        const double rs = (double)realSize(opt_.real);
        const double n = P_.n_dim, no = P_.num_obs, np = P_.num_points, ni = P_.num_images;
        const double packed = n * (n + 1) / 2;
        double b = 0;
        b += no * (2 * rs + 12) + 4 * (double)P_.model_obs.size();  // 观测与索引表
        b += 16 * ni + 16 * (double)(P_.groups.size() + P_.members.size());
        b += 2 * (P_.pose_dim + P_.exts.size() + P_.total_intr) * rs;  // 参数与备份
        b += 3 * np * rs * 3;                                      // 点参数、备份与 Bp0
        b += 4 * (np + 1);
        b += ((double)P_.jc_total + 8 * no) * rs;                  // Jc, Jp, res
        b += (9 + 3 + 9) * np * rs;                                // App, Bp, W
        b += 2 * n * rs;                                           // g, y
        if (withDense) {
            b += packed * rs;
            if (pairTables)
                b += 8.0 * pairEntries * 1.01 + 6 * no * rs;       // 图像对条目与 Y
        }
        if (withCG && tcN_) {
            const double tn = tcN_, tc = tn * (tn + 1) / 2 + 42.0 * P_.num_frames;
            b += (std::max(withDense ? packed : 0.0, tc) - (withDense ? packed : 0.0)) * rs;
            b += std::max(0.0, 34.0 * tn - n) * rs + 8.0 * tcEntries_ * 1.01;
        }
        if (withCG)
            b += (4 * n + 3 * np + ni * (double)bBlk_ +
                  (ni + (double)P_.members.size() + (double)P_.groups.size()) * kCamBlk) * rs +
                 4 * (ni + 1) + 4 * no + 12 * (no / 1024 + ni) +
                 16 * (ni + (double)P_.groups.size());  // 分块与预条件块表
        return b / (1024.0 * 1024.0);
    }

    // ---------------- 提交预算 ----------------

    // 代价单位约为 RTX 5070 在 fp64、24 维 rig 相机块下的一纳秒，测自 6946 图的 profile；wide_ 调整较窄相机档的代价。
    static constexpr double kWLaunch = 2000, kWPoint = 0.25, kWVec = 1, kWImage = 5;
    static constexpr double kWCost = 1.5, kWJac = 5.5, kWDp = 0.6, kWYPrep = 0.3;
    static constexpr double kWCamDiag = 12.7, kWGather = 0.65, kWScatter = 1.2;
    static constexpr double kWSchurObs = 40, kWPairEntry = 3, kWFlop = 2.6e-3, kWTcSchur = 6;
    // schur_obs 按观测遍历轨迹；kWSchurObs 对应测量数据 sum t^2 / sum t = 5.8 的有效轨迹长度。
    static constexpr double kSchurObsT = 5.8;
    // 首次提交计时前保守假设设备慢 64 倍。
    static constexpr double kPriorRate = 1e9 / 64;

    static std::mutex& budgetMutex() {
        static std::mutex m;
        return m;
    }
    // 每设备、每标量配置共享预算估计，使新的 BA 求解器继承上次测量。
    spirula::SubmitBudget& budgetLocked() {
        static std::map<std::string, spirula::SubmitBudget> m;
        return m.try_emplace(ctx_.selector() + realCfgName(opt_.real), kPriorRate).first->second;
    }
    double budgetLimit() {
        std::lock_guard<std::mutex> g(budgetMutex());
        return budgetLocked().limit();
    }
    // 实际耗时与内核模型代价之比，未探测时为 0。
    double& kernelScaleLocked(const std::string& name) {
        static std::map<std::string, double> m;
        return m[ctx_.selector() + realCfgName(opt_.real) + name];
    }

    void beginSeg() {
        cb_ = ctx_.begin();
        open_ = 0;
        segCg_ = pollCg_;
        segTop_.clear();
        segTopWork_ = 0;
    }
    // CG 收敛后内核立即返回，只有回读标志尚未置位时才计入预算测量；两计算单元集显曾因空任务低估耗时而丢失设备。
    double endSeg(bool record = true) {
        if (segCg_) ctx_.recordDownload(cb_, bCgScal_, 8 * realSize(opt_.real), 0, kDlCgScal);
        const auto t0 = std::chrono::steady_clock::now();
        VkCommandBuffer cb = cb_;
        cb_ = VK_NULL_HANDLE;
        ctx_.submit(cb);
        const double dt =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        cgNoop_ = segCg_ && cgFlagStaged();
        segCg_ = false;
        std::lock_guard<std::mutex> g(budgetMutex());
        spirula::SubmitBudget& b = budgetLocked();
        if (!record || cgNoop_) return dt;
        // 主要由单内核构成的提交段更新该内核比例，防止全局速率漂移导致超额；df 的 jac 曾在集显上偏差三倍，每次调整限制为两倍。
        double* scale = segTop_.empty() ? nullptr : &kernelScaleLocked(segTop_);
        if (scale && *scale > 0 && segTopWork_ > 0.8 * open_ && dt > b.target() / 16 &&
            b.rate() > 0) {
            const double ratio = dt * b.rate() / open_;
            *scale *= ratio > 1 ? std::min(ratio, 2.0) : std::sqrt(ratio);
            *scale = std::min(std::max(*scale, 1.0 / 16), 256.0);
        } else {
            b.record(open_, dt);
        }
        return dt;
    }
    void split() {
        endSeg();
        beginSeg();
    }
    // 屏障后、记录更多工作前检查预算；超额则先提交已有命令，返回是否提交。
    bool room(double w) {
        const bool full = open_ > 0 && open_ + w > budgetLimit();
        if (full) split();
        open_ += w;
        return full;
    }

    // 按预算分块执行 n 项，由 at 设置范围；首次大内核仅用 1/32 预算单独计时，避免设备差异，双计算单元集显的 df schur_obs 曾慢于模型 21 倍。
    template <class At>
    void launch(const std::string& name, uint32_t n, uint32_t per_group, double w_item,
                const Push& p, At at) {
        if (n == 0) return;
        double lim, scale, rate;
        bool measured;
        auto read = [&] {
            std::lock_guard<std::mutex> g(budgetMutex());
            lim = budgetLocked().limit();
            rate = budgetLocked().rate();
            measured = budgetLocked().measured();
            scale = kernelScaleLocked(name);
        };
        read();
        const bool big = n * w_item > lim / 32;
        if (scale == 0 && big && open_ > 0) {
            split();  // 待提交工作可能包含探测所需测量
            read();
        }
        // 最多分成 256 段；小启动受固定延迟限制，任一完整启动的 1/256 已远低于 2 s。
        auto rangeOf = [&](double units) {
            const double w = w_item * (scale > 0 ? scale : 1);
            uint64_t c = lim > 0 ? (uint64_t)(units / std::max(w, 1e-9)) : n;
            c = std::max<uint64_t>(c, n / 256);
            return std::max<uint64_t>(per_group, c / per_group * per_group);
        };
        uint64_t a = 0;
        if (scale == 0 && measured && big) {
            if (open_ > 0) split();
            const uint32_t c = (uint32_t)std::min<uint64_t>(rangeOf(lim / 32), n);
            Push q = p;
            at(q, 0, c);
            ctx_.dispatch(cb_, name, (c + per_group - 1) / per_group, q);
            open_ = kWLaunch + c * w_item;
            const double dt = endSeg(false);
            beginSeg();
            if (!cgNoop_) {
                std::lock_guard<std::mutex> g(budgetMutex());
                kernelScaleLocked(name) =
                    std::min(std::max(dt * rate / (c * w_item), 1.0 / 16), 256.0);
            }
            a = c;
        }
        while (a < n) {
            read();
            const uint32_t c = (uint32_t)std::min<uint64_t>(rangeOf(lim), n - a);
            const double w = c * w_item * (scale > 0 ? scale : 1);
            room(kWLaunch + w);
            if (name == segTop_) segTopWork_ += w;
            else if (w > segTopWork_) segTop_ = name, segTopWork_ = w;
            Push q = p;
            at(q, (uint32_t)a, c);
            ctx_.dispatch(cb_, name, (c + per_group - 1) / per_group, q);
            a += c;
        }
    }
    // u0 为总量、u4 为当前范围起点的内核。
    static void atBase(Push& q, uint32_t first, uint32_t) { q.u4 = first; }

    // 逐模型内核的 u0/u1 已用于数量与偏移。
    template <class Range>
    static auto atModel(const Range& mr) {
        return [&mr](Push& q, uint32_t first, uint32_t count) {
            q.u0 = count;
            q.u1 = mr.offset + first;
        };
    }

    bool cgFlagStaged() {
        std::vector<double> v;
        unpackReals(v, (const uint8_t*)ctx_.stagingDownloadPtr() + kDlCgScal, 8, opt_.real);
        return v[6] > 0.5;
    }

    // ---------------- 命令记录 ----------------

    // chol_update 同时分解下一对角块，因此 chol_diag 仅处理首块；每块三角求解合并为一次分派。
    void recordCholesky() {
        const uint32_t n = P_.n_dim, bs = 32;
        const uint32_t nb = (n + bs - 1) / bs;
        recordFactor(n);
        Push p;
        p.u0 = n;
        for (uint32_t k = 0; k < nb; k++) {
            p.u1 = k;
            p.u2 = (k + 1) * bs < n ? n - (k + 1) * bs : 0;
            room(kWLaunch + 2.0 * bs * p.u2 * kWFlop);
            ctx_.dispatch(cb_, "tri_fwd", std::max(1u, (p.u2 + 255) / 256), p);
            ctx_.barrier(cb_);
        }
        for (int k = (int)nb - 1; k >= 0; k--) {
            p.u1 = (uint32_t)k;
            p.u2 = (uint32_t)k * bs;
            room(kWLaunch + 2.0 * bs * p.u2 * kWFlop);
            ctx_.dispatch(cb_, "tri_bwd", std::max(1u, (p.u2 + 255) / 256), p);
            ctx_.barrier(cb_);
        }
    }

    // 原地分解 u_S 前 n×n 压缩三角；rel > 0 时，用 tc_reg 对角值替换低于指定比例的主元。
    void recordFactor(uint32_t n, float rel = 0) {
        const uint32_t bs = 32;
        const uint32_t nb = (n + bs - 1) / bs;
        const double tile = 2.0 * bs * bs * bs * kWFlop;
        Push p;
        p.u0 = n;
        p.u1 = 0;
        p.u3 = rel > 0 ? 1 : 0;
        p.f0 = rel;
        room(kWLaunch + tile);
        ctx_.dispatch(cb_, "chol_diag", 1, p);
        ctx_.barrier(cb_);
        for (uint32_t k = 0; k + 1 < nb; k++) {
            p.u1 = k;
            uint32_t below = nb - 1 - k;
            launch("chol_panel", below, 1, tile, p, atBase);
            ctx_.barrier(cb_);
            p.u2 = below * (below + 1) / 2;
            launch("chol_update", p.u2, 1, tile, p, atBase);
            ctx_.barrier(cb_);
        }
    }

    // 由本轮 B/W 构造 A_c = P^T S P，再原地求 Cholesky 因子逆，使每次应用仅需两次矩阵向量乘。
    void recordCoarse() {
        const uint32_t packedTc = tcN_ * (tcN_ + 1) / 2;
        Push p;
        p.u0 = P_.num_frames;
        p.u2 = tcK_;
        p.u3 = packedTc;
        room(2 * kWLaunch + (P_.num_frames + P_.num_images) * kWImage);
        ctx_.dispatch(cb_, "tc_basis", (P_.num_frames + 63) / 64, p);
        ctx_.fillZero(cb_, bS_, 0, (VkDeviceSize)packedTc * realSize(opt_.real));
        ctx_.barrier(cb_);
        p.u0 = P_.num_images;
        ctx_.dispatch(cb_, std::string("tc_bpart") + cgSuffix_, (P_.num_images + 63) / 64, p);
        p.u0 = tcChunks_;
        p.u1 = P_.num_pair_chunks;
        launch("tc_schur", tcChunks_, 1,
               kWTcSchur * (double)tcEntries_ / std::max(1u, tcChunks_), p, atBase);
        ctx_.barrier(cb_);
        Push q;
        q.u0 = tcN_;
        q.f0 = opt_.real == RealCfg::F32 ? 1e-4f : 1e-8f;
        room(kWLaunch + tcN_ * kWVec);
        ctx_.dispatch(cb_, "tc_reg", (tcN_ + 255) / 256, q);
        ctx_.barrier(cb_);
        recordFactor(tcN_, opt_.real == RealCfg::F32 ? 1e-3f : 1e-6f);

        const uint32_t bs = 32, nb = (tcN_ + bs - 1) / bs;
        const double tile = 2.0 * bs * bs * bs * kWFlop;
        Push d;
        d.u0 = tcN_;
        room(kWLaunch + nb * tile);
        ctx_.dispatch(cb_, "tc_dinv", nb, d);
        ctx_.barrier(cb_);
        d.u2 = 2 * tcN_;  // 临时行位于 u_y 中两个向量之后
        for (uint32_t i = 1; i < nb; i++) {
            d.u1 = i;
            room(2 * kWLaunch + i * (i + 1) / 2.0 * tile);
            ctx_.dispatch(cb_, "tc_inv_row", i, d);
            ctx_.barrier(cb_);
            ctx_.dispatch(cb_, "tc_inv_copy", (bs * bs * i + 255) / 256, d);
            ctx_.barrier(cb_);
        }
    }

    // 块 Jacobi 预条件后补充 z += P A_c^-1 P^T r。
    void recordCoarseApply() {
        Push p;
        p.u0 = tcN_;
        p.u1 = P_.num_frames;
        p.u2 = tcK_;
        p.u3 = tcN_ * (tcN_ + 1) / 2;
        room(4 * kWLaunch + (6.0 * P_.num_frames + (double)tcN_ * tcN_) * kWVec);
        ctx_.dispatch(cb_, "tc_restrict", (tcN_ + 255) / 256, p);
        ctx_.barrier(cb_);
        ctx_.dispatch(cb_, "tc_lmul", (tcN_ + 7) / 8, p);
        ctx_.barrier(cb_);
        ctx_.dispatch(cb_, "tc_ltmul", (tcN_ + 31) / 32, p);
        ctx_.barrier(cb_);
        p.u0 = P_.pose_dim;
        ctx_.dispatch(cb_, "tc_prolong", (P_.pose_dim + 255) / 256, p);
        ctx_.barrier(cb_);
    }

    // 将先验块加入稠密 S 或 CG 预条件器，梯度加入 g；普通读改写必须在先前构建内核的屏障之后。
    void recordPriorAdd(const char* blocks_kernel) {
        Push pe;
        pe.u0 = prior_.numEntries();
        room(2 * kWLaunch + (prior_.numEntries() + P_.pose_dim) * kWVec);
        ctx_.dispatch(cb_, blocks_kernel, (prior_.numEntries() + 255) / 256, pe);
        Push pg;
        pg.u0 = P_.pose_dim;
        ctx_.dispatch(cb_, "prior_add_g", (P_.pose_dim + 255) / 256, pg);
    }

    void recordCost() {
        ctx_.fillZero(cb_, bCost_);
        ctx_.barrier(cb_);
        for (auto& mr : P_.model_ranges) {
            Push p;
            p.f0 = opt_.loss_param;
            launch(costEntry(mr), mr.count, 256, kWCost, p, atModel(mr));
        }
        ctx_.barrier(cb_);
    }

    void recordAssembly(float damping, bool reuse, LinSolve path) {
        const bool dense = path != LinSolve::CG;
        if (reuse) {
            // 拒绝步恢复参数后复用 Jc/Jp/res/App，跳过雅可比重算；S/g 始终由 Schur 内核重建，仅回代会覆盖的 Bp 需要恢复快照。
            ctx_.copy(cb_, bBp0_, bBp_, bBp_.size);
            if (dense) {
                ctx_.fillZero(cb_, bS_);
                ctx_.fillZero(cb_, bG_);
            }
            ctx_.barrier(cb_);
        } else {
            // 备份参数以便拒绝时恢复
            ctx_.copy(cb_, bPoses_, bPosesBak_, bPoses_.size);
            ctx_.copy(cb_, bIntr_, bIntrBak_, bIntr_.size);
            ctx_.copy(cb_, bPoints_, bPointsBak_, bPoints_.size);
            if (!P_.exts.empty()) ctx_.copy(cb_, bExts_, bExtsBak_, bExts_.size);

            if (dense) {
                ctx_.fillZero(cb_, bS_);
                ctx_.fillZero(cb_, bG_);
            }
            ctx_.fillZero(cb_, bApp_);
            ctx_.fillZero(cb_, bBp_);
            ctx_.barrier(cb_);

            for (auto& mr : P_.model_ranges) {
                Push p;
                p.f0 = opt_.loss_param;
                launch(jacEntry(mr), mr.count, 128, kWJac * wide_, p, atModel(mr));
            }
            ctx_.barrier(cb_);

            ctx_.copy(cb_, bBp_, bBp0_, bBp_.size);
            ctx_.barrier(cb_);
        }

        {
            Push q;
            q.u0 = P_.num_points;
            q.f0 = damping;
            room(kWLaunch + P_.num_points * kWPoint);
            ctx_.dispatch(cb_, "point_prep", (P_.num_points + 255) / 256, q);
            if (path == LinSolve::CG) {  // 分块 cg_cam_diag 使用原子累加
                ctx_.fillZero(cb_, bCgB_);
                ctx_.fillZero(cb_, bCgM_);
                ctx_.fillZero(cb_, bG_);
            }
        }
        ctx_.barrier(cb_);

        {
            Push p;
            p.f0 = damping;
            if (path == LinSolve::DensePair) {
                p.u0 = P_.num_obs;
                launch("y_prep", P_.num_obs, 256, kWYPrep, p, atBase);
                ctx_.barrier(cb_);
                p.u0 = P_.num_pair_chunks;
                const double entries = P_.pair_entries.size() / 2.0;
                launch(std::string("schur_pair") + schurSuffix_, P_.num_pair_chunks, 1,
                       kWPairEntry * wide_ * wide_ * entries / std::max(1u, P_.num_pair_chunks),
                       p, atBase);
            } else if (path == LinSolve::DenseObs) {
                p.u0 = P_.num_obs;
                launch(std::string("schur_obs") + schurSuffix_, P_.num_obs, 128,
                       kWSchurObs * wide_ * wide_ * meanTrackT_ / kSchurObsT, p, atBase);
            }
            if (dense && hasPriors_) {
                ctx_.barrier(cb_);
                recordPriorAdd("prior_add_s");
            }
            if (!dense) {
                p.u0 = P_.num_cam_chunks;
                p.u1 = P_.prec_exclusive ? 1 : 0;
                p.u2 = P_.num_frames;  // 成员块位于帧块之后，随后为相机组
                p.u3 = P_.num_frames + (uint32_t)P_.members.size();
                launch(std::string("cg_cam_diag") + cgSuffix_, P_.num_cam_chunks, 1,
                       kWCamDiag * wide_ * wide_ * P_.num_obs / std::max(1u, P_.num_cam_chunks),
                       p, atBase);
                ctx_.barrier(cb_);
                if (hasPriors_) {
                    recordPriorAdd("prior_add_m");
                    ctx_.barrier(cb_);
                }
                p.u0 = P_.num_prec_blocks;
                room(kWLaunch + P_.num_prec_blocks * kWImage);
                ctx_.dispatch(cb_, "cg_prec_fact", (P_.num_prec_blocks + 255) / 256, p);
                if (tcBuild_) recordCoarse();
            }
        }
        ctx_.barrier(cb_);
    }

    // 记录设备 PCG 循环；收敛标志置位后内核不再执行，预算分段处回读标志并停止记录剩余迭代。
    void recordPCG(uint32_t maxit) {
        const uint32_t n = P_.n_dim;
        const uint32_t ng = (n + 255) / 256;
        const uint32_t npart = ng;
        const uint32_t nib = (P_.num_prec_blocks + 255) / 256;
        // cg_bmul 对共享列累加，须先清零成员与分组尾部；帧也共享时需清零整个向量。
        const bool rigs = P_.hasRigs();
        const VkDeviceSize intrOff = rigs ? 0 : (VkDeviceSize)P_.pose_dim * realSize(opt_.real);
        const VkDeviceSize intrSize =
            (VkDeviceSize)(rigs ? n : n - P_.pose_dim) * realSize(opt_.real);
        const bool zeroIntr = !P_.prec_exclusive && intrSize > 0;
        const double wVecs = 5 * kWLaunch + 3.0 * n * kWVec;
        room(wVecs + P_.num_prec_blocks * kWImage);
        Push pn;
        pn.u0 = n;
        ctx_.dispatch(cb_, "cg_init", ng, pn);
        ctx_.barrier(cb_);
        Push pc;
        pc.u0 = P_.num_prec_blocks;
        pc.u1 = 0;  // 标志刚刚清零
        ctx_.dispatch(cb_, "cg_prec_apply", nib, pc);
        ctx_.barrier(cb_);
        if (tcUse_) recordCoarseApply();
        Push pr;
        pr.u0 = n;
        pr.u1 = 1;
        pr.u2 = npart;
        pr.u3 = 0;
        ctx_.dispatch(cb_, "cg_red2", ng, pr);
        ctx_.barrier(cb_);
        Push pf;
        pf.u0 = npart;
        pf.u1 = 0;
        pf.u2 = npart;
        pf.f0 = (float)opt_.cg_tol;
        pf.f1 = (float)opt_.cg_model_tol;
        ctx_.dispatch(cb_, "cg_fin", 1, pf);
        ctx_.barrier(cb_);
        ctx_.dispatch(cb_, "cg_copy", ng, pn);
        ctx_.barrier(cb_);
        pc.u1 = 1;
        pr.u3 = 1;
        const double wTrack = (double)P_.num_obs / std::max(1u, P_.num_points);
        const double wChunk = (double)P_.num_obs / std::max(1u, P_.num_cam_chunks);
        pollCg_ = true;
        for (uint32_t it = 0; it < maxit; it++) {
            if (room(2 * wVecs + P_.num_images * kWImage) && cgNoop_) break;
            segCg_ = true;
            Push pg;
            pg.u0 = P_.num_points;
            launch(std::string("cg_gather") + cgSuffix_, P_.num_points, 256,
                   kWGather * wide_ * wTrack, pg, atBase);
            ctx_.barrier(cb_);
            Push ps;
            ps.u0 = P_.num_images;
            ps.u1 = P_.prec_exclusive ? 1 : 0;
            if (zeroIntr) {
                ctx_.fillZero(cb_, bCgSp_, intrOff, intrSize);
                ctx_.barrier(cb_);
            }
            ctx_.dispatch(cb_, std::string("cg_bmul") + cgSuffix_, P_.num_images, ps);
            ctx_.barrier(cb_);
            ps.u0 = P_.num_cam_chunks;
            launch(std::string("cg_scatter") + cgSuffix_, P_.num_cam_chunks, 1,
                   kWScatter * wide_ * wChunk, ps, atBase);
            ctx_.barrier(cb_);
            if (hasPriors_) {
                Push pp;
                pp.u0 = P_.num_frames;
                room(kWLaunch + P_.num_frames * kWImage);
                ctx_.dispatch(cb_, "prior_matvec", (P_.num_frames + 255) / 256, pp);
                ctx_.barrier(cb_);
            }
            pr.u1 = 0;
            ctx_.dispatch(cb_, "cg_red2", ng, pr);
            ctx_.barrier(cb_);
            pf.u1 = 1;
            ctx_.dispatch(cb_, "cg_fin", 1, pf);
            ctx_.barrier(cb_);
            ctx_.dispatch(cb_, "cg_axpy", ng, pn);
            ctx_.barrier(cb_);
            ctx_.dispatch(cb_, "cg_prec_apply", nib, pc);
            ctx_.barrier(cb_);
            if (tcUse_) recordCoarseApply();
            pr.u1 = 1;
            ctx_.dispatch(cb_, "cg_red2", ng, pr);
            ctx_.barrier(cb_);
            pf.u1 = 2;
            ctx_.dispatch(cb_, "cg_fin", 1, pf);
            ctx_.barrier(cb_);
            ctx_.dispatch(cb_, "cg_updp", ng, pn);
            ctx_.barrier(cb_);
        }
        pollCg_ = false;
    }

    // 拒绝步或 CG 回退时先标记待恢复，在下一命令缓冲中合并执行三次参数复制，避免独立提交的栅栏往返。
    // 两者之间不执行其他工作；退出循环前必须刷新恢复。
    void recordRestore() {
        ctx_.copy(cb_, bPosesBak_, bPoses_, bPoses_.size);
        ctx_.copy(cb_, bIntrBak_, bIntr_, bIntr_.size);
        ctx_.copy(cb_, bPointsBak_, bPoints_, bPoints_.size);
        if (!P_.exts.empty()) ctx_.copy(cb_, bExtsBak_, bExts_, bExts_.size);
        ctx_.barrier(cb_);
    }

    // 循环结束即将回读参数，不能继续延迟，单独提交待执行的恢复。
    void flushRestore() {
        if (!restore_pending_) return;
        restore_pending_ = false;
        beginSeg();
        recordRestore();
        endSeg();
    }

    void recordIteration(float damping, bool reuse, LinSolve path) {
        if (restore_pending_) {
            restore_pending_ = false;
            recordRestore();
        }
        recordAssembly(damping, reuse, path);

        if (path == LinSolve::CG)
            recordPCG(cgMaxit_);
        else
            recordCholesky();

        {
            Push p;
            p.u0 = P_.num_obs;
            launch(std::string("dp_accum") + schurSuffix_, P_.num_obs, 256, kWDp * wide_, p,
                   atBase);
        }
        ctx_.barrier(cb_);

        {
            room(4 * kWLaunch + P_.num_points * kWPoint);
            Push p;
            p.u0 = P_.num_points;
            ctx_.dispatch(cb_, "point_update", (P_.num_points + 255) / 256, p);
            Push q;
            q.u0 = P_.pose_dim;
            ctx_.dispatch(cb_, "cam_update", (P_.pose_dim + 255) / 256, q);
            if (!P_.members.empty()) {
                Push e;
                e.u0 = (uint32_t)P_.members.size();
                ctx_.dispatch(cb_, "ext_update", ((uint32_t)P_.members.size() + 63) / 64, e);
            }
            if (!P_.groups.empty()) {
                Push r;
                r.u0 = (uint32_t)P_.groups.size();
                ctx_.dispatch(cb_, "intr_update", ((uint32_t)P_.groups.size() + 63) / 64, r);
            }
        }
        ctx_.barrier(cb_);

        recordCost();
        // 将 LM 所需两次回读合入同一命令缓冲，使每轮提交数减半；小型四十图问题主要受提交延迟限制，bottom-up 原子阶段可能执行约五千轮。
        ctx_.barrier(cb_);
        ctx_.recordDownload(cb_, bCost_, realSize(opt_.real), 0, kDlCost);
        if (path == LinSolve::CG)
            ctx_.recordDownload(cb_, bCgScal_, 8 * realSize(opt_.real), 0, kDlCgScal);
        if (hasPriors_) {
            const VkDeviceSize pb = P_.poses.size() * realSize(opt_.real);
            ctx_.recordDownload(cb_, bPoses_, pb, 0, kDlPoses);
            if (!P_.exts.empty())
                ctx_.recordDownload(cb_, bExts_, P_.exts.size() * realSize(opt_.real), 0,
                                    kDlPoses + pb);
        }
    }

    // 合并回读在暂存缓冲中的偏移；启用先验时后接试探位姿与外参。
    static constexpr VkDeviceSize kDlCost = 0, kDlCgScal = 64, kDlPoses = 128;

    double readCost() {
        std::vector<double> v;
        unpackReals(v, (const uint8_t*)ctx_.stagingDownloadPtr() + kDlCost, 1, opt_.real);
        return v[0];
    }

    void readCgStatus(bool& converged, double& iters) {
        std::vector<double> v;
        unpackReals(v, (const uint8_t*)ctx_.stagingDownloadPtr() + kDlCgScal, 8, opt_.real);
        // 须同时满足标志置位与容差；仅置位也可能表示数值失效，未置位表示达到记录的迭代上限
        converged = v[6] > 0.5 && v[4] <= v[5];
        iters = v[7];
    }

    std::vector<double> downloadG() {
        std::vector<uint8_t> raw(P_.n_dim * realSize(opt_.real));
        ctx_.download(bG_, raw.data(), raw.size());
        std::vector<double> v;
        unpackReals(v, raw.data(), P_.n_dim, opt_.real);
        return v;
    }

    BAProblem& P_;
    SolverOptions opt_;
    // 本次求解的规范 UUID，init 前或无可用设备的 CPU 路径为空；能力缓存以此为键，使同设备求解共用探测。
    std::string selector_;
    std::unique_ptr<bacpu::Solver> cpu_;  // 主机求解时非空
    const char* schurSuffix_ = "_c";  // Schur 内核自由度档位，由 pickTiers 选择
    const char* cgSuffix_ = "_w";     // CG 内核自由度档位
    uint32_t bBlk_ = kCamBlk;         // 当前档位的逐图像 B 块步长
    double wide_ = 1;                 // 最宽相机块相对 rig 档 24 维的比例
    double meanTrackT_ = kSchurObsT;  // 轨迹统计量 sum t^2 / sum t
    SolverStats stats_;
    std::unique_ptr<VkContext> owned_;      // 共享上下文时为空
    VkContext& ctx_;
    std::vector<GpuBuffer*> ownBufs_;   // 本实例缓冲，共享上下文时由析构释放

    LinSolve densePath_ = LinSolve::DenseObs;
    bool useCG_ = false;       // 当前使用 CG，可能回退稠密路径
    bool cgAllocated_ = false; // CG 缓冲与索引表已存在
    bool haveFallback_ = false;
    uint32_t cgMaxit_ = 100;
    // 粗层校正的每簇帧数、粗维度（0 禁用）及本次求解是否启用。
    uint32_t tcK_ = 0, tcN_ = 0;
    uint64_t tcEntries_ = 0;
    uint32_t tcChunks_ = 0;
    std::vector<uint32_t> tcEnt_, tcChk_;  // 在设备上位于 pair-Schur 表之后
    bool tcUse_ = false, tcBuild_ = false, tcHave_ = false, tcOff_ = false;
    int tcAge_ = 0;
    double tcLambda_ = 0;
    double lastCg_ = 0;  // 上次 CG 求解的迭代数
    static constexpr double kTcMinIters = 12;
    // 阻尼低于约 1e-9 时，病态尺度下舍入误差会使 S 非正定；22042 图模型中深度 2e-9 的点曾产生 1e23 Gram 项。1e-8 仍保留八位 Gauss-Newton 精度。
    static constexpr double kMinDamping = 1e-8;

    GpuBuffer bObs_, bObsImage_, bObsPoint_, bImageInfo_, bGroupInfo_, bMemberInfo_;
    GpuBuffer bPoses_, bExts_, bIntr_, bPoints_, bObsRanges_, bModelObs_, bJcOff_;
    GpuBuffer bJp_, bS_, bG_, bApp_, bBp_, bCost_;
    GpuBuffer bPosesBak_, bExtsBak_, bIntrBak_, bPointsBak_;
    bool restore_pending_ = false;  // 当前缓冲仍保留已拒绝步的参数
    VkCommandBuffer cb_ = VK_NULL_HANDLE;  // 正在记录 beginSeg 到 endSeg 的命令
    double open_ = 0;                      // cb_ 已记录的预算工作量
    bool pollCg_ = false;                  // 正在记录 PCG 循环
    bool segCg_ = false;                   // cb_ 包含 PCG 循环内核
    bool cgNoop_ = false;                  // 最近提交的 PCG 已检测到收敛
    std::string segTop_;                   // cb_ 中工作量最大的 launch 内核
    double segTopWork_ = 0;
    GpuBuffer bBp0_, bJc_, bRes_;
    GpuBuffer bPairEntries_, bPairChunks_, bW_, bYp_, bY_;
    GpuBuffer bCamRanges_, bCamObs_, bCamChunks_, bCgR_, bCgZ_, bCgP_, bCgSp_;
    GpuBuffer bCgV_, bCgB_, bCgM_, bCgScal_, bCgPart_, bPrecBlocks_;
    // 先验装配器、设备表及其求值参数的主机镜像，分别保存已接受值与当前试探值。
    sfm::PriorAssembler prior_;
    bool hasPriors_ = false;
    GpuBuffer bPriorRows_, bPriorCols_, bPriorErow_, bPriorBlk_, bPriorG_;
    std::vector<double> hostPoses_, hostExts_, trialPoses_, trialExts_;
};
