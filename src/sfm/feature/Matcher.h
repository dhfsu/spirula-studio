// 可替换特征匹配接口与 GPU 暴力实现；输入两组特征，输出筛选后的候选对应，不在接口层限定描述子类型或维度。
// 图像配对策略独立定义，可与任一匹配器组合。
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/SubmitBudget.h"
#include "sfm/core/Features.h"
#include "sfm/core/Matches.h"
#include "sfm/vk/EmbeddedSpirv.h"
#include "sfm/vk/VkContext.h"

namespace sfm {

struct MatchOptions {
    float max_ratio = 0.8f;     // Lowe 比值：best_dist < ratio*second_dist
    // 单位浮点描述子还可设绝对余弦阈值，0 禁用；独立保留两选项，以表达仅用余弦筛选等策略。
    float min_similarity = 0.0f;
    bool cross_check = true;    // 仅保留互为最近邻的对应
    uint32_t max_num_matches = 32768;  // 每图像对匹配上限，0 不限制
    int device = -1;
    // 规范 uuid:<hex> 优先于序号，空值沿用共享选择规则。
    std::string device_selector;
    // 每 GPU 提交的图像对数；一批共用命令缓冲与结果回读，将栅栏往返开销摊到整批（D22）。
    int batch_pairs = 64;
    // 常驻描述子显存预算；每图 8192 特征约 1 MB，默认约容纳 1500 图，超出后整块回收并按批重新上传。
    size_t descriptor_budget_bytes = 1536ull << 20;
};

struct IFeatureMatcher {
    virtual ~IFeatureMatcher() = default;
    // 候选匹配索引分别指向 a/b 的关键点。
    virtual std::vector<FeatureMatch> match(const FeatureSet& a, const FeatureSet& b) = 0;
    virtual const char* name() const = 0;

    // 批量匹配 pairs[begin..end)，按图像对顺序追加结果；GPU 实现可合并上传与提交，默认逐对调用。
    virtual void matchBatch(const std::vector<FeatureSet>& feats,
                            const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t begin,
                            size_t end, std::vector<std::vector<FeatureMatch>>& out) {
        out.resize(end - begin);
        for (size_t k = begin; k < end; k++)
            out[k - begin] = match(feats[pairs[k].first], feats[pairs[k].second]);
    }
};

// GPU 暴力匹配器将多图描述子常驻设备，一批图像对共用命令缓冲。
// 100 图穷举的上传量约 100 MB，而逐对上传为 9.9 GB；距离使用精确的打包 uint8x4 点积。
class BruteForceMatcher : public IFeatureMatcher {
public:
    explicit BruteForceMatcher(const MatchOptions& opt = {}) : opt_(opt) {
        // 硬件支持时使用 DP4A，否则采用整数等价展开；Intel Gen11/12 缺失、Apple 可能仅模拟。
        dot4_ = VkContext::probeCaps(deviceOnlyOpt(opt.device, opt.device_selector))
                    .intDotProductFast;
        // SS_SFM_NO_DOT4=1 可在支持设备上强制回退，验证整数结果一致。
        if (spirula::env_on("SFM_NO_DOT4")) dot4_ = false;
        VkContextOptions vo;
        vo.selector = opt.device_selector;
        vo.deviceIndex = opt.device;
        vo.needIntDotProduct = dot4_;
        ctx_.init(vo);
    }

    const char* name() const override { return "brute-force"; }

    // 供自测和单次图像对使用的双图会话。
    std::vector<FeatureMatch> match(const FeatureSet& a, const FeatureSet& b) override {
        std::vector<FeatureSet> two;  // 此非热点路径允许复制
        two.push_back(a);
        two.push_back(b);
        std::vector<std::pair<uint32_t, uint32_t>> one{{0u, 1u}};
        std::vector<std::vector<FeatureMatch>> out;
        matchBatch(two, one, 0, 1, out);
        return out.empty() ? std::vector<FeatureMatch>() : std::move(out[0]);
    }

    void matchBatch(const std::vector<FeatureSet>& feats,
                    const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t begin,
                    size_t end, std::vector<std::vector<FeatureMatch>>& out) override {
        matchBatch(asView(feats), pairs, begin, end, out);
    }

    // 以指针视图引用来自不同数组的特征，避免拼接查询子集与完整特征时复制描述子；1200 图数据曾因此额外占用约 1 GB。
    void matchBatch(const std::vector<const FeatureSet*>& feats,
                    const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t begin,
                    size_t end, std::vector<std::vector<FeatureMatch>>& out) {
        out.assign(end - begin, {});
        if (!prepare(feats, pairs, begin, end)) return;
        for (size_t b = begin; b < end;) {
            size_t e = chunkEnd(feats, pairs, b, end);
            matchChunk(feats, pairs, b, e, &out, nullptr, begin);
            b = e;
        }
    }

    // 仅统计匹配数而不构造列表，供数十万候选对评分，避免为取 size 创建并丢弃大量向量。
    void countBatch(const std::vector<const FeatureSet*>& feats,
                    const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t begin,
                    size_t end, std::vector<uint32_t>& out) {
        out.assign(end - begin, 0u);
        if (!prepare(feats, pairs, begin, end)) return;
        for (size_t b = begin; b < end;) {
            size_t e = chunkEnd(feats, pairs, b, end);
            matchChunk(feats, pairs, b, e, nullptr, &out, begin);
            b = e;
        }
    }

private:
    // 每次将连续数组包装为指针视图，O(n) 指针写入相对 GPU 批量开销很小。
    const std::vector<const FeatureSet*>& asView(const std::vector<FeatureSet>& feats) {
        view_.resize(feats.size());
        for (size_t i = 0; i < feats.size(); i++) view_[i] = &feats[i];
        return view_;
    }

    // 共享输入检查与首次定尺寸，false 表示无需处理。
    bool prepare(const std::vector<const FeatureSet*>& feats,
                 const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t begin,
                 size_t end) {
        if (begin >= end) return false;
        for (size_t k = begin; k < end; k++) {
            checkDescriptors(*feats[pairs[k].first]);
            checkDescriptors(*feats[pairs[k].second]);
        }
        ensureSetup(feats);
        return true;
    }

    // 按结果字节数与耗时分块；双计算单元 RADV 集显执行 64 对 8192² 交叉匹配约 2 s，已达看门狗限制，每块至少一对。
    size_t chunkEnd(const std::vector<const FeatureSet*>& feats,
                    const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t b,
                    size_t end) {
        const uint64_t resCap =
            std::min<uint64_t>(resultCap_, VkContext::stagingCapacity() / 16);
        const bool cc = opt_.cross_check;
        if (stamp_.size() < feats.size()) stamp_.assign(feats.size(), 0);
        ++epoch_;
        const double workCap = budget_.limit();
        uint64_t res = 0, desc = used_;
        double work = 0;
        size_t e = b;
        for (; e < end; e++) {
            const uint32_t ia = pairs[e].first, ib = pairs[e].second;
            const uint32_t na = feats[ia]->count(), nb = feats[ib]->count();
            const uint64_t addRes = (na && nb) ? (uint64_t)na + (cc ? nb : 0) : 0;
            const double addWork = pairWork(na, nb);
            uint64_t addDesc = 0;
            for (uint32_t img : {ia, ib})
                if (stamp_[img] != epoch_) {
                    stamp_[img] = epoch_;
                    if (!resident_.count(img)) addDesc += feats[img]->count();
                }
            if (e > b && (res + addRes > resCap || desc + addDesc > descCap_ ||
                          work + addWork > workCap))
                break;
            res += addRes;
            desc += addDesc;
            work += addWork;
        }
        return e;
    }

    // 按单对需计算的描述子字数估计 GPU 工作量，作为 SubmitBudget 单位。
    double pairWork(uint32_t na, uint32_t nb) const {
        return (double)na * nb * desc_words_;
    }

    // out 与 counts 必须恰有一个非空。
    void matchChunk(const std::vector<const FeatureSet*>& feats,
                    const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t begin,
                    size_t end, std::vector<std::vector<FeatureMatch>>* out,
                    std::vector<uint32_t>* counts, size_t outBase) {
        ensureResident(feats, pairs, begin, end);

        // 各分派写 nQuery 个 uint4，整批一次回读；无交叉检查时无需列归约或训练侧结果槽位。
        const bool cc = opt_.cross_check;
        struct Slot { uint32_t a, b, na, nb; uint32_t oa, ob; };
        std::vector<Slot> slots;
        slots.reserve(end - begin);
        uint64_t outCount = 0;
        double work = 0;
        for (size_t k = begin; k < end; k++) {
            uint32_t ia = pairs[k].first, ib = pairs[k].second;
            uint32_t na = feats[ia]->count(), nb = feats[ib]->count();
            Slot s{ia, ib, na, nb, (uint32_t)outCount, (uint32_t)(outCount + na)};
            if (na == 0 || nb == 0) s.na = s.nb = 0;  // 没有待分派任务
            else outCount += (uint64_t)na + (cc ? nb : 0);
            work += pairWork(s.na, s.nb);
            slots.push_back(s);
        }
        if (outCount == 0) return;
        if (outCount > resultCap_)  // 分块应保证此界限，触发则是内部错误
            throw std::runtime_error("match result buffer too small for a chunk");

        VkCommandBuffer cb = ctx_.begin();
        if (normDirty_.first != normDirty_.second) {
            Push p;
            p.u0 = normDirty_.first;
            p.u1 = normDirty_.second - normDirty_.first;
            ctx_.dispatch(cb, "descriptor_norms", (p.u1 + 63) / 64, p);
            ctx_.barrier(cb);
            normDirty_ = {0, 0};
        }
        // 每对先计算距离矩阵再归约列候选，两者及相邻图像对共用 colPartial，必须以屏障隔开。
        for (const Slot& s : slots) {
            if (s.na == 0 || s.nb == 0) continue;
            // 交叉检查内核宽 64，行内核使用 kRowThreads；reduce_cols 读取的 u6 仍按 64 宽计算。
            uint32_t numWG = (s.na + 63) / 64;
            Push p;
            p.u0 = resident_[s.a];
            p.u1 = s.na;
            p.u2 = resident_[s.b];
            p.u3 = s.nb;
            p.u4 = s.oa;
            p.u5 = s.ob;
            p.u6 = numWG;
            if (cc) {
                // colPartial 跨图像对复用，当前矩阵与列归约完成后才能开始下一对。
                ctx_.dispatch(cb, "match_pair", numWG, p);
                ctx_.barrier(cb);
                ctx_.dispatch(cb, "reduce_cols", (s.nb + 63) / 64, p);
                ctx_.barrier(cb);
            } else {
                // 仅行匹配写入互斥结果区，批内图像对无数据依赖，可不加屏障，避免大量小评分分派被流水线排空串行化。
                ctx_.dispatch(cb, "match_rows", (s.na + kRowThreads - 1) / kRowThreads, p);
            }
        }
        // 分派与回读合并到同一命令缓冲，每块只需一次提交和栅栏；直接读映射暂存区，过大时回退分块下载。
        const VkDeviceSize bytes = (VkDeviceSize)outCount * 16;
        const bool fused = bytes <= VkContext::stagingCapacity();
        if (fused) {
            ctx_.barrier(cb);
            ctx_.recordDownload(cb, bResult_, bytes);
        }
        const auto t0 = std::chrono::steady_clock::now();
        ctx_.submit(cb);
        budget_.record(work, std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0).count());

        const uint32_t* res;
        if (fused) {
            res = (const uint32_t*)ctx_.stagingDownloadPtr();
        } else {
            res_.resize((size_t)outCount * 4);
            ctx_.download(bResult_, res_.data(), bytes);
            res = res_.data();
        }

        for (size_t k = 0; k < slots.size(); k++) {
            const Slot& s = slots[k];
            if (s.na == 0 || s.nb == 0) continue;
            const uint32_t* rA = res + (size_t)s.oa * 4;
            const uint32_t* rB = res + (size_t)s.ob * 4;
            if (out) (*out)[begin - outBase + k] = reduce(rA, rB, s.na, s.nb);
            else (*counts)[begin - outBase + k] = countMatches(rA, rB, s.na, s.nb);
        }
    }

    // 浮点描述子归一化后均匀仿射量化为 uint8，复用打包点积；共同缩放保持距离顺序及比值判据，绝对阈值另行换算。
    // ALIKED 分量约 N(0,0.088)，实测最大 0.392，取半量程 0.4 几乎不截断，量化残差约为单位描述子范数的 1%。
    static constexpr float kQuantHalfRange = 0.4f;
    static constexpr float kQuantSteps = 127.0f;

    static uint8_t quantize(float v) {
        const float q = v * (kQuantSteps / kQuantHalfRange) + 128.0f;
        return (uint8_t)std::lround(std::min(255.0f, std::max(0.0f, q)));
    }
    // 单位描述子满足 ||a-b||^2 = 2 - 2 cos，量化 d2 按 (steps/range)^2 缩放，据此恢复余弦相似度。
    static float similarityFromD2(uint32_t d2) {
        const float k = kQuantHalfRange / kQuantSteps;
        return 1.0f - 0.5f * (float)d2 * k * k;
    }

    // SIFT/ALIKED 每描述子 128 字节，DeDoDe-G 为 256；从首个非空 FeatureSet 确定，设备编译两种宽度。
    int descBytes() const { return desc_words_ * 4; }
    // 行内核工作组宽度须与 bruteforce.slang 的 TQR 一致；训练流量为 ceil(nQuery/宽度)*nTrain，误用 64 会产生四倍冗余工作组。
    static constexpr uint32_t kRowThreads = 256;

    // 每分量一个字节：SIFT 原生 uint8，浮点描述子上传时归一化并量化。
    void checkDescriptors(const FeatureSet& f) {
        if (f.count() == 0) return;
        if (desc_words_ == 0 && (f.dim == 128 || f.dim == 256))
            desc_words_ = (int)f.dim / 4;
        if (f.dim != 128 && f.dim != 256)
            throw std::runtime_error(
                "brute-force matcher expects 128-D or 256-D descriptors, got " +
                std::to_string(f.dim));
        if ((int)f.dim != desc_words_ * 4)
            throw std::runtime_error(
                "brute-force matcher was set up for " + std::to_string(desc_words_ * 4) +
                "-D descriptors and this image has " + std::to_string(f.dim) +
                "; one batch cannot mix widths");
        if (f.dtype != DType::U8 && f.dtype != DType::F32)
            throw std::runtime_error("brute-force matcher expects uint8 or f32 descriptors");
    }

    // 按 Lowe 比值与互近邻交叉检查判定匹配，GPU 仅负责距离计算。
    std::vector<FeatureMatch> reduce(const uint32_t* rA, const uint32_t* rB, uint32_t na,
                                     uint32_t nb) const {
        std::vector<FeatureMatch> out;
        const float r2 = opt_.max_ratio * opt_.max_ratio;
        for (uint32_t i = 0; i < na; i++) {
            uint32_t j = rA[4 * i + 0];
            uint32_t bestD2 = rA[4 * i + 1], secondD2 = rA[4 * i + 2];
            if (j >= nb) continue;
            if (secondD2 != 0xffffffffu && (float)bestD2 >= r2 * (float)secondD2) continue;
            if (opt_.min_similarity > 0 && similarityFromD2(bestD2) < opt_.min_similarity)
                continue;
            if (opt_.cross_check && rB[4 * j + 0] != i) continue;
            out.push_back({i, j, std::sqrt((float)bestD2)});
        }
        if (opt_.max_num_matches > 0 && out.size() > opt_.max_num_matches) {
            std::partial_sort(out.begin(), out.begin() + opt_.max_num_matches, out.end(),
                              [](const FeatureMatch& x, const FeatureMatch& y) {
                                  return x.distance < y.distance;
                              });
            out.resize(opt_.max_num_matches);
        }
        return out;
    }

    // 与 reduce 使用相同接受条件，仅统计数量；相邻放置以便保持一致。
    uint32_t countMatches(const uint32_t* rA, const uint32_t* rB, uint32_t na,
                          uint32_t nb) const {
        uint32_t n = 0;
        const float r2 = opt_.max_ratio * opt_.max_ratio;
        for (uint32_t i = 0; i < na; i++) {
            uint32_t j = rA[4 * i + 0];
            uint32_t bestD2 = rA[4 * i + 1], secondD2 = rA[4 * i + 2];
            if (j >= nb) continue;
            if (secondD2 != 0xffffffffu && (float)bestD2 >= r2 * (float)secondD2) continue;
            if (opt_.min_similarity > 0 && similarityFromD2(bestD2) < opt_.min_similarity)
                continue;
            if (opt_.cross_check && rB[4 * j + 0] != i) continue;
            n++;
        }
        if (opt_.max_num_matches > 0 && n > opt_.max_num_matches) n = opt_.max_num_matches;
        return n;
    }

    // 三个缓冲与描述符必须先于流水线创建，布局由 createDescriptors 建立；按完整特征集合一次定尺寸，避免活动流水线下重新分配。
    void ensureSetup(const std::vector<const FeatureSet*>& feats) {
        // 全部图像为空时仍需合法缓冲尺寸与除数，采用普通描述子宽度。
        if (desc_words_ == 0) desc_words_ = 32;
        uint64_t total = 0, maxCount = 0;
        for (const FeatureSet* f : feats) {
            total += f->count();
            maxCount = std::max<uint64_t>(maxCount, f->count());
        }
        if (setup_) {
            // 缓冲仅分配一次，一个匹配器对应一组容量；后续更大特征集合须明确拒绝，避免越界。
            if (maxCount > setupMaxCount_)
                throw std::runtime_error(
                    "matcher was sized for a smaller feature set; use a fresh BruteForceMatcher");
            return;
        }
        setupMaxCount_ = maxCount;

        // 单批最多引用 2*batch_pairs 张不同图像，常驻容量至少覆盖全部，不能由首次双图调用过度缩小。
        const uint64_t batchFloor =
            (uint64_t)std::max(1, opt_.batch_pairs) * 2 * std::max<uint64_t>(maxCount, 1);
        const uint64_t budget =
            std::max<uint64_t>(opt_.descriptor_budget_bytes / descBytes(), batchFloor);
        descCap_ = (uint32_t)std::min<uint64_t>(std::max<uint64_t>(total, batchFloor), budget);
        // 按每对两侧均达到最大特征数估计最坏结果容量。
        resultCap_ = (uint32_t)std::max<uint64_t>(
            1, (uint64_t)std::max(1, opt_.batch_pairs) * 2 * std::max<uint64_t>(maxCount, 1));
        // 列候选容量为 ceil(nA/64) × nB，跨对复用，仅按最大单对分配。
        const uint64_t mc = std::max<uint64_t>(maxCount, 1);
        uint64_t colCap = ((mc + 63) / 64) * mc;

        bDesc_ = ctx_.createBuffer((VkDeviceSize)descCap_ * descBytes());
        bNorm_ = ctx_.createBuffer((VkDeviceSize)descCap_ * 4);
        bResult_ = ctx_.createBuffer((VkDeviceSize)resultCap_ * 16);
        bCol_ = ctx_.createBuffer((VkDeviceSize)colCap * 4);
        ctx_.createDescriptors({bDesc_.buf, bNorm_.buf, bResult_.buf, bCol_.buf});

        size_t words = 0;
        const std::string blob = std::string(dot4_ ? "match" : "match_nodot") +
                                 (desc_words_ == 64 ? "_d256" : "");
        const uint32_t* code = findSpirv(blob.c_str(), &words);
        if (!code)
            throw std::runtime_error(blob + " shader not built into this binary");
        ctx_.loadPipelines(code, words * 4,
                           {"match_pair", "match_rows", "reduce_cols", "descriptor_norms"});
        setup_ = true;
    }

    // 使本批所有图像常驻，新图像描述子连续上传；容量不足时回收整块，穷举工作集覆盖全部数据，局部淘汰收益有限。
    void ensureResident(const std::vector<const FeatureSet*>& feats,
                        const std::vector<std::pair<uint32_t, uint32_t>>& pairs, size_t begin,
                        size_t end) {
        const uint32_t capDesc = descCap_;
        std::vector<uint32_t> need;
        for (size_t k = begin; k < end; k++)
            for (uint32_t img : {pairs[k].first, pairs[k].second})
                if (!resident_.count(img)) need.push_back(img);
        std::sort(need.begin(), need.end());
        need.erase(std::unique(need.begin(), need.end()), need.end());
        if (need.empty()) return;

        uint32_t want = 0;
        for (uint32_t img : need) want += feats[img]->count();
        if (used_ + want > capDesc) {  // 回收常驻块
            resident_.clear();
            used_ = 0;
            normDirty_ = {0, 0};
            need.clear();
            for (size_t k = begin; k < end; k++)
                for (uint32_t img : {pairs[k].first, pairs[k].second}) need.push_back(img);
            std::sort(need.begin(), need.end());
            need.erase(std::unique(need.begin(), need.end()), need.end());
            want = 0;
            for (uint32_t img : need) want += feats[img]->count();
            if (want > capDesc)
                throw std::runtime_error(  // ensureSetup 的 batchFloor 应排除此情况
                    "descriptor budget too small for one batch of pairs");
        }

        // 新图像描述子汇集为连续块，一次上传并一次计算范数，避免逐图提交。
        const uint32_t first = used_;
        const size_t dsz = (size_t)descBytes();
        blob_.resize((size_t)want * dsz);
        uint8_t* blob = blob_.data();
        size_t off = 0;
        for (uint32_t img : need) {
            const FeatureSet& f = *feats[img];
            resident_[img] = used_;
            used_ += f.count();
            if (!f.count()) continue;
            const size_t bytes = (size_t)f.count() * dsz;
            if (f.dtype == DType::U8) {
                memcpy(blob + off, f.descriptors.data(), bytes);
            } else {
                // 上传前执行 L2 归一化；ALIKED 已归一化，DeDoDe 未归一化，量化范围与余弦恢复均要求单位范数。
                const float* src = reinterpret_cast<const float*>(f.descriptors.data());
                for (uint32_t r = 0; r < f.count(); r++) {
                    const float* row = src + (size_t)r * dsz;
                    double sq = 0;
                    for (size_t i = 0; i < dsz; i++) sq += (double)row[i] * row[i];
                    const float inv = sq > 0 ? (float)(1.0 / std::sqrt(sq)) : 1.0f;
                    uint8_t* dst = blob + off + (size_t)r * dsz;
                    for (size_t i = 0; i < dsz; i++) dst[i] = quantize(row[i] * inv);
                }
            }
            off += bytes;
        }
        ctx_.upload(bDesc_, blob, (VkDeviceSize)want * dsz, (VkDeviceSize)first * dsz);
        normDirty_ = {first, used_};
    }

    MatchOptions opt_;
    spirula::SubmitBudget budget_;
    bool dot4_ = true;  // 设备支持 VK_KHR_shader_integer_dot_product
    VkContext ctx_;
    GpuBuffer bDesc_, bNorm_, bResult_, bCol_;
    bool setup_ = false;
    std::vector<const FeatureSet*> view_;      // vector<FeatureSet> 重载使用的指针临时数组
    std::vector<uint32_t> stamp_;             // chunkEnd 记录本块已出现图像的标记
    uint32_t epoch_ = 0;
    std::vector<uint32_t> res_;               // 跨块复用的下载缓冲
    std::vector<uint8_t> blob_;               // 跨块复用的上传缓冲
    std::map<uint32_t, uint32_t> resident_;   // 图像索引到首描述子索引的映射
    std::pair<uint32_t, uint32_t> normDirty_{0, 0};  // 待计算 ||d||^2 的描述子区间
    int desc_words_ = 0;                      // 从首个特征集确定，32 或 64
    uint32_t used_ = 0, descCap_ = 0, resultCap_ = 0;
    uint64_t setupMaxCount_ = 0;  // 缓冲支持的最大特征数量
};

}  // 命名空间 sfm
