// GPU SIFT 的主机调度，构造高斯金字塔布局并驱动金字塔、DoG、极值、方向及描述子阶段。
// 设备通过原子计数追加列表，阶段间回读数量决定后续分派；参数参考 COLMAP，金字塔常量必须与 sift.slang 一致。
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/SubmitBudget.h"
#include "sfm/core/Log.h"
#include "i18n/catalog/Sfm.h"
#include "sfm/core/Features.h"
#include "sfm/core/Image.h"
#include "sfm/vk/EmbeddedSpirv.h"
#include "sfm/vk/VkContext.h"

namespace sfm {

struct SiftOptions {
    int max_num_features = 8192;      // 按尺度保留 top-K，与 COLMAP 一致
    int num_octaves = 4;
    int max_num_orientations = 2;
    double peak_threshold = 0.02 / 3.0;
    double edge_threshold = 10.0;
    int device = -1;
    // 规范 uuid:<hex> 优先于序号，空值沿用共享设备选择。
    std::string device_selector;
    bool profile = false;
    bool verbose = true;
    std::string spv_path;             // 覆盖嵌入的 sift 模块
    // 下列设备列表容量饱和时发出警告。
    uint32_t max_raw_keypoints = 262144;
    uint32_t max_oriented_keypoints = 262144;
};

class SiftExtractor {
public:
    // 金字塔常量必须与 sift.slang 同步。
    static constexpr int S = 3;
    static constexpr int GAUSS_PER_OCT = S + 3;  // 6
    static constexpr int DOG_PER_OCT = S + 2;    // 5
    static constexpr float SIGMA0 = 1.6f;
    static constexpr int FIRST_OCTAVE = -1;
    static constexpr int KP_STRIDE = 6;
    static constexpr int OKP_STRIDE = 8;
    static constexpr int kNumBins = 2048;  // top-K 使用 log2(scale) 直方图

    explicit SiftExtractor(const SiftOptions& opt) : opt_(opt) {
        VkContextOptions vo;
        vo.selector = opt.device_selector;
        vo.deviceIndex = opt.device;
        vo.profile = opt.profile;
        ctx_.init(vo);
    }

    VkContext& ctx() { return ctx_; }

    // 首次创建缓冲与流水线，后续复用；遇到更大图像扩展尺寸相关缓冲并重绑描述符，批量按最大图优先可只分配一次。
    FeatureSet extract(const GrayImage& img) {
        if (img.width < 4 || img.height < 4)
            throw std::runtime_error("image too small for SIFT");
        planPyramid(img.width, img.height);
        ensureAllocated();
        uploadInputs(img);
        runPyramid();
        uint32_t nkp = runExtrema();
        uint32_t nokp = runOrient(nkp);
        uint32_t nsel = runSelect(nokp);
        runDescriptor(nsel);
        FeatureSet fs = readback(img.width, img.height, nsel);
        if (opt_.profile) ctx_.printProfile();
        return fs;
    }

private:
    // ---------------- 金字塔布局 ----------------
    struct Level { uint32_t off, w, h; };

    void planPyramid(int w0, int h0) {
        W0_ = w0 * 2;  // first_octave = -1
        H0_ = h0 * 2;
        planKey_ = {W0_, H0_};
        int O = 0;
        while (O < opt_.num_octaves && (W0_ >> O) >= 8 && (H0_ >> O) >= 8) O++;
        octaves_ = std::max(1, O);

        gLevels_.clear();
        dLevels_.clear();
        uint32_t goff = 0, doff = 0;
        for (int o = 0; o < octaves_; o++) {
            uint32_t w = (uint32_t)(W0_ >> o), h = (uint32_t)(H0_ >> o);
            for (int s = 0; s < GAUSS_PER_OCT; s++) {
                gLevels_.push_back({goff, w, h});
                goff += w * h;
            }
            for (int d = 0; d < DOG_PER_OCT; d++) {
                dLevels_.push_back({doff, w, h});
                doff += w * h;
            }
        }
        gaussFloats_ = goff;
        dogFloats_ = doff;

        // 第 0 步将上采样图像从 octave-0 的 sigma=1.0 模糊至 SIGMA0；后续为各 octave 相同的增量模糊核。
        weights_.clear();
        stepOff_.clear();
        stepRad_.clear();
        auto addKernel = [&](double dsigma) {
            int r = std::max(1, (int)std::ceil(4.0 * dsigma));
            stepOff_.push_back((uint32_t)weights_.size());
            stepRad_.push_back((uint32_t)r);
            double sum = 0;
            std::vector<double> k(2 * r + 1);
            for (int i = -r; i <= r; i++) {
                double v = std::exp(-0.5 * (i / dsigma) * (i / dsigma));
                k[i + r] = v;
                sum += v;
            }
            for (double v : k) weights_.push_back((float)(v / sum));
        };
        double sigmaNominal = 1.0;  // 上采样后为 0.5 * 2，以 octave-0 像素计
        addKernel(std::sqrt(std::max(SIGMA0 * SIGMA0 - sigmaNominal * sigmaNominal, 0.01)));
        for (int s = 1; s < GAUSS_PER_OCT; s++) {
            double sp = SIGMA0 * std::pow(2.0, (s - 1) / (double)S);
            double sc = SIGMA0 * std::pow(2.0, s / (double)S);
            addKernel(std::sqrt(sc * sc - sp * sp));
        }
    }

    const Level& gL(int o, int s) const { return gLevels_[o * GAUSS_PER_OCT + s]; }
    const Level& dL(int o, int d) const { return dLevels_[o * DOG_PER_OCT + d]; }

    // ---------------- 分配与描述符 ----------------
    // 仅首次或更大图像时扩展尺寸相关缓冲，关键点列表等按选项固定；最大图优先使每批只需一次分配。
    void ensureAllocated() {
        size_t needImg = (size_t)(W0_ / 2) * (H0_ / 2);
        if (setup_ && gaussFloats_ <= capGauss_ && dogFloats_ <= capDog_ && needImg <= capImg_ &&
            (size_t)W0_ * H0_ <= capTmp_ && gLevels_.size() <= capGlev_ &&
            dLevels_.size() <= capDlev_ && weights_.size() <= capW_)
            return;
        allocate();
        setup_ = true;
        capGauss_ = gaussFloats_;
        capDog_ = dogFloats_;
        capImg_ = needImg;
        capTmp_ = (size_t)W0_ * H0_;
        capGlev_ = gLevels_.size();
        capDlev_ = dLevels_.size();
        capW_ = weights_.size();
    }

    void allocate() {
        plannedOnce_ = false;  // 新缓冲为空，需重新上传布局表
        bImg_ = ctx_.createBuffer((VkDeviceSize)(W0_ / 2) * (H0_ / 2) * 4);
        bGauss_ = ctx_.createBuffer((VkDeviceSize)gaussFloats_ * 4);
        bDog_ = ctx_.createBuffer((VkDeviceSize)dogFloats_ * 4);
        bTmp_ = ctx_.createBuffer((VkDeviceSize)W0_ * H0_ * 4);
        bWeights_ = ctx_.createBuffer(std::max<VkDeviceSize>(16, weights_.size() * 4));
        bGlev_ = ctx_.createBuffer((VkDeviceSize)gLevels_.size() * 16);
        bDlev_ = ctx_.createBuffer((VkDeviceSize)dLevels_.size() * 16);
        bKp_ = ctx_.createBuffer((VkDeviceSize)opt_.max_raw_keypoints * KP_STRIDE * 4);
        bKpCnt_ = ctx_.createBuffer(16);
        bOkp_ = ctx_.createBuffer((VkDeviceSize)opt_.max_oriented_keypoints * OKP_STRIDE * 4);
        bOkpCnt_ = ctx_.createBuffer(16);
        bDesc_ = ctx_.createBuffer((VkDeviceSize)opt_.max_oriented_keypoints * 32 * 4);
        bHist_ = ctx_.createBuffer((VkDeviceSize)kNumBins * 4);
        bFokp_ = ctx_.createBuffer((VkDeviceSize)opt_.max_oriented_keypoints * OKP_STRIDE * 4);
        bSelCnt_ = ctx_.createBuffer(16);

        ctx_.createDescriptors({bImg_.buf, bGauss_.buf, bDog_.buf, bTmp_.buf, bWeights_.buf,
                                bGlev_.buf, bDlev_.buf, bKp_.buf, bKpCnt_.buf, bOkp_.buf,
                                bOkpCnt_.buf, bDesc_.buf, bHist_.buf, bFokp_.buf, bSelCnt_.buf});

        // 流水线仅依赖固定描述符布局，缓冲扩展重绑定后仍兼容，只需加载一次。
        if (!pipelinesLoaded_) {
            size_t words = 0;
            if (!opt_.spv_path.empty()) {
                ctx_.loadPipelines(opt_.spv_path, kEntries());
            } else {
                const uint32_t* code = findSpirv("sift", &words);
                if (!code) throw std::runtime_error("sift shader not built into this binary");
                ctx_.loadPipelines(code, words * 4, kEntries());
            }
            pipelinesLoaded_ = true;
        }
    }

    static std::vector<std::string> kEntries() {
        return {"upsample", "blur_h", "blur_v", "downsample",   "dog_diff",  "extrema",
                "orient",   "scale_hist", "select_topk", "descriptor"};
    }

    // 图像每次变化，模糊权重与层表仅在金字塔布局改变时上传，避免每图三次无效栅栏往返。
    void uploadInputs(const GrayImage& img) {
        ctx_.upload(bImg_, img.data.data(), img.data.size() * 4);
        if (planKey_ == lastPlanKey_ && plannedOnce_) return;
        ctx_.upload(bWeights_, weights_.data(), weights_.size() * 4);
        std::vector<uint32_t> gt(gLevels_.size() * 4), dt(dLevels_.size() * 4);
        for (size_t i = 0; i < gLevels_.size(); i++) {
            gt[4 * i + 0] = gLevels_[i].off;
            gt[4 * i + 1] = gLevels_[i].w;
            gt[4 * i + 2] = gLevels_[i].h;
            gt[4 * i + 3] = 0;
        }
        for (size_t i = 0; i < dLevels_.size(); i++) {
            dt[4 * i + 0] = dLevels_[i].off;
            dt[4 * i + 1] = dLevels_[i].w;
            dt[4 * i + 2] = dLevels_[i].h;
            dt[4 * i + 3] = 0;
        }
        ctx_.upload(bGlev_, gt.data(), gt.size() * 4);
        ctx_.upload(bDlev_, dt.data(), dt.size() * 4);
        lastPlanKey_ = planKey_;
        plannedOnce_ = true;
    }

    // ---------------- 分派辅助函数 ----------------
    static Push pk(uint32_t a, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0, uint32_t e = 0,
                   uint32_t f = 0, uint32_t g = 0, uint32_t h = 0) {
        Push p;
        uint32_t v[8] = {a, b, c, d, e, f, g, h};
        std::memcpy(&p, v, sizeof v);
        return p;
    }
    static uint32_t fbits(double x) {
        float xf = (float)x;
        uint32_t u;
        std::memcpy(&u, &xf, 4);
        return u;
    }
    static uint32_t grid(uint32_t n, uint32_t local) { return (n + local - 1) / local; }

    void img2d(VkCommandBuffer cb, const char* name, uint32_t w, uint32_t h, const Push& p) {
        ctx_.dispatch(cb, name, grid(w, 16), p, grid(h, 16));
    }

    // 双计算单元 RADV 集显上 3200 px 金字塔耗时 0.83 s，5000 px 会超过 2 s 看门狗，须按预算分段提交。
    void runPyramid() {
        VkCommandBuffer cb = ctx_.begin();
        for (int o = 0; o < octaves_; o++) {
            const Level& g0 = gL(o, 0);
            if (o == 0) {
                // 原图上采样到 gauss(0,0)，再原地模糊至 SIGMA0
                img2d(cb, "upsample", g0.w, g0.h,
                      pk(g0.off, (uint32_t)(W0_ / 2), (uint32_t)(H0_ / 2), g0.w, g0.h));
                ctx_.barrier(cb);
                blur(cb, g0.off, g0.off, g0.w, g0.h, 0);
            } else {
                const Level& prevTop = gL(o - 1, S);
                img2d(cb, "downsample", g0.w, g0.h,
                      pk(prevTop.off, g0.off, prevTop.w, g0.w, g0.h));
                ctx_.barrier(cb);
            }
            for (int s = 1; s < GAUSS_PER_OCT; s++) {
                const Level& a = gL(o, s - 1);
                const Level& b = gL(o, s);
                blur(cb, a.off, b.off, b.w, b.h, s);
            }
            for (int d = 0; d < DOG_PER_OCT; d++) {
                const Level& lo = gL(o, d);
                const Level& hi = gL(o, d + 1);
                const Level& dd = dL(o, d);
                img2d(cb, "dog_diff", dd.w, dd.h, pk(lo.off, hi.off, dd.off, dd.w, dd.h));
                pyramidWork_ += (double)dd.w * dd.h;
            }
            ctx_.barrier(cb);
        }
        submitTimed(cb, pyramidBudget_, pyramidWork_);
    }

    // 使用 step 核执行可分离模糊，源缓冲不能与临时缓冲混用
    void blur(VkCommandBuffer& cb, uint32_t srcOff, uint32_t dstOff, uint32_t w, uint32_t h,
              int step) {
        uint32_t woff = stepOff_[step], r = stepRad_[step];
        img2d(cb, "blur_h", w, h, pk(srcOff, w, h, r, woff));
        ctx_.barrier(cb);
        img2d(cb, "blur_v", w, h, pk(dstOff, w, h, r, woff));
        ctx_.barrier(cb);
        pyramidWork_ += 2.0 * w * h * (2 * r + 1);
        if (pyramidWork_ >= pyramidBudget_.limit()) {
            submitTimed(cb, pyramidBudget_, pyramidWork_);
            cb = ctx_.begin();
        }
    }

    void submitTimed(VkCommandBuffer cb, spirula::SubmitBudget& budget, double& work) {
        const auto t0 = std::chrono::steady_clock::now();
        ctx_.submit(cb);
        budget.record(work, std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count());
        work = 0;
    }

    uint32_t runExtrema() {
        VkCommandBuffer cb = ctx_.begin();
        ctx_.fillZero(cb, bKpCnt_);
        ctx_.barrier(cb);
        for (int o = 0; o < octaves_; o++) {
            const Level& base = dL(o, 0);
            for (int d = 1; d <= S; d++)
                img2d(cb, "extrema", base.w, base.h,
                      pk((uint32_t)o, (uint32_t)d, base.off, base.w, base.h,
                         opt_.max_raw_keypoints, fbits(opt_.peak_threshold),
                         fbits(opt_.edge_threshold)));
        }
        ctx_.submit(cb);
        uint32_t n = 0;
        ctx_.download(bKpCnt_, &n, 4);
        if (n > opt_.max_raw_keypoints) {
            if (opt_.verbose)
                slog::warn(slog::Tag::Extract, spirula::i18n::msg::sfm::sift_saturated_raw,
                           {(long long)n, (long long)opt_.max_raw_keypoints});
            n = opt_.max_raw_keypoints;
        }
        if (opt_.verbose)
            slog::err(slog::Tag::Extract, spirula::i18n::msg::sfm::sift_raw,
                      {(long long)octaves_, (long long)n});
        return n;
    }

    // 方向与描述子每关键点一线程；双计算单元集显单图曾分别耗时 392/222 ms，因此按关键点范围分段。
    uint32_t runOrient(uint32_t nkp) {
        VkCommandBuffer cb = ctx_.begin();
        ctx_.fillZero(cb, bOkpCnt_);
        ctx_.barrier(cb);
        for (uint32_t k0 = 0;;) {
            const uint32_t n = (uint32_t)std::min<int64_t>(nkp - k0, orientBudget_.chunk(4096, nkp));
            if (n > 0)
                ctx_.dispatch(cb, "orient", grid(n, 64),
                              pk(k0 + n, (uint32_t)opt_.max_num_orientations,
                                 opt_.max_oriented_keypoints, k0));
            double work = n;
            submitTimed(cb, orientBudget_, work);
            k0 += n;
            if (k0 >= nkp) break;
            cb = ctx_.begin();
        }
        uint32_t n = 0;
        ctx_.download(bOkpCnt_, &n, 4);
        if (n > opt_.max_oriented_keypoints) {
            if (opt_.verbose)
                slog::warn(slog::Tag::Extract, spirula::i18n::msg::sfm::sift_saturated_oriented,
                           {(long long)n, (long long)opt_.max_oriented_keypoints});
            n = opt_.max_oriented_keypoints;
        }
        return n;
    }

    // GPU 用尺度直方图选阈值并压缩到 fokp，仅为保留点计算描述子；末桶可略超过 K，主机随后精确截断。
    uint32_t runSelect(uint32_t nokp) {
        if (nokp == 0) return 0;
        uint32_t K = (uint32_t)opt_.max_num_features;
        float threshold = 0.0f;  // 尺度严格为正，保留全部

        if (opt_.max_num_features > 0 && nokp > K) {
            // 由金字塔几何确定 log2(scale) 直方图范围。
            double lo = 0.5 * SIGMA0 * std::exp2((double)FIRST_OCTAVE);
            double hi = SIGMA0 * std::exp2((S + 1.0) / S) *
                        std::exp2((double)(octaves_ - 1 + FIRST_OCTAVE)) * 1.5;
            double logMin = std::log2(lo), logMax = std::log2(hi);
            double invRange = 1.0 / (logMax - logMin);

            VkCommandBuffer cb = ctx_.begin();
            ctx_.fillZero(cb, bHist_);
            ctx_.barrier(cb);
            ctx_.dispatch(cb, "scale_hist", grid(nokp, 64),
                          pk(nokp, kNumBins, fbits(logMin), fbits(invRange)));
            ctx_.submit(cb);

            std::vector<uint32_t> hist(kNumBins);
            ctx_.download(bHist_, hist.data(), hist.size() * 4);
            // 从大尺度端累计到至少 K，保留最后一个桶，使超量最小且不会误选为空。
            uint64_t run = 0;
            int thrBin = 0;
            for (int b = kNumBins - 1; b >= 0; b--) {
                if (run >= K) break;
                run += hist[b];
                thrBin = b;
            }
            threshold = (float)std::exp2(logMin + (double)thrBin / kNumBins * (logMax - logMin));
        }

        VkCommandBuffer cb = ctx_.begin();
        ctx_.fillZero(cb, bSelCnt_);
        ctx_.barrier(cb);
        ctx_.dispatch(cb, "select_topk", grid(nokp, 64),
                      pk(nokp, fbits((double)threshold), opt_.max_oriented_keypoints));
        ctx_.submit(cb);
        uint32_t nsel = 0;
        ctx_.download(bSelCnt_, &nsel, 4);
        nsel = std::min(nsel, opt_.max_oriented_keypoints);
        if (opt_.verbose)
            slog::err(slog::Tag::Extract, spirula::i18n::msg::sfm::sift_selected,
                      {(long long)nokp, (long long)nsel, slog::num(threshold, 3)});
        return nsel;
    }

    void runDescriptor(uint32_t nsel) {
        for (uint32_t k0 = 0; k0 < nsel;) {
            const uint32_t n = (uint32_t)std::min<int64_t>(nsel - k0, descBudget_.chunk(1024, nsel));
            VkCommandBuffer cb = ctx_.begin();
            ctx_.dispatch(cb, "descriptor", grid(n, 64), pk(k0 + n, k0));
            double work = n;
            submitTimed(cb, descBudget_, work);
            k0 += n;
        }
    }

    FeatureSet readback(int w0, int h0, uint32_t nsel) {
        FeatureSet fs;
        fs.width = w0;
        fs.height = h0;
        fs.dim = 128;
        fs.dtype = DType::U8;
        if (nsel == 0) return fs;

        std::vector<float> okp((size_t)nsel * OKP_STRIDE);
        ctx_.download(bFokp_, okp.data(), okp.size() * 4);
        std::vector<uint8_t> desc((size_t)nsel * 128);
        ctx_.download(bDesc_, desc.data(), desc.size());

        std::vector<Keypoint> kps(nsel);
        for (uint32_t i = 0; i < nsel; i++) {
            const float* p = &okp[(size_t)i * OKP_STRIDE];
            kps[i] = {p[0], p[1], p[2], p[3], 0.0f};
        }

        // 主机按尺度精确截断，再按位置输出；两次排序都必须为全序，以消除 GPU 原子追加顺序的不确定性（D16）。
        // 不能按尺度编号，否则下游按索引打破平局时会偏向大尺度、定位较差的特征。
        auto byScale = [&](uint32_t a, uint32_t b) {
            const Keypoint& p = kps[a];
            const Keypoint& q = kps[b];
            if (p.scale != q.scale) return p.scale > q.scale;
            if (p.x != q.x) return p.x < q.x;
            if (p.y != q.y) return p.y < q.y;
            return p.orientation < q.orientation;
        };
        auto byPosition = [&](uint32_t a, uint32_t b) {
            const Keypoint& p = kps[a];
            const Keypoint& q = kps[b];
            if (p.x != q.x) return p.x < q.x;
            if (p.y != q.y) return p.y < q.y;
            if (p.scale != q.scale) return p.scale > q.scale;
            return p.orientation < q.orientation;
        };
        std::vector<uint32_t> idx(nsel);
        for (uint32_t i = 0; i < nsel; i++) idx[i] = i;
        uint32_t keep = nsel;
        if (opt_.max_num_features > 0 && nsel > (uint32_t)opt_.max_num_features) {
            keep = (uint32_t)opt_.max_num_features;
            std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), byScale);
            idx.resize(keep);
        }
        std::sort(idx.begin(), idx.end(), byPosition);

        fs.keypoints.resize(keep);
        fs.descriptors.resize((size_t)keep * 128);
        for (uint32_t i = 0; i < keep; i++) {
            fs.keypoints[i] = kps[idx[i]];
            std::memcpy(&fs.descriptors[(size_t)i * 128], &desc[(size_t)idx[i] * 128], 128);
        }
        if (opt_.verbose)
            slog::err(slog::Tag::Extract, spirula::i18n::msg::sfm::sift_features,
                      {(long long)keep});
        return fs;
    }

    SiftOptions opt_;
    VkContext ctx_;
    // 金字塔工作量按像素数×核采样数估计，方向与描述子按关键点数估计。
    spirula::SubmitBudget pyramidBudget_, orientBudget_, descBudget_;
    double pyramidWork_ = 0;
    int W0_ = 0, H0_ = 0, octaves_ = 0;
    uint32_t gaussFloats_ = 0, dogFloats_ = 0;
    std::vector<Level> gLevels_, dLevels_;
    std::vector<float> weights_;
    std::vector<uint32_t> stepOff_, stepRad_;

    GpuBuffer bImg_, bGauss_, bDog_, bTmp_, bWeights_, bGlev_, bDlev_;
    GpuBuffer bKp_, bKpCnt_, bOkp_, bOkpCnt_, bDesc_;
    GpuBuffer bHist_, bFokp_, bSelCnt_;

    bool setup_ = false, pipelinesLoaded_ = false;
    // 当前设备权重和层表对应的金字塔布局。
    std::pair<int, int> planKey_{0, 0}, lastPlanKey_{-1, -1};
    bool plannedOnce_ = false;
    size_t capGauss_ = 0, capDog_ = 0, capImg_ = 0, capTmp_ = 0, capGlev_ = 0, capDlev_ = 0,
           capW_ = 0;
};

}  // 命名空间 sfm
