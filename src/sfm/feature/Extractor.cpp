#include "sfm/core/Log.h"
#include "i18n/catalog/Sfm.h"
#include "sfm/feature/Extractor.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#if SS_HAVE_ALIKED
#include "aliked/Aliked.h"
#endif
#if SS_HAVE_LOMA
#include "loma/Loma.h"
#endif
#if SS_HAVE_ALIKED || SS_HAVE_LOMA
#include "nn/Device.h"
#endif

namespace sfm {
namespace {

#if SS_HAVE_ALIKED || SS_HAVE_LOMA
// 加载学习模型前，将解析出的 UUID 交给 NN；空值沿用 NN 自身的设备选择优先级。
inline void configureLearnedDevice(const std::string& selector) {
    if (!selector.empty()) nn::configure_device(selector);
}
#endif

class SiftFrontend : public IFeatureExtractor {
public:
    explicit SiftFrontend(const SiftOptions& opt) : ext_(opt) {}
    FeatureSet extract(const GrayImage& img) override { return ext_.extract(img); }
    const char* name() const override { return "sift"; }

private:
    SiftExtractor ext_;
};

#if SS_HAVE_ALIKED

// ALIKED 输入沿用加载器为点云着色提供的 RGB 字节，仅按 1/255 归一化。
// 检测器不提供尺度或方向，检测分数存入 FeatureSet::rank 和 features.bin v5；scale 保持 0，避免伪造尺度影响后续计算。
class AlikedFrontend : public IFeatureExtractor {
public:
    explicit AlikedFrontend(const AlikedOptions& opt) : opt_(opt) {
        configureLearnedDevice(opt.device_selector);
        ext_.load(opt.model);
        aopts_.max_num_features = opt.max_num_features;
        aopts_.min_score = (float)opt.min_score;
    }

    const char* name() const override { return "aliked"; }
    bool wantsColor() const override { return true; }

    FeatureSet extract(const GrayImage& img) override {
        FeatureSet fs;
        fs.width = img.width;
        fs.height = img.height;
        fs.dim = (uint32_t)ext_.descriptorDim();
        fs.dtype = DType::F32;
        if (!img.hasColor())
            throw std::runtime_error(
                "ALIKED needs a colour image; the loader decoded luma only");

        const aliked::Features f =
            ext_.extract(img.rgb.data(), img.width, img.height, aopts_);

        // 按位置采用与 GPU SIFT 相同的规范顺序（D16）；沿用提取器的分数排序会让匹配数量上限和首个三维对应点等平局判定偏向高分特征。
        std::vector<uint32_t> idx(f.keypoints.size());
        for (uint32_t i = 0; i < idx.size(); i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
            const aliked::Keypoint& p = f.keypoints[a];
            const aliked::Keypoint& q = f.keypoints[b];
            if (p.x != q.x) return p.x < q.x;
            if (p.y != q.y) return p.y < q.y;
            return p.score > q.score;
        });

        const size_t n = idx.size();
        fs.keypoints.resize(n);
        fs.descriptors.resize(n * fs.dim * sizeof(float));
        float* dst = reinterpret_cast<float*>(fs.descriptors.data());
        for (size_t i = 0; i < n; i++) {
            const aliked::Keypoint& k = f.keypoints[idx[i]];
            fs.keypoints[i] = {k.x, k.y, /*scale=*/0.0f, /*orientation=*/0.0f, k.score};
            std::memcpy(dst + i * fs.dim, &f.descriptors[(size_t)idx[i] * fs.dim],
                        fs.dim * sizeof(float));
        }
        if (opt_.verbose)
            slog::err(slog::Tag::Extract, spirula::i18n::msg::sfm::sift_features,
                      {(long long)n});
        return fs;
    }

private:
    AlikedOptions           opt_;
    aliked::Extractor       ext_;
    aliked::ExtractOptions  aopts_;
};

#endif  // SS_HAVE_ALIKED

#if SS_HAVE_LOMA

// LoMa 描述子因版本不同为 128 维或 256 维，且未归一化；余弦阈值需自行归一化，COLMAP 的 LOMA_BRUTEFORCE 也如此。
class LomaFrontend : public IFeatureExtractor {
public:
    explicit LomaFrontend(const LomaOptions& opt) : opt_(opt) {
        configureLearnedDevice(opt.device_selector);
        ext_.load(opt.detector_model.empty() ? "loma-dad" : opt.detector_model,
                  opt.descriptor_model);
        lopts_.max_num_features = opt.max_num_features;
        lopts_.min_score = (float)opt.min_score;
    }

    const char* name() const override { return "loma"; }
    bool wantsColor() const override { return true; }

    FeatureSet extract(const GrayImage& img) override {
        FeatureSet fs;
        fs.width = img.width;
        fs.height = img.height;
        fs.dim = (uint32_t)ext_.descriptorDim();
        fs.dtype = DType::F32;
        if (!img.hasColor())
            throw std::runtime_error(
                "LoMa needs a colour image; the loader decoded luma only");

        const loma::Features f =
            ext_.extract(img.rgb.data(), img.width, img.height, lopts_);

        // 按位置采用与 GPU SIFT 相同的规范顺序（D16）；沿用提取器的分数排序会使后续平局判定偏向高分特征。
        std::vector<uint32_t> idx(f.keypoints.size());
        for (uint32_t i = 0; i < idx.size(); i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
            const loma::Keypoint& p = f.keypoints[a];
            const loma::Keypoint& q = f.keypoints[b];
            if (p.x != q.x) return p.x < q.x;
            if (p.y != q.y) return p.y < q.y;
            return p.score > q.score;
        });

        const size_t n = idx.size();
        fs.keypoints.resize(n);
        fs.descriptors.resize(n * fs.dim * sizeof(float));
        float* dst = reinterpret_cast<float*>(fs.descriptors.data());
        for (size_t i = 0; i < n; i++) {
            const loma::Keypoint& k = f.keypoints[idx[i]];
            fs.keypoints[i] = {k.x, k.y, /*scale=*/0.0f, /*orientation=*/0.0f, k.score};
            std::memcpy(dst + i * fs.dim, &f.descriptors[(size_t)idx[i] * fs.dim],
                        fs.dim * sizeof(float));
        }
        if (opt_.verbose)
            slog::err(slog::Tag::Extract, spirula::i18n::msg::sfm::sift_features,
                      {(long long)n});
        return fs;
    }

private:
    LomaOptions          opt_;
    loma::Extractor      ext_;
    loma::ExtractOptions lopts_;
};

#endif  // SS_HAVE_LOMA

}  // 匿名命名空间

bool isAlikedType(const std::string& type) { return type.rfind("aliked", 0) == 0; }
// 显式列出五个已发布版本，使拼写错误在解析时报告，避免运行到一半下载时才返回 404。
bool isLomaType(const std::string& type) {
    return type == "loma-b" || type == "loma-b128" || type == "loma-r" ||
           type == "loma-l" || type == "loma-g";
}

int lomaDescriptorDim(const std::string& variant) {
#if SS_HAVE_LOMA
    const std::string d = loma::descriptor_for_matcher(variant);
    if (d == "loma-dedode-b") return 128;
    if (d == "loma-dedode-g") return 256;
#else
    (void)variant;
#endif
    return 0;
}

int defaultMaxImageSize(const std::string& type) {
    return (isAlikedType(type) || isLomaType(type)) ? 1600 : 3200;
}

std::unique_ptr<IFeatureExtractor> createFeatureExtractor(const std::string& type,
                                                          const SiftOptions& sift,
                                                          const AlikedOptions& aliked,
                                                          const LomaOptions& loma_opt) {
    if (type == "sift") return std::make_unique<SiftFrontend>(sift);
    if (isAlikedType(type)) {
#if SS_HAVE_ALIKED
        AlikedOptions opt = aliked;
        // --features 已指定权重名称；仅在使用本地文件时需要显式设置 --aliked-model。
        if (opt.model.empty() || isAlikedType(opt.model)) opt.model = type;
        return std::make_unique<AlikedFrontend>(opt);
#else
        throw std::runtime_error(
            "this build has no learned frontend: '" + type +
            "' is not available in this standalone SfM build");
#endif
    }
    if (isLomaType(type)) {
#if SS_HAVE_LOMA
        LomaOptions opt = loma_opt;
        opt.variant = type;
        // --features 指定版本及其描述子：loma-b128 使用 DeDoDe-B 训练，其余四种使用 DeDoDe-G。
        if (opt.descriptor_model.empty())
            opt.descriptor_model = loma::descriptor_for_matcher(type);
        if (opt.descriptor_model.empty())
            throw std::runtime_error(
                "unknown LoMa variant '" + type +
                "' (expected loma-b, loma-b128, loma-r, loma-l or loma-g)");
        return std::make_unique<LomaFrontend>(opt);
#else
        (void)loma_opt;
        throw std::runtime_error(
            "this build has no learned frontend: '" + type +
            "' is not available in this standalone SfM build");
#endif
    }
    (void)loma_opt;
    throw std::runtime_error("unknown feature type '" + type +
                             "' (expected sift, aliked-n16rot, aliked-n32, loma-b or "
                             "loma-b128)");
}

}  // 命名空间 sfm
