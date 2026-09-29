#include "sfm/feature/LearnedMatcher.h"

#include "sfm/feature/Extractor.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#if SS_HAVE_ALIKED
#include "aliked/model/LightGlue.h"
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
// LightGlue 与 LoMa 使用推理层的进程级共享设备；device 字段不能单独选择设备，必须在加载权重前提交本次运行解析出的设备标识。
inline void configureLearnedDevice(const std::string& selector) {
    if (!selector.empty()) nn::configure_device(selector);
}
#endif

#if SS_HAVE_ALIKED

class LightGlueMatcher : public IFeatureMatcher {
public:
    explicit LightGlueMatcher(const MatchOptions& match, const LightGlueOptions& opt)
        : match_(match), opt_(opt) {
        configureLearnedDevice(opt.device_selector);
        lg_.load(opt.model);
        mopt_.min_score = (float)opt.min_score;
    }

    const char* name() const override { return "lightglue"; }

    std::vector<FeatureMatch> match(const FeatureSet& a, const FeatureSet& b) override {
        std::vector<FeatureMatch> out;
        if (a.count() == 0 || b.count() == 0) return out;
        check(a);
        check(b);

        std::vector<float> ka, kb;
        const std::vector<aliked::Match> m =
            lg_.match(view(a, ka), view(b, kb), mopt_);

        out.reserve(m.size());
        for (const aliked::Match& x : m) {
            // FeatureMatch 存储按升序排序的距离，用于 max_num_matches 截断；将 LightGlue 的 (0, 1] 置信度取补值即可保持所需顺序。
            out.push_back({x.i, x.j, 1.0f - x.score});
        }
        if (match_.max_num_matches > 0 && out.size() > match_.max_num_matches) {
            std::partial_sort(out.begin(), out.begin() + match_.max_num_matches, out.end(),
                              [](const FeatureMatch& p, const FeatureMatch& q) {
                                  return p.distance < q.distance;
                              });
            out.resize(match_.max_num_matches);
        }
        return out;
    }

private:
    static void check(const FeatureSet& f) {
        if (f.dtype != DType::F32)
            throw std::runtime_error(
                "LightGlue needs float descriptors; these are uint8. It is trained for "
                "a specific frontend -- use --features aliked-n16rot or aliked-n32");
    }

    // scaleKeypoints 已将关键点映射到原图像素坐标，与 width/height 一致；误传 extract_width 会使位置编码产生缩放错误。
    // xy 必须打包为 [n, 2]；Keypoint 含五个浮点数，其 x/y 不能直接视作连续坐标对。
    static aliked::MatchInput view(const FeatureSet& f, std::vector<float>& xy) {
        xy.resize((size_t)f.count() * 2);
        for (uint32_t i = 0; i < f.count(); i++) {
            xy[(size_t)i * 2] = f.keypoints[i].x;
            xy[(size_t)i * 2 + 1] = f.keypoints[i].y;
        }
        aliked::MatchInput in;
        in.keypoints = xy.data();
        in.descriptors = reinterpret_cast<const float*>(f.descriptors.data());
        in.n = f.count();
        in.width = f.width;
        in.height = f.height;
        return in;
    }

    MatchOptions           match_;
    LightGlueOptions       opt_;
    aliked::Matcher        lg_;
    aliked::MatchOptions   mopt_;
};

#endif  // SS_HAVE_ALIKED

#if SS_HAVE_LOMA

// LoMa 与 LightGlue 共用匹配接口；每个版本只针对固定宽度训练，描述子宽度不符属于形状错误，必须拒绝。
class LomaFeatureMatcher : public IFeatureMatcher {
public:
    LomaFeatureMatcher(const MatchOptions& match, const std::string& model,
                       const LomaMatchOptions& opt)
        : match_(match) {
        configureLearnedDevice(opt.device_selector);
        m_.load(model);
        mopt_.min_score = (float)opt.min_score;
    }

    const char* name() const override { return "loma"; }

    std::vector<FeatureMatch> match(const FeatureSet& a, const FeatureSet& b) override {
        std::vector<FeatureMatch> out;
        if (a.count() == 0 || b.count() == 0) return out;
        check(a);
        check(b);

        std::vector<float> ka, kb;
        const std::vector<loma::Match> m = m_.match(view(a, ka), view(b, kb), mopt_);

        out.reserve(m.size());
        // 下游按距离升序排序；将匹配器的 (0, 1] 置信度取补值，保持所需顺序。
        for (const loma::Match& x : m) out.push_back({x.i, x.j, 1.0f - x.score});
        if (match_.max_num_matches > 0 && out.size() > match_.max_num_matches) {
            std::partial_sort(out.begin(), out.begin() + match_.max_num_matches, out.end(),
                              [](const FeatureMatch& p, const FeatureMatch& q) {
                                  return p.distance < q.distance;
                              });
            out.resize(match_.max_num_matches);
        }
        return out;
    }

private:
    void check(const FeatureSet& f) const {
        if (f.dtype != DType::F32)
            throw std::runtime_error(
                "the LoMa matcher needs float descriptors; these are uint8. Use "
                "--features loma-b128 or loma-b");
        if ((int)f.dim != m_.descriptorDim())
            throw std::runtime_error(
                "the LoMa matcher wants " + std::to_string(m_.descriptorDim()) +
                "-D descriptors and these are " + std::to_string(f.dim) +
                "-D; --features and --matcher name different variants");
    }

    // scaleKeypoints 已将关键点映射到 width/height 所描述的原图像素坐标（D46）；误用 extract_width 会破坏位置编码。
    static loma::MatchInput view(const FeatureSet& f, std::vector<float>& xy) {
        xy.resize((size_t)f.count() * 2);
        for (uint32_t i = 0; i < f.count(); i++) {
            xy[(size_t)i * 2] = f.keypoints[i].x;
            xy[(size_t)i * 2 + 1] = f.keypoints[i].y;
        }
        loma::MatchInput in;
        in.keypoints = xy.data();
        in.descriptors = reinterpret_cast<const float*>(f.descriptors.data());
        in.n = f.count();
        in.width = f.width;
        in.height = f.height;
        return in;
    }

    MatchOptions       match_;
    loma::Matcher      m_;
    loma::MatchOptions mopt_;
};

#endif  // SS_HAVE_LOMA

}  // 匿名命名空间

bool isLearnedMatcher(const std::string& type) {
    return type == "lightglue" || isLomaType(type);
}

std::unique_ptr<IFeatureMatcher> createFeatureMatcher(const std::string& type,
                                                      const MatchOptions& match,
                                                      const LightGlueOptions& lightglue,
                                                      const LomaMatchOptions& loma_opt) {
    if (type == "bruteforce") return std::make_unique<BruteForceMatcher>(match);
    if (type == "lightglue") {
#if SS_HAVE_ALIKED
        return std::make_unique<LightGlueMatcher>(match, lightglue);
#else
        throw std::runtime_error(
            "this build has no learned matcher: --matcher lightglue needs the "
            "not available in this standalone SfM build");
#endif
    }
    if (isLomaType(type)) {
#if SS_HAVE_LOMA
        // --matcher 已指定版本；仅在使用本地文件时需要显式设置 --loma-matcher-model。
        const std::string model = loma_opt.model.empty() ? type : loma_opt.model;
        return std::make_unique<LomaFeatureMatcher>(match, model, loma_opt);
#else
        (void)loma_opt;
        throw std::runtime_error(
            "this build has no learned matcher: --matcher " + type +
            " needs the not available in this standalone SfM build");
#endif
    }
    (void)loma_opt;
    throw std::runtime_error("unknown matcher '" + type +
                             "' (expected bruteforce, lightglue, loma-b or loma-b128)");
}

}  // 命名空间 sfm
