// 双视图几何验证，稳健估计基础矩阵和单应性，按 H 内点比例选择模型并输出内点与类别；可由内参恢复相对位姿。
#pragma once

#include <vector>

#include "sfm/geometry/Essential.h"
#include "sfm/geometry/Fundamental.h"
#include "sfm/geometry/Homography.h"
#include "sfm/geometry/LinAlg.h"
#include "sfm/optim/Ransac.h"

namespace sfm {

enum class TwoViewConfig {
    Undefined = 0,
    Degenerate,          // 内点不足
    Uncalibrated,        // 基础矩阵最能解释数据
    PlanarOrPanoramic,   // 单应性最能解释数据，可能为平面或纯旋转
};

inline const char* twoViewConfigName(TwoViewConfig c) {
    switch (c) {
        case TwoViewConfig::Degenerate: return "degenerate";
        case TwoViewConfig::Uncalibrated: return "uncalibrated";
        case TwoViewConfig::PlanarOrPanoramic: return "planar/panoramic";
        default: return "undefined";
    }
}

struct TwoViewOptions {
    // 置信度与试验数参考 COLMAP，内点半径采用提取分辨率的 3 px，由六场景、两种提取尺度扫描选定；不能与源图像素阈值直接比较（D47）。
    RansacOptions ransac;
    int min_num_inliers = 15;
    double max_H_inlier_ratio = 0.8;
    // 初次验证始终拟合单应性；种子搜索可沿用已有非平面结论，避免在更少内点上重跑昂贵 H-RANSAC。
    bool estimate_homography = true;
    // 可选从 F 内点恢复标定位姿。
    bool recover_pose = false;
    Mat3 K1 = mat3Identity(), K2 = mat3Identity();
    TwoViewOptions() {
        ransac.max_error = 3.0;
        ransac.min_inlier_ratio = 0.0;
    }
};

struct TwoViewGeometry {
    TwoViewConfig config = TwoViewConfig::Undefined;
    std::vector<char> inlier_mask;  // 对应全部输入匹配
    int num_inliers = 0;
    // 像素路径保存 F/H；单位视线路径保存一般秩二极线矩阵和射线单应性，前者保留七点解的额外两自由度（D45）。
    Mat3 F = {}, H = {};
    bool has_pose = false;
    Pose pose;  // 相机 1 到相机 2 的相对位姿，平移单位化
};

namespace detail {
struct HModel {
    Mat3 H = mat3Identity(), Hinv = mat3Identity();
};

// 两路径共用收尾：按 COLMAP 的 H 内点比例选择模型并复制内点。
inline void selectTwoViewModel(TwoViewGeometry& g, const RansacReport<Mat3>& fRep,
                               const RansacReport<HModel>& hRep, const TwoViewOptions& opt) {
    g.F = fRep.model;
    g.H = hRep.model.H;
    if (std::max(fRep.num_inliers, hRep.num_inliers) < opt.min_num_inliers) {
        g.config = TwoViewConfig::Degenerate;
        return;
    }
    // H 内点比例较高表示平面或纯旋转，极线几何退化；COLMAP 默认阈值为 0.8。
    double hRatio = opt.estimate_homography
                        ? (double)hRep.num_inliers / std::max(1, fRep.num_inliers)
                        : 0.0;
    if (hRatio > opt.max_H_inlier_ratio) {
        g.config = TwoViewConfig::PlanarOrPanoramic;
        g.inlier_mask = hRep.inlier_mask;
        g.num_inliers = hRep.num_inliers;
    } else {
        g.config = TwoViewConfig::Uncalibrated;
        g.inlier_mask = fRep.inlier_mask;
        g.num_inliers = fRep.num_inliers;
    }
}
}  // 命名空间 detail

// 单位视线验证对所有视场适用，max_error 在此以弧度表示，由调用方按焦距转换像素预算。
// 类别标签与像素路径一致，使数据库和建图器无需区分验证坐标系。
inline TwoViewGeometry estimateTwoViewBearing(const std::vector<Vec3>& b1,
                                              const std::vector<Vec3>& b2,
                                              const TwoViewOptions& opt) {
    TwoViewGeometry g;
    int n = (int)b1.size();
    if (n < opt.min_num_inliers) return g;

    auto fFit = [&](const std::vector<int>& s) { return estimateEpipolar7Bearing(b1, b2, s); };
    auto fRefit = [&](const std::vector<int>& s) { return estimateEpipolar8Bearing(b1, b2, s); };
    auto fRes = [&](const Mat3& E, int i) { return sampsonSqBearing(E, b1[i], b2[i]); };
    RansacReport<Mat3> fRep = loransac<Mat3>(n, 7, fFit, fRefit, fRes, opt.ransac);

    RansacReport<detail::HModel> hRep;
    if (opt.estimate_homography) {
        auto hFit = [&](const std::vector<int>& s) {
            std::vector<detail::HModel> out;
            for (const Mat3& H : estimateHomographyBearing(b1, b2, s))
                out.push_back({H, inverse3(H)});
            return out;
        };
        auto hRes = [&](const detail::HModel& m, int i) {
            return angularTransferSq(m.H, m.Hinv, b1[i], b2[i]);
        };
        hRep = loransac<detail::HModel>(n, 4, hFit, hFit, hRes, opt.ransac);
    }

    detail::selectTwoViewModel(g, fRep, hRep, opt);
    if (g.config == TwoViewConfig::Degenerate || g.config == TwoViewConfig::Undefined) return g;

    if (opt.recover_pose && g.config == TwoViewConfig::Uncalibrated && g.num_inliers >= 5) {
        std::vector<int> idx;
        for (int i = 0; i < n; i++)
            if (g.inlier_mask[i]) idx.push_back(i);
        // 此时才施加七点拟合未固定的本质矩阵约束。
        int cnt = 0;
        g.pose = recoverRelativePoseBearing(b1, b2, idx, projectToEssential(g.F), cnt);
        g.has_pose = cnt > 0;
    }
    return g;
}

// 验证单个图像对，p1[k] 与 p2[k] 为对应关键点像素坐标。
inline TwoViewGeometry estimateTwoView(const std::vector<Vec2>& p1, const std::vector<Vec2>& p2,
                                       const TwoViewOptions& opt) {
    TwoViewGeometry g;
    int n = (int)p1.size();
    if (n < opt.min_num_inliers) return g;

    // ---------------- 基础矩阵 ----------------
    auto fFit = [&](const std::vector<int>& s) { return estimateFundamental7(p1, p2, s); };
    auto fRefit = [&](const std::vector<int>& s) { return estimateFundamental8(p1, p2, s); };
    auto fRes = [&](const Mat3& F, int i) { return sampsonSq(F, p1[i], p2[i]); };
    RansacReport<Mat3> fRep = loransac<Mat3>(n, 7, fFit, fRefit, fRes, opt.ransac);

    // ---------------- 单应性 ----------------
    RansacReport<detail::HModel> hRep;
    if (opt.estimate_homography) {
        auto hFit = [&](const std::vector<int>& s) {
            std::vector<detail::HModel> out;
            for (const Mat3& H : estimateHomography(p1, p2, s)) out.push_back({H, inverse3(H)});
            return out;
        };
        auto hRes = [&](const detail::HModel& m, int i) {
            return symmetricTransferSq(m.H, m.Hinv, p1[i], p2[i]);
        };
        hRep = loransac<detail::HModel>(n, 4, hFit, hFit, hRes, opt.ransac);
    }

    detail::selectTwoViewModel(g, fRep, hRep, opt);
    if (g.config == TwoViewConfig::Degenerate || g.config == TwoViewConfig::Undefined) return g;

    // 可选由 F 内点恢复标定相对位姿。
    if (opt.recover_pose && g.config == TwoViewConfig::Uncalibrated && g.num_inliers >= 5) {
        Mat3 K1inv = inverse3(opt.K1), K2inv = inverse3(opt.K2);
        std::vector<Vec2> n1(n), n2(n);
        std::vector<int> idx;
        for (int i = 0; i < n; i++) {
            Vec3 a = mul(K1inv, Vec3{p1[i].x, p1[i].y, 1});
            Vec3 b = mul(K2inv, Vec3{p2[i].x, p2[i].y, 1});
            n1[i] = {a.x / a.z, a.y / a.z};
            n2[i] = {b.x / b.z, b.y / b.z};
            if (g.inlier_mask[i]) idx.push_back(i);
        }
        Mat3 E = essentialFromFundamental(g.F, opt.K1, opt.K2);
        int cnt = 0;
        g.pose = recoverRelativePose(n1, n2, idx, E, cnt);
        g.has_pose = cnt > 0;
    }
    return g;
}

}  // 命名空间 sfm
