// 无外部测量时，将相机平均向上轴对齐 +Z，以相机中心居中并归一化最大坐标，再按点云地面调平。
// 相机向上轴是持机方式的统计猜测，可用 level=cameras 或 no-orient 调整。
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/SceneAlign.h"
#include "sfm/core/Exif.h"
#include "sfm/core/Model.h"
#include "sfm/core/Pose.h"

namespace sfm {

// 世界向上方向为世界到相机 R 的第二行取负，平均后未归一化；use_exif 使用各图方向标签，处理竖拍 90 度差。
inline Vec3 meanCameraUp(const Reconstruction& rec, bool use_exif = false) {
    Vec3 up{0, 0, 0};
    for (const auto& kv : rec.images) {
        const Image& im = kv.second;
        if (!im.registered) continue;
        double u[3] = {0, -1, 0};
        if (use_exif) exifUpInCamera(im.exif_orientation, u);
        // R^T u，将相机向上向量写到世界坐标。
        for (int c = 0; c < 3; c++)
            up = up + Vec3{im.pose.R[3 * c] * u[c], im.pose.R[3 * c + 1] * u[c],
                           im.pose.R[3 * c + 2] * u[c]};
    }
    return up;
}

// 用 Rodrigues 绕 up×z 将单位 up 旋至 +Z。
inline Mat3 rotationUpToZ(const Vec3& up) {
    Vec3 axis{up.y, -up.x, 0.0};
    const double s = std::sqrt(axis.x * axis.x + axis.y * axis.y);
    const double c = up.z;
    Mat3 R = mat3Identity();
    if (s > 1e-12) {
        axis = axis * (1.0 / s);
        const double C = 1.0 - c;
        R = Mat3{c + axis.x * axis.x * C,       axis.x * axis.y * C - axis.z * s, axis.y * s,
                 axis.x * axis.y * C + axis.z * s, c + axis.y * axis.y * C,      -axis.x * s,
                 -axis.y * s,                   axis.x * s,                       c};
    } else if (c < 0.0) {
        R = Mat3{1, 0, 0, 0, -1, 0, 0, 0, -1};   // up==-z 时翻转
    }
    return R;
}

// 按 R 旋转并围绕相机中心归一化，不足两张已配准图时返回恒等。
inline Sim3 normalizingTransform(const Reconstruction& rec, const Mat3& R) {
    Sim3 T;
    std::vector<Vec3> centers;
    Vec3 mid{0, 0, 0};
    for (const auto& kv : rec.images) {
        const Image& im = kv.second;
        if (!im.registered) continue;
        Vec3 c = mul(transpose(im.pose.R), im.pose.t) * -1.0;
        centers.push_back(c);
        mid = mid + c;
    }
    if (centers.size() < 2) return T;
    mid = mid * (1.0 / (double)centers.size());

    // 按最大坐标分量而非欧氏范数缩放到 1，与训练器规范保持一致。
    double max_abs = 0.0;
    for (const Vec3& p : centers) {
        const Vec3 d = mul(R, p - mid);
        max_abs = std::max(max_abs, std::abs(d.x));
        max_abs = std::max(max_abs, std::abs(d.y));
        max_abs = std::max(max_abs, std::abs(d.z));
    }
    T.scale = max_abs > 1e-12 ? 1.0 / max_abs : 1.0;
    T.R = R;
    T.t = mul(R, mid) * -T.scale;
    return T;
}

// 将 up 对齐 +Z 的相同变换，up 为零时返回恒等。
inline Sim3 normalizingTransform(const Reconstruction& rec, const Vec3& up) {
    const double un = up.norm();
    if (!(un > 1e-12)) return Sim3{};
    return normalizingTransform(rec, rotationUpToZ(up * (1.0 / un)));
}

// 为磁盘读取模型补 EXIF 方向，若已有图像携带变换标签则以特征为权威，不覆盖。
inline int fillExifOrientations(std::vector<Reconstruction>& models,
                                const std::string& imagedir) {
    if (imagedir.empty()) return 0;
    for (const Reconstruction& m : models)
        for (const auto& kv : m.images)
            if (kv.second.exif_orientation != 1) return 0;
    int read = 0;
    for (Reconstruction& m : models)
        for (auto& kv : m.images)
            if (kv.second.registered) {
                kv.second.exif_orientation =
                    (uint8_t)exifOrientation(imagedir + "/" + kv.second.name);
                read++;
            }
    return read;
}

// 从相机自身获取向上方向。
inline Sim3 uprightTransform(const Reconstruction& rec, bool use_exif = false) {
    return normalizingTransform(rec, meanCameraUp(rec, use_exif));
}

// 世界规范变换只改位姿、三维点及相应 rig 平移尺度，内参、二维像素和重投影误差不随世界单位改变。
inline void applySim3(Reconstruction& rec, const Sim3& T) {
    for (auto& kv : rec.images)
        if (kv.second.registered) kv.second.pose = transformPose(T, kv.second.pose);
    for (auto& kv : rec.points3D) kv.second.xyz = transformPoint(T, kv.second.xyz);
    transformRigs(rec.rigs, T.scale);
}

// 使用 SceneAlign 的地面几何调平，以当前 +Z 为先验，拒绝偏离超过 60 度的地面。
struct GroundFit {
    Sim3 T;               // 未找到地面时为恒等变换
    bool found = false;
    double share = 0.0;   // 落在地面上的点占比
};

// full 模式将地面调平到 z=0、墙面对轴、水平投影居中并保持尺度；否则仅沿 Z 平移。pre 先作用于采样点。
inline GroundFit groundTransform(const Reconstruction& rec, bool full,
                                 const Sim3& pre = Sim3{}) {
    GroundFit out;
    std::vector<double> pts;
    const size_t step = std::max<size_t>(1, rec.points3D.size() / 250000);
    size_t k = 0;
    for (const auto& kv : rec.points3D) {
        if (k++ % step) continue;
        const Vec3 p = transformPoint(pre, kv.second.xyz);
        pts.insert(pts.end(), {p.x, p.y, p.z});
    }
    const int64_t n = (int64_t)pts.size() / 3;
    if (n < 16) return out;
    namespace al = spirula::align;
    al::AutoAlignOptions opt;
    opt.tol = 0.01 * al::robust_extent(pts.data(), n);
    opt.yaw = opt.centre = full;
    const double up[3] = {0, 0, 1};
    const al::AutoAlignResult r = al::auto_align(pts.data(), n, up, nullptr, nullptr, opt);
    if (!r.ground || !(opt.tol > 0.0)) return out;
    out.found = true;
    out.share = r.ground_share;
    if (full) {
        for (int i = 0; i < 9; i++) out.T.R[(size_t)i] = r.T.R[i];
        out.T.t = Vec3{r.T.t[0], r.T.t[1], r.T.t[2]};
        return out;
    }
    // 水平投影中心处的平面高度。
    std::vector<double> xs, ys;
    for (int64_t i = 0; i < n; i++) {
        xs.push_back(pts[(size_t)i * 3]);
        ys.push_back(pts[(size_t)i * 3 + 1]);
    }
    std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
    std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
    const al::Plane& g = r.plane;
    const double z = -(g.n[0] * xs[xs.size() / 2] + g.n[1] * ys[ys.size() / 2] + g.d) / g.n[2];
    out.T.t = Vec3{0, 0, -z};
    return out;
}

}  // 命名空间 sfm
