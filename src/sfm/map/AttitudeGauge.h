// 根据图像记录的俯仰、滚转确定向上方向，偏航确定北向；与重建矛盾的姿态集合拒绝使用。
// 这是姿态测量而非按持机方向猜测，适用于俯拍或云台倒置。
#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "sfm/core/Attitude.h"
#include "sfm/core/Exif.h"
#include "sfm/core/Model.h"
#include "sfm/core/Pose.h"
#include "sfm/map/ImuExtrinsic.h"
#include "sfm/map/Orient.h"

namespace sfm {

// 每张含姿态的已配准图像，相机到东、北、上坐标的旋转。
struct AttitudeRef {
    std::vector<uint32_t> image_ids;
    std::vector<Mat3> world_from_cam;
    int registered = 0;
};

// pixels_turned 表示已应用 EXIF 像素旋转，模型相机系与原姿态记录坐标系不同。
inline AttitudeRef attitudeRefFromImages(const Reconstruction& rec, const std::string& image_dir,
                                         bool pixels_turned) {
    AttitudeRef ref;
    for (const auto& kv : rec.images) {
        if (!kv.second.registered) continue;
        ref.registered++;
        const std::string path = (std::filesystem::path(image_dir) / kv.second.name).string();
        const CameraAttitude a = readCameraAttitude(path);
        if (!a.valid) continue;
        if (pixels_turned && !exifTransform(exifOrientation(path)).identity()) continue;
        ref.image_ids.push_back(kv.first);
        ref.world_from_cam.push_back(attitudeWorldFromCamera(a));
    }
    return ref;
}

enum class AttitudeFail { None, Few, Disagree };

struct AttitudeFit {
    bool ok = false;        // T 将模型调平
    bool north = false;     // 并将北向旋至 +Y
    AttitudeFail reason = AttitudeFail::Few;
    AttitudeFail north_reason = AttitudeFail::None;
    Sim3 T;
    UpConsensus up;         // 位于模型坐标系
    UpConsensus heading;    // 绕调平后 +Z 旋转的 (cos,sin,0)
};

// 偏离超过 10 度视为离群；离群多于内点时拒绝整组，而非强行平均。
inline bool attitudeMajority(const UpConsensus& u) {
    return u.ok && 2 * u.outliers <= u.votes;
}

inline AttitudeFit fitAttitudeGauge(const Reconstruction& rec, const AttitudeRef& ref,
                                    bool north) {
    AttitudeFit fit;
    std::vector<Mat3> W, R;
    std::vector<Vec3> votes;
    for (size_t k = 0; k < ref.image_ids.size(); k++) {
        const auto it = rec.images.find(ref.image_ids[k]);
        if (it == rec.images.end() || !it->second.registered) continue;
        const Mat3& w = ref.world_from_cam[k];
        W.push_back(w);
        R.push_back(it->second.pose.R);
        // 世界 +Z 在相机中的方向为相机到世界矩阵的第三行。
        votes.push_back(mul(transpose(R.back()), Vec3{w[6], w[7], w[8]}));
    }
    if (votes.size() < 3) return fit;
    fit.up = consensusUp(votes);
    if (!attitudeMajority(fit.up)) {
        fit.reason = AttitudeFail::Disagree;
        return fit;
    }
    const Mat3 level = rotationUpToZ(fit.up.up);
    Mat3 turn = mat3Identity();
    if (north) {
        std::vector<Vec3> hv;
        for (size_t k = 0; k < W.size(); k++) {
            // 调平模型到东、北、上的变换，在噪声范围内仅绕 +Z 旋转。
            const Mat3 E = mul(W[k], mul(R[k], transpose(level)));
            const double a = std::atan2(E[3] - E[1], E[0] + E[4]);
            hv.push_back({std::cos(a), std::sin(a), 0});
        }
        fit.heading = consensusUp(hv);
        if (attitudeMajority(fit.heading)) {
            const double c = fit.heading.up.x, s = fit.heading.up.y, n = std::hypot(c, s);
            turn = Mat3{c / n, -s / n, 0, s / n, c / n, 0, 0, 0, 1};
            fit.north = true;
        } else {
            fit.north_reason = AttitudeFail::Disagree;
        }
    }
    fit.T = normalizingTransform(rec, mul(turn, level));
    fit.ok = true;
    fit.reason = AttitudeFail::None;
    return fit;
}

}  // 命名空间 sfm
