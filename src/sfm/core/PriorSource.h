// 传感器先验统一接口，提供图像间相对旋转、邻接关系、公制位置及给定模型位姿下的 BA 因子。
// 建图与验证仅依赖此接口，IMU/GPS 由 SensorPriors 实现，也可接入其他里程计。
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "sfm/ba/Priors.h"
#include "sfm/core/Pose.h"

namespace sfm {

// 先验源所见的已配准图像：ID、相机组及模型坐标中的世界到相机位姿。
struct PosedImage {
    uint32_t image = 0;
    uint32_t camera = 0;
    Pose pose;
};

class PriorSource {
public:
    virtual ~PriorSource() = default;
    // 判断来源是否能为图像提供任何信息。
    virtual bool has(uint32_t img) const = 0;
    // 相机间相对旋转 R_j ~ R_ji R_i，并提供以弧度表示的 1-sigma 不确定度。
    virtual bool relativeRotation(uint32_t i, uint32_t j, Mat3& R_ji, double& sigma) const = 0;
    // 与 img 直接相关的图像，最近者优先。
    virtual std::vector<uint32_t> neighbours(uint32_t img) const = 0;
    // 在图像当前坐标规范中生成约束，按给定位姿拟合所需向上轴、尺度或公制坐标，保持输入图像 ID。
    virtual PosePriors factors(const std::vector<PosedImage>& imgs) = 0;
    // 来源公制坐标中的位置，GPS 使用东、北、上，供按拍摄位置配对。
    virtual bool position(uint32_t img, Vec3& p) const {
        (void)img;
        (void)p;
        return false;
    }
};

// 子数据库的先验适配：输入输出均为局部 ID，内部封装源仅使用全局 ID，供原子重建使用。
class RemappedPriorSource : public PriorSource {
public:
    RemappedPriorSource(PriorSource& inner, std::vector<uint32_t> to_global)
        : inner_(inner), to_global_(std::move(to_global)) {
        for (uint32_t i = 0; i < to_global_.size(); i++) to_local_.emplace(to_global_[i], i);
    }
    bool has(uint32_t img) const override {
        return img < to_global_.size() && inner_.has(to_global_[img]);
    }
    bool relativeRotation(uint32_t i, uint32_t j, Mat3& R, double& sigma) const override {
        if (i >= to_global_.size() || j >= to_global_.size()) return false;
        return inner_.relativeRotation(to_global_[i], to_global_[j], R, sigma);
    }
    std::vector<uint32_t> neighbours(uint32_t img) const override {
        std::vector<uint32_t> out;
        if (img >= to_global_.size()) return out;
        for (uint32_t g : inner_.neighbours(to_global_[img])) {
            auto it = to_local_.find(g);
            if (it != to_local_.end()) out.push_back(it->second);
        }
        return out;
    }
    PosePriors factors(const std::vector<PosedImage>& imgs) override {
        std::vector<PosedImage> g = imgs;
        for (PosedImage& p : g) p.image = p.image < to_global_.size() ? to_global_[p.image] : ~0u;
        PosePriors pr = inner_.factors(g);
        auto local = [&](uint32_t& id) {
            auto it = to_local_.find(id);
            if (it == to_local_.end()) return false;
            id = it->second;
            return true;
        };
        PosePriors out;
        out.up_w = pr.up_w;
        out.huber = pr.huber;
        for (PriorRotation r : pr.rotations)
            if (local(r.i) && local(r.j)) out.rotations.push_back(r);
        for (PriorUp u : pr.ups)
            if (local(u.i)) out.ups.push_back(u);
        for (PriorCentre c : pr.centres) {
            bool ok = true;
            for (int k = 0; k < c.n; k++) ok = ok && local(c.img[k]);
            if (ok) out.centres.push_back(c);
        }
        return out;
    }
    bool position(uint32_t img, Vec3& p) const override {
        return img < to_global_.size() && inner_.position(to_global_[img], p);
    }

private:
    PriorSource& inner_;
    std::vector<uint32_t> to_global_;
    std::unordered_map<uint32_t, uint32_t> to_local_;
};

}  // 命名空间 sfm
