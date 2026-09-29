// 重建模型及 COLMAP 二进制读写，保存相机、图像位姿和二维点、带轨迹的三维点；假设主机为小端。
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "sfm/core/Camera.h"
#include "sfm/core/Pose.h"
#include "sfm/core/Rig.h"
#include "sfm/geometry/LinAlg.h"

namespace sfm {

static constexpr uint64_t kInvalidPoint3D = ~0ull;

struct TrackElement {
    uint32_t image_id = 0;
    uint32_t point2D_idx = 0;
};

struct Point3D {
    Vec3 xyz;
    uint8_t rgb[3] = {128, 128, 128};
    double error = 0;
    std::vector<TrackElement> track;
};

struct Image {
    uint32_t id = 0;
    uint32_t camera_id = 0;
    std::string name;
    Pose pose;                 // 世界坐标系 -> 相机坐标系
    bool registered = false;
    // EXIF Orientation，无标签或像素已旋转时为 1；仅规范对齐使用，磁盘模型本身不保存此信息。
    uint8_t exif_orientation = 1;
    std::vector<Vec2> points2D;              // 全部特征的关键点坐标
    std::vector<uint64_t> point3D_ids;       // 与关键点一一对应，无三维点时为 kInvalidPoint3D

    uint32_t numPoint3D() const {
        uint32_t n = 0;
        for (uint64_t id : point3D_ids)
            if (id != kInvalidPoint3D) n++;
        return n;
    }
};

// COLMAP 相机模型 ID 统一来自 Camera.h 的元数据表，读写层不再维护独立枚举。

struct Reconstruction {
    std::map<uint32_t, Camera> cameras;
    std::map<uint32_t, Image> images;
    std::map<uint64_t, Point3D> points3D;
    uint64_t next_point3D_id = 1;
    // 逐 rig 的模型外参标定，使用模型单位；尚无帧配准时为空。
    std::vector<RigCalib> rigs;
    // 脱离帧、拥有独立位姿的图像；建图器保持整帧而不填充此集合，求解与合并仍遵守它。
    std::set<uint32_t> rig_detached;

    uint32_t numRegistered() const {
        uint32_t n = 0;
        for (const auto& kv : images)
            if (kv.second.registered) n++;
        return n;
    }

    uint64_t addPoint3D(const Vec3& xyz, const std::vector<TrackElement>& track) {
        uint64_t id = next_point3D_id++;
        Point3D p;
        p.xyz = xyz;
        p.track = track;
        points3D[id] = p;
        for (const TrackElement& e : track) images[e.image_id].point3D_ids[e.point2D_idx] = id;
        return id;
    }

    void writeBinary(const std::string& dir) const;
    static Reconstruction readBinary(const std::string& dir);
};

// 复制 keep 指定的图像并裁剪轨迹，其余图像恢复未配准，少于两个观测的三维点删除，供模型拆分使用（D45）。
inline Reconstruction subsetModel(const Reconstruction& m, const std::set<uint32_t>& keep) {
    Reconstruction out = m;
    for (auto& kv : out.images) {
        if (keep.count(kv.first)) continue;
        kv.second.registered = false;
        kv.second.pose = {mat3Identity(), {0, 0, 0}};
        std::fill(kv.second.point3D_ids.begin(), kv.second.point3D_ids.end(), kInvalidPoint3D);
    }
    std::vector<uint64_t> dead;
    for (auto& kv : out.points3D) {
        std::vector<TrackElement> t;
        for (const TrackElement& e : kv.second.track)
            if (keep.count(e.image_id)) t.push_back(e);
        if (t.size() < 2) { dead.push_back(kv.first); continue; }
        kv.second.track = std::move(t);
    }
    for (uint64_t pid : dead) {
        for (const TrackElement& e : m.points3D.at(pid).track) {
            auto it = out.images.find(e.image_id);
            if (it != out.images.end() && e.point2D_idx < it->second.point3D_ids.size() &&
                it->second.point3D_ids[e.point2D_idx] == pid)
                it->second.point3D_ids[e.point2D_idx] = kInvalidPoint3D;
        }
        out.points3D.erase(pid);
    }
    return out;
}

// 由对应图持有者提供两图像是否具有真实匹配支持的查询，用于重复结构检查；建图器有对应图，合并器没有。
using MatchedFn = std::function<bool(uint32_t, uint32_t)>;

// 排序后的图像可见三维点，便于快速求交。
inline std::vector<uint64_t> observedPoints(const Image& im) {
    std::vector<uint64_t> v;
    for (uint64_t p : im.point3D_ids)
        if (p != kInvalidPoint3D) v.push_back(p);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

inline size_t sharedPoints(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    size_t n = 0, i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) i++;
        else if (b[j] < a[i]) j++;
        else { n++; i++; j++; }
    }
    return n;
}

// ---------------- 小端二进制辅助函数 ----------------
namespace detail {
template <class T>
void wr(std::ostream& f, T v) {
    f.write((const char*)&v, sizeof(T));
}
template <class T>
T rd(std::istream& f) {
    T v{};
    f.read((char*)&v, sizeof(T));
    return v;
}
}  // 命名空间 detail

inline void Reconstruction::writeBinary(const std::string& dir) const {
    using namespace detail;
    {
        std::ofstream f(dir + "/cameras.bin", std::ios::binary);
        if (!f) throw std::runtime_error("cannot write cameras.bin in " + dir);
        wr<uint64_t>(f, cameras.size());
        for (const auto& kv : cameras) {
            const Camera& c = kv.second;
            wr<uint32_t>(f, c.id);
            wr<int32_t>(f, camColmapId(c.model));
            wr<uint64_t>(f, c.width);
            wr<uint64_t>(f, c.height);
            double ps[12];
            packColmap(c, ps);
            for (int i = 0; i < camColmapParams(c.model); i++) wr<double>(f, ps[i]);
        }
    }
    {
        std::ofstream f(dir + "/images.bin", std::ios::binary);
        if (!f) throw std::runtime_error("cannot write images.bin in " + dir);
        uint64_t nreg = numRegistered();
        wr<uint64_t>(f, nreg);
        for (const auto& kv : images) {
            const Image& im = kv.second;
            if (!im.registered) continue;
            Quat q = rotationToQuaternion(im.pose.R);
            wr<uint32_t>(f, im.id);
            for (double v : {q[0], q[1], q[2], q[3]}) wr<double>(f, v);
            for (double v : {im.pose.t.x, im.pose.t.y, im.pose.t.z}) wr<double>(f, v);
            wr<uint32_t>(f, im.camera_id);
            f.write(im.name.c_str(), im.name.size() + 1);  // 以 NUL 结尾
            wr<uint64_t>(f, im.points2D.size());
            for (size_t i = 0; i < im.points2D.size(); i++) {
                wr<double>(f, im.points2D[i].x);
                wr<double>(f, im.points2D[i].y);
                wr<uint64_t>(f, im.point3D_ids[i]);
            }
        }
    }
    {
        std::ofstream f(dir + "/points3D.bin", std::ios::binary);
        if (!f) throw std::runtime_error("cannot write points3D.bin in " + dir);
        wr<uint64_t>(f, points3D.size());
        for (const auto& kv : points3D) {
            const Point3D& p = kv.second;
            wr<uint64_t>(f, kv.first);
            for (double v : {p.xyz.x, p.xyz.y, p.xyz.z}) wr<double>(f, v);
            f.write((const char*)p.rgb, 3);
            wr<double>(f, p.error);
            wr<uint64_t>(f, p.track.size());
            for (const TrackElement& e : p.track) {
                wr<uint32_t>(f, e.image_id);
                wr<uint32_t>(f, e.point2D_idx);
            }
        }
    }
}

inline Reconstruction Reconstruction::readBinary(const std::string& dir) {
    using namespace detail;
    Reconstruction r;
    {
        std::ifstream f(dir + "/cameras.bin", std::ios::binary);
        if (!f) throw std::runtime_error("cannot read cameras.bin");
        uint64_t n = rd<uint64_t>(f);
        for (uint64_t i = 0; i < n; i++) {
            Camera c;
            c.id = rd<uint32_t>(f);
            int32_t model = rd<int32_t>(f);
            c.width = (int)rd<uint64_t>(f);
            c.height = (int)rd<uint64_t>(f);
            c.model = camFromColmapId(model);
            double ps[12];
            for (int k = 0; k < camColmapParams(c.model); k++) ps[k] = rd<double>(f);
            unpackColmap(c, ps);
            r.cameras[c.id] = c;
        }
    }
    {
        std::ifstream f(dir + "/images.bin", std::ios::binary);
        if (!f) throw std::runtime_error("cannot read images.bin");
        uint64_t n = rd<uint64_t>(f);
        for (uint64_t i = 0; i < n; i++) {
            Image im;
            im.registered = true;
            im.id = rd<uint32_t>(f);
            Quat q = {rd<double>(f), rd<double>(f), rd<double>(f), rd<double>(f)};
            im.pose.R = quaternionToRotation(q);
            im.pose.t = {rd<double>(f), rd<double>(f), rd<double>(f)};
            im.camera_id = rd<uint32_t>(f);
            std::string name;
            char ch;
            while (f.get(ch) && ch != '\0') name += ch;
            im.name = name;
            uint64_t np = rd<uint64_t>(f);
            im.points2D.resize(np);
            im.point3D_ids.resize(np);
            for (uint64_t k = 0; k < np; k++) {
                im.points2D[k] = {rd<double>(f), rd<double>(f)};
                im.point3D_ids[k] = rd<uint64_t>(f);
            }
            r.images[im.id] = im;
        }
    }
    {
        std::ifstream f(dir + "/points3D.bin", std::ios::binary);
        if (!f) throw std::runtime_error("cannot read points3D.bin");
        uint64_t n = rd<uint64_t>(f);
        for (uint64_t i = 0; i < n; i++) {
            uint64_t id = rd<uint64_t>(f);
            Point3D p;
            p.xyz = {rd<double>(f), rd<double>(f), rd<double>(f)};
            p.rgb[0] = rd<uint8_t>(f); p.rgb[1] = rd<uint8_t>(f); p.rgb[2] = rd<uint8_t>(f);
            p.error = rd<double>(f);
            uint64_t tl = rd<uint64_t>(f);
            p.track.resize(tl);
            for (uint64_t k = 0; k < tl; k++) {
                p.track[k].image_id = rd<uint32_t>(f);
                p.track[k].point2D_idx = rd<uint32_t>(f);
            }
            r.points3D[id] = p;
            r.next_point3D_id = std::max(r.next_point3D_id, id + 1);
        }
    }
    // 部分导出器删除未三角化关键点却保留旧轨迹索引；Hierarchical 3DGS campus 的 2970 万项中有 2660 万越界，因此不一致时从图像重建轨迹。
    auto agrees = [&](uint64_t id, const TrackElement& e) {
        auto it = r.images.find(e.image_id);
        return it != r.images.end() && e.point2D_idx < it->second.point3D_ids.size() &&
               it->second.point3D_ids[e.point2D_idx] == id;
    };
    bool consistent = true;
    for (const auto& kv : r.points3D)
        for (const TrackElement& e : kv.second.track) consistent = consistent && agrees(kv.first, e);
    if (!consistent) {
        std::fprintf(stderr, "[model] %s: tracks disagree with images.bin; rebuilt from the images\n",
                     dir.c_str());
        for (auto& kv : r.points3D) kv.second.track.clear();
        for (auto& kv : r.images)
            for (uint32_t k = 0; k < kv.second.point3D_ids.size(); k++) {
                uint64_t& id = kv.second.point3D_ids[k];
                if (id == kInvalidPoint3D) continue;
                auto it = r.points3D.find(id);
                if (it == r.points3D.end()) id = kInvalidPoint3D;
                else it->second.track.push_back({kv.first, k});
            }
    }
    return r;
}

}  // 命名空间 sfm
