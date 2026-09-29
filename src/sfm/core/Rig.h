// 固定相对位姿的相机装置；每镜头一个成员，同一时刻各成员图像组成一帧。BA 优化逐帧位姿及跨帧共享的成员 cam_from_rig。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sfm/core/Pose.h"

namespace sfm {

constexpr uint32_t kNoImage = UINT32_MAX;
constexpr uint32_t kNoRig = UINT32_MAX;

// 外参优化掩码的 0–2 位为轴角，3–5 位为平移；axial 约束装置原点位于镜头光轴，仅优化旋转与 t.z。
enum RigDof : uint8_t {
    kRigDofNone = 0,
    kRigDofRotation = 0x07,
    kRigDofTranslation = 0x38,
    kRigDofBaseline = 0x20,
    kRigDofAxial = 0x27,
    kRigDofAll = 0x3F,
};

inline constexpr const char* kRigDofNames =
    "all, axial, baseline, rotation, translation or none";

inline bool parseRigDof(const std::string& s, uint8_t& out) {
    static const std::pair<const char*, uint8_t> kNames[] = {
        {"all", kRigDofAll},           {"axial", kRigDofAxial},
        {"baseline", kRigDofBaseline}, {"rotation", kRigDofRotation},
        {"translation", kRigDofTranslation}, {"none", kRigDofNone}};
    for (const auto& n : kNames)
        if (s == n.first) {
            out = n.second;
            return true;
        }
    return false;
}

inline const char* rigDofName(uint8_t dof) {
    switch (dof) {
        case kRigDofAxial: return "axial";
        case kRigDofBaseline: return "baseline";
        case kRigDofRotation: return "rotation";
        case kRigDofTranslation: return "translation";
        case kRigDofNone: return "none";
        default: return "all";
    }
}

// 用户定义的成员：图像目录内路径前缀及可选已知外参。
struct RigMemberDef {
    std::string prefix;
    bool has_ext = false;
    Pose ext;             // cam_from_rig，使用用户单位
    bool ext_fixed = true;
    uint8_t dof = kRigDofAll;
};

// 指定 captures 后，成员路径相对各采集前缀；多个视频可共享同一物理 rig 定义，帧键同时包含采集标识与名称。
struct RigDef {
    std::string name;
    std::vector<RigMemberDef> members;
    std::vector<std::string> captures;  // * 表示每个顶层文件夹
    std::string kind;                   // 空值或 dual-fisheye，由 applyRigKind 解释
};

// 背靠背双鱼眼的名义旋转；Insta360 X、DJI Osmo 360 与 PortalCam 标定结果距此约 0.8–1.4 度。
inline Pose dualFisheyeNominal() {
    return {Mat3{-1, 0, 0, 0, 1, 0, 0, 0, -1}, {0, 0, 0}};
}

// 根据 kind 补齐未给定外参；双鱼眼初始绕图像竖轴相差 180 度，默认优化全部六自由度，除非 refine 另行指定。
inline std::string applyRigKind(RigDef& d) {
    if (d.kind.empty()) return {};
    if (d.kind != "dual-fisheye") return "unknown rig kind '" + d.kind + "' (dual-fisheye)";
    if (d.members.size() < 2) return "rig kind dual-fisheye needs two members";
    RigMemberDef& a = d.members[0];
    RigMemberDef& b = d.members[1];
    if (!a.has_ext && !b.has_ext) {
        a.has_ext = b.has_ext = true;
        a.ext = {mat3Identity(), {0, 0, 0}};
        b.ext = dualFisheyeNominal();
        a.ext_fixed = b.ext_fixed = false;
        // b.dof = kRigDofAxial;
    }
    return {};
}

// 按数据库解析后的 rig，逐帧保存图像 ID。
struct RigSpec {
    std::string name;
    std::string kind;
    std::vector<RigMemberDef> members;
    std::vector<std::vector<uint32_t>> frames;  // frames[f][m]，缺失图像为 kNoImage
    std::vector<std::string> frame_keys;        // 同帧图像共享的键
    bool anyKnownExt() const {
        for (const RigMemberDef& m : members)
            if (m.has_ext) return true;
        return false;
    }
};

struct RigSlot {
    uint32_t rig = kNoRig, frame = 0, member = 0;
    bool valid() const { return rig != kNoRig; }
};

// 本次运行全部 rig 及逐图像反向查询表。
struct RigTable {
    std::vector<RigSpec> rigs;
    std::vector<RigSlot> of_image;  // 按图像 ID 索引

    bool empty() const { return rigs.empty(); }
    RigSlot slot(uint32_t img) const {
        return img < of_image.size() ? of_image[img] : RigSlot{};
    }
    const std::vector<uint32_t>& frameOf(RigSlot s) const { return rigs[s.rig].frames[s.frame]; }

    // 由帧表构建 of_image，图像重复归属时抛错。
    void index(size_t num_images) {
        of_image.assign(num_images, RigSlot{});
        for (uint32_t r = 0; r < rigs.size(); r++) {
            const RigSpec& rig = rigs[r];
            for (uint32_t f = 0; f < rig.frames.size(); f++) {
                if (rig.frames[f].size() != rig.members.size())
                    throw std::runtime_error("rig " + rig.name + ": frame " + std::to_string(f) +
                                             " does not have one slot per member");
                for (uint32_t m = 0; m < rig.frames[f].size(); m++) {
                    const uint32_t img = rig.frames[f][m];
                    if (img == kNoImage) continue;
                    if (img >= num_images)
                        throw std::runtime_error("rig " + rig.name + ": image id out of range");
                    if (of_image[img].valid()) {
                        const RigSlot& o = of_image[img];
                        throw std::runtime_error(
                            "image " + std::to_string(img) + " is in two rig frames (rig " +
                            rigs[o.rig].name + " frame " + std::to_string(o.frame) + " and rig " +
                            rig.name + " frame " + std::to_string(f) + "): conflicting rigs");
                    }
                    of_image[img] = {r, f, m};
                }
            }
        }
    }

    // 将 rig 映射到子数据库；local[g] 为全局图像 g 的局部 ID，原子分组缺失时为 kNoImage。
    RigTable subset(const std::vector<uint32_t>& local, size_t num_local) const {
        RigTable out;
        for (const RigSpec& rig : rigs) {
            RigSpec s;
            s.name = rig.name;
            s.kind = rig.kind;
            s.members = rig.members;
            for (size_t f = 0; f < rig.frames.size(); f++) {
                std::vector<uint32_t> fr(rig.members.size(), kNoImage);
                size_t n = 0;
                for (size_t m = 0; m < fr.size(); m++) {
                    const uint32_t g = rig.frames[f][m];
                    if (g == kNoImage || g >= local.size() || local[g] == kNoImage) continue;
                    fr[m] = local[g];
                    n++;
                }
                if (!n) continue;
                s.frames.push_back(std::move(fr));
                if (f < rig.frame_keys.size()) s.frame_keys.push_back(rig.frame_keys[f]);
            }
            out.rigs.push_back(std::move(s));
        }
        out.index(num_local);
        return out;
    }

    // 图像 ID 整体增加 offset，用于联合 BA 拼接多个模型。
    RigTable shifted(uint32_t offset, size_t num_images) const {
        RigTable out = *this;
        for (RigSpec& rig : out.rigs)
            for (auto& fr : rig.frames)
                for (uint32_t& img : fr)
                    if (img != kNoImage) img += offset;
        out.index(num_images);
        return out;
    }
};

// --rig [KIND=][CAPTURES:]MEMBERS，两部分均为逗号分隔前缀，如 cam0,cam1、vid1,vid2:cam0,cam1、*:cam0,cam1；成功返回空字符串。
inline std::string parseRigArg(const std::string& arg, RigDef& out) {
    out = RigDef{};
    std::string v = arg;
    if (v.compare(0, 13, "dual-fisheye=") == 0) {
        out.kind = "dual-fisheye";
        v.erase(0, 13);
    }
    auto split = [](const std::string& s, std::vector<std::string>& into) {
        for (size_t i = 0;;) {
            size_t c = s.find(',', i);
            if (c == std::string::npos) c = s.size();
            std::string p = s.substr(i, c - i);
            while (!p.empty() && (p.back() == '/' || p.back() == '\\')) p.pop_back();
            if (p.empty()) return false;
            into.push_back(p);
            if (c == s.size()) break;
            i = c + 1;
        }
        return true;
    };
    const size_t colon = v.find(':');
    std::vector<std::string> members;
    if (colon != std::string::npos && !split(v.substr(0, colon), out.captures))
        return "--rig '" + v + "': empty capture prefix";
    if (!split(colon == std::string::npos ? v : v.substr(colon + 1), members))
        return "--rig '" + v + "': empty member prefix";
    for (const std::string& p : members) {
        RigMemberDef m;
        m.prefix = p;
        out.members.push_back(m);
    }
    if (out.members.size() < 2) return "--rig '" + v + "': a rig needs at least two members";
    return applyRigKind(out);
}

// ---------------- 按图像名称解析定义 ----------------

namespace rig_detail {

inline bool prefixMatches(const std::string& name, const std::string& prefix) {
    if (prefix.empty()) return true;
    if (name.size() < prefix.size() || name.compare(0, prefix.size(), prefix) != 0) return false;
    return name.size() == prefix.size() || name[prefix.size()] == '/';
}

// 帧键为成员前缀下的完整相对路径；数据库已去除扩展名，1756371636.290711 等时间戳的小数部分必须保留。
inline std::string frameKey(const std::string& name, const std::string& prefix) {
    std::string rest = prefix.empty() ? name : name.substr(prefix.size());
    if (!rest.empty() && rest[0] == '/') rest.erase(0, 1);
    return rest;
}

}  // 命名空间 rig_detail

// 按 ID 对应的 names 组帧，最长成员前缀优先；图像属于两个 rig 或成员无匹配图像时均报错。
inline RigTable buildRigTable(const std::vector<std::string>& names,
                              const std::vector<RigDef>& defs) {
    RigTable out;
    for (const RigDef& d : defs) {
        if (d.members.size() < 2)
            throw std::runtime_error("rig " + d.name + ": a rig needs at least two members");
        RigSpec s;
        s.name = d.name.empty() ? "rig" + std::to_string(out.rigs.size()) : d.name;
        s.kind = d.kind;
        s.members = d.members;
        // 成员相对指定采集前缀，* 展开为全部顶层目录；未指定时使用空采集前缀。
        std::vector<std::string> captures = d.captures;
        if (captures.size() == 1 && captures[0] == "*") {
            std::set<std::string> tops;
            for (const std::string& n : names) {
                const size_t slash = n.find('/');
                if (slash != std::string::npos) tops.insert(n.substr(0, slash));
            }
            captures.assign(tops.begin(), tops.end());
            if (captures.empty())
                throw std::runtime_error("rig " + s.name + ": '*' found no folders to be captures");
        }
        if (captures.empty()) captures.push_back("");
        std::map<std::string, size_t> key_to_frame;
        std::vector<size_t> used(d.members.size(), 0);
        for (uint32_t img = 0; img < names.size(); img++) {
            int best = -1, best_cap = -1;
            size_t best_len = 0;
            for (size_t ci = 0; ci < captures.size(); ci++)
                for (size_t m = 0; m < d.members.size(); m++) {
                    const std::string pre = captures[ci].empty()
                                                ? d.members[m].prefix
                                                : captures[ci] + "/" + d.members[m].prefix;
                    if (!rig_detail::prefixMatches(names[img], pre)) continue;
                    if (best < 0 || pre.size() > best_len) {
                        best = (int)m;
                        best_cap = (int)ci;
                        best_len = pre.size();
                    }
                }
            if (best < 0) continue;
            const std::string pre = captures[best_cap].empty()
                                        ? d.members[best].prefix
                                        : captures[best_cap] + "/" + d.members[best].prefix;
            std::string key = rig_detail::frameKey(names[img], pre);
            if (!captures[best_cap].empty()) key = std::to_string(best_cap) + "/" + key;
            auto it = key_to_frame.find(key);
            if (it == key_to_frame.end()) {
                it = key_to_frame.emplace(key, s.frames.size()).first;
                s.frames.emplace_back(d.members.size(), kNoImage);
                s.frame_keys.push_back(key);
            }
            std::vector<uint32_t>& fr = s.frames[it->second];
            if (fr[best] != kNoImage)
                throw std::runtime_error("rig " + s.name + ": images " + names[fr[best]] +
                                         " and " + names[img] + " both fill member " +
                                         d.members[best].prefix + " of frame " + key);
            fr[best] = img;
            used[best]++;
        }
        for (size_t m = 0; m < d.members.size(); m++)
            if (!used[m])
                throw std::runtime_error("rig " + s.name + ": no image matches member '" +
                                         d.members[m].prefix + "'");
        out.rigs.push_back(std::move(s));
    }
    out.index(names.size());
    return out;
}

// ---------------- 模型保存的标定 ----------------

inline Pose invertPose(const Pose& p) {
    Mat3 Rt = transpose(p.R);
    Vec3 t = mul(Rt, p.t);
    return {Rt, {-t.x, -t.y, -t.z}};
}

// 由 world -> A 和 world -> B 求 A -> B。
inline Pose relativePose(const Pose& a, const Pose& b) {
    Mat3 R = mul(b.R, transpose(a.R));
    return {R, b.t - mul(R, a.t)};
}

inline double rotationAngleDeg(const Mat3& R) {
    const double tr = std::max(-1.0, std::min(1.0, (R[0] + R[4] + R[8] - 1.0) * 0.5));
    return std::acos(tr) * 180.0 / M_PI;
}

// rig 坐标系采用参考成员的相机坐标系，cam_from_rig[ref] 为恒等；平移使用模型单位，随模型尺度变化。
struct RigCalib {
    int ref = -1;
    std::vector<Pose> cam_from_rig;
    std::vector<uint8_t> established;  // 逐成员是否可作为约束
    std::vector<uint8_t> fixed;        // 逐成员是否在 BA 中固定
    std::vector<uint32_t> support;     // 支持该估计的帧数
    std::vector<double> spread_deg;    // 支持帧的角度偏差中位数
    std::vector<uint32_t> declined_at; // 上次报告不同步时的帧数
    double user_scale = 0;             // 每用户单位对应的模型单位数，0 表示未知

    bool usable(uint32_t m) const { return ref >= 0 && m < established.size() && established[m]; }
    size_t numEstablished() const {
        size_t n = 0;
        for (uint8_t e : established) n += e ? 1 : 0;
        return n;
    }
    void resize(size_t n) {
        cam_from_rig.assign(n, Pose{mat3Identity(), {0, 0, 0}});
        established.assign(n, 0);
        fixed.assign(n, 0);
        support.assign(n, 0);
        spread_deg.assign(n, 0);
        declined_at.assign(n, 0);
    }
    // 在成员图像位姿与 rig 帧位姿之间转换。
    Pose rigFromWorld(uint32_t m, const Pose& cam_from_world) const {
        return composePose(invertPose(cam_from_rig[m]), cam_from_world);
    }
    Pose camFromWorld(uint32_t m, const Pose& rig_from_world) const {
        return composePose(cam_from_rig[m], rig_from_world);
    }
    // 由 from 成员位姿推算 to 成员位姿。
    Pose predict(uint32_t from, uint32_t to, const Pose& cam_from_world) const {
        return camFromWorld(to, rigFromWorld(from, cam_from_world));
    }
};

// 模型规范变换时，成员旋转不变、平移按尺度变化，与 transformPose 的复合一致。
inline void transformRigs(std::vector<RigCalib>& rigs, double scale) {
    for (RigCalib& c : rigs)
        for (Pose& p : c.cam_from_rig) p.t = p.t * scale;
}

struct RigCalibOptions {
    // 成员外参可信前所需共同配准帧数及一致比例；不同步镜头仅约 55–80% 帧一致，刚性同步装置应几乎全部一致。
    int min_frames = 3;
    double min_inlier_frac = 0.9;
    // 相对位姿角度偏差中位数超过阈值则不视为刚性；真实 rig 通常约 0.2–0.45 度。
    double max_spread_deg = 1.0;
};

namespace rig_detail {

// 对旋转求弦平均，再投影回 SO(3)。
inline Mat3 meanRotation(const std::vector<Mat3>& Rs, const std::vector<char>* mask = nullptr) {
    Mat3 acc{};
    for (size_t i = 0; i < Rs.size(); i++) {
        if (mask && !(*mask)[i]) continue;
        for (int k = 0; k < 9; k++) acc[k] += Rs[i][k];
    }
    Svd3 s = svd3(acc);
    Mat3 D = mat3Identity();
    D[8] = det3(s.U) * det3(s.V) < 0 ? -1.0 : 1.0;
    return mul(mul(s.U, D), transpose(s.V));
}

inline double medianOf(std::vector<double> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

}  // 命名空间 rig_detail

// 对共同配准帧的 cam_ref -> cam_m 做稳健平均，返回内点数，spread 为其偏差中位数。
inline int averageRelativePoses(const std::vector<Pose>& rel, const RigCalibOptions& opt,
                                Pose& out, double& spread, const Mat3* fixed_R = nullptr) {
    using rig_detail::meanRotation;
    using rig_detail::medianOf;
    const size_t n = rel.size();
    if (!n) return 0;
    std::vector<Mat3> Rs(n);
    for (size_t i = 0; i < n; i++) Rs[i] = rel[i].R;
    Mat3 R0 = fixed_R ? *fixed_R : meanRotation(Rs);
    std::vector<double> dev(n);
    for (size_t i = 0; i < n; i++) dev[i] = rotationAngleDeg(mul(Rs[i], transpose(R0)));
    const double med = medianOf(dev);
    const double thr = std::max(3.0 * opt.max_spread_deg, 3.0 * med);
    std::vector<char> in(n, 0);
    int count = 0;
    for (size_t i = 0; i < n; i++) count += (in[i] = dev[i] <= thr) ? 1 : 0;
    if (!count) return 0;
    if (!fixed_R) {
        R0 = meanRotation(Rs, &in);
        for (size_t i = 0; i < n; i++) dev[i] = rotationAngleDeg(mul(Rs[i], transpose(R0)));
    }
    std::vector<double> tx, ty, tz, dv;
    for (size_t i = 0; i < n; i++) {
        if (!in[i]) continue;
        // 相对平均旋转表达平移，避免逐帧旋转误差污染基线投票。
        tx.push_back(rel[i].t.x);
        ty.push_back(rel[i].t.y);
        tz.push_back(rel[i].t.z);
        dv.push_back(dev[i]);
    }
    out.R = R0;
    out.t = {medianOf(tx), medianOf(ty), medianOf(tz)};
    spread = medianOf(dv);
    return count;
}

}  // 命名空间 sfm
