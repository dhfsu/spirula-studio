// 重建模型合并：通过共享图像位姿估计 Sim(3)，变换相机与点，按共享特征拼接轨迹并过滤。
// alignReconstructions 负责像素评分的稳健对齐，mergeInto 执行变换，MergeSession 决定顺序和接受；在副本上验证，通过后才提交。
// 两模型必须来自同一特征集合，保证 point2D_idx 含义一致；共享 ID 的名称或关键点数不符时拒绝（D43）。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sfm/core/Model.h"
#include "sfm/core/Pose.h"
#include "sfm/geometry/LinAlg.h"
#include "sfm/geometry/Triangulation.h"
#include "sfm/optim/Ransac.h"
#include "sfm/core/Log.h"

namespace sfm {

// ---------------- 重复结构与折叠（D45）----------------
// 相似房间或立面可能被错误叠到同处且内部自洽；同位同向图像却无共同三维结构是缺失对应信号。
// 实测折叠同侧共享点中位数 454，跨侧为 0，73% 少于 5；参考 Heinly 等的重复结构检测。

// 对应图持有者提供已验证匹配支持，避免将轨迹碎裂误判为折叠；缺少此判据曾把正确的 261 图模型切成 144+96+20。
struct DuplicateOptions {
    // 相机中心距离小于模型最近邻距离中位数的此倍数时视为同位，无需世界单位。
    double radius_nn = 2.0;
    double min_view_cos = 0.5;      // 主光轴夹角 60 度
    int min_image_points = 50;      // 观测结构不足的图像不能作为证据
    double max_shared_frac = 0.02;  // 共享点数/min(观测数) 低于此值视为冲突
    // 至少需足够冲突对才考虑折叠，少量偶发冲突不触发判断。
    int min_conflicts = 3;
    // 冲突率默认不作判据，0 禁用；可靠判据是切分损失的共视比例，保留参数仅便于测量。
    double min_conflict_ratio = 0.0;
    // 切分损失的共视权重占比是拒绝切割的条件，冲突须先触发候选切分；真实重叠副本本就几乎不共享结构。
    // 合成真折叠切割损失为零，某正确 1243 图模型切割损失 1.30% 却丢 568 图、65 分 AUC，因此门限需保留充分间隔。
    double max_cut_fraction = 0.005;
    // 切出的组须有足够图像与外组同位，才真是重叠副本，否则并回主组（D67）。
    // 6112 图数据仅六对冲突曾以 0.08% 共视损失切走 358 图，使覆盖 90%->84%，说明低切割成本本身不够。
    double min_fold_overlap = 0.5;
};

struct DuplicateReport {
    size_t colocated = 0;   // 检查的同位同向图像对数
    size_t conflicts = 0;   // 几乎无共同结构且从未匹配的对数
    size_t unmatched_but_seen = 0;  // 无共同点但曾匹配的对数
    std::vector<std::pair<uint32_t, uint32_t>> pairs;  // 冲突图像对
    // 下项保存全部同位对，用于判断切出组是否真与其他组重叠，不能只看构造切分的冲突。
    std::vector<std::pair<uint32_t, uint32_t>> colocated_pairs;
    double ratio() const { return colocated ? (double)conflicts / (double)colocated : 0.0; }
    bool duplicated(const DuplicateOptions& o) const {
        return (int)conflicts >= o.min_conflicts && ratio() >= o.min_conflict_ratio;
    }
};

inline DuplicateReport findDuplicateStructure(const Reconstruction& m,
                                              const DuplicateOptions& opt = {},
                                              const MatchedFn& matched = nullptr) {
    DuplicateReport rep;
    std::vector<uint32_t> ids;
    std::vector<Vec3> C, fwd;
    std::vector<std::vector<uint64_t>> pts;
    for (const auto& kv : m.images) {
        if (!kv.second.registered) continue;
        std::vector<uint64_t> p = observedPoints(kv.second);
        if ((int)p.size() < opt.min_image_points) continue;
        ids.push_back(kv.first);
        C.push_back(cameraCenter(kv.second.pose));
        // R 第三行为世界坐标中的主光轴
        const Mat3& R = kv.second.pose.R;
        fwd.push_back({R[6], R[7], R[8]});
        pts.push_back(std::move(p));
    }
    const size_t n = ids.size();
    if (n < 4) return rep;

    // 以最近邻距离中位数定义模型自身长度尺度。
    std::vector<double> nn(n, 1e300);
    for (size_t i = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++) {
            double d2 = (C[i] - C[j]).dot(C[i] - C[j]);
            nn[i] = std::min(nn[i], d2);
            nn[j] = std::min(nn[j], d2);
        }
    std::vector<double> sorted = nn;
    std::sort(sorted.begin(), sorted.end());
    const double med = std::sqrt(std::max(1e-24, sorted[sorted.size() / 2]));
    const double r2 = (opt.radius_nn * med) * (opt.radius_nn * med);

    for (size_t i = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++) {
            Vec3 d = C[i] - C[j];
            if (d.dot(d) > r2) continue;
            if (fwd[i].dot(fwd[j]) < opt.min_view_cos) continue;
            rep.colocated++;
            rep.colocated_pairs.push_back({ids[i], ids[j]});
            size_t sh = sharedPoints(pts[i], pts[j]);
            double frac = (double)sh / (double)std::min(pts[i].size(), pts[j].size());
            if (frac >= opt.max_shared_frac) continue;
            // 无共同点且从未验证匹配才算冲突，否则可能只是轨迹未融合。
            if (matched && matched(ids[i], ids[j])) {
                rep.unmatched_but_seen++;
                continue;
            }
            rep.conflicts++;
            rep.pairs.push_back({ids[i], ids[j]});
        }
    return rep;
}

// 按共视边由强到弱执行带冲突约束的 Kruskal，禁止把冲突图像并到同组，分离错误叠合的场景副本。
// 过小组撤销配准；cut_out 记录共视损失，供调用方判断切分是否可信。
struct DuplicateCut {
    size_t groups = 0;   // Kruskal 产生的连通组数
    uint64_t severed = 0;  // 跨组共视权重
    uint64_t kept = 0;     // 组内共视权重
    // 切分丢弃的共视比例由跨组权重除以总权重计算。
    double fraction() const {
        const uint64_t t = severed + kept;
        return t ? (double)severed / (double)t : 0.0;
    }
    // 因无空间重叠而并回的组数，以及保留组中最小重叠比例；无数据时为 1。
    size_t reattached = 0;
    double min_overlap = 1.0;
};

// 既有折叠冲突，又能低成本切开才接受；两条件缺一不可。
inline bool foldSplitAccepted(const DuplicateReport& rep, const DuplicateCut& cut,
                              const DuplicateOptions& opt) {
    return rep.duplicated(opt) && cut.groups > 1 && cut.fraction() <= opt.max_cut_fraction;
}

inline std::vector<Reconstruction> splitDuplicateStructure(const Reconstruction& m,
                                                           const DuplicateReport& rep,
                                                           size_t min_group,
                                                           size_t* dropped_out = nullptr,
                                                           DuplicateCut* cut_out = nullptr,
                                                           double min_fold_overlap = 0.0) {
    if (rep.pairs.empty()) return {m};
    std::vector<uint32_t> ids;
    std::map<uint32_t, size_t> pos;
    for (const auto& kv : m.images)
        if (kv.second.registered) { pos[kv.first] = ids.size(); ids.push_back(kv.first); }
    if (ids.size() < 2) return {m};

    // 从轨迹构造共视图，成本为 sum |track|²，而非图像数平方。
    std::map<std::pair<size_t, size_t>, uint32_t> covis;
    for (const auto& kv : m.points3D) {
        const std::vector<TrackElement>& t = kv.second.track;
        if (t.size() > 64) continue;  // 所有图像都看到的点不提供分组区分度
        for (size_t a = 0; a < t.size(); a++)
            for (size_t b = a + 1; b < t.size(); b++) {
                auto ia = pos.find(t[a].image_id), ib = pos.find(t[b].image_id);
                if (ia == pos.end() || ib == pos.end()) continue;
                size_t x = ia->second, y = ib->second;
                if (x > y) std::swap(x, y);
                covis[{x, y}]++;
            }
    }
    struct Edge { uint32_t w; size_t a, b; };
    std::vector<Edge> edges;
    edges.reserve(covis.size());
    for (const auto& kv : covis) edges.push_back({kv.second, kv.first.first, kv.first.second});
    std::sort(edges.begin(), edges.end(),
              [](const Edge& x, const Edge& y) { return x.w > y.w; });

    std::vector<size_t> parent(ids.size());
    for (size_t i = 0; i < parent.size(); i++) parent[i] = i;
    std::function<size_t(size_t)> find = [&](size_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    std::vector<std::pair<size_t, size_t>> conflict;
    for (const auto& p : rep.pairs) {
        auto a = pos.find(p.first), b = pos.find(p.second);
        if (a != pos.end() && b != pos.end()) conflict.push_back({a->second, b->second});
    }
    for (const Edge& e : edges) {
        size_t ra = find(e.a), rb = find(e.b);
        if (ra == rb) continue;
        bool forbidden = false;
        for (const auto& c : conflict) {
            size_t rc = find(c.first), rd = find(c.second);
            if ((rc == ra && rd == rb) || (rc == rb && rd == ra)) { forbidden = true; break; }
        }
        if (forbidden) continue;
        parent[ra] = rb;
    }

    std::map<size_t, std::vector<uint32_t>> groups;
    for (size_t i = 0; i < ids.size(); i++) groups[find(i)].push_back(ids[i]);
    std::vector<std::vector<uint32_t>> gs;
    for (auto& kv : groups) gs.push_back(std::move(kv.second));
    std::sort(gs.begin(), gs.end(),
              [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
                  return a.size() > b.size();
              });
    if (gs.size() <= 1) return {m};

    // 切出组若没有与其他组明显空间重叠，则只是被误切的连续区域，应并回最大组（D67）。
    if (min_fold_overlap > 0.0 && !rep.colocated_pairs.empty()) {
        std::map<uint32_t, size_t> group_of;
        for (size_t g = 0; g < gs.size(); g++)
            for (uint32_t id : gs[g]) group_of[id] = g;
        std::vector<std::set<uint32_t>> covered(gs.size());
        for (const auto& p : rep.colocated_pairs) {
            auto a = group_of.find(p.first), b = group_of.find(p.second);
            if (a == group_of.end() || b == group_of.end() || a->second == b->second) continue;
            covered[a->second].insert(p.first);
            covered[b->second].insert(p.second);
        }
        std::vector<std::vector<uint32_t>> keep;
        keep.push_back(std::move(gs[0]));  // 最大组作为其他组的回退归属
        for (size_t g = 1; g < gs.size(); g++) {
            const double ov = (double)covered[g].size() / (double)gs[g].size();
            if (ov >= min_fold_overlap) {
                if (cut_out) cut_out->min_overlap = std::min(cut_out->min_overlap, ov);
                keep.push_back(std::move(gs[g]));
            } else {
                if (cut_out) cut_out->reattached++;
                keep[0].insert(keep[0].end(), gs[g].begin(), gs[g].end());
            }
        }
        gs = std::move(keep);
        std::sort(gs.begin(), gs.end(),
                  [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
                      return a.size() > b.size();
                  });
        if (gs.size() <= 1) return {m};
    }

    // 按重新并回后的实际分组计算切割代价，不能沿用 Kruskal 原提议；每条边恰好属于组内或跨组。
    if (cut_out) {
        std::vector<size_t> gid(ids.size(), (size_t)-1);
        for (size_t g = 0; g < gs.size(); g++)
            for (uint32_t id : gs[g]) gid[pos.at(id)] = g;
        for (const auto& kv : covis) {
            if (gid[kv.first.first] == gid[kv.first.second]) cut_out->kept += kv.second;
            else cut_out->severed += kv.second;
        }
        cut_out->groups = gs.size();
    }

    std::vector<Reconstruction> parts;
    size_t dropped = 0;
    for (const std::vector<uint32_t>& g : gs) {
        if (g.size() < min_group) { dropped += g.size(); continue; }
        parts.push_back(subsetModel(m, std::set<uint32_t>(g.begin(), g.end())));
    }
    if (dropped_out) *dropped_out = dropped;
    if (parts.empty()) return {m};
    return parts;
}

// 为 MergeOptions::validate 提前声明，定义位于下方 mergeInto 附近。
struct MergeCounts;

struct MergeOptions {
    // 对齐 RANSAC 使用像素内点阈值；独立优化模型的共享位姿可有更大误差，因此通常比建图过滤更宽松，COLMAP 合并默认 8 px。
    int min_common_images = 3;         // 两个共享位姿确定 Sim3，第三个提供冗余验证
    double max_reproj_error = 8.0;
    double min_inlier_ratio = 0.3;     // COLMAP 的 kMinInlierObservations
    int max_alignment_points = 100;    // 每共享图像采样的三维点数
    int ransac_max_trials = 1000;
    unsigned seed = 0;
    // 拼接轨迹尚未经过跨接缝 BA，合并后过滤沿用建图标准。
    double filter_reproj_error = 4.0;
    double min_tri_angle_deg = 1.5;
    // 错误重复结构对齐可能通过 RANSAC，因此还检查新增图像是否失去观测，以及锚点原观测是否被破坏；任一失败都回滚。
    int min_image_points = 5;          // 图像点数低于此值视为无充分支持
    double max_hollow_ratio = 0.34;    // 占新增图像的比例
    double max_anchor_obs_loss = 0.2;  // 占锚点原观测的比例
    double max_splice_conflict_ratio = 0.5;  // 共享点中允许的冲突比例
    // 大量共享位姿可靠时，可精化后用独立证据仲裁形状漂移（D64）；5356 图数据的两模型共享 1291 图，但 900671 共享点中 578129 个不一致。
    // 下项为 0 时恢复直接拒绝。
    int splice_arbitrate_inliers = 20;
    // 可检查合并新引入的折叠，不计锚点已有冲突；默认关闭逐次检查，最终统一切割更便宜。
    bool check_duplicate = false;
    DuplicateOptions duplicate;
    // 共享点冲突按独立像素阈值判断，不能随对齐阈值一同放宽，否则难对齐数据的错误检查也会失效。
    double splice_tolerance = 8.0;
    // 内置检查后调用外部验证器，以未参与对齐的对应图证据判断接缝；空字符串接受，否则返回原因。
    // 验证器可先精化 merged 再复判，通过后提交精化结果，避免只能拒绝轻微漂移的正确合并（D45/D64）。
    std::function<std::string(Reconstruction& merged, const Reconstruction& src,
                              const Sim3& transform, const MergeCounts& counts)> validate;
    bool verbose = true;
    // 不同模型持有同帧不同镜头时，可通过 rig 标定形成位姿对应，无需共享同一图像。
    const RigTable* rigs = nullptr;
};

// ---------------- Umeyama 最小二乘相似变换 ----------------
// 求 src 到 dst 的变换，最小化 sum |dst_i-(s R src_i+t)|²。
inline bool estimateSim3(const std::vector<Vec3>& src, const std::vector<Vec3>& dst, Sim3& out) {
    const size_t n = src.size();
    if (n < 3 || dst.size() != n) return false;
    Vec3 ms{0, 0, 0}, md{0, 0, 0};
    for (size_t i = 0; i < n; i++) { ms = ms + src[i]; md = md + dst[i]; }
    ms = ms * (1.0 / n);
    md = md * (1.0 / n);
    Mat3 sigma{};
    double var_s = 0;
    for (size_t i = 0; i < n; i++) {
        Vec3 a = src[i] - ms, b = dst[i] - md;
        var_s += a.dot(a);
        const double av[3] = {a.x, a.y, a.z}, bv[3] = {b.x, b.y, b.z};
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) sigma[3 * r + c] += bv[r] * av[c];
    }
    var_s /= (double)n;
    for (double& v : sigma) v /= (double)n;
    if (!(var_s > 1e-12)) return false;  // 全部源中心重合

    Svd3 s = svd3(sigma);
    // det(U)det(V)<0 表示 U V^T 为反射，翻转最小奇异方向得到最佳正旋转。
    Mat3 D = mat3Identity();
    const double flip = det3(s.U) * det3(s.V) < 0 ? -1.0 : 1.0;
    D[8] = flip;
    out.R = mul(mul(s.U, D), transpose(s.V));
    const double trace = s.s.x + s.s.y + flip * s.s.z;
    out.scale = trace / var_s;
    if (!(out.scale > 1e-12) || !std::isfinite(out.scale)) return false;
    out.t = md - mul(out.R, ms) * out.scale;
    for (double v : out.R)
        if (!std::isfinite(v)) return false;
    return std::isfinite(out.t.x) && std::isfinite(out.t.y) && std::isfinite(out.t.z);
}

// 共享图像同时提供旋转与中心：单个位姿即可给出 T.R=R_dst^T R_src，再由两个中心求尺度和平移。
// 仅用三个中心的 Umeyama 在走廊或立面共线轨迹中无法确定绕线旋转，因此使用完整位姿。
inline bool estimateSim3FromPoses(const std::vector<Pose>& src, const std::vector<Pose>& dst,
                                  Sim3& out) {
    const size_t n = src.size();
    if (n < 2 || dst.size() != n) return false;
    // 逐图旋转估计求弦平均，再投影回 SO(3)。
    Mat3 acc{};
    for (size_t i = 0; i < n; i++) {
        Mat3 Ri = mul(transpose(dst[i].R), src[i].R);
        for (int k = 0; k < 9; k++) acc[k] += Ri[k];
    }
    Svd3 s = svd3(acc);
    Mat3 D = mat3Identity();
    D[8] = det3(s.U) * det3(s.V) < 0 ? -1.0 : 1.0;
    out.R = mul(mul(s.U, D), transpose(s.V));

    // 旋转固定后，中心的一维最小二乘确定尺度与平移。
    std::vector<Vec3> cs(n), cd(n);
    Vec3 ms{0, 0, 0}, md{0, 0, 0};
    for (size_t i = 0; i < n; i++) {
        cs[i] = cameraCenter(src[i]);
        cd[i] = cameraCenter(dst[i]);
        ms = ms + cs[i];
        md = md + cd[i];
    }
    ms = ms * (1.0 / n);
    md = md * (1.0 / n);
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        Vec3 a = mul(out.R, cs[i] - ms), b = cd[i] - md;
        num += a.dot(b);
        den += a.dot(a);
    }
    if (!(den > 1e-12)) return false;  // 所有共享图像位于同一位置
    out.scale = num / den;
    if (!(out.scale > 1e-12) || !std::isfinite(out.scale)) return false;
    out.t = md - mul(out.R, ms) * out.scale;
    return std::isfinite(out.t.x) && std::isfinite(out.t.y) && std::isfinite(out.t.z);
}

// ---------------- 独立重建模型的重投影辅助函数 ----------------
// 合并没有 FeatureSet，使用模型 points2D 保存的相同关键点坐标。
inline double reprojErrorAt(const Camera& cam, const Pose& pose, const Vec2& obs, const Vec3& X) {
    Vec3 pc = mul(pose.R, X) + pose.t;
    // 正深度与建图器一致，针孔按 z，宽角按观测射线方向（D33）。
    if (cam.wideFov()) {
        if (pc.dot(cam.bearing(obs)) <= 0) return 1e30;
    } else if (pc.z < 1e-8) {
        return 1e30;
    }
    Vec2 px = cam.project(pc);
    return std::hypot(px.x - obs.x, px.y - obs.y);
}

inline size_t countObservations(const Reconstruction& rec) {
    size_t n = 0;
    for (const auto& kv : rec.points3D) n += kv.second.track.size();
    return n;
}

// 对独立模型过滤重投影差的观测与缺视差点；max_err 使用提取像素，由相机尺度换算（D47）。
inline void filterModel(Reconstruction& rec, double max_err, double min_ang_deg,
                        size_t& removed_obs, size_t& removed_pts) {
    removed_obs = removed_pts = 0;
    std::map<uint32_t, Vec3> centers;
    for (const auto& kv : rec.images)
        if (kv.second.registered) centers[kv.first] = cameraCenter(kv.second.pose);
    const double min_ang = min_ang_deg * M_PI / 180.0;
    std::vector<uint64_t> drop;
    for (auto& kv : rec.points3D) {
        Point3D& pt = kv.second;
        std::vector<TrackElement> keep;
        for (const TrackElement& e : pt.track) {
            auto im = rec.images.find(e.image_id);
            if (im == rec.images.end() || !im->second.registered ||
                e.point2D_idx >= im->second.points2D.size()) {
                removed_obs++;
                continue;
            }
            const Camera& cam = rec.cameras.at(im->second.camera_id);
            // 将提取像素 max_err 换为当前相机源图像素。
            if (reprojErrorAt(cam, im->second.pose, im->second.points2D[e.point2D_idx], pt.xyz) <=
                cam.errPx(max_err)) {
                keep.push_back(e);
            } else {
                im->second.point3D_ids[e.point2D_idx] = kInvalidPoint3D;
                removed_obs++;
            }
        }
        pt.track = keep;
        bool degenerate = pt.track.size() < 2;
        if (!degenerate) {
            double best = 0;
            for (size_t i = 0; i + 1 < pt.track.size() && best < min_ang; i++)
                for (size_t j = i + 1; j < pt.track.size() && best < min_ang; j++)
                    best = std::max(best, triangulationAngle(pt.xyz, centers[pt.track[i].image_id],
                                                             centers[pt.track[j].image_id]));
            degenerate = best < min_ang;
        }
        if (degenerate) {
            for (const TrackElement& e : pt.track) {
                auto im = rec.images.find(e.image_id);
                if (im != rec.images.end() && e.point2D_idx < im->second.point3D_ids.size())
                    im->second.point3D_ids[e.point2D_idx] = kInvalidPoint3D;
                removed_obs++;
            }
            drop.push_back(kv.first);
        }
    }
    for (uint64_t id : drop) {
        rec.points3D.erase(id);
        removed_pts++;
    }
}

// ---------------- 模型对齐 ----------------

struct AlignmentResult {
    Sim3 transform;                // 源世界坐标 -> 目标世界坐标
    size_t common_images = 0;      // 可用于对齐的共享图像数
    size_t inliers = 0;
    double mean_error = 0;         // 内点平均误差，单位像素
    bool success = false;
    std::string reason;
    // 标记是否由共享三维结构而非共享图像估计变换，并记录三维对应数；结构对齐可没有共同相机（D70）。
    size_t structure_pairs = 0;
    bool from_structure = false;
    size_t rig_views = 0;   // 共同位姿中由 rig 提供的数量
};

// ---------------- 相机装置 ----------------

// 普通图像以 ID 为键，rig 图像以帧为键，使同帧不同镜头计为一次重叠。
inline uint64_t overlapKey(const RigTable* rigs, uint32_t img) {
    if (!rigs) return img;
    const RigSlot sl = rigs->slot(img);
    if (!sl.valid()) return img;
    uint64_t k = 1ull << 40;
    for (uint32_t r = 0; r < sl.rig; r++) k += rigs->rigs[r].frames.size();
    return k + sl.frame;
}

// 由 src 中已标定、已配准的 rig 伙伴推算 dst_image 在 src 坐标中的位姿，无可用伙伴时失败。
inline bool rigPoseInSrc(const Reconstruction& src, const RigTable& rigs, uint32_t dst_image,
                         Pose& out) {
    const RigSlot sl = rigs.slot(dst_image);
    if (!sl.valid() || sl.rig >= src.rigs.size()) return false;
    if (src.rig_detached.count(dst_image)) return false;
    const RigCalib& c = src.rigs[sl.rig];
    if (!c.usable(sl.member)) return false;
    const std::vector<uint32_t>& fr = rigs.frameOf(sl);
    int best = -1;
    uint32_t best_pts = 0;
    for (uint32_t m = 0; m < fr.size(); m++) {
        if (m == sl.member || fr[m] == kNoImage || !c.usable(m)) continue;
        if (src.rig_detached.count(fr[m])) continue;
        auto it = src.images.find(fr[m]);
        if (it == src.images.end() || !it->second.registered) continue;
        const uint32_t n = it->second.numPoint3D();
        if (best < 0 || n > best_pts) { best = (int)m; best_pts = n; }
    }
    if (best < 0) return false;
    out = c.predict((uint32_t)best, sl.member, src.images.at(fr[best]).pose);
    return true;
}

// 由有结构支持的伙伴推算的图像虽自身观测不足，位姿仍由 rig 约束成立。
inline bool rigCovered(const Reconstruction& m, const RigTable* rigs, uint32_t img, int min_pts) {
    if (!rigs || m.rig_detached.count(img)) return false;
    const RigSlot sl = rigs->slot(img);
    if (!sl.valid() || sl.rig >= m.rigs.size() || !m.rigs[sl.rig].usable(sl.member)) return false;
    const std::vector<uint32_t>& fr = rigs->frameOf(sl);
    for (uint32_t k = 0; k < fr.size(); k++) {
        if (k == sl.member || fr[k] == kNoImage || !m.rigs[sl.rig].usable(k)) continue;
        auto it = m.images.find(fr[k]);
        if (it != m.images.end() && it->second.registered &&
            (int)it->second.numPoint3D() >= min_pts)
            return true;
    }
    return false;
}

// 合并标定优先保留目标已有项，缺项采用按目标尺度换算的来源标定。
inline void mergeRigCalibs(std::vector<RigCalib>& dst, const std::vector<RigCalib>& src,
                           double scale) {
    if (dst.size() < src.size()) dst.resize(src.size());
    for (size_t r = 0; r < src.size(); r++) {
        const RigCalib& s = src[r];
        RigCalib& d = dst[r];
        if (s.ref < 0) continue;
        if (d.ref < 0) {
            d = s;
            for (Pose& p : d.cam_from_rig) p.t = p.t * scale;
            continue;
        }
        if (d.ref != s.ref) continue;  // 两套参考坐标由后续求解协调
        for (size_t m = 0; m < s.cam_from_rig.size() && m < d.cam_from_rig.size(); m++) {
            if (d.established[m] || !s.established[m]) continue;
            d.cam_from_rig[m] = s.cam_from_rig[m];
            d.cam_from_rig[m].t = d.cam_from_rig[m].t * scale;
            d.established[m] = 1;
            d.fixed[m] = s.fixed[m];
            d.support[m] = s.support[m];
            d.spread_deg[m] = s.spread_deg[m];
        }
    }
}

// 位姿对应表示 src 为 dst_image 推算的位置。
struct PoseCorr {
    Pose src_pose;
    uint32_t dst_image = 0;
    bool by_rig = false;
};

// 收集共同图像及双方持有不同镜头的共同 rig 帧；后者使用变换源坐标系的 src 标定。
inline std::vector<PoseCorr> poseCorrespondences(const Reconstruction& src,
                                                 const Reconstruction& dst,
                                                 const RigTable* rigs) {
    std::vector<PoseCorr> out;
    std::set<uint32_t> covered;
    for (const auto& kv : src.images) {
        if (!kv.second.registered) continue;
        auto it = dst.images.find(kv.first);
        if (it == dst.images.end() || !it->second.registered) continue;
        out.push_back({kv.second.pose, kv.first, false});
        covered.insert(kv.first);
    }
    if (!rigs) return out;
    for (const auto& kv : dst.images) {
        if (!kv.second.registered || covered.count(kv.first)) continue;
        Pose p;
        if (rigPoseInSrc(src, *rigs, kv.first, p)) out.push_back({p, kv.first, true});
    }
    return out;
}

// 共同图像按 src ID 排序，要求同 ID 名称与关键点数一致，确保两模型来自相同特征；mismatched 记录违反项。
inline std::vector<uint32_t> sharedImages(const Reconstruction& a, const Reconstruction& b,
                                          size_t* mismatched = nullptr) {
    std::vector<uint32_t> out;
    size_t bad = 0;
    for (const auto& kv : a.images) {
        if (!kv.second.registered) continue;
        auto it = b.images.find(kv.first);
        if (it == b.images.end() || !it->second.registered) continue;
        if (kv.second.points2D.size() != it->second.points2D.size() ||
            (!kv.second.name.empty() && !it->second.name.empty() &&
             kv.second.name != it->second.name)) {
            bad++;
            continue;
        }
        out.push_back(kv.first);
    }
    if (mismatched) *mismatched = bad;
    return out;
}

// 用最少两个共享位姿估计 src->dst 相似变换，并以预测相机对目标三维点的重投影像素误差做 RANSAC。
// 像素评分无需已知场景尺度，错误尺度、旋转与平移均能反映在残差中。
inline AlignmentResult alignReconstructions(const Reconstruction& src, const Reconstruction& dst,
                                            const MergeOptions& opt) {
    AlignmentResult r;
    size_t mismatched = 0;
    std::vector<uint32_t> shared = sharedImages(src, dst, &mismatched);
    if (mismatched) {
        r.reason = std::to_string(mismatched) +
                   " shared image id(s) disagree on name or keypoint count: the models "
                   "were not built from the same features";
        return r;
    }

    // 每对应保存双方位姿及目标观测的采样三维点，控制评分成本。
    struct View {
        Pose src_pose, dst_pose;
        const Camera* cam = nullptr;
        std::vector<Vec3> pts;
        std::vector<Vec2> obs;
    };
    std::vector<View> views;
    const std::vector<PoseCorr> corr = poseCorrespondences(src, dst, opt.rigs);
    views.reserve(corr.size());
    for (const PoseCorr& pc : corr) {
        const uint32_t id = pc.dst_image;
        const Image& di = dst.images.at(id);
        auto cam = dst.cameras.find(di.camera_id);
        if (cam == dst.cameras.end()) continue;
        if (pc.by_rig) r.rig_views++;
        View v;
        v.src_pose = pc.src_pose;
        v.dst_pose = di.pose;
        v.cam = &cam->second;
        std::vector<uint32_t> feats;
        for (uint32_t f = 0; f < (uint32_t)di.point3D_ids.size(); f++)
            if (di.point3D_ids[f] != kInvalidPoint3D && dst.points3D.count(di.point3D_ids[f]))
                feats.push_back(f);
        if (feats.empty()) continue;  // 图像缺少可用于评分的点
        const size_t stride =
            std::max<size_t>(1, feats.size() / std::max(1, opt.max_alignment_points));
        for (size_t k = 0; k < feats.size(); k += stride) {
            v.pts.push_back(dst.points3D.at(di.point3D_ids[feats[k]]).xyz);
            v.obs.push_back(di.points2D[feats[k]]);
        }
        views.push_back(std::move(v));
    }
    r.common_images = views.size();
    if ((int)views.size() < std::max(2, opt.min_common_images)) {
        r.reason = std::to_string(views.size()) + " usable shared image(s), need " +
                   std::to_string(std::max(2, opt.min_common_images));
        return r;
    }

    // 预测位姿对目标点的平均像素误差；相机后方点采用有限截断值，避免单点决定整视图。
    const double cap = 10.0 * opt.max_reproj_error;
    auto viewError = [&](const Sim3& T, const View& v) {
        Pose pred = transformPose(T, v.src_pose);
        double sum = 0;
        for (size_t k = 0; k < v.pts.size(); k++)
            // 按提取像素评分，使不同缩放比例相机处于同一误差尺度（D47）。
            sum += std::min(cap, reprojErrorAt(*v.cam, pred, v.obs[k], v.pts[k]) /
                                     v.cam->pixel_scale);
        return sum / (double)v.pts.size();
    };

    auto fit = [&](const std::vector<int>& idx) {
        std::vector<Pose> s, d;
        s.reserve(idx.size());
        d.reserve(idx.size());
        for (int i : idx) {
            s.push_back(views[i].src_pose);
            d.push_back(views[i].dst_pose);
        }
        Sim3 T;
        std::vector<Sim3> out;
        if (estimateSim3FromPoses(s, d, T)) out.push_back(T);
        return out;
    };
    auto res = [&](const Sim3& T, int i) {
        double e = viewError(T, views[i]);
        return e * e;
    };
    RansacOptions ro;
    ro.max_error = opt.max_reproj_error;
    ro.max_num_trials = opt.ransac_max_trials;
    ro.seed = opt.seed;
    // 每假设两个位姿，使共享图像较少时仍能进行真正的 RANSAC。
    RansacReport<Sim3> rep = loransac<Sim3>((int)views.size(), 2, fit, fit, res, ro);

    if (!rep.success || (int)rep.num_inliers < std::max(2, opt.min_common_images)) {
        r.reason = "alignment found only " + std::to_string(rep.num_inliers) + "/" +
                   std::to_string(views.size()) + " consistent shared image(s)";
        return r;
    }
    const double ratio = (double)rep.num_inliers / (double)views.size();
    if (ratio < opt.min_inlier_ratio) {
        char buf[128];
        snprintf(buf, sizeof buf, "only %d/%zu shared images agree (%.0f%%, need %.0f%%)",
                 rep.num_inliers, views.size(), 100 * ratio, 100 * opt.min_inlier_ratio);
        r.reason = buf;
        return r;
    }
    double sum = 0;
    for (size_t i = 0; i < views.size(); i++)
        if (rep.inlier_mask[i]) sum += viewError(rep.model, views[i]);
    r.transform = rep.model;
    r.inliers = rep.num_inliers;
    r.mean_error = sum / (double)rep.num_inliers;
    r.success = true;
    return r;
}

// ---------------- 执行合并 ----------------

struct MergeCounts {
    size_t images_added = 0;
    size_t points_added = 0;    // 转为目标新点的来源点数
    size_t points_spliced = 0;  // 拼入目标已有点的来源点数
    size_t points_dropped = 0;  // 剩余有效观测不足的来源点数
    size_t obs_added = 0;
    size_t obs_filtered = 0;
    size_t points_filtered = 0;
    size_t hollow_images = 0;   // 低于最少支持点数的新增图像
    // 下项统计双方位置不一致的拼接点，是错误对齐的重要信号。
    size_t splice_conflicts = 0;
};

// 应用 T 并拼接、过滤，是否接受由调用方决定；共享特征对应的点按轨迹长度平均，至少两个自由观测可建立新点。
// 来源点若同时指向两个不同目标点则有歧义，直接丢弃。
inline MergeCounts mergeInto(Reconstruction& dst, const Reconstruction& src, const Sim3& T,
                             const MergeOptions& opt) {
    MergeCounts c;
    // 双方同 ID 相机采用目标内参，来源图像可能略变内参，由合并后 BA 协调。
    for (const auto& kv : src.cameras)
        if (!dst.cameras.count(kv.first)) dst.cameras[kv.first] = kv.second;
    mergeRigCalibs(dst.rigs, src.rigs, T.scale);
    dst.rig_detached.insert(src.rig_detached.begin(), src.rig_detached.end());

    std::vector<uint32_t> added;
    for (const auto& kv : src.images) {
        if (!kv.second.registered || dst.images.count(kv.first)) continue;
        Image im = kv.second;
        im.pose = transformPose(T, im.pose);
        im.point3D_ids.assign(im.points2D.size(), kInvalidPoint3D);
        dst.images[kv.first] = std::move(im);
        added.push_back(kv.first);
    }
    c.images_added = added.size();

    for (const auto& kv : src.points3D) {
        const Point3D& sp = kv.second;
        std::vector<TrackElement> free_track;  // 尚无目标三维点的观测
        std::set<uint64_t> existing;
        size_t bound = 0;
        for (const TrackElement& e : sp.track) {
            auto di = dst.images.find(e.image_id);
            if (di == dst.images.end() || !di->second.registered ||
                e.point2D_idx >= di->second.point3D_ids.size())
                continue;
            uint64_t id = di->second.point3D_ids[e.point2D_idx];
            if (id == kInvalidPoint3D) free_track.push_back(e);
            else { existing.insert(id); bound++; }
        }
        const Vec3 X = transformPoint(T, sp.xyz);
        if (existing.size() == 1 && free_track.size() + bound >= 2) {
            Point3D& tp = dst.points3D.at(*existing.begin());
            // 双方三角化的同一特征必须在变换后解释目标原观测；轨迹按索引相遇不保证几何一致，采样原观测可揭示错位。
            for (size_t k = 0, tested = 0; k < tp.track.size() && tested < 3; k++) {
                const TrackElement& e = tp.track[k];
                auto di = dst.images.find(e.image_id);
                if (di == dst.images.end() || e.point2D_idx >= di->second.points2D.size()) continue;
                auto cam = dst.cameras.find(di->second.camera_id);
                if (cam == dst.cameras.end()) continue;
                tested++;
                if (reprojErrorAt(cam->second, di->second.pose,
                                  di->second.points2D[e.point2D_idx], X) >
                    cam->second.errPx(opt.splice_tolerance)) {
                    c.splice_conflicts++;
                    break;
                }
            }
            // 同轨迹同图像只能有一个特征，优先保留已有观测。
            std::set<uint32_t> in_track;
            for (const TrackElement& e : tp.track) in_track.insert(e.image_id);
            size_t attached = 0;
            for (const TrackElement& e : free_track) {
                if (!in_track.insert(e.image_id).second) continue;
                tp.track.push_back(e);
                dst.images[e.image_id].point3D_ids[e.point2D_idx] = *existing.begin();
                attached++;
            }
            // 同一点的两个位置按各自轨迹长度加权平均，与 COLMAP 一致。
            const double wo = (double)(tp.track.size() - attached), wn = (double)sp.track.size();
            if (wo + wn > 0) {
                tp.xyz = (tp.xyz * wo + X * wn) * (1.0 / (wo + wn));
                for (int k = 0; k < 3; k++)
                    tp.rgb[k] = (uint8_t)std::lround((tp.rgb[k] * wo + sp.rgb[k] * wn) / (wo + wn));
            }
            c.obs_added += attached;
            c.points_spliced++;
        } else if (existing.empty() && free_track.size() >= 2) {
            uint64_t id = dst.addPoint3D(X, free_track);
            dst.points3D[id].rgb[0] = sp.rgb[0];
            dst.points3D[id].rgb[1] = sp.rgb[1];
            dst.points3D[id].rgb[2] = sp.rgb[2];
            c.obs_added += free_track.size();
            c.points_added++;
        } else {
            c.points_dropped++;
        }
    }

    filterModel(dst, opt.filter_reproj_error, opt.min_tri_angle_deg, c.obs_filtered,
                c.points_filtered);
    for (uint32_t id : added)
        if ((int)dst.images.at(id).numPoint3D() < opt.min_image_points &&
            !rigCovered(dst, opt.rigs, id, opt.min_image_points))
            c.hollow_images++;
    return c;
}

// ---------------- 合并策略 ----------------

// 自动策略排序的合并候选，也供 GUI 选择，统一通过 tryMerge 执行。
struct MergeCandidate {
    size_t dst = 0, src = 0;     // 将 src 合入 dst
    size_t common_images = 0;
};

struct MergeAttempt {
    size_t dst = 0, src = 0;
    AlignmentResult alignment;
    MergeCounts counts;
    bool merged = false;
    std::string reason;          // 未合并时的拒绝原因
};

// MergeSession 持有模型，支持自动循环或交互逐步合并；外部可预览估计变换，也可提供变换，仍执行完整验证。
// 模型索引稳定，被吸收者保留无效槽；成功提交后释放来源模型，界面若需撤销须自行保存副本。
class MergeSession {
public:
    explicit MergeSession(std::vector<Reconstruction> models, MergeOptions opt = {})
        : models_(std::move(models)), alive_(models_.size(), 1), opt_(opt) {}

    size_t numModels() const { return models_.size(); }
    bool alive(size_t i) const { return i < alive_.size() && alive_[i]; }
    const Reconstruction& model(size_t i) const { return models_.at(i); }
    const std::vector<MergeAttempt>& log() const { return log_; }
    const MergeOptions& options() const { return opt_; }
    MergeOptions& options() { return opt_; }

    size_t commonImages(size_t a, size_t b) const {
        if (!alive(a) || !alive(b) || a == b) return 0;
        if (!opt_.rigs) return sharedImages(models_[a], models_[b]).size();
        std::set<uint64_t> keys;
        for (const auto& kv : models_[a].images)
            if (kv.second.registered) keys.insert(overlapKey(opt_.rigs, kv.first));
        size_t n = 0;
        std::set<uint64_t> seen;
        for (const auto& kv : models_[b].images)
            if (kv.second.registered) {
                const uint64_t k = overlapKey(opt_.rigs, kv.first);
                if (keys.count(k) && seen.insert(k).second) n++;
            }
        return n;
    }

    // 优先共同图像最多的模型对，再优先更大锚点；大模型保持规范和内参，小模型移动。
    std::vector<MergeCandidate> candidates() const {
        // 用重叠键到模型列表的倒排索引计数，避免模型两两求交；rig 使用帧键。
        std::unordered_map<uint64_t, std::vector<uint32_t>> holders;
        for (size_t i = 0; i < models_.size(); i++) {
            if (!alive_[i]) continue;
            for (const auto& kv : models_[i].images) {
                if (!kv.second.registered) continue;
                std::vector<uint32_t>& h = holders[overlapKey(opt_.rigs, kv.first)];
                if (h.empty() || h.back() != (uint32_t)i) h.push_back((uint32_t)i);
            }
        }
        std::unordered_map<uint64_t, size_t> shared;  // (lo << 32 | hi) -> 共享图像数
        for (const auto& kv : holders) {
            const std::vector<uint32_t>& v = kv.second;  // 构造时已保持升序
            for (size_t a = 0; a + 1 < v.size(); a++)
                for (size_t b = a + 1; b < v.size(); b++)
                    shared[((uint64_t)v[a] << 32) | v[b]]++;
        }
        std::vector<MergeCandidate> out;
        for (const auto& kv : shared) {
            if ((int)kv.second < std::max(3, opt_.min_common_images)) continue;
            const size_t i = (size_t)(kv.first >> 32), j = (size_t)(kv.first & 0xffffffffu);
            MergeCandidate c;
            // 已配准图像更多者作锚点，平局保留较早且通常三维点更多者。
            const bool i_first = models_[i].numRegistered() >= models_[j].numRegistered();
            c.dst = i_first ? i : j;
            c.src = i_first ? j : i;
            c.common_images = kv.second;
            out.push_back(c);
        }
        // 哈希遍历无序，先按全序排序，再按优先级排序以保证复现。
        std::sort(out.begin(), out.end(), [](const MergeCandidate& a, const MergeCandidate& b) {
            return a.dst != b.dst ? a.dst < b.dst : a.src < b.src;
        });
        std::stable_sort(out.begin(), out.end(),
                         [&](const MergeCandidate& a, const MergeCandidate& b) {
                             if (a.common_images != b.common_images)
                                 return a.common_images > b.common_images;
                             const uint32_t ra = models_[a.dst].numRegistered();
                             const uint32_t rb = models_[b.dst].numRegistered();
                             if (ra != rb) return ra > rb;
                             return a.dst < b.dst;
                         });
        return out;
    }

    // 合入 src，验证失败则保持双方原状；外部提供位姿或结构对齐结果可跳过估计，但不能跳过接受检查，携带内点数供拼接仲裁使用。
    MergeAttempt tryMerge(size_t dst, size_t src, const AlignmentResult& alignment) {
        return tryMergeImpl(dst, src, &alignment, nullptr);
    }
    MergeAttempt tryMerge(size_t dst, size_t src, const Sim3* alignment = nullptr) {
        return tryMergeImpl(dst, src, nullptr, alignment);
    }

private:
    MergeAttempt tryMergeImpl(size_t dst, size_t src, const AlignmentResult* full,
                              const Sim3* alignment) {
        MergeAttempt a;
        a.dst = dst;
        a.src = src;
        if (dst == src || !alive(dst) || !alive(src)) {
            a.reason = "not two live models";
            log_.push_back(a);
            return log_.back();
        }
        if (full) {
            a.alignment = *full;
        } else if (alignment) {
            a.alignment.transform = *alignment;
            a.alignment.common_images = commonImages(dst, src);
            a.alignment.success = true;
        } else {
            a.alignment = alignReconstructions(models_[src], models_[dst], opt_);
            if (!a.alignment.success) {
                a.reason = a.alignment.reason;
                log_.push_back(a);
                return log_.back();
            }
        }

        // 在目标副本上尝试合并，通过才提交；合并同时重写位姿、点和轨迹，复制是可靠回滚所需成本。
        const size_t anchor_obs = countObservations(models_[dst]);
        const uint32_t anchor_imgs = models_[dst].numRegistered();
        Reconstruction merged = models_[dst];
        MergeCounts c = mergeInto(merged, models_[src], a.alignment.transform, opt_);
        a.counts = c;

        char buf[192];
        // 同特征的双侧三维位置须一致；来源自身轨迹即使放错位置也可能内部自洽，共享点检查提供更强证据。
        const bool contested =
            c.points_spliced &&
            c.splice_conflicts > opt_.max_splice_conflict_ratio * (double)c.points_spliced;
        if (contested && !(opt_.validate && opt_.splice_arbitrate_inliers > 0 &&
                           (int)a.alignment.inliers >= opt_.splice_arbitrate_inliers)) {
            snprintf(buf, sizeof buf,
                     "the models disagree about %zu of the %zu points they both triangulated",
                     c.splice_conflicts, c.points_spliced);
            a.reason = buf;
            log_.push_back(a);
            return log_.back();
        }
        if (c.images_added && c.hollow_images > opt_.max_hollow_ratio * (double)c.images_added) {
            snprintf(buf, sizeof buf,
                     "%zu of the %zu images it added kept fewer than %d points", c.hollow_images,
                     c.images_added, opt_.min_image_points);
            a.reason = buf;
            log_.push_back(a);
            return log_.back();
        }
        const size_t after = countObservations(merged);
        // 锚点原观测应保留，净损失表示来源几何正在破坏原有模型。
        if (anchor_obs && after + (size_t)(opt_.max_anchor_obs_loss * anchor_obs) < anchor_obs) {
            snprintf(buf, sizeof buf, "the merged model lost %.0f%% of the anchor's observations",
                     100.0 * (double)(anchor_obs - after) / (double)anchor_obs);
            a.reason = buf;
            log_.push_back(a);
            return log_.back();
        }

        if (opt_.check_duplicate) {
            DuplicateReport before = findDuplicateStructure(models_[dst], opt_.duplicate);
            DuplicateReport after = findDuplicateStructure(merged, opt_.duplicate);
            if (after.duplicated(opt_.duplicate) &&
                after.conflicts > before.conflicts + (size_t)opt_.duplicate.min_conflicts) {
                snprintf(buf, sizeof buf,
                         "the merged model puts %zu pairs of images in the same place with no "
                         "structure in common (%zu before the merge, of %zu co-located pairs)",
                         after.conflicts, before.conflicts, after.colocated);
                a.reason = buf;
                log_.push_back(a);
                return log_.back();
            }
        }
        if (opt_.validate) {
            std::string why = opt_.validate(merged, models_[src], a.alignment.transform, c);
            if (!why.empty()) {
                a.reason = why;
                log_.push_back(a);
                return log_.back();
            }
        }

        models_[dst] = std::move(merged);
        models_[src] = Reconstruction();  // 释放内容，保留槽位用于日志引用
        alive_[src] = 0;
        a.merged = true;
        if (opt_.verbose)
            slog::diag(slog::Tag::Merge,
                       "[merge] model %zu <- model %zu: %zu %s (%zu inliers, %.2f px%s), "
                       "+%zu images, %u -> %u, +%zu points, %zu spliced (%zu disagreed)",
                       dst, src,
                       a.alignment.from_structure ? a.alignment.structure_pairs
                       : a.alignment.common_images,
                       a.alignment.from_structure ? "shared points" : "shared images",
                       a.alignment.inliers, a.alignment.mean_error,
                       a.alignment.rig_views ? ", some through the rig" : "",
                       c.images_added, anchor_imgs,
                       models_[dst].numRegistered(), c.points_added, c.points_spliced,
                       c.splice_conflicts);
        log_.push_back(a);
        return log_.back();
    }

public:
    // 循环合并直到无可接受候选，缓存失败对避免相同输入重试；模型被成功合并改变后重新激活相关失败对。
    size_t runAuto() {
        size_t merges = 0;
        while (true) {
            bool progressed = false;
            for (const MergeCandidate& c : candidates()) {
                if (failed_.count({c.dst, c.src}) || failed_.count({c.src, c.dst})) continue;
                MergeAttempt a = tryMerge(c.dst, c.src);
                if (a.merged) {
                    merges++;
                    progressed = true;
                    // 模型增长后可能获得更多共享图像，相关失败对值得重试。
                    for (auto it = failed_.begin(); it != failed_.end();)
                        it = (it->first == c.dst || it->second == c.dst) ? failed_.erase(it)
                                                                         : std::next(it);
                    break;
                }
                failed_.insert({c.dst, c.src});
                if (opt_.verbose)
                    slog::diag(slog::Tag::Merge, "[merge] model %zu <- model %zu refused: %s",
                               c.dst, c.src,
                               a.reason.c_str());
            }
            if (!progressed) break;
        }
        return merges;
    }

    // 仅对吸收过其他模型的结果执行跨接缝 BA，未改变模型应保持原数值。
    std::set<size_t> changed() const {
        std::set<size_t> s;
        for (const MergeAttempt& a : log_)
            if (a.merged && alive_[a.dst]) s.insert(a.dst);
        return s;
    }
    Reconstruction& modelMut(size_t i) { return models_.at(i); }

    // 幸存模型按三维点数降序返回，与 COLMAP 输出顺序一致。
    std::vector<Reconstruction> take() {
        std::vector<Reconstruction> out;
        for (size_t i = 0; i < models_.size(); i++)
            if (alive_[i]) out.push_back(std::move(models_[i]));
        std::stable_sort(out.begin(), out.end(),
                         [](const Reconstruction& a, const Reconstruction& b) {
                             return a.points3D.size() > b.points3D.size();
                         });
        return out;
    }

private:
    std::vector<Reconstruction> models_;
    std::vector<char> alive_;
    MergeOptions opt_;
    std::vector<MergeAttempt> log_;
    std::set<std::pair<size_t, size_t>> failed_;
};

}  // 命名空间 sfm
