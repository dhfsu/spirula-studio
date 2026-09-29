// 公制规范用相机中心与米制参考拟合 Sim(3)。尺度和旋转不确定度仅报告，不作接受门限，因为相关参考噪声可使其低估 3.9–4.5 倍。
// 接受依据是参考空间跨度和是否近共线（D74）。
#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "sfm/core/Exif.h"
#include "sfm/core/Model.h"
#include "sfm/core/Pose.h"
#include "sfm/map/Orient.h"
#include "sfm/geometry/LinAlg.h"
#include "sfm/optim/Ransac.h"
#include "sfm/map/Merge.h"

namespace sfm {

// 模型相机中心与米制目标按索引配对，image_ids 仅供报告，不参与拟合。
struct MetricRef {
    std::vector<Vec3> centres;
    std::vector<Vec3> targets;
    std::vector<uint32_t> image_ids;
};

enum class MetricFail { None, Pairs, Spread, Inliers, Collinear };

// 水平模式只用平面位置并沿用已有倾斜；百米采集中的米级 GPS 高度偏差会使完整三维拟合倾斜数度（D75）。
enum class MetricAxes { Full, Horizontal };

struct MetricFit {
    bool ok = false;
    MetricFail reason = MetricFail::Pairs;
    Sim3 T;
    int n = 0;                  // 输入的相机对应数
    int inliers = 0;
    double max_error = 0;       // 米
    double rms = 0;             // 内点误差，单位米
    double scale_unc = 0;       // std(ds/s)，百分比，仅供参考
    double rot_unc_deg = 0;     // 最弱约束主轴的角度不确定度，仅供参考
    double spread = 0;          // 参考位置的 RMS 半径，单位米
    double perp_frac = 0;       // 最弱约束轴的横向跨度占整体比例
    std::vector<char> inlier_mask;
};

// 旋转误差相对尺度误差约按 1/perp_frac 放大，0.05 拒绝超过约二十倍放大的近共线参考；这是单次飞行测量形成的启发值（D74）。
inline constexpr double kMetricMinPerpFraction = 0.05;

namespace detail {

inline Vec3 meanOf(const std::vector<Vec3>& v) {
    Vec3 m{0, 0, 0};
    for (const Vec3& p : v) m = m + p;
    return v.empty() ? m : m * (1.0 / (double)v.size());
}

}  // 命名空间 detail

// 仅拟合尺度、绕 +Z 航向和位置；不能简单压平目标后使用完整 Sim3，否则会把模型倾斜到目标平面。
inline bool estimateSim3Yaw(const std::vector<Vec3>& src, const std::vector<Vec3>& dst,
                            Sim3& out) {
    const size_t n = src.size();
    if (n < 2 || dst.size() != n) return false;
    Vec3 ms{0, 0, 0}, md{0, 0, 0};
    for (size_t i = 0; i < n; i++) { ms = ms + src[i]; md = md + dst[i]; }
    ms = ms * (1.0 / (double)n);
    md = md * (1.0 / (double)n);
    double var_s = 0, dot = 0, cross = 0;
    for (size_t i = 0; i < n; i++) {
        const Vec3 a = src[i] - ms, b = dst[i] - md;
        var_s += a.x * a.x + a.y * a.y;
        dot += a.x * b.x + a.y * b.y;
        cross += a.x * b.y - a.y * b.x;
    }
    const double h = std::sqrt(dot * dot + cross * cross);
    if (!(var_s > 1e-12) || !(h > 1e-12)) return false;
    const double c = dot / h, s = cross / h;
    out.scale = h / var_s;
    if (!(out.scale > 1e-12) || !std::isfinite(out.scale)) return false;
    out.R = Mat3{c, -s, 0, s, c, 0, 0, 0, 1};
    out.t = md - mul(out.R, ms) * out.scale;
    return std::isfinite(out.t.x) && std::isfinite(out.t.y) && std::isfinite(out.t.z);
}

// 拟合 centres 到 targets，证据不足时给出拒绝原因；内点半径为绝对米制阈值与参考 RMS 半径比例阈值的较大者。
inline MetricFit fitMetricGauge(const MetricRef& ref, double max_error,
                                MetricAxes axes = MetricAxes::Full,
                                double max_error_frac = 0.0) {
    MetricFit out;
    out.max_error = max_error;
    const bool flat = axes == MetricAxes::Horizontal;
    const int n = (int)ref.centres.size();
    out.n = n;
    if (n < 3 || (int)ref.targets.size() != n) {
        out.reason = MetricFail::Pairs;
        return out;
    }

    // 没有空间跨度的参考不约束尺度，任意拟合都可能通过 RMS 门限，必须拒绝。
    const Vec3 pbar = detail::meanOf(ref.targets);
    double var_t = 0;
    for (const Vec3& p : ref.targets) {
        const Vec3 d = p - pbar;
        var_t += flat ? d.x * d.x + d.y * d.y : d.dot(d);
    }
    out.spread = std::sqrt(var_t / (double)n);
    if (!(out.spread > max_error)) {
        out.reason = MetricFail::Spread;
        return out;
    }
    // GPS 与模型漂移随范围增大，误差半径在绝对下限之上按跨度缩放。
    max_error = std::max(max_error, max_error_frac * out.spread);
    out.max_error = max_error;

    auto fit_fn = [&](const std::vector<int>& idx) {
        std::vector<Sim3> models;
        std::vector<Vec3> s, d;
        s.reserve(idx.size());
        d.reserve(idx.size());
        for (int i : idx) {
            s.push_back(ref.centres[i]);
            d.push_back(ref.targets[i]);
        }
        Sim3 T;
        if (flat ? estimateSim3Yaw(s, d, T) : estimateSim3(s, d, T)) models.push_back(T);
        return models;
    };
    auto res_fn = [&](const Sim3& T, int i) {
        const Vec3 r = ref.targets[i] - transformPoint(T, ref.centres[i]);
        return flat ? r.x * r.x + r.y * r.y : r.dot(r);
    };

    RansacOptions opt;
    opt.max_error = max_error;
    opt.max_num_trials = 1000;
    opt.seed = 0;
    RansacReport<Sim3> rep = loransac<Sim3>(n, flat ? 2 : 3, fit_fn, fit_fn, res_fn, opt);
    out.inlier_mask = rep.inlier_mask;
    out.inliers = std::max(rep.num_inliers, 0);
    const int need = std::max(3, (n + 1) / 2);
    if (!rep.success || out.inliers < need) {
        out.reason = MetricFail::Inliers;
        return out;
    }

    std::vector<int> keep;
    keep.reserve(out.inliers);
    for (int i = 0; i < n; i++)
        if (out.inlier_mask[i]) keep.push_back(i);
    std::vector<Sim3> refit = fit_fn(keep);
    if (refit.empty()) {
        out.reason = MetricFail::Inliers;
        return out;
    }
    out.T = refit.front();

    const int m = (int)keep.size();
    double ss = 0;
    Vec3 cbar{0, 0, 0};
    for (int i : keep) {
        ss += res_fn(out.T, i);
        cbar = cbar + ref.centres[i];
    }
    cbar = cbar * (1.0 / (double)m);
    out.rms = std::sqrt(ss / (double)m);
    // 三维拟合从 3m 残差估七参数，水平从 2m 估四参数，噪声按剩余自由度归一化。
    const double sigma = std::sqrt(ss / (double)(flat ? 2 * m - 4 : 3 * m - 7));

    std::vector<double> C(9, 0.0);
    for (int i : keep) {
        const Vec3 a = ref.centres[i] - cbar;
        const double v[3] = {a.x, a.y, flat ? 0.0 : a.z};
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) C[3 * r + c] += v[r] * v[c];
    }
    const double s2 = out.T.scale * out.T.scale;
    for (double& v : C) v *= s2 / (double)m;
    std::vector<double> lam, V;
    jacobiEigenSymmetric(C, 3, lam, V);
    const double tr = lam[0] + lam[1] + lam[2];

    out.scale_unc = 100.0 * sigma / (std::sqrt((double)m) * std::sqrt(tr));
    // 绕主轴旋转仅受横向分布约束，无横向跨度则不可观；水平模式只绕竖轴，受整个平面半径约束。
    double worst = 0, perp_min = 0;
    for (int k = 0; k < 3; k++) {
        const double perp = flat ? tr : tr - lam[k];
        worst = std::max(worst, perp > 0.0 ? sigma / std::sqrt((double)m * perp)
                                           : std::numeric_limits<double>::infinity());
        if (k == 0 || perp < perp_min) perp_min = perp;
    }
    out.rot_unc_deg = worst * 180.0 / M_PI;
    out.perp_frac = tr > 0.0 ? std::sqrt(std::max(perp_min, 0.0) / tr) : 0.0;

    // 水平拟合不引入此检查针对的倾斜。
    if (!flat && !(out.perp_frac >= kMetricMinPerpFraction)) {
        out.reason = MetricFail::Collinear;
        return out;
    }
    out.ok = true;
    out.reason = MetricFail::None;
    return out;
}

// 配对统计，解释参考与已配准图像为何数量不同。
struct MetricPairCounts {
    int matched = 0;
    int unmatched_file = 0;    // 没有对应已配准图像的参考位置数
    int unmatched_model = 0;   // 没有参考位置的已配准图像数
};

// 参考文件采用每行 image_name X Y Z，单位米，支持 # 注释并跳过空行，错误带行号。
inline bool readMetricPositions(const std::string& path, std::map<std::string, Vec3>& out,
                                std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    out.clear();
    std::string line;
    for (int lineno = 1; std::getline(f, line); lineno++) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream is(line);
        std::string name;
        if (!(is >> name)) continue;
        double x, y, z;
        std::string extra;
        if (!(is >> x >> y >> z) || (is >> extra)) {
            err = "line " + std::to_string(lineno) + ": expected `image_name X Y Z`";
            return false;
        }
        if (!out.emplace(name, Vec3{x, y, z}).second) {
            err = "line " + std::to_string(lineno) + ": " + name + " appears twice";
            return false;
        }
    }
    if (out.empty()) {
        err = "no positions in " + path;
        return false;
    }
    return true;
}

// 先按完整模型图像名，再按去扩展名路径配对；不能仅用基本文件名，rig 不同目录常有同名帧。
inline MetricPairCounts pairMetricRef(const Reconstruction& rec,
                                      const std::map<std::string, Vec3>& positions,
                                      MetricRef& ref) {
    MetricPairCounts c;
    std::vector<char> used(positions.size(), 0);
    for (const auto& kv : rec.images) {
        if (!kv.second.registered) continue;
        auto it = positions.find(kv.second.name);
        if (it == positions.end()) {
            const size_t dot = kv.second.name.find_last_of('.');
            const size_t slash = kv.second.name.find_last_of('/');
            if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
                it = positions.find(kv.second.name.substr(0, dot));
        }
        if (it == positions.end()) {
            c.unmatched_model++;
            continue;
        }
        used[(size_t)std::distance(positions.begin(), it)] = 1;
        ref.centres.push_back(cameraCenter(kv.second.pose));
        ref.targets.push_back(it->second);
        ref.image_ids.push_back(kv.first);
        c.matched++;
    }
    for (char u : used)
        if (!u) c.unmatched_file++;
    return c;
}


struct Geodetic {
    double lat_deg = 0, lon_deg = 0, alt_m = 0;
};

// WGS-84 转 ECEF 使用 double，地心坐标约 6.4e6 m，此处 float 的半 ULP 已达 0.25 m。
inline Vec3 ecefFromGeodetic(double lat_deg, double lon_deg, double h) {
    constexpr double a = 6378137.0, f = 1.0 / 298.257223563;
    constexpr double e2 = f * (2.0 - f);
    const double p = lat_deg * M_PI / 180.0, l = lon_deg * M_PI / 180.0;
    const double sp = std::sin(p), cp = std::cos(p);
    const double N = a / std::sqrt(1.0 - e2 * sp * sp);
    return {(N + h) * cp * std::cos(l), (N + h) * cp * std::sin(l), (N * (1.0 - e2) + h) * sp};
}

// 以 origin 为中心的右手东、北、上米制坐标，北向为正；不适用于跨日界线或约 100 km 以上范围。
inline std::vector<Vec3> enuFromGeodetic(const std::vector<Geodetic>& g,
                                         const Geodetic& origin) {
    std::vector<Vec3> out;
    if (g.empty()) return out;
    const double lat0 = origin.lat_deg, lon0 = origin.lon_deg, h0 = origin.alt_m;
    const Vec3 o = ecefFromGeodetic(lat0, lon0, h0);
    const double p = lat0 * M_PI / 180.0, l = lon0 * M_PI / 180.0;
    const double sp = std::sin(p), cp = std::cos(p), sl = std::sin(l), cl = std::cos(l);
    out.reserve(g.size());
    for (const Geodetic& q : g) {
        const Vec3 d = ecefFromGeodetic(q.lat_deg, q.lon_deg, q.alt_m) - o;
        out.push_back({-sl * d.x + cl * d.y,
                       -sp * cl * d.x - sp * sl * d.y + cp * d.z,
                       cp * cl * d.x + cp * sl * d.y + sp * d.z});
    }
    return out;
}

// 以定位均值为原点建立同类坐标；原点变化也改变旋转，不只是平移。
inline std::vector<Vec3> enuFromGeodetic(const std::vector<Geodetic>& g) {
    if (g.empty()) return {};
    Geodetic o;
    for (const Geodetic& q : g) {
        o.lat_deg += q.lat_deg;
        o.lon_deg += q.lon_deg;
        o.alt_m += q.alt_m;
    }
    const double inv = 1.0 / (double)g.size();
    o.lat_deg *= inv;
    o.lon_deg *= inv;
    o.alt_m *= inv;
    return enuFromGeodetic(g, o);
}


// 图像 EXIF GPS 的读取统计。
struct MetricGpsCounts {
    int matched = 0;
    int no_gps = 0;
    int no_alt = 0;   // 有位置但无高度时按海平面处理
};

// 从已配准图像 EXIF 生成局部 ENU 米制参考，高度可缺省。
inline MetricGpsCounts metricRefFromGps(const Reconstruction& rec, const std::string& image_dir,
                                        MetricRef& ref) {
    MetricGpsCounts c;
    std::vector<Geodetic> g;
    std::vector<Vec3> centres;
    std::vector<uint32_t> ids;
    for (const auto& kv : rec.images) {
        if (!kv.second.registered) continue;
        const ExifData e =
            readExif((std::filesystem::path(image_dir) / kv.second.name).string());
        if (!e.has_gps) {
            c.no_gps++;
            continue;
        }
        if (!e.has_alt) c.no_alt++;
        g.push_back({e.lat_deg, e.lon_deg, e.alt_m});
        centres.push_back(cameraCenter(kv.second.pose));
        ids.push_back(kv.first);
        c.matched++;
    }
    std::vector<Vec3> enu = enuFromGeodetic(g);
    for (size_t i = 0; i < enu.size(); i++) {
        ref.centres.push_back(centres[i]);
        ref.targets.push_back(enu[i]);
        ref.image_ids.push_back(ids[i]);
    }
    return c;
}


// 比较参考 +Z 与相机估计向上方向；手持或云台通常仅几度，数十度提示参考或采集倾斜。
inline double metricUpDisagreementDeg(const Reconstruction& rec) {
    const Sim3 up = uprightTransform(rec);
    return std::acos(std::max(-1.0, std::min(1.0, up.R[8]))) * 180.0 / M_PI;
}

}  // 命名空间 sfm
