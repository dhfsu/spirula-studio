#pragma once
// 通过重建标定视频 IMU 到镜头的旋转：陀螺积分与相机位姿的相对旋转满足手眼约束 A X = X B，各帧重力方向需映射到同一世界向量。
// 两类约束对 X 的九个元素均为线性，求零空间向量后投影到正交矩阵。
// 同时估计规范对齐所需的 IMU 时钟偏移与稳健向上方向。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "sfm/core/Pose.h"
#include "sfm/core/SensorTimeline.h"
#include "sfm/geometry/LinAlg.h"

namespace sfm {

// 已配准图像，时间采用所属采集序列的时钟。
struct SensorFrame {
    uint32_t image_id = 0;
    double t = 0;
    Mat3 R;      // 世界坐标系 -> 相机坐标系
    Vec3 c;      // 相机中心，单位为模型单位
    int group = 0;     // 镜头分组；每组一个 IMU 到相机的旋转
    int capture = 0;   // 所属时间线
    UpVote up;         // t 时刻的 IMU 坐标系向量
};

enum class ExtrinsicFail { None, Frames, Pairs, Disagree, NoStream, Degenerate };

struct ExtrinsicFit {
    bool ok = false;
    ExtrinsicFail reason = ExtrinsicFail::None;
    Mat3 R_ci = mat3Identity();   // 相机 <- IMU；IMU 轴为左手系时 det = -1
    bool mirrored = false;
    // 运动主要绕单轴旋转时，X 绕该轴的自由度不可观；确定符号后仍能使用向上方向，但不能使用预积分位置。
    bool degenerate = false;
    double gyro_sign = 1.0;       // 拟合采用的积分符号
    int frames = 0, rot_pairs = 0, grav_pairs = 0;
    double rms = 0;               // 加权残差，以标准差为单位
    double sig_rot_deg = 0, sig_grav_deg = 0;   // 各类约束估计出的噪声
    double gap = 0;               // 次小特征值与最小特征值之比
    Vec3 sigma_deg;               // 扰动 X Exp(d) 的逐轴不确定度
};

namespace extrinsic_detail {

// 满足 M vec(X) = R X a 的 M 各行；vec 按行展开。
inline void rowsRXa(const Mat3& R, const Vec3& a, double sign, std::vector<double>& rows) {
    const double av[3] = {a.x, a.y, a.z};
    for (int r = 0; r < 3; r++) {
        std::vector<double> row(9, 0.0);
        for (int m = 0; m < 3; m++)
            for (int n = 0; n < 3; n++) row[(size_t)(3 * m + n)] += sign * R[3 * r + m] * av[n];
        rows.insert(rows.end(), row.begin(), row.end());
    }
}

// 满足 M vec(X) = vec(A X - X B) 的 M 各行。
inline void rowsAXminusXB(const Mat3& A, const Mat3& B, std::vector<double>& rows) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            std::vector<double> row(9, 0.0);
            for (int m = 0; m < 3; m++) row[(size_t)(3 * m + c)] += A[3 * r + m];
            for (int n = 0; n < 3; n++) row[(size_t)(3 * r + n)] -= B[3 * n + c];
            rows.insert(rows.end(), row.begin(), row.end());
        }
}

// 最近正交矩阵，保留极分解因子的行列式符号；约束无法区分 X 与 -X，由调用方决定。
inline Mat3 nearestOrthogonal(const Mat3& X) {
    const Svd3 s = svd3(X);
    return mul(s.U, transpose(s.V));
}

inline double angleDeg(const Vec3& a, const Vec3& b) {
    const double d = std::max(-1.0, std::min(1.0, a.dot(b) / (a.norm() * b.norm())));
    return std::acos(d) * 180.0 / M_PI;
}

inline double rotAngleDeg(const Mat3& R) {
    const double tr = std::max(-1.0, std::min(1.0, (R[0] + R[4] + R[8] - 1.0) * 0.5));
    return std::acos(tr) * 180.0 / M_PI;
}

inline double medianOf(std::vector<double> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + (long)(v.size() / 2), v.end());
    return v[v.size() / 2];
}

struct Constraint {
    std::vector<double> rows;   // 每行 9 列
    int nrows = 0;
    Mat3 A, B;    // 旋转对
    Mat3 Rj, Rk;  // 重力对
    Vec3 aj, ak;
    bool gravity = false;
    double w = 1;
    double residual_deg(const Mat3& X) const {
        if (gravity) return angleDeg(mul(transpose(Rj), mul(X, aj)), mul(transpose(Rk), mul(X, ak)));
        return rotAngleDeg(mul(transpose(A), mul(X, mul(B, transpose(X)))));
    }
};

inline Mat3 solveNullSpace(const std::vector<Constraint>& cs, double& lam0, double& lam1,
                           double& trace) {
    std::vector<double> N(81, 0.0);
    for (const Constraint& c : cs)
        for (int r = 0; r < c.nrows; r++) {
            const double* row = &c.rows[(size_t)r * 9];
            for (int i = 0; i < 9; i++)
                for (int j = 0; j < 9; j++) N[(size_t)i * 9 + j] += c.w * row[i] * row[j];
        }
    trace = 0;
    for (int i = 0; i < 9; i++) trace += N[(size_t)i * 10];
    std::vector<double> w, V;
    jacobiEigenSymmetric(N, 9, w, V);
    int i0 = 0;
    for (int i = 1; i < 9; i++)
        if (w[i] < w[i0]) i0 = i;
    lam0 = w[i0];
    lam1 = 1e300;
    for (int i = 0; i < 9; i++)
        if (i != i0) lam1 = std::min(lam1, w[i]);
    Mat3 X;
    for (int i = 0; i < 9; i++) X[i] = V[(size_t)i * 9 + i0];
    return nearestOrthogonal(X);
}

}  // 命名空间 extrinsic_detail

// 两时刻间的相机相对旋转 A = R(t0) R(t1)^T；R 为世界到相机的旋转，来自模型位姿或已验证图像对。
struct RotationPairObs {
    double t0 = 0, t1 = 0;
    Mat3 A;
};

inline std::vector<RotationPairObs> consecutiveRotationPairs(const std::vector<SensorFrame>& frames) {
    std::vector<RotationPairObs> out;
    for (size_t k = 1; k < frames.size(); k++) {
        const SensorFrame& fj = frames[k - 1];
        const SensorFrame& fk = frames[k];
        if (fj.group != fk.group || fj.capture != fk.capture) continue;
        out.push_back({fj.t, fk.t, mul(fj.R, transpose(fk.R))});
    }
    return out;
}

// 联合 rot 的手眼约束与 frames 的重力、位姿约束；建图前仅有双视图旋转，frames 可为空。mean_up_w 确定约束未定的符号。
inline ExtrinsicFit calibrateImuExtrinsicFrom(const std::vector<RotationPairObs>& rot,
                                              const std::vector<SensorFrame>& frames,
                                              const SensorTimeline& tl, const Vec3& mean_up_w) {
    using namespace extrinsic_detail;
    ExtrinsicFit best;
    best.frames = (int)frames.size();
    if (rot.size() + frames.size() < 3) {
        best.reason = ExtrinsicFail::Frames;
        return best;
    }
    const bool attitude_only = !tl.hasGyro() && tl.hasRotation();
    double best_score = 1e300;
    ExtrinsicFail fail = ExtrinsicFail::Pairs;
    for (double sign : {1.0, -1.0}) {
        if (sign < 0 && attitude_only) break;
        std::vector<Constraint> cs;
        int rot_pairs = 0, grav_pairs = 0;
        for (const RotationPairObs& rp : rot) {
            const double dt = rp.t1 - rp.t0;
            if (dt < 0.02 || dt > 3.0) continue;
            Mat3 B;
            if (!tl.rotationBetween(rp.t0, rp.t1, B, sign)) continue;
            Constraint c;
            c.A = rp.A;
            c.B = B;
            rowsAXminusXB(c.A, c.B, c.rows);
            c.nrows = 9;
            cs.push_back(std::move(c));
            rot_pairs++;
        }
        for (int stride : {1, 3, 9, 27, 81}) {
            for (size_t k = (size_t)stride; k < frames.size(); k++) {
                const SensorFrame& fj = frames[k - (size_t)stride];
                const SensorFrame& fk = frames[k];
                if (!fj.up.ok || !fk.up.ok) continue;
                Constraint c;
                c.gravity = true;
                c.Rj = fj.R;
                c.Rk = fk.R;
                c.aj = fj.up.up;
                c.ak = fk.up.up;
                rowsRXa(transpose(fj.R), fj.up.up, 1.0, c.rows);
                std::vector<double> neg;
                rowsRXa(transpose(fk.R), fk.up.up, -1.0, neg);
                for (int i = 0; i < 27; i++) c.rows[(size_t)i] += neg[(size_t)i];
                c.nrows = 3;
                cs.push_back(std::move(c));
                grav_pairs++;
            }
        }
        if (rot_pairs + grav_pairs < 6) continue;
        double max_turn = 0;
        for (const Constraint& c : cs)
            if (!c.gravity) max_turn = std::max(max_turn, rotAngleDeg(c.A));
        if (rot_pairs > 0 && max_turn < 3.0) {
            fail = ExtrinsicFail::Degenerate;
            continue;
        }

        // 陀螺旋转对精度约十分之一度，重力投票误差为数度；各类约束分别估计 sigma，并按 1/sigma^2 加权，避免重力投票淹没旋转约束。
        Mat3 X = mat3Identity();
        double lam0 = 0, lam1 = 0, trace = 0, rms = 0, sig_rot = 1, sig_grav = 1;
        for (int round = 0; round < 6; round++) {
            X = solveNullSpace(cs, lam0, lam1, trace);
            std::vector<double> res, rr, rg;
            for (const Constraint& c : cs) {
                res.push_back(c.residual_deg(X));
                (c.gravity ? rg : rr).push_back(res.back());
            }
            sig_rot = std::max(1.4826 * medianOf(rr), 0.2);
            sig_grav = std::max(1.4826 * medianOf(rg), 1.0);
            double ss = 0, sw = 0;
            for (size_t i = 0; i < cs.size(); i++) {
                const double sg = cs[i].gravity ? sig_grav : sig_rot;
                const double r = res[i] / sg;
                const double huber = r <= 1.345 ? 1.0 : 1.345 / r;
                cs[i].w = huber / (sg * sg);
                ss += huber * r * r;
                sw += huber;
            }
            rms = sw > 0 ? std::sqrt(ss / sw) : 0;   // 以标准差为单位
        }
        // 使用旋转对自身以度为单位的残差比较假设，避免放大 sigma 的假设仅凭较小归一化 RMS 胜出。
        const double score = rot_pairs > 0 ? sig_rot : sig_grav;
        if (rot_pairs > 0 ? sig_rot > 5.0 : sig_grav > 10.0) {
            if (fail != ExtrinsicFail::Degenerate) fail = ExtrinsicFail::Disagree;
            continue;
        }
        if (!(lam1 > 1e-9 * trace)) {
            fail = ExtrinsicFail::Degenerate;
            continue;
        }
        // 单轴旋转无法区分两种符号；取反假设必须明显更优，不能仅凭噪声胜出。
        if (best.ok && score >= 0.7 * best_score) continue;
        best_score = score;

        ExtrinsicFit f;
        f.ok = true;
        f.frames = (int)frames.size();
        f.rot_pairs = rot_pairs;
        f.grav_pairs = grav_pairs;
        f.rms = rms;
        f.sig_rot_deg = sig_rot;
        f.sig_grav_deg = sig_grav;
        f.gyro_sign = sign;
        f.gap = lam1 / std::max(lam0, 1e-12 * trace);
        f.degenerate = f.gap < 3.0;
        // 经 X 变换的投票须与相机平均向上方向一致；仅靠旋转对不能区分 X 与 -X，需由具备位姿的调用方确定，SensorPriors.h 按模型处理。
        double agree = 0;
        for (const SensorFrame& fr : frames)
            if (fr.up.ok) agree += mul(transpose(fr.R), mul(X, fr.up.up)).dot(mean_up_w);
        if (agree < 0) X = mat3Scale(X, -1.0);
        f.R_ci = X;
        f.mirrored = det3(X) < 0;
        // 由加权残差曲率估计 X Exp(d) 的逐轴不确定度。
        std::vector<double> H(9, 0.0);
        const double h = 1e-3;
        for (const Constraint& c : cs) {
            double J[3];
            for (int ax = 0; ax < 3; ax++) {
                Vec3 d{0, 0, 0};
                (&d.x)[ax] = h;
                const double rp = c.residual_deg(mul(X, so3Exp(d)));
                const double rm = c.residual_deg(mul(X, so3Exp(d * -1.0)));
                J[ax] = (rp - rm) / (2 * h);
            }
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) H[(size_t)a * 3 + b] += c.w * J[a] * J[b];
        }
        std::vector<double> w, V;
        jacobiEigenSymmetric(H, 3, w, V);
        for (int ax = 0; ax < 3; ax++) {
            double var = 0;
            for (int i = 0; i < 3; i++) {
                const double v = V[(size_t)ax * 3 + i];
                var += w[i] > 1e-12 ? v * v / w[i] : 1e12;
            }
            (&f.sigma_deg.x)[ax] = std::sqrt(var) * 180.0 / M_PI;
        }
        best = f;
    }
    if (!best.ok) best.reason = fail;
    return best;
}

// frames 属于同一组，已按时间排序并填入向上方向投票。
inline ExtrinsicFit calibrateImuExtrinsic(const std::vector<SensorFrame>& frames,
                                          const SensorTimeline& tl, const Vec3& mean_up_w) {
    if (frames.size() < 3) {
        ExtrinsicFit f;
        f.frames = (int)frames.size();
        f.reason = ExtrinsicFail::Frames;
        return f;
    }
    return calibrateImuExtrinsicFrom(consecutiveRotationPairs(frames), frames, tl, mean_up_w);
}

// 比较两时刻间的旋转角以估计 IMU 相对视频的时钟偏移；旋转角与外参无关，因此可先于标定估计。无明确极小值时返回 0 且 found 为 false。
struct TimeOffsetFit {
    bool found = false;
    double offset = 0;      // 需加到视频时间上的秒数
    double gain = 0;        // 消除的角度不一致比例
    int pairs = 0;
};

inline TimeOffsetFit estimateTimeOffsetFrom(const std::vector<RotationPairObs>& rot,
                                            const SensorTimeline& tl, double range = 0.2) {
    using namespace extrinsic_detail;
    TimeOffsetFit fit;
    struct Pair { double t0, t1, vis; };
    // 同一段分散骑行重建仅用 25 对数据，曾分别得到 -111 ms 和 39 ms。
    constexpr size_t kMinPairs = 30;
    std::vector<Pair> pairs;
    for (const RotationPairObs& rp : rot) {
        const double dt = rp.t1 - rp.t0;
        if (dt < 0.02 || dt > 3.0) continue;
        const double vis = rotAngleDeg(rp.A);
        if (vis < 2.0) continue;
        pairs.push_back({rp.t0, rp.t1, vis});
    }
    fit.pairs = (int)pairs.size();
    if (pairs.size() < kMinPairs) return fit;
    // 角度差的平方，剔除最差的十分之一，避免单个位姿错误移动极小值。
    auto cost = [&](double d) {
        std::vector<double> r;
        for (const Pair& p : pairs) {
            Mat3 R;
            if (!tl.rotationBetween(p.t0 + d, p.t1 + d, R)) continue;
            const double e = rotAngleDeg(R) - p.vis;
            r.push_back(e * e);
        }
        if (r.size() < pairs.size() / 2) return 1e300;
        std::sort(r.begin(), r.end());
        double c = 0;
        const size_t n = r.size() - r.size() / 10;
        for (size_t i = 0; i < n; i++) c += r[i];
        return c / (double)n;
    };
    const double c0 = cost(0.0);
    double best_d = 0, best_c = c0;
    for (double d = -range; d <= range + 1e-9; d += 0.01) {
        const double c = cost(d);
        if (c < best_c) { best_c = c; best_d = d; }
    }
    const double step = 0.002;
    for (double d = best_d - 0.01; d <= best_d + 0.01 + 1e-9; d += step) {
        const double c = cost(d);
        if (c < best_c) { best_c = c; best_d = d; }
    }
    const double cm = cost(best_d - step), cp = cost(best_d + step);
    const double denom = cm - 2 * best_c + cp;
    if (denom > 0) best_d += 0.5 * step * (cm - cp) / denom;
    if (!(c0 > 0) || std::fabs(best_d) >= range - 0.005) return fit;
    fit.gain = 1.0 - best_c / c0;
    fit.found = fit.gain > 0.2;
    fit.offset = fit.found ? best_d : 0.0;
    return fit;
}

inline TimeOffsetFit estimateTimeOffset(const std::vector<SensorFrame>& frames,
                                        const SensorTimeline& tl, double range = 0.2) {
    return estimateTimeOffsetFrom(consecutiveRotationPairs(frames), tl, range);
}

// 稳健平均各帧投票得到世界向上方向。
struct UpConsensus {
    bool ok = false;
    Vec3 up{0, 0, 1};
    double spread_deg = 0;   // 投票角度残差的中位数
    int votes = 0, outliers = 0;
    std::vector<double> residual_deg;   // 对应各输入投票，缺失时为 -1
};

inline UpConsensus consensusUp(const std::vector<Vec3>& votes_w) {
    using namespace extrinsic_detail;
    UpConsensus u;
    u.residual_deg.assign(votes_w.size(), -1.0);
    Vec3 m{0, 0, 0};
    for (const Vec3& v : votes_w) m = m + v;
    if (!(m.norm() > 1e-9)) return u;
    m = m.normalized();
    std::vector<double> w(votes_w.size(), 1.0), res(votes_w.size(), 0.0);
    for (int round = 0; round < 6; round++) {
        for (size_t i = 0; i < votes_w.size(); i++) res[i] = angleDeg(votes_w[i], m);
        const double sigma = std::max(1.4826 * medianOf(res), 0.3);
        Vec3 s{0, 0, 0};
        for (size_t i = 0; i < votes_w.size(); i++) {
            const double r = res[i] / sigma;
            w[i] = r <= 1.345 ? 1.0 : 1.345 / r;
            s = s + votes_w[i] * w[i];
        }
        if (!(s.norm() > 1e-9)) return u;
        m = s.normalized();
    }
    for (size_t i = 0; i < votes_w.size(); i++) {
        res[i] = angleDeg(votes_w[i], m);
        u.residual_deg[i] = res[i];
        if (res[i] > 10.0) u.outliers++;
    }
    u.up = m;
    u.votes = (int)votes_w.size();
    u.spread_deg = medianOf(res);
    u.ok = u.votes >= 3 && u.votes - u.outliers >= 3;
    return u;
}

}  // 命名空间 sfm
