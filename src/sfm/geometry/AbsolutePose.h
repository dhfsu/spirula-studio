// 绝对位姿估计：单相机由二维、三维对应执行 P3P，rig 联合各镜头对应执行 GP3P，均使用 LO-RANSAC。
// 局部优化采用至少六点的 DLT 或整帧内点精化；单位视线统一支持鱼眼，残差使用归一化单位。
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "sfm/core/Pose.h"
#include "sfm/geometry/Essential.h"
#include "sfm/geometry/GP3P.h"
#include "sfm/geometry/LinAlg.h"
#include "sfm/geometry/P3P.h"
#include "sfm/optim/Ransac.h"

namespace sfm {

// 由至少六组世界点 X 与单位视线 b 执行 DLT，采用前向透视形式 (b.x/b.z,b.y/b.z)，返回零或一个候选。
inline std::vector<Pose> estimatePoseDLT(const std::vector<Vec3>& X, const std::vector<Vec3>& b,
                                         const std::vector<int>& idx) {
    if (idx.size() < 6) return {};
    // 对世界点去中心并各向同性缩放以改善条件数。
    Vec3 c{};
    for (int i : idx) c = c + X[i];
    c = c * (1.0 / idx.size());
    double meanDist = 0;
    for (int i : idx) meanDist += (X[i] - c).norm();
    meanDist /= idx.size();
    double sw = meanDist > 1e-12 ? std::sqrt(3.0) / meanDist : 1.0;

    size_t n = idx.size();
    std::vector<double> A(2 * n * 12, 0.0);
    for (size_t k = 0; k < n; k++) {
        int i = idx[k];
        Vec3 W = (X[i] - c) * sw;  // 归一化世界坐标
        double u = b[i].x / b[i].z, v = b[i].y / b[i].z;
        double Xh[4] = {W.x, W.y, W.z, 1.0};
        double* r0 = &A[(2 * k) * 12];
        double* r1 = &A[(2 * k + 1) * 12];
        for (int t = 0; t < 4; t++) {
            r0[t] = Xh[t];
            r0[8 + t] = -u * Xh[t];
            r1[4 + t] = Xh[t];
            r1[8 + t] = -v * Xh[t];
        }
    }
    std::vector<double> p = nullspaceVector(A, (int)(2 * n), 12);
    // P_n 将归一化世界齐次坐标映射到图像，恢复尺度时 P=P_n*S，S=[[sw I,-sw c],[0,1]]。
    Mat3 M = {p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]};
    Vec3 p4 = {p[3], p[7], p[11]};
    // 未归一化旋转为 M*sw，平移为 p4-M*(sw c)，在下方合并处理。
    Mat3 Rraw = {M[0] * sw, M[1] * sw, M[2] * sw, M[3] * sw, M[4] * sw,
                 M[5] * sw, M[6] * sw, M[7] * sw, M[8] * sw};
    Vec3 traw = {p4.x - (M[0] * c.x + M[1] * c.y + M[2] * c.z) * sw,
                 p4.y - (M[3] * c.x + M[4] * c.y + M[5] * c.z) * sw,
                 p4.z - (M[6] * c.x + M[7] * c.y + M[8] * c.z) * sw};

    // 零空间向量有任意非零尺度与符号；令旋转块 det>0 排除镜像解，同时确定右手性与正深度。
    if (det3(Rraw) < 0) {
        for (int i = 0; i < 9; i++) Rraw[i] = -Rraw[i];
        traw = {-traw.x, -traw.y, -traw.z};
    }
    double lambda = (Vec3{Rraw[0], Rraw[3], Rraw[6]}.norm() + Vec3{Rraw[1], Rraw[4], Rraw[7]}.norm() +
                     Vec3{Rraw[2], Rraw[5], Rraw[8]}.norm()) / 3.0;
    if (lambda < 1e-12) return {};
    Mat3 Rn;
    for (int i = 0; i < 9; i++) Rn[i] = Rraw[i] / lambda;
    Pose pose;
    pose.R = nearestRotation(Rn);
    pose.t = traw * (1.0 / lambda);
    return {pose};
}

// 前向视线使用归一化平面误差平方，宽角使用夹角正弦平方，两者小角度尺度一致且后者可覆盖 p.z<=0。
inline double pnpResidualSqAt(const Vec3& p, const Vec3& b) {
    if (b.z > 0.1) {
        if (p.z < 1e-8) return 1e30;  // 前半球正深度检查
        double du = p.x / p.z - b.x / b.z, dv = p.y / p.z - b.y / b.z;
        return du * du + dv * dv;
    }
    if (p.dot(b) <= 0) return 1e30;   // 沿观测射线检查正深度
    Vec3 ph = p.normalized();         // b 已为单位向量
    Vec3 cr = ph.cross(b);
    return cr.dot(cr);                // sin^2(angle)
}

inline double pnpResidualSq(const Pose& pose, const Vec3& X, const Vec3& b) {
    return pnpResidualSqAt(mul(pose.R, X) + pose.t, b);
}

struct PnPResult {
    Pose pose;
    std::vector<char> inlier_mask;
    int num_inliers = 0;
    bool success = false;
};

namespace pose_detail {

// 在位姿 p 和焦距比例 s 下计算二维残差；正深度失败返回无梯度大常量，只能拒绝步骤，不能引导优化。
inline void residualPair(const Pose& p, const Vec3& X, const Vec3& bi, double s, double* r) {
    Vec3 pc = mul(p.R, X) + p.t;
    if (bi.z > 0.1) {
        if (pc.z < 1e-8) { r[0] = r[1] = 1e3; return; }
        r[0] = pc.x / pc.z - bi.x / (bi.z * s);
        r[1] = pc.y / pc.z - bi.y / (bi.z * s);
        return;
    }
    if (pc.dot(bi) <= 0) { r[0] = r[1] = 1e3; return; }
    // 方向误差的切平面分量，与 sin^2 形式一致
    Vec3 e1 = (std::fabs(bi.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0}).cross(bi).normalized();
    Vec3 e2 = bi.cross(e1);
    Vec3 ph = pc.normalized();
    r[0] = ph.dot(e1);
    r[1] = ph.dot(e2);
}

// 对轴角增量、平移及可选对数焦距比例执行 LM，雅可比用中心差分；每对应两维残差，1e3 表示正深度失败。
template <class Resid>
bool lmRefine(int n, const Resid& resid, int NP, Pose& pose, double& s0, int max_iters) {
    auto cost = [&](const Pose& p, double s) {
        double c = 0, r[2];
        for (int i = 0; i < n; i++) { resid(p, s, i, r); c += r[0] * r[0] + r[1] * r[1]; }
        return c;
    };
    // 更新 R <- exp(w)R0，t <- t0+dt，s <- s0*exp(ds)，乘法焦距更新保证正值。
    auto stepP = [&](const Pose& p0, const double* d) {
        Pose p;
        p.R = mul(angleAxisToRotation({d[0], d[1], d[2]}), p0.R);
        p.t = {p0.t.x + d[3], p0.t.y + d[4], p0.t.z + d[5]};
        return p;
    };
    auto stepS = [&](double s, const double* d) { return NP == 7 ? s * std::exp(d[6]) : s; };

    double lambda = 1e-4, c0 = cost(pose, s0);
    for (int it = 0; it < max_iters; it++) {
        // 逐点累加数值雅可比的 J^T J 与 J^T r。
        double JtJ[49] = {0}, Jtr[7] = {0};
        const double h = 1e-6;
        for (int i = 0; i < n; i++) {
            double J[2][7], rp[2], rm[2], r0[2];
            resid(pose, s0, i, r0);
            if (r0[0] >= 1e3) continue;
            for (int k = 0; k < NP; k++) {
                double d[7] = {0, 0, 0, 0, 0, 0, 0};
                d[k] = h;
                resid(stepP(pose, d), stepS(s0, d), i, rp);
                d[k] = -h;
                resid(stepP(pose, d), stepS(s0, d), i, rm);
                J[0][k] = (rp[0] - rm[0]) / (2 * h);
                J[1][k] = (rp[1] - rm[1]) / (2 * h);
            }
            for (int a = 0; a < NP; a++) {
                for (int c = 0; c < NP; c++)
                    JtJ[NP * a + c] += J[0][a] * J[0][c] + J[1][a] * J[1][c];
                Jtr[a] += J[0][a] * r0[0] + J[1][a] * r0[1];
            }
        }
        // 用高斯消元求 (JtJ + lambda diag)d = -Jtr。
        double A[49], g[7], d[7];
        bool solved = false;
        for (int tries = 0; tries < 8 && !solved; tries++) {
            for (int a = 0; a < NP * NP; a++) A[a] = JtJ[a];
            for (int a = 0; a < NP; a++) {
                A[(NP + 1) * a] += lambda * std::max(JtJ[(NP + 1) * a], 1e-12);
                g[a] = -Jtr[a];
            }
            solved = true;
            for (int col = 0; col < NP && solved; col++) {
                int piv = col;
                for (int rw = col + 1; rw < NP; rw++)
                    if (std::fabs(A[NP * rw + col]) > std::fabs(A[NP * piv + col])) piv = rw;
                if (std::fabs(A[NP * piv + col]) < 1e-14) { solved = false; break; }
                if (piv != col) {
                    for (int c = 0; c < NP; c++) std::swap(A[NP * piv + c], A[NP * col + c]);
                    std::swap(g[piv], g[col]);
                }
                for (int rw = col + 1; rw < NP; rw++) {
                    double m = A[NP * rw + col] / A[NP * col + col];
                    for (int c = col; c < NP; c++) A[NP * rw + c] -= m * A[NP * col + c];
                    g[rw] -= m * g[col];
                }
            }
            if (!solved) lambda *= 10;
        }
        if (!solved) break;
        for (int a = NP - 1; a >= 0; a--) {
            double sum = g[a];
            for (int c = a + 1; c < NP; c++) sum -= A[NP * a + c] * d[c];
            d[a] = sum / A[NP * a + a];
        }
        Pose trialP = stepP(pose, d);
        double trialS = stepS(s0, d);
        double c1 = cost(trialP, trialS);
        if (c1 < c0) {
            double dn = 0;
            for (int a = 0; a < NP; a++) dn += d[a] * d[a];
            pose = trialP;
            s0 = trialS;
            lambda = std::max(lambda * 0.3, 1e-10);
            bool converged = c0 - c1 < 1e-12 * std::max(1.0, c0) || dn < 1e-20;
            c0 = c1;
            if (converged) break;
        } else {
            lambda *= 10;
            if (lambda > 1e8) break;
        }
    }
    return true;
}

}  // 命名空间 pose_detail

// 在内点掩码上按 RANSAC 同一残差执行 LM；可选第七参数 focal_scale 将视线解释为 f0*s，宽角视线保持固定。
inline bool refinePose(const std::vector<Vec3>& X, const std::vector<Vec3>& b,
                       const std::vector<char>& mask, Pose& pose, double* focal_scale = nullptr,
                       int max_iters = 30) {
    std::vector<int> idx;
    for (size_t i = 0; i < X.size(); i++)
        if (mask.empty() || mask[i]) idx.push_back((int)i);
    if (idx.size() < 4) return false;
    auto resid = [&](const Pose& p, double s, int j, double* r) {
        pose_detail::residualPair(p, X[idx[j]], b[idx[j]], s, r);
    };
    double s0 = focal_scale ? *focal_scale : 1.0;
    const bool ok = pose_detail::lmRefine((int)idx.size(), resid, focal_scale ? 7 : 6, pose, s0,
                                          max_iters);
    if (focal_scale) *focal_scale = s0;
    return ok;
}

// rig 成员携带对应点、内点掩码、外参及 1/内点半径权重，使不同镜头残差处于同一尺度。
struct FrameMember {
    const std::vector<Vec3>* X;
    const std::vector<Vec3>* b;
    const std::vector<char>* mask;
    Pose cam_from_rig;
    double weight = 1.0;
};

// 整帧共用一个 rig_from_world 位姿，经各成员外参解释其内点，即使镜头视野不重叠也可共同约束帧。
inline bool refineFramePose(const std::vector<FrameMember>& members, Pose& rig_from_world,
                            int max_iters = 30) {
    std::vector<std::pair<int, int>> idx;
    for (size_t m = 0; m < members.size(); m++)
        for (size_t i = 0; i < members[m].X->size(); i++)
            if (members[m].mask->empty() || (*members[m].mask)[i]) idx.emplace_back((int)m, (int)i);
    if (idx.size() < 4) return false;
    auto resid = [&](const Pose& F, double s, int j, double* r) {
        const FrameMember& m = members[idx[j].first];
        const int i = idx[j].second;
        pose_detail::residualPair(composePose(m.cam_from_rig, F), (*m.X)[i], (*m.b)[i], s, r);
        if (r[0] < 1e3) {  // 1e3 为正深度失败标记，不随尺度变化
            r[0] *= m.weight;
            r[1] *= m.weight;
        }
    };
    double s0 = 1.0;
    return pose_detail::lmRefine((int)idx.size(), resid, 6, rig_from_world, s0, max_iters);
}

// 对世界点与单位视线执行 LO-RANSAC PnP，像素阈值按 focal 归一化；max_trials 限制预算，批量审查可降低上限以避免为噪声对应做深度搜索。
inline PnPResult ransacPnP(const std::vector<Vec3>& X, const std::vector<Vec3>& b, double focal,
                           double max_error_px = 4.0, unsigned seed = 0, int max_trials = 3000) {
    PnPResult out;
    int n = (int)X.size();
    if (n < 4) return out;
    // 最小解使用对共面和细长点集稳健的 P3P；局部重拟合使用 DLT，仅改善时接受。
    auto fit = [&](const std::vector<int>& s) {
        std::array<Vec3, 3> br, Xs;
        for (int k = 0; k < 3; k++) { br[k] = b[s[k]]; Xs[k] = X[s[k]]; }
        return p3p(br, Xs);
    };
    auto refit = [&](const std::vector<int>& s) { return estimatePoseDLT(X, b, s); };
    auto res = [&](const Pose& p, int i) { return pnpResidualSq(p, X[i], b[i]); };
    RansacOptions ro;
    ro.max_error = max_error_px / focal;  // 残差使用归一化单位
    ro.seed = seed;
    ro.min_num_trials = std::min(100, max_trials);
    ro.max_num_trials = max_trials;
    RansacReport<Pose> rep = loransac<Pose>(n, 3, fit, refit, res, ro);
    out.pose = rep.model;
    out.inlier_mask = rep.inlier_mask;
    out.num_inliers = rep.num_inliers;
    out.success = rep.success;
    return out;
}

// ---------------- 广义相机装置 PnP ----------------

// 每成员提供世界点、镜头坐标单位视线、rig 外参及镜头内点角半径 errRad。
struct RigPnPMember {
    const std::vector<Vec3>* X;
    const std::vector<Vec3>* b;
    Pose cam_from_rig;
    double max_error = 0;
};

struct RigPnPResult {
    Pose rig_from_world;
    int num_inliers = 0;
    bool success = false;
};

// LO-RANSAC 联合全部成员对应，假设必须解释整帧；射线不共心时使用广义 GP3P（D78）。
inline RigPnPResult ransacRigPnP(const std::vector<RigPnPMember>& members, unsigned seed = 0,
                                 int max_trials = 3000) {
    RigPnPResult out;
    struct Entry {
        int m;
        int i;
        RigRay ray;
    };
    std::vector<Entry> pool;
    std::vector<double> inv2(members.size(), 0.0);
    std::vector<int> start(members.size(), 0), count(members.size(), 0);
    for (size_t m = 0; m < members.size(); m++) {
        const RigPnPMember& mem = members[m];
        start[m] = (int)pool.size();
        if (!(mem.max_error > 0)) continue;
        inv2[m] = 1.0 / (mem.max_error * mem.max_error);
        const Mat3 rig_from_cam = transpose(mem.cam_from_rig.R);
        const Vec3 centre = cameraCenter(mem.cam_from_rig);
        for (size_t i = 0; i < mem.X->size(); i++)
            pool.push_back({(int)m, (int)i, {centre, mul(rig_from_cam, (*mem.b)[i]).normalized()}});
        count[m] = (int)pool.size() - start[m];
    }
    if (pool.size() < 3) return out;

    auto res = [&](const Pose& F, int k) {
        const Entry& e = pool[k];
        const RigPnPMember& mem = members[e.m];
        const Vec3 rig = mul(F.R, (*mem.X)[e.i]) + F.t;
        return pnpResidualSqAt(mul(mem.cam_from_rig.R, rig) + mem.cam_from_rig.t,
                               (*mem.b)[e.i]) *
               inv2[e.m];
    };
    // 首个样本所在镜头若有足够对应，优先从同镜头取满；重建估计的 rig 外参通常只有角度级精度，单镜头射线几何更精确。
    std::mt19937 sampler(seed + 1);
    auto fit = [&](const std::vector<int>& s) {
        int take[3] = {s[0], s[1], s[2]};
        const int m = pool[s[0]].m;
        if (count[m] >= 3) {
            std::uniform_int_distribution<int> own(start[m], start[m] + count[m] - 1);
            for (int k = 1; k < 3; k++) {
                do take[k] = own(sampler);
                while (take[k] == take[0] || (k == 2 && take[k] == take[1]));
            }
        }
        std::array<RigRay, 3> rays;
        std::array<Vec3, 3> Xs;
        for (int k = 0; k < 3; k++) {
            const Entry& e = pool[take[k]];
            rays[k] = e.ray;
            Xs[k] = (*members[e.m].X)[e.i];
        }
        return gp3p(rays, Xs);
    };
    std::vector<std::vector<char>> masks(members.size());
    std::vector<FrameMember> fm(members.size());
    for (size_t m = 0; m < members.size(); m++) {
        masks[m].resize(members[m].X->size());
        fm[m] = {members[m].X, members[m].b, &masks[m], members[m].cam_from_rig,
                 inv2[m] > 0 ? 1.0 / members[m].max_error : 1.0};
    }
    auto refit = [&](const std::vector<int>& idx, const Pose& seed) {
        for (std::vector<char>& v : masks) std::fill(v.begin(), v.end(), 0);
        for (int k : idx) masks[pool[k].m][pool[k].i] = 1;
        Pose F = seed;
        std::vector<Pose> got;
        // 局部优化每次改善都会运行，数值雅可比每点每轮需 13 次残差，因此只用 10 轮 LM，而非默认 30。
        if (refineFramePose(fm, F, 10)) got.push_back(F);
        return got;
    };

    RansacOptions ro;
    ro.max_error = 1.0;  // res 已除以成员自身内点半径
    ro.seed = seed;
    ro.min_num_trials = std::min(100, max_trials);
    ro.max_num_trials = max_trials;
    RansacReport<Pose> rep = loransac<Pose>((int)pool.size(), 3, fit, refit, res, ro);
    if (!rep.success) return out;
    out.rig_from_world = rep.model;
    out.num_inliers = rep.num_inliers;
    out.success = true;
    return out;
}

}  // 命名空间 sfm
