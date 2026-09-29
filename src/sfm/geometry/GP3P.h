// 广义 P3P 用 rig 各镜头共同提供的三组对应估计 rig_from_world；射线在 rig 系中为 o_i+lambda_i*d_i。
// 射线共心时没有广义基线，退回普通 P3P。
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "sfm/geometry/Essential.h"  // 使用 Pose 与 nearestRotation
#include "sfm/geometry/LinAlg.h"
#include "sfm/geometry/P3P.h"

namespace sfm {

// rig 坐标系中的广义相机射线。
struct RigRay {
    Vec3 o;  // 成员镜头中心
    Vec3 d;  // 单位方向
};

namespace gp3p_detail {

// 每点对满足 lambda_i^2+lambda_j^2-2c li lj+2p li-2q lj+e=0，由世界点距与两条射线共同确定。
struct PairEq {
    double c = 0, p = 0, q = 0, e = 0;
};

// 隐藏深度多项式的稠密系数，低次在前；完整消元最高八次，数组固定大小。
struct Poly {
    double c[9] = {0};
    int n = 0;
};

inline Poly polyC(double a0) {
    Poly p;
    p.c[0] = a0;
    return p;
}
inline Poly polyL(double a0, double a1) {
    Poly p;
    p.c[0] = a0;
    p.c[1] = a1;
    p.n = 1;
    return p;
}
inline Poly polyQ(double a0, double a1, double a2) {
    Poly p;
    p.c[0] = a0;
    p.c[1] = a1;
    p.c[2] = a2;
    p.n = 2;
    return p;
}

inline Poly mulP(const Poly& a, const Poly& b) {
    Poly r;
    r.n = a.n + b.n;
    for (int i = 0; i <= a.n; i++)
        for (int j = 0; j <= b.n; j++) r.c[i + j] += a.c[i] * b.c[j];
    return r;
}
inline Poly addP(const Poly& a, const Poly& b) {
    Poly r;
    r.n = a.n > b.n ? a.n : b.n;
    for (int i = 0; i <= r.n; i++) r.c[i] = (i <= a.n ? a.c[i] : 0) + (i <= b.n ? b.c[i] : 0);
    return r;
}
inline Poly subP(const Poly& a, const Poly& b) {
    Poly r;
    r.n = a.n > b.n ? a.n : b.n;
    for (int i = 0; i <= r.n; i++) r.c[i] = (i <= a.n ? a.c[i] : 0) - (i <= b.n ? b.c[i] : 0);
    return r;
}

// 用 Aberth-Ehrlich 求实根，按根模长几何平均缩放变量；原始八次系数跨度可达 10^10，未缩放时约 7% 样本丢失全部根。
inline int realRoots(const Poly& poly, double* out, double imag_tol = 1e-4) {
    double amax = 0;
    for (int i = 0; i <= poly.n; i++) amax = std::max(amax, std::fabs(poly.c[i]));
    if (amax <= 0) return 0;
    int n = poly.n;
    while (n > 0 && std::fabs(poly.c[n]) < 1e-13 * amax) n--;
    if (n == 0) return 0;

    double a[9];
    for (int i = 0; i <= n; i++) a[i] = poly.c[i] / poly.c[n];
    double rho = std::pow(std::fabs(a[0]), 1.0 / n);
    if (!(rho > 1e-150) || !std::isfinite(rho)) {
        rho = 0;
        for (int k = 1; k <= n; k++)
            rho = std::max(rho, std::pow(std::fabs(a[n - k]), 1.0 / k));
        if (!(rho > 1e-150) || !std::isfinite(rho)) rho = 1;
    }
    for (int i = n - 1, p = 1; i >= 0; i--, p++) a[i] /= std::pow(rho, (double)p);

    auto evalReal = [&](double x) {
        double v = a[n];
        for (int i = n - 1; i >= 0; i--) v = v * x + a[i];
        return v;
    };
    auto evalRealD = [&](double x) {
        double v = n * a[n];
        for (int i = n - 1; i >= 1; i--) v = v * x + i * a[i];
        return v;
    };

    int found = 0;
    auto emit = [&](double x) {
        for (int it = 0; it < 12; it++) {
            const double d = evalRealD(x);
            if (std::fabs(d) < 1e-300) break;
            const double step = evalReal(x) / d;
            x -= step;
            if (std::fabs(step) < 1e-15 * (1.0 + std::fabs(x))) break;
        }
        x *= rho;
        for (int i = 0; i < found; i++)
            if (std::fabs(out[i] - x) < 1e-9 * (1.0 + std::fabs(x))) return;
        out[found++] = x;
    };

    if (n == 1) {
        emit(-a[0]);
        return found;
    }
    if (n == 2) {
        const double disc = a[1] * a[1] - 4 * a[0];
        if (disc < 0) return 0;
        const double s = std::sqrt(disc);
        emit(0.5 * (-a[1] + s));
        emit(0.5 * (-a[1] - s));
        return found;
    }

    double zr[9], zi[9];
    bool done[9];
    for (int k = 0; k < n; k++) {
        const double th = 2.0 * M_PI * k / n + 0.6;
        zr[k] = std::cos(th);
        zi[k] = std::sin(th);
        done[k] = false;
    }
    // 逐根停止迭代，避免近重根在最低位往返震荡；统一等待最慢根曾使 45% 样本耗尽迭代预算。
    for (int it = 0; it < 40; it++) {
        int live = 0;
        for (int k = 0; k < n; k++) {
            if (done[k]) continue;
            live++;
            double vr = a[n], vi = 0, dr = 0, di = 0;
            for (int i = n - 1; i >= 0; i--) {
                const double ndr = dr * zr[k] - di * zi[k] + vr;
                di = dr * zi[k] + di * zr[k] + vi;
                dr = ndr;
                const double nvr = vr * zr[k] - vi * zi[k] + a[i];
                vi = vr * zi[k] + vi * zr[k];
                vr = nvr;
            }
            const double dn = dr * dr + di * di;
            if (dn < 1e-300) continue;
            const double qr = (vr * dr + vi * di) / dn, qi = (vi * dr - vr * di) / dn;
            double sr = 0, si = 0;
            for (int j = 0; j < n; j++) {
                if (j == k) continue;
                const double xr = zr[k] - zr[j], xi = zi[k] - zi[j];
                const double xn = xr * xr + xi * xi;
                if (xn < 1e-300) continue;
                sr += xr / xn;
                si -= xi / xn;
            }
            const double er = 1.0 - (qr * sr - qi * si), ei = -(qr * si + qi * sr);
            const double en = er * er + ei * ei;
            if (en < 1e-300) continue;
            const double wr = (qr * er + qi * ei) / en, wi = (qi * er - qr * ei) / en;
            zr[k] -= wr;
            zi[k] -= wi;
            if (std::fabs(wr) + std::fabs(wi) < 1e-13 * (1.0 + std::fabs(zr[k]))) done[k] = true;
        }
        if (!live) break;
    }
    for (int k = 0; k < n; k++)
        if (std::fabs(zi[k]) < imag_tol * (1.0 + std::fabs(zr[k]))) emit(zr[k]);
    return found;
}

inline double eqResidual(const PairEq& q, double li, double lj) {
    return li * li + lj * lj - 2 * q.c * li * lj + 2 * q.p * li - 2 * q.q * lj + q.e;
}

// 对三个距离方程做 Newton 精化，返回相对深度尺度的残差，补偿近重根八次多项式的有限精度。
inline double polishLambdas(const PairEq eq[3], double l[3]) {
    static const int I[3] = {0, 0, 1}, J[3] = {1, 2, 2};
    double r[3] = {0, 0, 0};
    for (int it = 0; it < 8; it++) {
        double sum = 0;
        for (int k = 0; k < 3; k++) {
            r[k] = eqResidual(eq[k], l[I[k]], l[J[k]]);
            sum += std::fabs(r[k]);
        }
        const double mag = 1.0 + l[0] * l[0] + l[1] * l[1] + l[2] * l[2];
        if (sum < 1e-13 * mag) return sum / mag;
        Mat3 Jm{};
        for (int k = 0; k < 3; k++) {
            Jm[3 * k + I[k]] = 2 * l[I[k]] - 2 * eq[k].c * l[J[k]] + 2 * eq[k].p;
            Jm[3 * k + J[k]] = 2 * l[J[k]] - 2 * eq[k].c * l[I[k]] - 2 * eq[k].q;
        }
        bool ok = false;
        const Mat3 Ji = inverse3(Jm, &ok);
        if (!ok) break;
        const Vec3 d = mul(Ji, Vec3{r[0], r[1], r[2]});
        l[0] -= d.x;
        l[1] -= d.y;
        l[2] -= d.z;
    }
    double sum = 0;
    for (int k = 0; k < 3; k++) sum += std::fabs(eqResidual(eq[k], l[I[k]], l[J[k]]));
    return sum / (1.0 + l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
}

}  // 命名空间 gp3p_detail

inline std::vector<Pose> gp3p(const std::array<RigRay, 3>& rays, const std::array<Vec3, 3>& X) {
    using namespace gp3p_detail;
    std::vector<Pose> out;

    const Vec3 W01 = X[0] - X[1], W02 = X[0] - X[2], W12 = X[1] - X[2];
    const double scale = (W01.norm() + W02.norm() + W12.norm()) / 3.0;
    if (!(scale > 0) || !std::isfinite(scale)) return out;
    const double inv = 1.0 / scale;

    Vec3 o[3], d[3];
    double baseline = 0;
    for (int i = 0; i < 3; i++) {
        o[i] = rays[i].o * inv;
        d[i] = rays[i].d.normalized();
    }
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++) baseline = std::max(baseline, (o[i] - o[j]).norm());

    // 共点射线无基线，八次消元退化，应在公共中心使用普通 P3P。
    if (baseline < 1e-5) {
        const Vec3 centre = (rays[0].o + rays[1].o + rays[2].o) * (1.0 / 3.0);
        out = p3p({d[0], d[1], d[2]}, X);
        for (Pose& p : out) p.t = p.t + centre;
        return out;
    }

    static const int I[3] = {0, 0, 1}, J[3] = {1, 2, 2};
    const double dist[3] = {W01.norm() * inv, W02.norm() * inv, W12.norm() * inv};
    PairEq eq[3];
    for (int k = 0; k < 3; k++) {
        const Vec3 u = o[I[k]] - o[J[k]];
        eq[k].c = d[I[k]].dot(d[J[k]]);
        eq[k].p = u.dot(d[I[k]]);
        eq[k].q = u.dot(d[J[k]]);
        eq[k].e = u.dot(u) - dist[k] * dist[k];
    }

    // 隐藏 lambda_0 后，前两方程分别是 lambda_1/lambda_2 的首一二次式；约简第三式得到双线性关系，再用结式消元。
    const Poly b1 = polyL(-2 * eq[0].q, -2 * eq[0].c);
    const Poly c1 = polyQ(eq[0].e, 2 * eq[0].p, 1);
    const Poly b2 = polyL(-2 * eq[1].q, -2 * eq[1].c);
    const Poly c2 = polyQ(eq[1].e, 2 * eq[1].p, 1);
    const Poly A = polyC(-2 * eq[2].c);
    const Poly B = polyL(2 * eq[2].p + 2 * eq[0].q, 2 * eq[0].c);
    const Poly C = polyL(-2 * eq[2].q + 2 * eq[1].q, 2 * eq[1].c);
    const Poly D = polyQ(eq[2].e - eq[0].e - eq[1].e, -2 * (eq[0].p + eq[1].p), -2);

    // (B l1+D)^2-b2(B l1+D)(A l1+C)+c2(A l1+C)^2 为 lambda_1 二次式，须与 eq[0] 有共同根。
    const Poly alpha = addP(subP(mulP(B, B), mulP(mulP(b2, A), B)), mulP(c2, mulP(A, A)));
    const Poly beta = addP(subP(mulP(polyC(2), mulP(B, D)),
                                mulP(b2, addP(mulP(B, C), mulP(D, A)))),
                           mulP(polyC(2), mulP(c2, mulP(A, C))));
    const Poly gamma = addP(subP(mulP(D, D), mulP(mulP(b2, D), C)), mulP(c2, mulP(C, C)));
    const Poly r0 = subP(mulP(alpha, c1), gamma);
    const Poly r1 = subP(mulP(alpha, b1), beta);
    const Poly r2 = subP(mulP(beta, c1), mulP(gamma, b1));
    const Poly res = subP(mulP(r0, r0), mulP(r1, r2));

    const Vec3 w1 = W01 * inv, w2 = W02 * inv, w3 = w1.cross(w2);
    bool invertible = false;
    const Mat3 Wi = inverse3({w1.x, w2.x, w3.x, w1.y, w2.y, w3.y, w1.z, w2.z, w3.z}, &invertible);
    if (!invertible) return out;

    double roots[9];
    const int nr = realRoots(res, roots);
    for (int k = 0; k < nr; k++) {
        const double l0 = roots[k];
        if (!(l0 > 1e-9) || !std::isfinite(l0)) continue;
        const double bb1 = b1.c[0] + b1.c[1] * l0, cc1 = c1.c[0] + l0 * (c1.c[1] + l0);
        const double bb2 = b2.c[0] + b2.c[1] * l0, cc2 = c2.c[0] + l0 * (c2.c[1] + l0);
        // 重根判别式接近零，舍入可使其略正或略负，后续 Newton 均可恢复。
        const double disc1 = bb1 * bb1 - 4 * cc1, disc2 = bb2 * bb2 - 4 * cc2;
        if (disc1 < -1e-8 || disc2 < -1e-8) continue;
        const double s1 = std::sqrt(std::max(disc1, 0.0)), s2 = std::sqrt(std::max(disc2, 0.0));
        for (int u = 0; u < 2; u++) {
            const double l1 = 0.5 * (-bb1 + (u ? -s1 : s1));
            if (!(l1 > 1e-9)) continue;
            for (int v = 0; v < 2; v++) {
                double lam[3] = {l0, l1, 0.5 * (-bb2 + (v ? -s2 : s2))};
                if (!(lam[2] > 1e-9)) continue;
                if (std::fabs(eqResidual(eq[2], lam[1], lam[2])) >
                    0.05 * (1 + lam[1] * lam[1] + lam[2] * lam[2]))
                    continue;
                if (polishLambdas(eq, lam) > 1e-10) continue;
                if (!(lam[0] > 0 && lam[1] > 0 && lam[2] > 0)) continue;

                const Vec3 P0 = o[0] + d[0] * lam[0];
                const Vec3 v1 = P0 - (o[1] + d[1] * lam[1]);
                const Vec3 v2 = P0 - (o[2] + d[2] * lam[2]);
                const Vec3 v3 = v1.cross(v2);
                const Mat3 R = nearestRotation(
                    mul(Mat3{v1.x, v2.x, v3.x, v1.y, v2.y, v3.y, v1.z, v2.z, v3.z}, Wi));
                const Pose pose{R, P0 * scale - mul(R, X[0])};
                bool dup = false;
                for (const Pose& q : out) {
                    double dr = 0;
                    for (int i = 0; i < 9; i++) dr += std::fabs(q.R[i] - R[i]);
                    dup = dup || (dr < 1e-8 && (q.t - pose.t).norm() < 1e-8 * scale);
                }
                if (!dup) out.push_back(pose);
            }
        }
    }
    return out;
}

}  // 命名空间 sfm
