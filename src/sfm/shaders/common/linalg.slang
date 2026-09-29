#pragma once
// Real 上的小型可微向量与矩阵操作，采用自定义 RVec2/RVec3，使 DF 与 float/double 共用同一路径。

#include "real.slang"

struct RVec2 : IDifferentiable {
    typealias Differential = RVec2;
    Real x; Real y;
    [Differentiable] static RVec2 dzero() { RVec2 v = { Real(0.0), Real(0.0) }; return v; }
    [Differentiable] static RVec2 dadd(RVec2 a, RVec2 b) { RVec2 v = { a.x + b.x, a.y + b.y }; return v; }
};
struct RVec3 : IDifferentiable {
    typealias Differential = RVec3;
    Real x; Real y; Real z;
    [Differentiable] static RVec3 dzero() { RVec3 v = { Real(0.0), Real(0.0), Real(0.0) }; return v; }
    [Differentiable] static RVec3 dadd(RVec3 a, RVec3 b) { RVec3 v = { a.x + b.x, a.y + b.y, a.z + b.z }; return v; }
};

[Differentiable] RVec2 rvec2(Real x, Real y) { RVec2 v = { x, y }; return v; }
[Differentiable] RVec3 rvec3(Real x, Real y, Real z) { RVec3 v = { x, y, z }; return v; }

[Differentiable] RVec3 operator +(RVec3 a, RVec3 b) { return rvec3(a.x + b.x, a.y + b.y, a.z + b.z); }
[Differentiable] RVec3 operator -(RVec3 a, RVec3 b) { return rvec3(a.x - b.x, a.y - b.y, a.z - b.z); }
[Differentiable] RVec3 operator *(RVec3 a, Real s) { return rvec3(a.x * s, a.y * s, a.z * s); }
[Differentiable] RVec2 operator -(RVec2 a, RVec2 b) { return rvec2(a.x - b.x, a.y - b.y); }

// 合并的可微 sincos 每次仅做一次参数约简与多项式对，反向同样复用，返回 RVec2(sin,cos)。
[BackwardDerivative(rsincos_bwd)]
RVec2 rsincos(Real x) {
    Real s, c;
    rsincos_raw(x, s, c);
    return rvec2(s, c);
}
void rsincos_bwd(inout DifferentialPair<Real> x, RVec2 g) {
    Real s, c;
    rsincos_raw(x.p, s, c);
    x = diffPair(x.p, g.x * c - g.y * s);
}

[Differentiable] Real dot3(RVec3 a, RVec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
[Differentiable] RVec3 cross3(RVec3 a, RVec3 b) {
    return rvec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
[Differentiable] Real length3(RVec3 a) { return sqrt(dot3(a, a)); }

// 行主序 Real[9] 对称 3×3 矩阵的伴随公式逆。
void inverse3(Real m[9], out Real r[9]) {
    Real c00 = m[4] * m[8] - m[5] * m[7];
    Real c01 = m[5] * m[6] - m[3] * m[8];
    Real c02 = m[3] * m[7] - m[4] * m[6];
    Real det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    Real inv_det = Real(1.0) / det;
    r[0] = c00 * inv_det;
    r[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    r[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    r[3] = c01 * inv_det;
    r[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    r[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    r[6] = c02 * inv_det;
    r[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    r[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
}
