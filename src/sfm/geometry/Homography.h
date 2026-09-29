// 单应性采用至少四点的归一化 DLT，RANSAC 使用平方像素单位的对称传递误差。
#pragma once

#include <cmath>
#include <vector>

#include "sfm/geometry/LinAlg.h"

namespace sfm {

// Hartley 归一化 DLT 求图像 1 到图像 2 的单应矩阵 H。
inline std::vector<Mat3> estimateHomography(const std::vector<Vec2>& p1,
                                            const std::vector<Vec2>& p2,
                                            const std::vector<int>& idx) {
    if (idx.size() < 4) return {};
    std::vector<Vec2> a, b;
    Mat3 T1 = hartleyNormalize(p1, idx, a);
    Mat3 T2 = hartleyNormalize(p2, idx, b);

    size_t n = idx.size();
    std::vector<double> A(2 * n * 9, 0.0);
    for (size_t i = 0; i < n; i++) {
        double x1 = a[i].x, y1 = a[i].y, x2 = b[i].x, y2 = b[i].y;
        double* r0 = &A[(2 * i) * 9];
        double* r1 = &A[(2 * i + 1) * 9];
        r0[0] = -x1; r0[1] = -y1; r0[2] = -1;
        r0[6] = x2 * x1; r0[7] = x2 * y1; r0[8] = x2;
        r1[3] = -x1; r1[4] = -y1; r1[5] = -1;
        r1[6] = y2 * x1; r1[7] = y2 * y1; r1[8] = y2;
    }
    auto nv = nullVectors9(A, (int)(2 * n), 1);
    if (nv.empty()) return {};
    Mat3 Hn;
    for (int i = 0; i < 9; i++) Hn[i] = nv[0][i];
    // 反归一化：H=T2^-1 Hn T1
    Mat3 H = mul(mul(inverse3(T2), Hn), T1);
    if (std::fabs(H[8]) > 1e-12)
        for (double& v : H) v /= H[8];
    return {H};
}

// 单位视线平面映射满足 b2~H b1，H=R+t n^T/d，直接对齐次三维向量做 DLT，无需 Hartley 归一化。
// 选择 H 符号使多数样本正向传递，避免把背向射线或相机后方平面当成同等有效解。
inline std::vector<Mat3> estimateHomographyBearing(const std::vector<Vec3>& p1,
                                                   const std::vector<Vec3>& p2,
                                                   const std::vector<int>& idx) {
    size_t n = idx.size();
    if (n < 4) return {};
    std::vector<double> A(2 * n * 9, 0.0);
    for (size_t i = 0; i < n; i++) {
        const Vec3& a = p1[idx[i]];
        const Vec3& b = p2[idx[i]];
        double* r0 = &A[(2 * i) * 9];
        double* r1 = &A[(2 * i + 1) * 9];
        r0[0] = -b.z * a.x; r0[1] = -b.z * a.y; r0[2] = -b.z * a.z;
        r0[6] = b.x * a.x;  r0[7] = b.x * a.y;  r0[8] = b.x * a.z;
        r1[3] = -b.z * a.x; r1[4] = -b.z * a.y; r1[5] = -b.z * a.z;
        r1[6] = b.y * a.x;  r1[7] = b.y * a.y;  r1[8] = b.y * a.z;
    }
    auto nv = nullVectors9(A, (int)(2 * n), 1);
    if (nv.empty()) return {};
    Mat3 H;
    for (int i = 0; i < 9; i++) H[i] = nv[0][i];
    int forward = 0;
    for (size_t i = 0; i < n; i++)
        forward += p2[idx[i]].dot(mul(H, p1[idx[i]])) > 0 ? 1 : -1;
    if (forward < 0) for (double& v : H) v = -v;
    return {H};
}

// 球面对称传递误差采用 tan^2(angle)=|u×v|^2/(u·v)^2，拒绝转到相反半球的射线。
// 在毫弧度阈值内与角度平方一致到七位，MSAC 又截断大残差，因此内点和评分等效，同时省去两次 atan2 与四次 sqrt。
inline double angularTransferSq(const Mat3& H, const Mat3& Hinv, const Vec3& a, const Vec3& b) {
    const Vec3 hb = mul(H, a), ha = mul(Hinv, b);
    const double d1 = hb.dot(b), d2 = ha.dot(a);
    if (d1 <= 0 || d2 <= 0) return 1e30;  // 传递到相反半球
    const Vec3 c1 = hb.cross(b), c2 = ha.cross(a);
    return c1.dot(c1) / (d1 * d1) + c2.dot(c2) / (d2 * d2);
}

inline Vec2 applyH(const Mat3& H, const Vec2& p) {
    Vec3 q = mul(H, Vec3{p.x, p.y, 1});
    if (std::fabs(q.z) < 1e-30) return {1e15, 1e15};
    return {q.x / q.z, q.y / q.z};
}

// 对称传递误差：|b-H a|^2+|a-H^-1 b|^2。
inline double symmetricTransferSq(const Mat3& H, const Mat3& Hinv, const Vec2& a, const Vec2& b) {
    Vec2 hb = applyH(H, a);
    Vec2 ha = applyH(Hinv, b);
    double d1 = (hb.x - b.x) * (hb.x - b.x) + (hb.y - b.y) * (hb.y - b.y);
    double d2 = (ha.x - a.x) * (ha.x - a.x) + (ha.y - a.y) * (ha.y - a.y);
    return d1 + d2;
}

}  // 命名空间 sfm
