// 由两个 3×4 投影矩阵执行 DLT 三角化，并检查视差角，对应 COLMAP 的 min_tri_angle 过滤。
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "sfm/geometry/LinAlg.h"

namespace sfm {

using Mat34 = std::array<double, 12>;  // 3×4，行主序

// 前向视线采用经典 DLT 行 x*P3-P1、y*P3-P2，x=b.x/b.z；宽角或负 b.z 使用 b×(P X)=0 的两个独立行，避免透视除法退化。
inline void dltRows(double* r0, double* r1, const Vec3& b, const Mat34& P) {
    if (std::fabs(b.z) > 0.1) {
        double x = b.x / b.z, y = b.y / b.z;
        for (int c = 0; c < 4; c++) {
            r0[c] = x * P[8 + c] - P[0 + c];
            r1[c] = y * P[8 + c] - P[4 + c];
        }
    } else {
        for (int c = 0; c < 4; c++) {
            r0[c] = b.y * P[8 + c] - b.z * P[4 + c];   // (b×PX) 的第 0 分量
            r1[c] = b.z * P[0 + c] - b.x * P[8 + c];   // (b×PX) 的第 1 分量
        }
    }
}

// 由两条单位视线和投影矩阵求齐次 DLT，再返回非齐次三维点；前向与宽角由 dltRows 统一处理。
inline Vec3 triangulateDLT(const Mat34& P1, const Mat34& P2, const Vec3& b1, const Vec3& b2) {
    double A[4][4];
    dltRows(A[0], A[1], b1, P1);
    dltRows(A[2], A[3], b2, P2);
    // 使用栈存储，重三角化每轮会调用数百万次。
    double AtA[16], w[4], V[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            double s = 0;
            for (int r = 0; r < 4; r++) s += A[r][i] * A[r][j];
            AtA[i * 4 + j] = s;
        }
    jacobiEigenSymmetric(AtA, 4, w, V);
    int mi = 0;
    for (int i = 1; i < 4; i++)
        if (w[i] < w[mi]) mi = i;
    double h[4];
    for (int r = 0; r < 4; r++) h[r] = V[r * 4 + mi];
    if (std::fabs(h[3]) < 1e-30) return {0, 0, 0};
    return {h[0] / h[3], h[1] / h[3], h[2] / h[3]};
}

// 点 X 处指向相机中心 c1/c2 的射线夹角，单位弧度。
inline double triangulationAngle(const Vec3& X, const Vec3& c1, const Vec3& c2) {
    Vec3 r1 = (X - c1).normalized();
    Vec3 r2 = (X - c2).normalized();
    double c = std::max(-1.0, std::min(1.0, r1.dot(r2)));
    return std::acos(c);
}

}  // 命名空间 sfm
