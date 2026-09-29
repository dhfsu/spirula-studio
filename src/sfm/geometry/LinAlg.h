// 主机小型稠密线性代数，不依赖 Eigen，提供二维/三维向量、3×3 矩阵、对称 Jacobi 特征分解和 3×3 SVD，供几何估计使用。
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sfm {

struct Vec2 {
    double x = 0, y = 0;
};
struct Vec3 {
    double x = 0, y = 0, z = 0;
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    double norm() const { return std::sqrt(dot(*this)); }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    Vec3 normalized() const {
        double n = norm();
        return n > 0 ? Vec3{x / n, y / n, z / n} : *this;
    }
};

// 3×3，行主序。
using Mat3 = std::array<double, 9>;

inline Mat3 mat3Identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }

// 叉乘矩阵 [v]_x，满足 [v]_x u=v×u。
inline Mat3 crossMatrix(const Vec3& v) {
    return {0, -v.z, v.y, v.z, 0, -v.x, -v.y, v.x, 0};
}

inline Vec3 mul(const Mat3& M, const Vec3& v) {
    return {M[0] * v.x + M[1] * v.y + M[2] * v.z, M[3] * v.x + M[4] * v.y + M[5] * v.z,
            M[6] * v.x + M[7] * v.y + M[8] * v.z};
}

inline Mat3 mul(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += A[3 * i + k] * B[3 * k + j];
            C[3 * i + j] = s;
        }
    return C;
}

inline Mat3 transpose(const Mat3& A) {
    return {A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]};
}

inline double det3(const Mat3& A) {
    return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
           A[2] * (A[3] * A[7] - A[4] * A[6]);
}

inline Mat3 inverse3(const Mat3& A, bool* ok = nullptr) {
    double d = det3(A);
    if (ok) *ok = std::fabs(d) > 1e-300;
    if (std::fabs(d) < 1e-300) return mat3Identity();
    double id = 1.0 / d;
    Mat3 C;
    C[0] = (A[4] * A[8] - A[5] * A[7]) * id;
    C[1] = (A[2] * A[7] - A[1] * A[8]) * id;
    C[2] = (A[1] * A[5] - A[2] * A[4]) * id;
    C[3] = (A[5] * A[6] - A[3] * A[8]) * id;
    C[4] = (A[0] * A[8] - A[2] * A[6]) * id;
    C[5] = (A[2] * A[3] - A[0] * A[5]) * id;
    C[6] = (A[3] * A[7] - A[4] * A[6]) * id;
    C[7] = (A[1] * A[6] - A[0] * A[7]) * id;
    C[8] = (A[0] * A[4] - A[1] * A[3]) * id;
    return C;
}

// 原地循环 Jacobi 分解对称 n×n 行主序 A；特征值写 w，特征向量为 V 的列，不排序且不分配内存。
inline void jacobiEigenSymmetric(double* A, int n, double* w, double* V) {
    for (int i = 0; i < n; i++) w[i] = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) V[i] = 0;
    for (int i = 0; i < n; i++) V[(size_t)i * n + i] = 1.0;

    // 按矩阵自身尺度判断收敛；绝对阈值可能使病态尺度矩阵每次耗尽 100 轮（D24）。
    double frob2 = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) frob2 += A[i] * A[i];
    if (!(frob2 > 0)) return;  // 零矩阵的特征值为 0，V=I
    const double tol = 1e-30 * frob2;

    for (int sweep = 0; sweep < 50; sweep++) {
        double off = 0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) off += A[(size_t)p * n + q] * A[(size_t)p * n + q];
        if (off <= tol) break;
        // 远离收敛时仅旋转足够大的非对角项。
        const double tresh = sweep < 3 ? 0.2 * off / ((double)n * n) : 0.0;

        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                double apq = A[(size_t)p * n + q];
                if (apq * apq <= tresh * tresh) continue;  // 同时跳过 apq==0
                double app = A[(size_t)p * n + p], aqq = A[(size_t)q * n + q];
                // 用 t=sign(theta)/(|theta|+sqrt(theta^2+1))，theta=cot(2 phi)，得到 |phi|<=pi/4 分支的相同旋转。
                // 一次 sqrt 替代 atan2/cos/sin，降低 RANSAC 高频内循环开销。
                double theta = 0.5 * (aqq - app) / apq;
                double t = (theta >= 0 ? 1.0 : -1.0) /
                           (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0);
                double s = t * c;
                for (int k = 0; k < n; k++) {
                    double akp = A[(size_t)k * n + p], akq = A[(size_t)k * n + q];
                    A[(size_t)k * n + p] = c * akp - s * akq;
                    A[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {
                    double apk = A[(size_t)p * n + k], aqk = A[(size_t)q * n + k];
                    A[(size_t)p * n + k] = c * apk - s * aqk;
                    A[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; k++) {
                    double vkp = V[(size_t)k * n + p], vkq = V[(size_t)k * n + q];
                    V[(size_t)k * n + p] = c * vkp - s * vkq;
                    V[(size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < n; i++) w[i] = A[(size_t)i * n + i];
}

inline void jacobiEigenSymmetric(std::vector<double>& A, int n, std::vector<double>& w,
                                 std::vector<double>& V) {
    w.assign(n, 0);
    V.assign((size_t)n * n, 0);
    jacobiEigenSymmetric(A.data(), n, w.data(), V.data());
}

// m<9 时对 A^T 做 Householder QR 直接求精确零空间，避免最小 RANSAC 样本反复迭代特征分解。
// A^T=Q^T R，Q 前 m 行张成行空间，后续行张成零空间；退化样本仍产生正交向量，由 RANSAC 评分淘汰。
inline std::vector<std::array<double, 9>> nullSpaceQR9(const std::vector<double>& A, int m,
                                                       int count) {
    double M[9][9] = {};  // A^T，尺寸 9×m
    for (int r = 0; r < m; r++)
        for (int c = 0; c < 9; c++) M[c][r] = A[(size_t)r * 9 + c];
    double Q[9][9] = {};
    for (int i = 0; i < 9; i++) Q[i][i] = 1.0;

    for (int k = 0; k < m; k++) {
        double nrm = 0;
        for (int i = k; i < 9; i++) nrm += M[i][k] * M[i][k];
        nrm = std::sqrt(nrm);
        if (!(nrm > 1e-300)) continue;  // 该列已消去
        // 沿较大分量方向反射，避免消减误差。
        const double alpha = M[k][k] > 0 ? -nrm : nrm;
        double v[9] = {};
        for (int i = k; i < 9; i++) v[i] = M[i][k];
        v[k] -= alpha;
        double vn = 0;
        for (int i = k; i < 9; i++) vn += v[i] * v[i];
        if (!(vn > 1e-300)) continue;
        for (int c = k; c < m; c++) {  // 将 H 应用到 M 剩余列
            double d = 0;
            for (int i = k; i < 9; i++) d += v[i] * M[i][c];
            d = 2 * d / vn;
            for (int i = k; i < 9; i++) M[i][c] -= d * v[i];
        }
        for (int c = 0; c < 9; c++) {  // 将 H 累积到 Q
            double d = 0;
            for (int i = k; i < 9; i++) d += v[i] * Q[i][c];
            d = 2 * d / vn;
            for (int i = k; i < 9; i++) Q[i][c] -= d * v[i];
        }
    }

    std::vector<std::array<double, 9>> out;
    for (int r = m; r < 9 && (int)out.size() < count; r++) {
        std::array<double, 9> v{};
        for (int c = 0; c < 9; c++) v[c] = Q[r][c];
        out.push_back(v);
    }
    return out;
}

// 求 A^T A 的 count 个最小特征值对应向量，按升序返回长度 9 的右零空间候选。
inline std::vector<std::array<double, 9>> nullVectors9(const std::vector<double>& A, int m,
                                                       int count) {
    // 欠定最小样本直接用精确零空间；超定重拟合需最小奇异向量，仍用特征分解。后者每对约十次，最小解约上万次（D27）。
    if (m < 9 && count <= 9 - m) return nullSpaceQR9(A, m, count);

    std::vector<double> ata((size_t)9 * 9, 0.0);
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++) {
            double s = 0;
            for (int r = 0; r < m; r++) s += A[(size_t)r * 9 + i] * A[(size_t)r * 9 + j];
            ata[(size_t)i * 9 + j] = s;
        }
    std::vector<double> w, V;
    jacobiEigenSymmetric(ata, 9, w, V);
    // 按特征值升序的索引
    std::array<int, 9> idx{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return w[a] < w[b]; });
    std::vector<std::array<double, 9>> out;
    for (int c = 0; c < count; c++) {
        std::array<double, 9> v{};
        for (int r = 0; r < 9; r++) v[r] = V[(size_t)r * 9 + idx[c]];
        out.push_back(v);
    }
    return out;
}

// 通过 A^T A 最小特征向量求 rows×cols 矩阵的最小右奇异向量，cols 约不超过 16。
inline std::vector<double> nullspaceVector(const std::vector<double>& A, int rows, int cols) {
    std::vector<double> ata((size_t)cols * cols, 0.0);
    for (int i = 0; i < cols; i++)
        for (int j = 0; j < cols; j++) {
            double s = 0;
            for (int r = 0; r < rows; r++) s += A[(size_t)r * cols + i] * A[(size_t)r * cols + j];
            ata[(size_t)i * cols + j] = s;
        }
    std::vector<double> w, V;
    jacobiEigenSymmetric(ata, cols, w, V);
    int mi = 0;
    for (int i = 1; i < cols; i++)
        if (w[i] < w[mi]) mi = i;
    std::vector<double> v(cols);
    for (int r = 0; r < cols; r++) v[r] = V[(size_t)r * cols + mi];
    return v;
}

// 3×3 SVD：A=U diag(s)V^T，奇异值降序，U/V 正交，由 A^T A 对称特征分解构造。
struct Svd3 {
    Mat3 U, V;
    Vec3 s;
};

inline Svd3 svd3(const Mat3& A) {
    // 由 eigen(A^T A) 得到 V 与 s^2
    Mat3 AtA = mul(transpose(A), A);
    std::vector<double> M(AtA.begin(), AtA.end()), w, Vv;
    jacobiEigenSymmetric(M, 3, w, Vv);
    int idx[3] = {0, 1, 2};
    std::sort(idx, idx + 3, [&](int a, int b) { return w[a] > w[b]; });

    Svd3 r;
    r.s = {0, 0, 0};
    Vec3 vcol[3];
    for (int c = 0; c < 3; c++) {
        (&r.s.x)[c] = std::sqrt(std::max(0.0, w[idx[c]]));
        vcol[c] = {Vv[0 * 3 + idx[c]], Vv[1 * 3 + idx[c]], Vv[2 * 3 + idx[c]]};
    }
    // V 的列
    for (int c = 0; c < 3; c++) {
        r.V[0 * 3 + c] = vcol[c].x;
        r.V[1 * 3 + c] = vcol[c].y;
        r.V[2 * 3 + c] = vcol[c].z;
    }
    // U 列为 A v_c/s_c，数值零奇异值需补齐方向；阈值必须相对 s_max，因 A^T A 误差开方后为约 1e-8*s_max。
    // 过小绝对阈值会误判秩并产生零列，曾破坏约 46% 的本质矩阵位姿分解（D25）。
    const double sigTol = 1e-6 * r.s.x;

    Vec3 ucol[3]{};
    bool have[3] = {false, false, false};

    // 从最大奇异值的可靠方向开始 Gram-Schmidt；除以小 s_c 会放大舍入，即使非零列也须重新正交化。
    for (int c = 0; c < 3; c++) {
        double sig = (&r.s.x)[c];
        if (sig <= sigTol) {
            (&r.s.x)[c] = 0.0;  // 数值为零，明确返回零
            continue;
        }
        Vec3 u = mul(A, vcol[c]) * (1.0 / sig);
        for (int d = 0; d < c; d++)
            if (have[d]) u = u - ucol[d] * u.dot(ucol[d]);
        double n = u.norm();
        if (n > 1e-12) {
            ucol[c] = u * (1.0 / n);
            have[c] = true;
        } else {
            (&r.s.x)[c] = 0.0;  // 与前列数值线性相关
        }
    }
    // 选择最远离现有张成空间的坐标轴补齐方向，适用于任意秩。
    const Vec3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int c = 0; c < 3; c++) {
        if (have[c]) continue;
        Vec3 best{0, 0, 1};
        double bestNorm = -1;
        for (const Vec3& axis : axes) {
            Vec3 v = axis;
            for (int d = 0; d < 3; d++)
                if (have[d]) v = v - ucol[d] * v.dot(ucol[d]);
            double n = v.norm();
            if (n > bestNorm) { bestNorm = n; best = v; }
        }
        ucol[c] = bestNorm > 1e-12 ? best.normalized() : Vec3{0, 0, 1};
        have[c] = true;  // 后续补齐方向保持与该列正交
    }
    for (int c = 0; c < 3; c++) {
        r.U[0 * 3 + c] = ucol[c].x;
        r.U[1 * 3 + c] = ucol[c].y;
        r.U[2 * 3 + c] = ucol[c].z;
    }
    return r;
}

}  // 命名空间 sfm
