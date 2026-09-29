#pragma once

// 三维相似变换 p' = s * R * p + t，s > 0，R 为行主序正旋转。
// 全程使用 double，避免距离原点数百万单位的地理模型在 float 中只剩米级精度；仅含头文件，供主机使用。

#include <cmath>

namespace spirula {

struct Sim3 {
    double s = 1.0;
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double t[3] = {0, 0, 0};

    bool is_identity(double eps = 1e-12) const {
        if (std::fabs(s - 1.0) > eps) return false;
        for (int i = 0; i < 9; i++)
            if (std::fabs(R[i] - (i % 4 == 0 ? 1.0 : 0.0)) > eps) return false;
        for (int i = 0; i < 3; i++)
            if (std::fabs(t[i]) > eps * (1.0 + std::fabs(t[i]))) return false;
        return true;
    }

    void apply(const double p[3], double out[3]) const {
        const double x = p[0], y = p[1], z = p[2];
        for (int r = 0; r < 3; r++)
            out[r] = s * (R[r*3+0]*x + R[r*3+1]*y + R[r*3+2]*z) + t[r];
    }
    // 方向仅旋转，不缩放或平移。
    void rotate(const double v[3], double out[3]) const {
        const double x = v[0], y = v[1], z = v[2];
        for (int r = 0; r < 3; r++)
            out[r] = R[r*3+0]*x + R[r*3+1]*y + R[r*3+2]*z;
    }

    // 行主序 3×4 [s*R | t]，与视口矩阵布局一致。
    template <typename T>
    void to_3x4(T out[12]) const {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) out[r*4+c] = (T)(s * R[r*3+c]);
            out[r*4+3] = (T)t[r];
        }
    }
    // 将 3×3 块视为 s*R，并重新正交化旋转，消除 float 往返造成的偏差。
    template <typename T>
    static Sim3 from_3x4(const T a[12]) {
        Sim3 o;
        double M[9];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) M[r*3+c] = (double)a[r*4+c];
            o.t[r] = (double)a[r*4+3];
        }
        const double det =
            M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) +
            M[2]*(M[3]*M[7]-M[4]*M[6]);
        o.s = std::cbrt(std::fabs(det));
        if (!(o.s > 1e-300)) { o.s = 1.0; return o; }
        for (int i = 0; i < 9; i++) o.R[i] = M[i] / o.s;
        o.orthonormalize();
        return o;
    }

    // 按行执行 Gram-Schmidt，第三行由叉积得到，保证舍入误差后仍为正旋转。
    void orthonormalize() {
        double* a = R; double* b = R + 3; double* c = R + 6;
        double n = std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        if (!(n > 1e-300)) return reset_rotation();
        for (int i = 0; i < 3; i++) a[i] /= n;
        double d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        for (int i = 0; i < 3; i++) b[i] -= d * a[i];
        n = std::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
        if (!(n > 1e-300)) return reset_rotation();
        for (int i = 0; i < 3; i++) b[i] /= n;
        c[0] = a[1]*b[2] - a[2]*b[1];
        c[1] = a[2]*b[0] - a[0]*b[2];
        c[2] = a[0]*b[1] - a[1]*b[0];
    }

    Sim3 inverse() const {
        Sim3 o;
        o.s = 1.0 / s;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) o.R[r*3+c] = R[c*3+r];
        for (int r = 0; r < 3; r++)
            o.t[r] = -o.s * (o.R[r*3+0]*t[0] + o.R[r*3+1]*t[1] + o.R[r*3+2]*t[2]);
        return o;
    }

    // 单位四元数 (w, x, y, z)，w >= 0。
    void quat(double q[4]) const {
        const double m00 = R[0], m01 = R[1], m02 = R[2];
        const double m10 = R[3], m11 = R[4], m12 = R[5];
        const double m20 = R[6], m21 = R[7], m22 = R[8];
        const double tr = m00 + m11 + m22;
        if (tr > 0) {
            const double k = 0.5 / std::sqrt(tr + 1.0);
            q[0] = 0.25 / k; q[1] = (m21 - m12) * k;
            q[2] = (m02 - m20) * k; q[3] = (m10 - m01) * k;
        } else if (m00 > m11 && m00 > m22) {
            const double k = 2.0 * std::sqrt(1.0 + m00 - m11 - m22);
            q[1] = 0.25 * k; q[2] = (m01 + m10) / k;
            q[3] = (m02 + m20) / k; q[0] = (m21 - m12) / k;
        } else if (m11 > m22) {
            const double k = 2.0 * std::sqrt(1.0 + m11 - m00 - m22);
            q[1] = (m01 + m10) / k; q[2] = 0.25 * k;
            q[3] = (m12 + m21) / k; q[0] = (m02 - m20) / k;
        } else {
            const double k = 2.0 * std::sqrt(1.0 + m22 - m00 - m11);
            q[1] = (m02 + m20) / k; q[2] = (m12 + m21) / k;
            q[3] = 0.25 * k; q[0] = (m10 - m01) / k;
        }
        double n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (q[0] < 0) n = -n;
        for (int i = 0; i < 4; i++) q[i] /= n;
    }

    static Sim3 translation(const double d[3]) {
        Sim3 o;
        for (int i = 0; i < 3; i++) o.t[i] = d[i];
        return o;
    }
    // 绕经过 pivot 的单位轴 axis 旋转 angle 弧度。
    static Sim3 rotation_about(const double axis[3], double angle,
                               const double pivot[3]) {
        Sim3 o;
        const double c = std::cos(angle), sn = std::sin(angle), k = 1.0 - c;
        const double x = axis[0], y = axis[1], z = axis[2];
        const double M[9] = {c + x*x*k,   x*y*k - z*sn, x*z*k + y*sn,
                             y*x*k + z*sn, c + y*y*k,   y*z*k - x*sn,
                             z*x*k - y*sn, z*y*k + x*sn, c + z*z*k};
        for (int i = 0; i < 9; i++) o.R[i] = M[i];
        o.pin(pivot);
        return o;
    }
    static Sim3 scale_about(double factor, const double pivot[3]) {
        Sim3 o;
        o.s = factor;
        o.pin(pivot);
        return o;
    }

private:
    void reset_rotation() {
        for (int i = 0; i < 9; i++) R[i] = i % 4 == 0 ? 1.0 : 0.0;
    }
    // 选择 t，使 pivot 映射到自身。
    void pin(const double pivot[3]) {
        double q[3];
        rotate(pivot, q);
        for (int i = 0; i < 3; i++) t[i] = pivot[i] - s * q[i];
    }
};

// 先应用 b，再应用 a：(a * b)(p) = a(b(p))。
inline Sim3 operator*(const Sim3& a, const Sim3& b) {
    Sim3 o;
    o.s = a.s * b.s;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            double v = 0.0;
            for (int k = 0; k < 3; k++) v += a.R[r*3+k] * b.R[k*3+c];
            o.R[r*3+c] = v;
        }
    a.apply(b.t, o.t);
    return o;
}

}  // 命名空间 spirula
