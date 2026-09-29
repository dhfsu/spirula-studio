#pragma once
// IMU 流形预积分，参考 Forster 等（2017）；在首时刻 IMU 坐标系累积旋转、速度与位置，并保存一阶偏置雅可比，后续无需重新积分样本。
// 对右手系样本采用中点积分。

#include <cmath>
#include <vector>

#include "sfm/core/Pose.h"
#include "sfm/geometry/LinAlg.h"

namespace sfm {

struct ImuSample {
    double t = 0;
    Vec3 w;   // rad/s
    Vec3 a;   // m/s^2，比力，包含重力
};

// 连续时间噪声密度，MEMS 默认值来自静止 X5 与 MAX 的测量。
struct ImuNoise {
    double gyro = 2e-3;    // rad/s/sqrt(Hz)
    double accel = 2e-2;   // m/s^2/sqrt(Hz)
};

inline Mat3 mat3Add(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 9; i++) C[i] = A[i] + B[i];
    return C;
}
inline Mat3 mat3Scale(const Mat3& A, double s) {
    Mat3 C;
    for (int i = 0; i < 9; i++) C[i] = A[i] * s;
    return C;
}

struct Preintegration {
    double dt = 0;
    Mat3 dR = mat3Identity();   // R_i(t0) <- i(t1)
    Vec3 dv, dp;                // t0 时刻 IMU 坐标系，包含重力
    Mat3 dR_dbg{}, dv_dbg{}, dv_dba{}, dp_dbg{}, dp_dba{};
    double var_r = 0, var_v = 0, var_p = 0;   // 各向同性方差，单位依次为 rad^2、m^2/s^2、m^2
    size_t samples = 0;

    bool ok() const { return samples >= 2 && dt > 0; }
    Mat3 rotation(const Vec3& dbg) const { return mul(dR, so3Exp(mul(dR_dbg, dbg))); }
    Vec3 velocity(const Vec3& dbg, const Vec3& dba) const {
        return dv + mul(dv_dbg, dbg) + mul(dv_dba, dba);
    }
    Vec3 position(const Vec3& dbg, const Vec3& dba) const {
        return dp + mul(dp_dbg, dbg) + mul(dp_dba, dba);
    }
};

// s 已按时间排序，调用方插值端点后覆盖 [t0,t1]；偏置在此扣除。
inline Preintegration preintegrate(const std::vector<ImuSample>& s, const Vec3& bg,
                                   const Vec3& ba, const ImuNoise& noise) {
    Preintegration P;
    P.samples = s.size();
    if (s.size() < 2) return P;
    Mat3 R = mat3Identity();
    for (size_t k = 0; k + 1 < s.size(); k++) {
        const double dt = s[k + 1].t - s[k].t;
        if (!(dt > 0)) continue;
        const Vec3 w = (s[k].w + s[k + 1].w) * 0.5 - bg;
        const Vec3 a = (s[k].a + s[k + 1].a) * 0.5 - ba;
        const Mat3 Ra = mul(R, crossMatrix(a));
        const Mat3 RaJ = mul(Ra, P.dR_dbg);

        P.dp = P.dp + P.dv * dt + mul(R, a) * (0.5 * dt * dt);
        P.dp_dba = mat3Add(P.dp_dba, mat3Add(mat3Scale(P.dv_dba, dt), mat3Scale(R, -0.5 * dt * dt)));
        P.dp_dbg = mat3Add(P.dp_dbg, mat3Add(mat3Scale(P.dv_dbg, dt), mat3Scale(RaJ, -0.5 * dt * dt)));
        P.dv = P.dv + mul(R, a) * dt;
        P.dv_dba = mat3Add(P.dv_dba, mat3Scale(R, -dt));
        P.dv_dbg = mat3Add(P.dv_dbg, mat3Scale(RaJ, -dt));

        const Vec3 phi = w * dt;
        const Mat3 E = so3Exp(phi);
        P.dR_dbg = mat3Add(mul(transpose(E), P.dR_dbg), mat3Scale(so3RightJacobian(phi), -dt));
        R = mul(R, E);

        const double an = a.norm();
        P.var_p += P.var_v * dt * dt + noise.accel * noise.accel * dt * dt * dt / 3.0;
        P.var_v += noise.accel * noise.accel * dt + an * an * dt * dt * P.var_r;
        P.var_r += noise.gyro * noise.gyro * dt;
        P.dt += dt;
    }
    P.dR = R;
    return P;
}

// 用融合姿态替代陀螺积分，每样本 R 为 R_i(t0) <- i(t)，没有陀螺偏置，相应雅可比保持零。
struct AttitudeSample {
    double t = 0;
    Vec3 a;   // m/s^2，比力
    Mat3 R = mat3Identity();
};

inline Preintegration preintegrateAttitude(const std::vector<AttitudeSample>& s, const Vec3& ba,
                                           const ImuNoise& noise) {
    Preintegration P;
    P.samples = s.size();
    if (s.size() < 2) return P;
    for (size_t k = 0; k + 1 < s.size(); k++) {
        const double dt = s[k + 1].t - s[k].t;
        if (!(dt > 0)) continue;
        const Mat3 R = mat3Scale(mat3Add(s[k].R, s[k + 1].R), 0.5);
        const Vec3 a = (s[k].a + s[k + 1].a) * 0.5 - ba;
        const Vec3 f = mul(R, a);

        P.dp = P.dp + P.dv * dt + f * (0.5 * dt * dt);
        P.dp_dba = mat3Add(P.dp_dba, mat3Add(mat3Scale(P.dv_dba, dt), mat3Scale(R, -0.5 * dt * dt)));
        P.dv = P.dv + f * dt;
        P.dv_dba = mat3Add(P.dv_dba, mat3Scale(R, -dt));

        const double an = a.norm();
        P.var_p += P.var_v * dt * dt + noise.accel * noise.accel * dt * dt * dt / 3.0;
        P.var_v += noise.accel * noise.accel * dt + an * an * dt * dt * P.var_r;
        P.var_r += noise.gyro * noise.gyro * dt;
        P.dt += dt;
    }
    P.dR = s.back().R;
    return P;
}

}  // 命名空间 sfm
