#pragma once
// 相机模型共享六自由度位姿与分组内参，各组可选不同模型。
// 新增模型需实现 ICameraModel、添加 ba.slang 代价/雅可比入口并注册 Problem.h；参数数须满足自由度上限。

#include "linalg.slang"

// 内参顺序与 Camera.h 的 packIntrinsics 一致，将主点移到末尾；BA 仅优化组内自由前缀 u_group_info.z。
// 雅可比仍计算全部参数，但固定参数不占系统列，避免乘法阻尼无法修复全零列造成的奇异性（D49/D50）。
interface ICameraModel : IDifferentiable {
    static const int kNumIntr;
    static const bool kPeriodicX;
    [Differentiable] static This fromArray(Real v[kNumIntr]);
    // p 为相机坐标点，返回投影像素。
    [Differentiable] RVec2 project(RVec3 p);
};

[Differentiable]
RVec3 angleAxisRotate(RVec3 axis, RVec3 p) {
    Real theta = length3(axis);
    RVec3 a = axis * (Real(1.0) / (theta + Real(1e-15)));
    RVec2 sc = rsincos(theta);
    Real st = sc.x, ct = sc.y;
    return p * ct + cross3(a, p) * st + a * (dot3(a, p) * (Real(1.0) - ct));
}

// pose=[轴角(3),平移(3)]
[Differentiable]
RVec3 applyPose(Real pose[6], RVec3 p_world) {
    RVec3 aa = rvec3(pose[0], pose[1], pose[2]);
    RVec3 t = rvec3(pose[3], pose[4], pose[5]);
    return angleAxisRotate(aa, p_world) + t;
}

// BAL/Snavely 模型使用对数焦距和两个径向畸变项，按 BAL 约定沿 -z 投影。
struct SnavelyModel : ICameraModel {
    static const int kNumIntr = 3;
    static const bool kPeriodicX = false;
    Real f_log; Real k1; Real k2;

    [Differentiable] static SnavelyModel fromArray(Real v[3]) {
        SnavelyModel m; m.f_log = v[0]; m.k1 = v[1]; m.k2 = v[2]; return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        Real xp = -p.x / p.z;
        Real yp = -p.y / p.z;
        Real r2 = xp * xp + yp * yp;
        Real distortion = Real(1.0) + r2 * (k1 + r2 * k2);
        Real f = rexp(f_log) * distortion;
        return rvec2(f * xp, f * yp);
    }
};

// 直接参数化焦距的同类模型，与 Ceres 的 BAL 示例一致。
struct SnavelyFModel : ICameraModel {
    static const int kNumIntr = 3;
    static const bool kPeriodicX = false;
    Real f; Real k1; Real k2;

    [Differentiable] static SnavelyFModel fromArray(Real v[3]) {
        SnavelyFModel m; m.f = v[0]; m.k1 = v[1]; m.k2 = v[2]; return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        Real xp = -p.x / p.z;
        Real yp = -p.y / p.z;
        Real r2 = xp * xp + yp * yp;
        Real fd = f * (Real(1.0) + r2 * (k1 + r2 * k2));
        return rvec2(fd * xp, fd * yp);
    }
};

// 含主点和单径向系数的五参数针孔模型，沿 +z，pixel=f*distorted+c，示例说明不同内参数量无需修改求解器。
struct PinholeRadialModel : ICameraModel {
    static const int kNumIntr = 5;
    static const bool kPeriodicX = false;
    Real f; Real cx; Real cy; Real k1; Real k2;

    [Differentiable] static PinholeRadialModel fromArray(Real v[5]) {
        PinholeRadialModel m;
        m.f = v[0]; m.k1 = v[1]; m.k2 = v[2]; m.cx = v[3]; m.cy = v[4];
        return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        Real xp = p.x / p.z;
        Real yp = p.y / p.z;
        Real r2 = xp * xp + yp * yp;
        Real fd = f * (Real(1.0) + r2 * (k1 + r2 * k2));
        return rvec2(fd * xp + cx, fd * yp + cy);
    }
};

// COLMAP SIMPLE_PINHOLE：单焦距、主点，无畸变。
struct SimplePinholeModel : ICameraModel {
    static const int kNumIntr = 3;
    static const bool kPeriodicX = false;
    Real f; Real cx; Real cy;
    [Differentiable] static SimplePinholeModel fromArray(Real v[3]) {
        SimplePinholeModel m; m.f = v[0]; m.cx = v[1]; m.cy = v[2]; return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        return rvec2(f * (p.x / p.z) + cx, f * (p.y / p.z) + cy);
    }
};

// COLMAP PINHOLE：双焦距、主点，无畸变。
struct PinholeModel : ICameraModel {
    static const int kNumIntr = 4;
    static const bool kPeriodicX = false;
    Real fx; Real fy; Real cx; Real cy;
    [Differentiable] static PinholeModel fromArray(Real v[4]) {
        PinholeModel m; m.fx = v[0]; m.fy = v[1]; m.cx = v[2]; m.cy = v[3]; return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        return rvec2(fx * (p.x / p.z) + cx, fy * (p.y / p.z) + cy);
    }
};

// COLMAP OPENCV：双焦距、主点、两径向和两切向 Brown-Conrady 系数，沿 +z；主机对应 Camera.h 的 OpenCV。
struct OpenCVModel : ICameraModel {
    static const int kNumIntr = 8;
    static const bool kPeriodicX = false;
    Real fx; Real fy; Real cx; Real cy; Real k1; Real k2; Real p1; Real p2;

    [Differentiable] static OpenCVModel fromArray(Real v[8]) {
        OpenCVModel m;
        m.fx = v[0]; m.fy = v[1];
        m.k1 = v[2]; m.k2 = v[3]; m.p1 = v[4]; m.p2 = v[5];
        m.cx = v[6]; m.cy = v[7];
        return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        Real xp = p.x / p.z;
        Real yp = p.y / p.z;
        Real r2 = xp * xp + yp * yp;
        Real radial = Real(1.0) + r2 * (k1 + r2 * k2);
        Real dx = xp * radial + Real(2.0) * p1 * xp * yp + p2 * (r2 + Real(2.0) * xp * xp);
        Real dy = yp * radial + p1 * (r2 + Real(2.0) * yp * yp) + Real(2.0) * p2 * xp * yp;
        return rvec2(fx * dx + cx, fy * dy + cy);
    }
};

// OPENCV_FISHEYE 使用 Kannala-Brandt 的四项角度径向多项式，支持超过 90 度的视线，主机对应 OpenCVFisheye。
struct FisheyeModel : ICameraModel {
    static const int kNumIntr = 8;
    static const bool kPeriodicX = false;
    Real fx; Real fy; Real cx; Real cy; Real k1; Real k2; Real k3; Real k4;

    [Differentiable] static FisheyeModel fromArray(Real v[8]) {
        FisheyeModel m;
        m.fx = v[0]; m.fy = v[1];
        m.k1 = v[2]; m.k2 = v[3]; m.k3 = v[4]; m.k4 = v[5];
        m.cx = v[6]; m.cy = v[7];
        return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        // sqrt 内的 epsilon 保证轴线上 r 及 thd/r 导数有限。
        Real r = sqrt(p.x * p.x + p.y * p.y + Real(1e-24));
        Real theta = rAtan2(r, p.z);           // [0,pi]，轴线上平滑
        Real t2 = theta * theta;
        Real thd = theta * (Real(1.0) + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
        Real scale = thd / r;
        return rvec2(fx * scale * p.x + cx, fy * scale * p.y + cy);
    }
};

// FULL_OPENCV 的径向项为 (1+k1 r²+k2 r⁴+k3 r⁶)/(1+k4 r²+k5 r⁴+k6 r⁶)，另加两切向项。
struct FullOpenCVModel : ICameraModel {
    static const int kNumIntr = 12;
    static const bool kPeriodicX = false;
    Real fx; Real fy; Real cx; Real cy;
    Real k1; Real k2; Real p1; Real p2; Real k3; Real k4; Real k5; Real k6;

    [Differentiable] static FullOpenCVModel fromArray(Real v[12]) {
        FullOpenCVModel m;
        m.fx = v[0]; m.fy = v[1];
        m.k1 = v[2]; m.k2 = v[3]; m.p1 = v[4]; m.p2 = v[5];
        m.k3 = v[6]; m.k4 = v[7]; m.k5 = v[8]; m.k6 = v[9];
        m.cx = v[10]; m.cy = v[11];
        return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        Real xp = p.x / p.z;
        Real yp = p.y / p.z;
        Real r2 = xp * xp + yp * yp;
        Real radial = (Real(1.0) + r2 * (k1 + r2 * (k2 + r2 * k3))) /
                      (Real(1.0) + r2 * (k4 + r2 * (k5 + r2 * k6)));
        Real dx = xp * radial + Real(2.0) * p1 * xp * yp + p2 * (r2 + Real(2.0) * xp * xp);
        Real dy = yp * radial + p1 * (r2 + Real(2.0) * yp * yp) + Real(2.0) * p2 * xp * yp;
        return rvec2(fx * dx + cx, fy * dy + cy);
    }
};

// THIN_PRISM_FISHEYE 在等距坐标 (uf,vf)=theta*(x,y)/r 上组合 KB 径向、切向和薄棱镜项，支持超过 90 度。
struct ThinPrismFisheyeModel : ICameraModel {
    static const int kNumIntr = 12;
    static const bool kPeriodicX = false;
    Real fx; Real fy; Real cx; Real cy;
    Real k1; Real k2; Real p1; Real p2; Real k3; Real k4; Real sx1; Real sy1;

    [Differentiable] static ThinPrismFisheyeModel fromArray(Real v[12]) {
        ThinPrismFisheyeModel m;
        m.fx = v[0]; m.fy = v[1];
        m.k1 = v[2]; m.k2 = v[3]; m.p1 = v[4]; m.p2 = v[5];
        m.k3 = v[6]; m.k4 = v[7]; m.sx1 = v[8]; m.sy1 = v[9];
        m.cx = v[10]; m.cy = v[11];
        return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        Real r = sqrt(p.x * p.x + p.y * p.y + Real(1e-24));
        Real theta = rAtan2(r, p.z);
        Real s = theta / r;
        Real uf = s * p.x;
        Real vf = s * p.y;
        Real rf2 = theta * theta;
        Real radial = rf2 * (k1 + rf2 * (k2 + rf2 * (k3 + rf2 * k4)));
        Real du = uf * radial + Real(2.0) * p1 * uf * vf + p2 * (rf2 + Real(2.0) * uf * uf) + sx1 * rf2;
        Real dv = vf * radial + p1 * (rf2 + Real(2.0) * vf * vf) + Real(2.0) * p2 * uf * vf + sy1 * rf2;
        return rvec2(fx * (uf + du) + cx, fy * (vf + dv) + cy);
    }
};

// EQUIRECTANGULAR 将方位角映射到宽度、仰角映射到高度，覆盖所有方向；两个尺寸参数不参与 BA。
struct EquirectModel : ICameraModel {
    static const int kNumIntr = 2;
    static const bool kPeriodicX = true;
    Real w; Real h;

    [Differentiable] static EquirectModel fromArray(Real v[2]) {
        EquirectModel m; m.w = v[0]; m.h = v[1]; return m;
    }
    [Differentiable] RVec2 project(RVec3 p) {
        // theta 从 +z 朝 +x，phi 从赤道朝 -y；epsilon 保证极点与原点的梯度有限。
        Real horiz = sqrt(p.x * p.x + p.z * p.z + Real(1e-24));
        Real theta = rAtan2(p.x, p.z);
        Real phi = rAtan2(-p.y, horiz);
        const Real kInv2Pi = Real(0.15915494309189535);
        const Real kInvPi = Real(0.3183098861837907);
        return rvec2((theta * kInv2Pi + Real(0.5)) * w, (Real(0.5) - phi * kInvPi) * h);
    }
};

[Differentiable]
RVec2 projectResidual<M : ICameraModel>(Real intr[M.kNumIntr], RVec3 p, no_diff RVec2 obs) {
    M cam = M.fromArray(intr);
    RVec2 r = cam.project(p) - obs;
    if (M.kPeriodicX) {
        Real half = Real(0.5) * intr[0];
        if (r.x > half) r.x = r.x - intr[0];
        else if (r.x < -half) r.x = r.x + intr[0];
    }
    return r;
}

// 通用相机模型的重投影残差。
[Differentiable]
RVec2 reproject<M : ICameraModel>(
    Real pose[6], Real intr[M.kNumIntr], RVec3 X, no_diff RVec2 obs
) {
    return projectResidual<M>(intr, applyPose(pose, X), obs);
}

// rig 路径中 pose 为 rig_from_world，ext 为成员 cam_from_rig。
[Differentiable]
RVec2 reprojectRig<M : ICameraModel>(
    Real pose[6], Real ext[6], Real intr[M.kNumIntr], RVec3 X, no_diff RVec2 obs
) {
    return projectResidual<M>(intr, applyPose(ext, applyPose(pose, X)), obs);
}
