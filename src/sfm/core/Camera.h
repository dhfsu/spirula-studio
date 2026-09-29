// 相机内参与模型元数据的唯一主机定义；投影数学与设备 camera.slang 对应，未使用参数置零。
// 模型 ID、参数数量、CLI 名称及打包布局集中管理；新增模型需同步设备结构、入口和 BA 注册表。
// 等距柱状投影覆盖整球，标定仅由图像尺寸决定，没有焦距或可优化内参；BA 布局将主点移到末尾，磁盘写出时恢复 COLMAP 顺序。
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "sfm/geometry/LinAlg.h"

namespace sfm {

enum class CamModel {
    SimplePinhole, Pinhole, Radial, OpenCV, OpenCVFisheye, FullOpenCV, ThinPrismFisheye,
    Equirect
};

struct Camera {
    uint32_t id = 0;
    int width = 0, height = 0;
    CamModel model = CamModel::Radial;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double k1 = 0, k2 = 0;   // 径向畸变
    double p1 = 0, p2 = 0;   // 切向畸变，供 OpenCV、FullOpenCV 与 ThinPrismFisheye 使用
    double k3 = 0, k4 = 0;   // 附加径向项：鱼眼 k3/k4，FullOpenCV 的 k3 与分母 k4
    double k5 = 0, k6 = 0;   // FullOpenCV 有理分母系数，与 k4 配合
    double sx1 = 0, sy1 = 0; // 薄棱镜项，仅 ThinPrismFisheye 使用

    // 一个提取像素对应的原图像素数，等于源尺寸与工作尺寸之比，未缩放时为 1。
    // 验证、重投影、损失和合并阈值均以提取像素定义，再据此换算，使不同质量和分辨率下的定位噪声尺度一致（D47）。
    // 仅运行时使用，不写入 cameras.bin；读回模型后从特征恢复。
    double pixel_scale = 1.0;

    // 将提取像素阈值转换为当前相机像素或角度，统一通过这两个接口。
    double errPx(double px) const { return px * pixel_scale; }
    double errRad(double px) const { return px * pixel_scale / std::max(1e-6, focal()); }

    bool isFisheye() const {
        return model == CamModel::OpenCVFisheye || model == CamModel::ThinPrismFisheye;
    }
    // 球面投影覆盖 4*pi 方向，无焦距与畸变，对应 COLMAP 的 IsSpherical()。
    bool isSpherical() const { return model == CamModel::Equirect; }
    // 可看到超过 90 度方向的相机必须使用 bearing 单位视线，不能用 unproject 的 z=1 坐标（D49）。
    bool wideFov() const { return isFisheye() || isSpherical(); }

    // 缺少 EXIF 时，针孔焦距默认 1.2*max(w,h)；鱼眼按约 180 度对角视场采用 f=diag/pi，避免针孔猜测过大约四倍而引发焦距与畸变退化。
    // 等距柱状投影由宽度对应 2*pi、高度对应 pi 精确标定，忽略外部焦距。
    static Camera defaultFor(uint32_t id, int w, int h, double focal = 0,
                             CamModel m = CamModel::Radial) {
        Camera c;
        c.id = id;
        c.width = w;
        c.height = h;
        c.model = m;
        if (m == CamModel::Equirect) {
            c.fx = w / (2.0 * M_PI);   // 每弧度方位角的像素数
            c.fy = h / M_PI;           // 每弧度仰角的像素数
            c.cx = w * 0.5;
            c.cy = h * 0.5;
            return c;
        }
        if (focal > 0)
            c.fx = c.fy = focal;
        else if (c.isFisheye())
            c.fx = c.fy = std::sqrt((double)w * w + (double)h * h) / M_PI;
        else
            c.fx = c.fy = 1.2 * std::max(w, h);
        c.cx = w * 0.5;
        c.cy = h * 0.5;
        return c;
    }

    // 供误差缩放和焦距搜索使用的单一焦距；双焦距模型在 BA 前满足 fx==fy。等距柱状投影返回 w/2pi，即每弧度像素数。
    double focal() const { return 0.5 * (fx + fy); }
    void setFocal(double f) { fx = fy = f; }

    // 鱼眼投影与视线共用 Kannala-Brandt 多项式 theta_d(theta)、其导数及 Newton 反解。
    double kbThetaD(double th) const {
        double t2 = th * th;
        return th * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
    }
    double kbThetaFromThetaD(double thd) const {
        double th = thd;  // 小畸变初值采用 theta_d ~= theta
        for (int i = 0; i < 10; i++) {
            double t2 = th * th;
            double f = th * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - thd;
            double fp = 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)));
            double next = th - f / fp;
            // 更新不再改变 th 时精确停止，结果与固定十轮完全相同；通常三四轮即可，鱼眼过滤对每个观测都会执行该反解。
            if (next == th) return th;
            th = next;
        }
        return th;
    }

    // 相机三维点到像素：针孔族采用 Brown-Conrady，FullOpenCV 附有理分母；鱼眼采用 KB 角度径向多项式，薄棱镜模型在等距坐标加入切向与棱镜项。
    // 两类鱼眼均支持 p.z <= 0 的宽角视线。
    Vec2 project(const Vec3& p) const {
        if (model == CamModel::Equirect) {
            // 方位角从 +z 朝 +x，仰角从赤道朝 -y；等距柱状投影内参为 (w/2pi,h/pi,w/2,h/2)，适用于全部方向。
            double theta = std::atan2(p.x, p.z);
            double phi = std::atan2(-p.y, std::hypot(p.x, p.z));
            return {fx * theta + cx, cy - fy * phi};
        }
        if (model == CamModel::OpenCVFisheye) {
            double r = std::hypot(p.x, p.y);
            double theta = std::atan2(r, p.z);  // 角度范围 [0, pi]，轴线上平滑
            double thd = kbThetaD(theta);
            double scale = r > 1e-12 ? thd / r : 0.0;  // 轴线上投影到主点
            return {fx * scale * p.x + cx, fy * scale * p.y + cy};
        }
        if (model == CamModel::ThinPrismFisheye) {
            double r = std::hypot(p.x, p.y);
            double theta = std::atan2(r, p.z);
            double s = r > 1e-12 ? theta / r : 0.0;
            double uf = s * p.x, vf = s * p.y;   // 等距坐标，|uf,vf| = theta
            double rf2 = theta * theta;
            double radial = rf2 * (k1 + rf2 * (k2 + rf2 * (k3 + rf2 * k4)));
            double du = uf * radial + 2.0 * p1 * uf * vf + p2 * (rf2 + 2.0 * uf * uf) + sx1 * rf2;
            double dv = vf * radial + p1 * (rf2 + 2.0 * vf * vf) + 2.0 * p2 * uf * vf + sy1 * rf2;
            return {fx * (uf + du) + cx, fy * (vf + dv) + cy};
        }
        double xp = p.x / p.z, yp = p.y / p.z;
        double r2 = xp * xp + yp * yp;
        double radial = (model == CamModel::FullOpenCV) ?
            (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))) :
            (1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4))));
        double dx = xp * radial + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
        double dy = yp * radial + p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;
        return {fx * dx + cx, fy * dy + cy};
    }

    // 通过固定点迭代去畸变，得到 z=1 的归一化坐标；鱼眼仅在前半球有效，宽角必须改用 bearing。
    Vec2 unproject(const Vec2& px) const {
        if (wideFov()) {  // 宽角必须使用单位视线，z=1 坐标在超过 90 度时失效
            Vec3 b = bearing(px);
            return {b.x / b.z, b.y / b.z};
        }
        double u = (px.x - cx) / fx, v = (px.y - cy) / fy;
        if (k1 == 0 && k2 == 0 && k3 == 0 && k4 == 0 && k5 == 0 && k6 == 0 &&
            p1 == 0 && p2 == 0)
            return {u, v};
        // 五轮固定点迭代在常见畸变范围内收敛，并保持径向模型数值一致；只有强畸变出现未收敛时才需调整轮数。
        const bool rational = model == CamModel::FullOpenCV;
        double xu = u, yu = v;
        for (int i = 0; i < 5; i++) {
            double r2 = xu * xu + yu * yu;
            double radial = rational ?
                (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))) :
                (1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4))));
            double dtx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu);
            double dty = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu;
            xu = (u - dtx) / radial;
            yu = (v - dty) / radial;
        }
        return {xu, yu};
    }

    // 像素到相机坐标系单位视线，是几何核心的统一表示；针孔为前向归一化射线，鱼眼可指向侧方或后方，供 PnP、三角化与相对位姿共用。
    Vec3 bearing(const Vec2& px) const {
        if (model == CamModel::Equirect) {
            double theta = (px.x - cx) / fx;
            double phi = (cy - px.y) / fy;
            double cp = std::cos(phi);
            return {cp * std::sin(theta), -std::sin(phi), cp * std::cos(theta)};
        }
        if (model == CamModel::OpenCVFisheye) {
            double u = (px.x - cx) / fx, v = (px.y - cy) / fy;
            double rd = std::hypot(u, v);  // = theta_d
            if (rd < 1e-12) return {0, 0, 1};
            double theta = kbThetaFromThetaD(rd);
            double s = std::sin(theta);
            return {s * u / rd, s * v / rd, std::cos(theta)};  // 单位向量，超过 90 度时 z < 0
        }
        if (model == CamModel::ThinPrismFisheye) {
            double u = (px.x - cx) / fx, v = (px.y - cy) / fy;
            // 正向关系 u = theta_d*dir + tangential+prism，theta_d = theta*(1+k1 theta^2+...)。
            // 先剥离小切向/棱镜项，再用一维 Newton 反解主导 KB 径向项；朴素二维固定点法在约 85 度后会发散。
            double uf = 0, vf = 0, theta = 0, dtx = 0, dty = 0;
            for (int it = 0; it < 6; it++) {
                double u2 = u - dtx, v2 = v - dty;
                double rd = std::hypot(u2, v2);  // = theta_d
                if (rd < 1e-12) { theta = uf = vf = 0; break; }
                theta = kbThetaFromThetaD(rd);   // theta_d -> theta，一维反解
                uf = theta * u2 / rd;
                vf = theta * v2 / rd;
                double rf2 = theta * theta;
                double ndtx = 2.0 * p1 * uf * vf + p2 * (rf2 + 2.0 * uf * uf) + sx1 * rf2;
                double ndty = p1 * (rf2 + 2.0 * vf * vf) + 2.0 * p2 * uf * vf + sy1 * rf2;
                // 修正量不再变化时即可精确停止，与内层 Newton 的固定点判据相同。
                if (ndtx == dtx && ndty == dty) break;
                dtx = ndtx;
                dty = ndty;
            }
            if (theta < 1e-12) return {0, 0, 1};
            double s = std::sin(theta) / theta;
            return {s * uf, s * vf, std::cos(theta)};  // 单位向量，超过 90 度时 z < 0
        }
        Vec2 n = unproject(px);
        return Vec3{n.x, n.y, 1.0}.normalized();
    }

    // 忽略畸变的 3×3 内参矩阵，用于 E/F 转换。
    Mat3 K() const { return {fx, 0, cx, 0, fy, cy, 0, 0, 1}; }
};

// ---------------- 模型元数据的唯一来源 ----------------
// 新增模型同步 CamModel、本表、设备结构、ba.slang 入口及 Problem.h 注册项。
struct CamModelInfo {
    CamModel model;
    int colmap_id;         // COLMAP cameras.bin 的模型 ID
    int ba_model;          // Problem.h 中 kModels 的索引
    int ba_params;         // BA 的 packIntrinsics 参数数量
    int ba_focal;          // 开头的焦距参数数量
    int ba_pp;             // 末尾的主点参数数量
    bool ba_refinable;     // false 表示 BA 读取但不优化参数
    int colmap_params;     // 写入 cameras.bin 的 packColmap 参数数量
    const char* cli_name;  // --camera-model 的选项值
};

// ba_model 索引必须与 Problem.h 的 kModels 顺序一致；BA 采用（焦距，附加参数，主点），使自由参数始终为前缀（D50）。
static constexpr CamModelInfo kCamModelInfo[] = {
    {CamModel::SimplePinhole,    0, 4,  3, 1, 2, true,   3, "simple-pinhole"},
    {CamModel::Pinhole,          1, 5,  4, 2, 2, true,   4, "pinhole"},
    {CamModel::Radial,           3, 2,  5, 1, 2, true,   5, "radial"},
    {CamModel::OpenCV,           4, 3,  8, 2, 2, true,   8, "opencv"},
    {CamModel::OpenCVFisheye,    5, 6,  8, 2, 2, true,   8, "opencv-fisheye"},
    {CamModel::FullOpenCV,       6, 7, 12, 2, 2, true,  12, "full-opencv"},
    {CamModel::ThinPrismFisheye, 10, 8, 12, 2, 2, true,  12, "thin-prism-fisheye"},
    {CamModel::Equirect,         17, 9,  2, 2, 0, false,  2, "equirectangular"},
};

inline const CamModelInfo& camInfo(CamModel m) {
    for (const CamModelInfo& i : kCamModelInfo)
        if (i.model == m) return i;
    throw std::runtime_error("unknown camera model");
}
inline int camNumParams(CamModel m) { return camInfo(m).ba_params; }  // BA 参数数量
// 可优化前缀长度由焦距、附加参数和主点开关决定；固定畸变也固定其后主点（D50/D72）。
inline int camNumFreeParams(CamModel m, bool refine_pp = false, bool refine_extra = true) {
    const CamModelInfo& i = camInfo(m);
    if (!i.ba_refinable) return 0;
    if (!refine_extra) return i.ba_focal;
    return refine_pp ? i.ba_params : i.ba_params - i.ba_pp;
}
// 模型畸变系数位于 BA 布局的中间块。
inline int camNumExtraParams(CamModel m) {
    const CamModelInfo& i = camInfo(m);
    return i.ba_params - i.ba_focal - i.ba_pp;
}
inline int camColmapParams(CamModel m) { return camInfo(m).colmap_params; }
inline int camBaModel(CamModel m) { return camInfo(m).ba_model; }
inline int camColmapId(CamModel m) { return camInfo(m).colmap_id; }

// 将 COLMAP 模型 ID 映射到本项目；不支持的模型明确抛错，不能静默替换。
inline CamModel camFromColmapId(int id) {
    for (const CamModelInfo& i : kCamModelInfo)
        if (i.colmap_id == id) return i.model;
    throw std::runtime_error("unsupported COLMAP camera model id " + std::to_string(id));
}
inline bool parseCamModelName(const std::string& s, CamModel& out) {
    for (const CamModelInfo& i : kCamModelInfo)
        if (s == i.cli_name) { out = i.model; return true; }
    return false;
}

// 相机字段与 BA 平铺参数互换，d 需容纳 camNumParams 项；布局为焦距、附加参数、cx、cy，使固定主点可表达为仅优化前 n 列。
// Bundle.h 共用此打包，packColmap 再恢复磁盘格式顺序。
inline void packIntrinsics(const Camera& c, double* d) {
    switch (c.model) {
        case CamModel::SimplePinhole: d[0] = c.focal();
                                      d[1] = c.cx; d[2] = c.cy; break;
        case CamModel::Pinhole:       d[0] = c.fx; d[1] = c.fy;
                                      d[2] = c.cx; d[3] = c.cy; break;
        case CamModel::Radial:        d[0] = c.focal(); d[1] = c.k1; d[2] = c.k2;
                                      d[3] = c.cx; d[4] = c.cy; break;
        case CamModel::OpenCV:        d[0] = c.fx; d[1] = c.fy;
                                      d[2] = c.k1; d[3] = c.k2; d[4] = c.p1; d[5] = c.p2;
                                      d[6] = c.cx; d[7] = c.cy; break;
        case CamModel::OpenCVFisheye: d[0] = c.fx; d[1] = c.fy;
                                      d[2] = c.k1; d[3] = c.k2; d[4] = c.k3; d[5] = c.k4;
                                      d[6] = c.cx; d[7] = c.cy; break;
        case CamModel::FullOpenCV:    d[0] = c.fx; d[1] = c.fy;
                                      d[2] = c.k1; d[3] = c.k2; d[4] = c.p1; d[5] = c.p2;
                                      d[6] = c.k3; d[7] = c.k4; d[8] = c.k5; d[9] = c.k6;
                                      d[10] = c.cx; d[11] = c.cy; break;
        case CamModel::ThinPrismFisheye:
                                      d[0] = c.fx; d[1] = c.fy;
                                      d[2] = c.k1; d[3] = c.k2; d[4] = c.p1; d[5] = c.p2;
                                      d[6] = c.k3; d[7] = c.k4; d[8] = c.sx1; d[9] = c.sy1;
                                      d[10] = c.cx; d[11] = c.cy; break;
        // COLMAP 的 (w,h) 对应 fx=w/2pi、fy=h/pi；没有可固定或优化的主点。
        case CamModel::Equirect:      d[0] = 2.0 * M_PI * c.fx; d[1] = M_PI * c.fy; break;
    }
}
inline void unpackIntrinsics(Camera& c, const double* d) {  // c.model 由调用方设置
    c.k1 = c.k2 = c.p1 = c.p2 = c.k3 = c.k4 = c.k5 = c.k6 = c.sx1 = c.sy1 = 0;
    switch (c.model) {
        case CamModel::SimplePinhole: c.setFocal(d[0]);
                                      c.cx = d[1]; c.cy = d[2]; break;
        case CamModel::Pinhole:       c.fx = d[0]; c.fy = d[1];
                                      c.cx = d[2]; c.cy = d[3]; break;
        case CamModel::Radial:        c.setFocal(d[0]); c.k1 = d[1]; c.k2 = d[2];
                                      c.cx = d[3]; c.cy = d[4]; break;
        case CamModel::OpenCV:        c.fx = d[0]; c.fy = d[1];
                                      c.k1 = d[2]; c.k2 = d[3]; c.p1 = d[4]; c.p2 = d[5];
                                      c.cx = d[6]; c.cy = d[7]; break;
        case CamModel::OpenCVFisheye: c.fx = d[0]; c.fy = d[1];
                                      c.k1 = d[2]; c.k2 = d[3]; c.k3 = d[4]; c.k4 = d[5];
                                      c.cx = d[6]; c.cy = d[7]; break;
        case CamModel::FullOpenCV:    c.fx = d[0]; c.fy = d[1];
                                      c.k1 = d[2]; c.k2 = d[3]; c.p1 = d[4]; c.p2 = d[5];
                                      c.k3 = d[6]; c.k4 = d[7]; c.k5 = d[8]; c.k6 = d[9];
                                      c.cx = d[10]; c.cy = d[11]; break;
        case CamModel::ThinPrismFisheye:
                                      c.fx = d[0]; c.fy = d[1];
                                      c.k1 = d[2]; c.k2 = d[3]; c.p1 = d[4]; c.p2 = d[5];
                                      c.k3 = d[6]; c.k4 = d[7]; c.sx1 = d[8]; c.sy1 = d[9];
                                      c.cx = d[10]; c.cy = d[11]; break;
        case CamModel::Equirect:      c.fx = d[0] / (2.0 * M_PI); c.fy = d[1] / M_PI;
                                      c.cx = d[0] * 0.5; c.cy = d[1] * 0.5; break;
    }
}

// 按模型 BA 顺序设置畸变列表；缺项置零，多余项忽略。
inline void setExtraParams(Camera& c, const std::vector<double>& v) {
    const int n = camNumExtraParams(c.model);
    if (n <= 0) return;
    double d[12];
    packIntrinsics(c, d);
    const int off = camInfo(c.model).ba_focal;
    for (int i = 0; i < n; i++) d[off + i] = i < (int)v.size() ? v[i] : 0.0;
    unpackIntrinsics(c, d);
}

// 写为 COLMAP cameras.bin 布局：焦距、cx、cy、附加参数，共 camColmapParams 项。
inline void packColmap(const Camera& c, double* d) {
    switch (c.model) {
        case CamModel::SimplePinhole: d[0] = c.focal(); d[1] = c.cx; d[2] = c.cy; break;
        case CamModel::Pinhole:       d[0] = c.fx; d[1] = c.fy; d[2] = c.cx; d[3] = c.cy; break;
        case CamModel::Radial:        d[0] = c.focal(); d[1] = c.cx; d[2] = c.cy;
                                      d[3] = c.k1; d[4] = c.k2; break;
        case CamModel::OpenCV:        d[0] = c.fx; d[1] = c.fy; d[2] = c.cx; d[3] = c.cy;
                                      d[4] = c.k1; d[5] = c.k2; d[6] = c.p1; d[7] = c.p2; break;
        case CamModel::OpenCVFisheye: d[0] = c.fx; d[1] = c.fy; d[2] = c.cx; d[3] = c.cy;
                                      d[4] = c.k1; d[5] = c.k2; d[6] = c.k3; d[7] = c.k4; break;
        case CamModel::FullOpenCV:    d[0] = c.fx; d[1] = c.fy; d[2] = c.cx; d[3] = c.cy;
                                      d[4] = c.k1; d[5] = c.k2; d[6] = c.p1; d[7] = c.p2;
                                      d[8] = c.k3; d[9] = c.k4; d[10] = c.k5; d[11] = c.k6; break;
        case CamModel::ThinPrismFisheye:
                                      d[0] = c.fx; d[1] = c.fy; d[2] = c.cx; d[3] = c.cy;
                                      d[4] = c.k1; d[5] = c.k2; d[6] = c.p1; d[7] = c.p2;
                                      d[8] = c.k3; d[9] = c.k4; d[10] = c.sx1; d[11] = c.sy1; break;
        // 不能用 2*pi*fx 反算宽度，浮点往返可能偏离整数一个 ULP。
        case CamModel::Equirect:      d[0] = c.width; d[1] = c.height; break;
    }
}
// 从 COLMAP 布局恢复相机字段。
inline void unpackColmap(Camera& c, const double* d) {
    c.k1 = c.k2 = c.p1 = c.p2 = c.k3 = c.k4 = c.k5 = c.k6 = c.sx1 = c.sy1 = 0;
    switch (c.model) {
        case CamModel::SimplePinhole: c.setFocal(d[0]); c.cx = d[1]; c.cy = d[2]; break;
        case CamModel::Pinhole:       c.fx = d[0]; c.fy = d[1]; c.cx = d[2]; c.cy = d[3]; break;
        case CamModel::Radial:        c.setFocal(d[0]); c.cx = d[1]; c.cy = d[2];
                                      c.k1 = d[3]; c.k2 = d[4]; break;
        case CamModel::OpenCV:        c.fx = d[0]; c.fy = d[1]; c.cx = d[2]; c.cy = d[3];
                                      c.k1 = d[4]; c.k2 = d[5]; c.p1 = d[6]; c.p2 = d[7]; break;
        case CamModel::OpenCVFisheye: c.fx = d[0]; c.fy = d[1]; c.cx = d[2]; c.cy = d[3];
                                      c.k1 = d[4]; c.k2 = d[5]; c.k3 = d[6]; c.k4 = d[7]; break;
        case CamModel::FullOpenCV:    c.fx = d[0]; c.fy = d[1]; c.cx = d[2]; c.cy = d[3];
                                      c.k1 = d[4]; c.k2 = d[5]; c.p1 = d[6]; c.p2 = d[7];
                                      c.k3 = d[8]; c.k4 = d[9]; c.k5 = d[10]; c.k6 = d[11]; break;
        case CamModel::ThinPrismFisheye:
                                      c.fx = d[0]; c.fy = d[1]; c.cx = d[2]; c.cy = d[3];
                                      c.k1 = d[4]; c.k2 = d[5]; c.p1 = d[6]; c.p2 = d[7];
                                      c.k3 = d[8]; c.k4 = d[9]; c.sx1 = d[10]; c.sy1 = d[11]; break;
        case CamModel::Equirect:      c.fx = d[0] / (2.0 * M_PI); c.fy = d[1] / M_PI;
                                      c.cx = d[0] * 0.5; c.cy = d[1] * 0.5; break;
    }
}

}  // 命名空间 sfm
