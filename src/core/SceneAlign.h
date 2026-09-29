#pragma once

// 从裸点数组估计地面、墙面和角点，以确定场景坐标系，无需窗口即可测试。
// 平面由 RANSAC 初始化并用内点最小二乘重拟合；点击处拟合在表面一致时向外扩展，更大地面范围可提高调平精度。

#include "core/Similarity.h"

#include <cstdint>
#include <vector>

namespace spirula {
namespace align {

// n . x + d = 0，n 为单位向量。
struct Plane {
    double n[3] = {0, 0, 1};
    double d = 0.0;
    int64_t inliers = 0;
    double distance(const double p[3]) const {
        return n[0]*p[0] + n[1]*p[1] + n[2]*p[2] + d;
    }
};

// 在 tol 内支持最多的平面；可选 inlier 输出逐点标记，没有三点支持时返回 false。
bool fit_plane(const double* pts, int64_t n, double tol, uint32_t seed,
               Plane& out, std::vector<uint8_t>* inlier = nullptr);

// 依次提取至多 k 个平面，每次移除内点；支持比例低于 min_frac 时结束。
std::vector<Plane> find_planes(const double* pts, int64_t n, double tol, int k,
                               double min_frac);

// 先拟合 at 周围 r0 内的表面，再持续将半径加倍，直到更大邻域不再与该平面一致。
bool fit_plane_at(const double* pts, int64_t n, const double at[3], double r0,
                  Plane& out);

// 求 at 附近至多三个相互垂直的表面并严格正交化；axes 各行为法向，corner 为交点，不足三个时使用投影后的点击点，返回平面数。
int fit_corner(const double* pts, int64_t n, const double at[3], double r0,
               double axes[9], double corner[3]);

// 将单位向量 a 旋至 b 的最短旋转，行主序。
void rotation_between(const double a[3], const double b[3], double R[9]);

struct AutoAlignOptions {
    double tol = 0.01;           // 平面厚度，单位与输入点一致
    bool yaw = true;             // 将墙面方向旋转到坐标轴
    bool centre = true;          // 将水平投影中心移到 x=y=0
};

struct AutoAlignResult {
    Sim3 T;
    bool ground = false, walls = false;
    double ground_share = 0.0;   // 距离地面不超过 tol 的点占比
    Plane plane;                 // 输入坐标系中的地面，法向朝上
};

// 各点到逐轴中位数中心的距离中位数的两倍；此尺度不易受漂浮点影响，可用于设置 AutoAlignOptions::tol。
double robust_extent(const double* pts, int64_t n);

// 地面置于 z=0、+Z 朝上，墙面与坐标轴对齐，水平投影居中；up 为空时采用 +Z。
// 可选 normals/weights 与点一一对应，提供后用法向估计墙面方向。
AutoAlignResult auto_align(const double* pts, int64_t n, const double* up,
                           const float* normals, const float* weights,
                           const AutoAlignOptions& opt);

}  // 命名空间 align
}  // 命名空间 spirula
