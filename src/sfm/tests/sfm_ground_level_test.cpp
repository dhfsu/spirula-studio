// 纯主机地面调平测试：相机倾斜的房间置于任意坐标规范，验证 groundTransform。
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "sfm/core/Model.h"
#include "sfm/core/Pose.h"
#include "sfm/map/Orient.h"
#include "sfm/tests/TestMain.h"

using namespace sfm;

// 相机位于 C，看向 dir，向上方向向 lean 倾斜。
static Pose lookAlong(const Vec3& C, const Vec3& dir, const Vec3& lean) {
    const Vec3 f = dir.normalized();
    const Vec3 down = (Vec3{0, 0, -1} + lean).normalized();
    const Vec3 r = down.cross(f).normalized();
    const Vec3 u = f.cross(r);
    const Mat3 R = {r.x, r.y, r.z, u.x, u.y, u.z, f.x, f.y, f.z};
    const Vec3 t = mul(R, C);
    return {R, {-t.x, -t.y, -t.z}};
}

// 6×4 地板、2.5 高四墙与桌子，相机高度 1.6，全部同向倾斜 9 度。
static Reconstruction room(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> g(0.0, 0.004);
    Reconstruction rec;
    uint64_t id = 1;
    auto add = [&](double x, double y, double z) {
        rec.points3D[id++].xyz = {x + g(rng), y + g(rng), z + g(rng)};
    };
    for (int i = 0; i < 6000; i++) add(6 * u(rng), 4 * u(rng), 0.0);
    for (int i = 0; i < 1500; i++) {
        add(0.0, 4 * u(rng), 2.5 * u(rng));
        add(6.0, 4 * u(rng), 2.5 * u(rng));
        add(6 * u(rng), 0.0, 2.5 * u(rng));
        add(6 * u(rng), 4.0, 2.5 * u(rng));
    }
    for (int i = 0; i < 800; i++) add(2.5 + u(rng), 1.5 + 0.8 * u(rng), 0.75);
    const Vec3 lean = {std::sin(9.0 * M_PI / 180.0), 0, 0};
    for (uint32_t k = 0; k < 40; k++) {
        const double a = 2.0 * M_PI * k / 40.0;
        const Vec3 C = {3.0 + 1.2 * std::cos(a), 2.0 + 0.8 * std::sin(a), 1.6};
        Image im;
        im.id = k;
        im.registered = true;
        im.pose = lookAlong(C, {std::cos(a), std::sin(a), -0.2}, lean);
        rec.images[k] = im;
    }
    return rec;
}

static double offIdentity(const Mat3& R) {
    double s = 0;
    for (int i = 0; i < 9; i++) s += std::fabs(R[(size_t)i] - (i % 4 == 0 ? 1.0 : 0.0));
    return s;
}

// 当前模型坐标中的地板平均高度及离散程度。
static void floorStats(const Reconstruction& rec, double& mean, double& spread) {
    double s = 0, s2 = 0;
    int n = 0;
    for (const auto& kv : rec.points3D) {
        if (kv.first > 6000) break;
        const double z = kv.second.xyz.z;
        s += z;
        s2 += z * z;
        n++;
    }
    mean = s / n;
    spread = std::sqrt(std::max(0.0, s2 / n - mean * mean));
}

int cmdGroundSelftest(int, char**) {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { printf("  FAIL: %s\n", what); fails++; }
        return ok;
    };
    std::mt19937 rng(5);
    Reconstruction rec = room(rng);
    Sim3 gauge;
    gauge.scale = 0.37;
    gauge.R = angleAxisToRotation(Vec3{0.4, -0.8, 0.3}.normalized() * 1.9);
    gauge.t = {5.0, -2.0, 7.0};
    applySim3(rec, gauge);

    // ---------------- T1：仅相机向上轴会保留持机倾斜 ----------------
    Reconstruction cams = rec;
    applySim3(cams, uprightTransform(cams));
    double mean = 0, spread = 0;
    floorStats(cams, mean, spread);
    printf("  T1: camera frame, floor spread %.4f\n", spread);
    check(spread > 0.05 * gauge.scale, "T1: cameras' up leaves the floor tilted");

    // ---------------- T2：地面调平到 z=0，尺度不变 ----------------
    const Sim3 T0 = uprightTransform(rec);
    const GroundFit g = groundTransform(rec, true, T0);
    check(g.found, "T2: a ground is found");
    applySim3(rec, composeSim3(g.T, T0));
    // 地面本身 4 mm 噪声换算到当前单位。
    const double noise = 0.004 * gauge.scale * T0.scale;
    floorStats(rec, mean, spread);
    printf("  T2: ground frame, floor at %.5f, spread %.5f, share %.2f\n", mean, spread,
           g.share);
    check(std::fabs(mean) < noise && spread < 1.1 * noise, "T2: floor at z = 0 and level");
    check(std::fabs(g.T.scale - 1.0) < 1e-12, "T2: scale untouched");
    double cz = 0;
    for (const auto& kv : rec.images)
        cz += (mul(transpose(kv.second.pose.R), kv.second.pose.t) * -1.0).z;
    cz /= (double)rec.images.size();
    check(cz > 0.0, "T2: cameras above the floor");

    // ---------------- T3：已有测量坐标只沿 Z 平移 ----------------
    Sim3 lift;
    lift.R = angleAxisToRotation(Vec3{0, 0, 1} * 0.6);
    lift.t = {0.3, -0.2, 0.45};
    applySim3(rec, lift);
    const GroundFit h = groundTransform(rec, false);
    check(h.found, "T3: a ground is found");
    check(offIdentity(h.T.R) < 1e-12 && h.T.t.x == 0.0 && h.T.t.y == 0.0,
          "T3: no turn and no sideways move");
    applySim3(rec, h.T);
    floorStats(rec, mean, spread);
    printf("  T3: after the height fix, floor at %.5f\n", mean);
    check(std::fabs(mean) < noise, "T3: floor back at z = 0");

    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

int main(int argc, char** argv) { return sfmTestMain(argc, argv, cmdGroundSelftest); }
