// 测试等距柱状 BA 接缝在代价、雅可比装配和求解中的行为。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "sfm/ba/CpuCamera.h"
#include "sfm/ba/Problem.h"
#include "sfm/ba/Solver.h"
#include "sfm/tests/TestMain.h"
#include "sfm/vk/EmbeddedSpirv.h"

namespace {

void project(const double pose[6], const double intr[2], const double X[3], double px[2]) {
    double p[3];
    bacpu::angleAxisRotate<double>(pose, X, p);
    for (int i = 0; i < 3; i++) p[i] += pose[3 + i];
    bacpu::EquirectModel::project<double>(intr, p, px);
}

BAProblem makeProblem() {
    constexpr uint32_t kImages = 2;
    constexpr uint32_t kPoints = 24;
    const double truth_poses[12] = {
        0.010, -0.018, 0.012, 0.000, 0.000, 0.000,
       -0.014,  0.021, 0.008, 0.120, -0.040, 0.060,
    };

    BAProblem P;
    P.num_images = kImages;
    P.num_points = kPoints;
    P.intr = {640.0, 480.0};
    P.groups = {{0, 12, 0, 9}};
    P.image_group.assign(kImages, 0);
    P.poses.assign(truth_poses, truth_poses + 12);
    P.points.resize(3 * kPoints);
    P.obs_ranges.assign(kPoints + 1, 0);

    for (uint32_t p = 0; p < kPoints; p++) {
        const double side = (p & 1) ? 1.0 : -1.0;
        const double theta = side * (M_PI - (2.0 + 0.2 * (p % 5)) * 2.0 * M_PI / P.intr[0]);
        const double phi = -0.22 + 0.04 * (p % 12);
        const double radius = 3.0 + 0.08 * p;
        double* X = &P.points[3 * p];
        X[0] = radius * std::cos(phi) * std::sin(theta);
        X[1] = -radius * std::sin(phi);
        X[2] = radius * std::cos(phi) * std::cos(theta);
        for (uint32_t image = 0; image < kImages; image++) {
            double px[2];
            project(&truth_poses[6 * image], P.intr.data(), X, px);
            P.obs_image.push_back(image);
            P.obs_point.push_back(p);
            P.obs_xy.push_back(px[0]);
            P.obs_xy.push_back(px[1]);
        }
        P.obs_ranges[p + 1] = (uint32_t)P.obs_image.size();
    }
    P.num_obs = (uint32_t)P.obs_image.size();
    P.pose_dim = 6 * kImages;
    P.total_intr = 2;
    P.free_intr = 0;
    P.n_dim = P.pose_dim;
    finalizeTables(P);

    P.poses[1] += 0.045;
    P.poses[7] -= 0.038;
    P.poses[3] += 0.025;
    P.poses[10] -= 0.020;
    for (uint32_t p = 0; p < kPoints; p++) {
        P.points[3 * p] += 0.006 * ((int)(p % 3) - 1);
        P.points[3 * p + 1] += 0.004 * ((int)(p % 5) - 2);
    }
    return P;
}

int seamCrossings(const BAProblem& P) {
    int count = 0;
    for (uint32_t o = 0; o < P.num_obs; o++) {
        const uint32_t image = P.obs_image[o];
        double px[2];
        project(&P.poses[6 * image], P.intr.data(), &P.points[3 * P.obs_point[o]], px);
        if (std::fabs(px[0] - P.obs_xy[2 * o]) > 0.5 * P.intr[0]) count++;
    }
    return count;
}

double relativeMax(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
    double diff = 0, scale = 1;
    for (size_t i = 0; i < a.size(); i++) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
        scale = std::max(scale, std::max(std::fabs(a[i]), std::fabs(b[i])));
    }
    return diff / scale;
}

struct Result {
    double initial = 0;
    double final = 0;
    std::vector<double> S, g, poses, points;
};

Result evaluate(BAProblem P, RealCfg real, const char* loss) {
    SolverOptions opt;
    opt.real = real;
    opt.loss = loss;
    opt.loss_param = 3.0f;
    opt.solver = SolverSel::Dense;
    opt.max_iters = 12;
    opt.init_damping = 1e-2;
    opt.rtol = 1e-10;
    opt.patience = 6;
    opt.verbose = false;

    BundleSolver solver(P, opt);
    solver.init();
    if (solver.real() != real) throw std::runtime_error("requested scalar mode unavailable");
    Result out;
    out.initial = solver.computeCost();
    solver.debugAssemble((float)opt.init_damping);
    out.S = solver.debugPackedS();
    out.g = solver.debugG();
    solver.solve();
    solver.downloadParams();
    out.final = solver.stats().final_cost;
    out.poses = P.poses;
    out.points = P.points;
    return out;
}

// 像素对规范变换不变，但位姿和点不变性不成立；未固定规范的两次求解可落在不同坐标系。
double reprojectionGap(const BAProblem& P, const Result& a, const Result& b) {
    double worst = 0;
    for (uint32_t o = 0; o < P.num_obs; o++) {
        const uint32_t image = P.obs_image[o], point = P.obs_point[o];
        double pa[2], pb[2];
        project(&a.poses[6 * image], P.intr.data(), &a.points[3 * point], pa);
        project(&b.poses[6 * image], P.intr.data(), &b.points[3 * point], pb);
        const double dx = std::remainder(pa[0] - pb[0], P.intr[0]);
        worst = std::max(worst, std::hypot(dx, pa[1] - pb[1]));
    }
    return worst;
}

bool variantBuilt(RealCfg real, const char* loss) {
    const std::string name = std::string("ba_") + realCfgName(real) + "_" + loss;
    size_t words = 0;
    return sfm::findSpirv(name.c_str(), &words) != nullptr;
}

bool testVariant(const BAProblem& base, RealCfg real, const char* loss) {
    if (!variantBuilt(real, loss)) {
        std::printf("%-8s %-8s SKIP (variant not built)\n", realCfgName(real), loss);
        return true;
    }
    const Result host = evaluate(base, RealCfg::CPU, loss);
    const Result device = evaluate(base, real, loss);
    const double cost_err = std::fabs(device.initial - host.initial) / std::max(1.0, host.initial);
    const double S_err = relativeMax(device.S, host.S);
    const double g_err = relativeMax(device.g, host.g);
    const double final_err = std::fabs(device.final - host.final) / std::max(1.0, host.final);
    const double reproj_px = reprojectionGap(base, device, host);
    // rAtan2 为近似，CPU/GPU 双精度投影一致性约为百万分之一。
    const double assembly_tol = real == RealCfg::F32 ? 2e-5 : 2e-6;
    const double solve_tol = real == RealCfg::F32 ? 1e-2 : 1e-5;
    const double reproj_tol_px = real == RealCfg::F32 ? 5e-3 : 5e-5;
    const bool descent = host.final < host.initial && device.final < device.initial;
    const bool ok = cost_err < assembly_tol && S_err < assembly_tol && g_err < assembly_tol &&
                    final_err < solve_tol && reproj_px < reproj_tol_px && descent;
    std::printf("%-8s %-8s cost %.2e  S %.2e  g %.2e  final %.2e  reproj %.2e px  %s\n",
                realCfgName(real), loss, cost_err, S_err, g_err, final_err, reproj_px,
                ok ? "PASS" : "FAIL");
    return ok;
}

int run(int, char**) {
    const BAProblem base = makeProblem();
    const int crossings = seamCrossings(base);
    if (crossings < 8) {
        std::fprintf(stderr, "FAIL: only %d raw residuals cross the seam\n", crossings);
        return 1;
    }
    std::printf("seam-crossing observations: %d/%u\n", crossings, base.num_obs);

    bool ok = true;
    for (const char* loss : {"trivial", "huber", "cauchy"})
        for (RealCfg real : {RealCfg::F32, RealCfg::F64})
            ok = testVariant(base, real, loss) && ok;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // 匿名命名空间

int main(int argc, char** argv) { return sfmTestMain(argc, argv, run); }
