// 纯主机公制规范测试，覆盖参考相机位置的 Sim(3)、不确定度与拒绝条件。
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "sfm/core/Camera.h"
#include "sfm/core/Model.h"
#include "sfm/map/Orient.h"
#include "sfm/core/Pose.h"
#include "sfm/map/MetricGauge.h"
#include "sfm/tests/TestMain.h"

using namespace sfm;

// 相机位于 C 看向 target，y 向上，场景辅助函数局部定义。
static Pose lookAt(const Vec3& C, const Vec3& target) {
    Vec3 f = (target - C).normalized();
    Vec3 up0 = {0, 1, 0};
    Vec3 r = up0.cross(f).normalized();
    Vec3 u = f.cross(r);
    Mat3 R = {r.x, r.y, r.z, u.x, u.y, u.z, f.x, f.y, f.z};
    Vec3 t = mul(R, C);
    return {R, {-t.x, -t.y, -t.z}};
}

// 半径 3 的圆弧叠加高度变化，保证中心覆盖三轴，平面与共线另设测试。
static std::vector<Vec3> arcCentres(int n) {
    std::vector<Vec3> c(n);
    for (int i = 0; i < n; i++) {
        const double a = 2.4 * M_PI * i / n;
        c[i] = {3.0 * std::cos(a), 0.8 * std::sin(3.0 * a), 3.0 * std::sin(a)};
    }
    return c;
}

static double chordal(const Mat3& A, const Mat3& B) {
    double s = 0;
    for (int i = 0; i < 9; i++) s += (A[i] - B[i]) * (A[i] - B[i]);
    return std::sqrt(s);
}

static Mat3 rotFromAxisAngle(const Vec3& axis, double ang) {
    return angleAxisToRotation(axis.normalized() * ang);
}

// 已知 Sim3 作用于中心后的参考位置。
static MetricRef makeRef(const std::vector<Vec3>& centres, const Sim3& T) {
    MetricRef ref;
    ref.centres = centres;
    ref.targets.reserve(centres.size());
    for (const Vec3& c : centres) ref.targets.push_back(transformPoint(T, c));
    return ref;
}

int cmdMetricSelftest(int, char**) {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { printf("  FAIL: %s\n", what); fails++; }
        return ok;
    };

    // ---------------- 两种非单位尺度的精确恢复 ----------------
    // 避免单位尺度掩盖错误方差归一化或多除三的问题。
    for (double s_true : {0.0731, 12.4}) {
        Sim3 T;
        T.scale = s_true;
        T.R = rotFromAxisAngle({0.3, -0.7, 0.5}, 0.9);
        T.t = {11.0, -4.0, 2.5};
        MetricRef ref = makeRef(arcCentres(24), T);
        MetricFit fit = fitMetricGauge(ref, 0.05);
        char what[96];
        snprintf(what, sizeof what, "T1 s=%g: ok", s_true);
        check(fit.ok, what);
        snprintf(what, sizeof what, "T1 s=%g: scale", s_true);
        check(std::fabs(fit.T.scale / s_true - 1.0) <= 1e-9, what);
        snprintf(what, sizeof what, "T1 s=%g: rotation", s_true);
        check(chordal(fit.T.R, T.R) <= 1e-9, what);
        snprintf(what, sizeof what, "T1 s=%g: rms", s_true);
        check(fit.rms <= 1e-9, what);
        snprintf(what, sizeof what, "T1 s=%g: all inliers", s_true);
        check(fit.inliers == fit.n && fit.n == 24, what);
    }

    // ---------------- 左手参考须被拒绝 ----------------
    // 镜像可产生 det=-1 且误差仅 3e-15 m，RMS 门限无法发现；须使用非共面样本。
    {
        int proper = 0, fitted = 0;
        double worst_rms = 0;
        for (int trial = 0; trial < 200; trial++) {
            std::vector<Vec3> src(24);
            for (int i = 0; i < 24; i++) {
                const double a = 2.4 * M_PI * i / 24;
                src[i] = {3.0 * std::cos(a), 0.02 * std::sin(3.0 * a), 3.0 * std::sin(a)};
            }
            Sim3 T;
            T.scale = 2.5;
            T.R = rotFromAxisAngle({0.3, -0.7, 0.5}, 0.9 + 0.005 * trial);
            T.t = {4.0, 1.0, -2.0};
            MetricRef ref;
            ref.centres = src;
            for (const Vec3& c : src)
                ref.targets.push_back(transformPoint(T, {c.x, -c.y, c.z}));
            MetricFit fit = fitMetricGauge(ref, 0.5);
            if (fit.ok) {
                fitted++;
                worst_rms = std::max(worst_rms, fit.rms);
                if (det3(fit.T.R) > 0.999999 && det3(fit.T.R) < 1.000001) proper++;
            }
        }
        printf("  T1b: fitted %d/200, worst rms %.4f m\n", fitted, worst_rms);
        check(fitted == 200, "T1b mirrored reference: every trial fitted");
        check(proper == 200, "T1b mirrored reference: det(R) = +1 on 200/200");
    }

    // ---------------- 大离群点与近门限内点 ----------------
    {
        Sim3 T;
        T.scale = 12.4;
        T.R = rotFromAxisAngle({0.2, 0.4, -0.9}, 1.7);
        T.t = {-3.0, 8.0, 1.0};
        // 阈值大于 3 m，使误把距离与平方阈值比较会明显放宽接受；3 倍偏移也不能靠整体平移同时捕获。
        const double max_err = 4.0;
        std::vector<Vec3> centres = arcCentres(40);
        MetricRef ref = makeRef(centres, T);
        std::vector<char> want(40, 1);
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> dir(-1.0, 1.0);
        for (int i = 0; i < 40; i++) {
            Vec3 d = Vec3{dir(rng), dir(rng), dir(rng)}.normalized();
            if (i % 10 == 3 || i % 10 == 7) {         // 20% 点位于三倍阈值外
                ref.targets[i] = ref.targets[i] + d * (3.0 * max_err);
                want[i] = 0;
            } else if (i % 10 == 5) {                 // 10% 点位于 0.3 倍阈值内
                ref.targets[i] = ref.targets[i] + d * (0.3 * max_err);
            }
        }
        MetricFit fit = fitMetricGauge(ref, max_err);
        check(fit.ok, "T2: ok");
        bool mask_exact = fit.inlier_mask.size() == want.size();
        for (size_t i = 0; mask_exact && i < want.size(); i++)
            mask_exact = (fit.inlier_mask[i] != 0) == (want[i] != 0);
        check(mask_exact, "T2: inlier mask equals the constructed set exactly");
        check(fit.inliers == 32, "T2: 32 inliers");
        {
            int wrong_in = 0, wrong_out = 0;
            for (size_t i = 0; i < want.size(); i++) {
                if (want[i] && !fit.inlier_mask[i]) wrong_out++;
                if (!want[i] && fit.inlier_mask[i]) wrong_in++;
            }
            printf("  T2: inliers %d (want 32), kept-outliers %d, dropped-inliers %d\n",
                   fit.inliers, wrong_in, wrong_out);
        }
        // 保留的扰动内点会使重拟合偏离真值有限距离，不能要求无噪声精度。
        printf("  T2: scale rel err %.3e\n", std::fabs(fit.T.scale / T.scale - 1.0));
        check(std::fabs(fit.T.scale / T.scale - 1.0) <= 5e-3, "T2: scale within 0.5 %");
    }

    // ---------------- 同几何三百次独立噪声试验 ----------------
    // 比较实际尺度与最弱轴旋转的统计离散和预测不确定度。
    {
        const int trials = 300, n = 50;
        const double sigma = 0.05, s_true = 0.7;
        Sim3 T;
        T.scale = s_true;
        T.R = rotFromAxisAngle({0.5, 0.2, 0.84}, 0.6);
        T.t = {2.0, -1.0, 0.5};
        std::vector<Vec3> centres = arcCentres(n);
        // 最弱旋转约束轴为横向跨度最小者，对应中心协方差最大特征值方向。
        Vec3 cbar{0, 0, 0};
        for (const Vec3& c : centres) cbar = cbar + c;
        cbar = cbar * (1.0 / n);
        std::vector<double> C(9, 0.0), lam, V;
        for (const Vec3& c : centres) {
            const Vec3 a = c - cbar;
            const double v[3] = {a.x, a.y, a.z};
            for (int r = 0; r < 3; r++)
                for (int q = 0; q < 3; q++) C[3 * r + q] += v[r] * v[q] / n;
        }
        jacobiEigenSymmetric(C, 3, lam, V);
        int kmax = 0;
        for (int k = 1; k < 3; k++)
            if (lam[k] > lam[kmax]) kmax = k;
        const Vec3 axis = mul(T.R, Vec3{V[kmax], V[3 + kmax], V[6 + kmax]});

        std::mt19937 rng(3);
        std::normal_distribution<double> nz(0.0, sigma);
        double sum_rel = 0, sum_rel2 = 0, sum_ax2 = 0;
        double pred_scale = 0, pred_rot = 0;
        int ok_count = 0;
        for (int k = 0; k < trials; k++) {
            MetricRef ref = makeRef(centres, T);
            for (Vec3& p : ref.targets) p = p + Vec3{nz(rng), nz(rng), nz(rng)};
            MetricFit fit = fitMetricGauge(ref, 1.0);   // 十一倍 sigma 门限，不拒绝任何点
            if (!fit.ok) continue;
            ok_count++;
            const double rel = fit.T.scale / s_true - 1.0;
            sum_rel += rel;
            sum_rel2 += rel * rel;
            const Vec3 dth = rotationToAngleAxis(mul(fit.T.R, transpose(T.R)));
            const double about = dth.dot(axis);
            sum_ax2 += about * about;
            pred_scale += fit.scale_unc / 100.0;
            pred_rot += fit.rot_unc_deg * M_PI / 180.0;
        }
        check(ok_count == trials, "T3: every trial fitted");
        const double emp_scale =
            std::sqrt(sum_rel2 / ok_count - (sum_rel / ok_count) * (sum_rel / ok_count));
        const double emp_rot = std::sqrt(sum_ax2 / ok_count);
        pred_scale /= ok_count;
        pred_rot /= ok_count;
        const double r_s = emp_scale / pred_scale, r_r = emp_rot / pred_rot;
        printf("  T3: scale emp/pred = %.4f (emp %.3e pred %.3e), "
               "rot emp/pred = %.4f (emp %.3e pred %.3e rad)\n",
               r_s, emp_scale, pred_scale, r_r, emp_rot, pred_rot);
        check(r_s >= 0.75 && r_s <= 1.33, "T3: scale uncertainty predicts the spread");
        check(r_r >= 0.75 && r_r <= 1.33, "T3: rotation uncertainty predicts the spread");
    }

    // ---------------- 按具体原因验证每种拒绝 ----------------
    {
        Sim3 T;
        T.scale = 4.0;
        T.R = mat3Identity();
        T.t = {0, 0, 0};
        for (int n : {0, 1, 2}) {
            MetricRef ref = makeRef(arcCentres(std::max(n, 1)), T);
            ref.centres.resize(n);
            ref.targets.resize(n);
            MetricFit fit = fitMetricGauge(ref, 0.5);
            check(!fit.ok && fit.reason == MetricFail::Pairs, "T4: n < 3 -> Pairs");
        }
        {   // 全部参考位置相同，无法确定尺度
            MetricRef ref = makeRef(arcCentres(20), T);
            for (Vec3& p : ref.targets) p = ref.targets[0];
            MetricFit fit = fitMetricGauge(ref, 0.5);
            check(!fit.ok && fit.reason == MetricFail::Spread, "T4: no spread -> Spread");
        }
        {   // 相机中心严格共线，绕线旋转自由
            MetricRef ref;
            for (int i = 0; i < 20; i++) {
                ref.centres.push_back({0.4 * i, 0, 0});
                ref.targets.push_back({1.6 * i, 0, 0});
            }
            MetricFit fit = fitMetricGauge(ref, 0.5);
            check(!fit.ok && fit.reason == MetricFail::Collinear, "T4: collinear -> Collinear");
        }
        {   // 全部参考误差远超 max_error
            MetricRef ref = makeRef(arcCentres(20), T);
            std::mt19937 rng(9);
            std::uniform_real_distribution<double> big(-500.0, 500.0);
            for (Vec3& p : ref.targets) p = {big(rng), big(rng), big(rng)};
            MetricFit fit = fitMetricGauge(ref, 0.05);
            check(!fit.ok && fit.reason == MetricFail::Inliers, "T4: all outliers -> Inliers");
            check(fit.inliers < (fit.n + 1) / 2, "T4: all outliers -> under half are inliers");
        }
        {   // 24 图中仅八图精确符合 Sim3，即使空间分散、预测不确定度很好，仍须因少数共识拒绝。
            Sim3 T2;
            T2.scale = 2.0;
            T2.R = rotFromAxisAngle({0.1, 0.2, 0.97}, 1.3);
            T2.t = {6.0, -2.0, 3.0};
            MetricRef ref = makeRef(arcCentres(24), T2);
            std::mt19937 rng(31);
            std::uniform_real_distribution<double> big(-300.0, 300.0);
            for (int i = 8; i < 24; i++) ref.targets[i] = {big(rng), big(rng), big(rng)};
            MetricFit fit = fitMetricGauge(ref, 0.2);
            printf("  T4: minority consensus %d/%d, scale unc %.4f %%, rot unc %.4f deg\n",
                   fit.inliers, fit.n, fit.scale_unc, fit.rot_unc_deg);
            check(fit.inliers == 8, "T4: the minority consensus is found");
            check(!fit.ok && fit.reason == MetricFail::Inliers,
                  "T4: a well-conditioned minority is still refused");
        }
    }

    // ---------------- 应用 Sim3 后中心对齐且重投影不变 ----------------
    {
        Sim3 T;
        T.scale = 0.0731;
        T.R = rotFromAxisAngle({0.6, -0.3, 0.74}, 1.1);
        T.t = {40.0, -12.0, 7.0};
        const int M = 18, N = 200;
        const int W = 1280, H = 960;
        Camera K = Camera::defaultFor(1, W, H, 1200);
        Reconstruction rec;
        rec.cameras[1] = K;
        std::mt19937 rng(13);
        std::uniform_real_distribution<double> ub(-2.0, 2.0);
        std::vector<Vec3> pts(N);
        for (Vec3& p : pts) p = {ub(rng), ub(rng), ub(rng)};
        std::vector<Vec3> centres = arcCentres(M);
        for (int i = 0; i < M; i++) {
            Image im;
            im.id = (uint32_t)(i + 1);
            im.camera_id = 1;
            im.name = "img" + std::to_string(i) + ".jpg";
            im.pose = lookAt(centres[i], {0, 0, 0});
            im.registered = true;
            im.points2D.resize(N);
            im.point3D_ids.assign(N, kInvalidPoint3D);
            rec.images[im.id] = im;
        }
        auto px = [&](uint32_t img, const Vec3& X) {
            const Pose& pose = rec.images.at(img).pose;
            return rec.cameras[1].project(mul(pose.R, X) + pose.t);
        };
        for (int j = 0; j < N; j++) {
            std::vector<TrackElement> track;
            for (int i = 0; i < M; i++) {
                rec.images[(uint32_t)(i + 1)].points2D[j] = px((uint32_t)(i + 1), pts[j]);
                track.push_back({(uint32_t)(i + 1), (uint32_t)j});
            }
            rec.addPoint3D(pts[j], track);
        }
        // 保存变换前投影，实际测量不变性。
        std::vector<Vec2> before;
        for (const auto& kv : rec.points3D)
            for (const TrackElement& e : kv.second.track)
                before.push_back(px(e.image_id, kv.second.xyz));

        MetricRef ref;
        for (int i = 0; i < M; i++) {
            ref.centres.push_back(centres[i]);
            ref.targets.push_back(transformPoint(T, centres[i]));
            ref.image_ids.push_back((uint32_t)(i + 1));
        }
        MetricFit fit = fitMetricGauge(ref, 0.05);
        check(fit.ok, "T8: ok");
        applySim3(rec, fit.T);
        double worst_c = 0;
        for (int i = 0; i < M; i++) {
            Vec3 c = cameraCenter(rec.images.at((uint32_t)(i + 1)).pose);
            worst_c = std::max(worst_c, (c - ref.targets[i]).norm());
        }
        check(worst_c <= 1e-9, "T8: every centre lands on its target");
        size_t k = 0;
        double worst_px = 0;
        for (const auto& kv : rec.points3D)
            for (const TrackElement& e : kv.second.track) {
                const Vec2 uv = px(e.image_id, kv.second.xyz);
                worst_px = std::max(worst_px, std::max(std::fabs(uv.x - before[k].x),
                                                       std::fabs(uv.y - before[k].y)));
                k++;
            }
        check(k == before.size() && k == (size_t)M * N,
              "T8: the same observations project after the transform");
        printf("  T8: worst centre %.3e m, worst reprojection %.3e px\n", worst_c, worst_px);
        // 使用与合并测试相同的 1e-6 px 容差；米制原点移位带来的消减使数值底约 1e-9 px。
        check(worst_px <= 1e-6, "T8: reprojection unchanged");
    }

    // ---------------- 共线门限独立于噪声 ----------------
    // 相同细长轨迹在相差 160 倍的噪声下均应拒绝，不能只靠预测角度不确定度。
    {
        Sim3 T;
        T.scale = 1.7;
        T.R = rotFromAxisAngle({0.0, 1.0, 0.0}, 0.35);
        T.t = {5.0, 5.0, 5.0};
        std::vector<Vec3> thin(60);
        for (int i = 0; i < 60; i++)
            thin[i] = {0.5 * i, 0.0009 * i * (i % 2 ? 1 : -1), 0.0007 * ((i / 3) % 5)};
        std::mt19937 rng(17);
        auto run = [&](const std::vector<Vec3>& c, double sigma) {
            std::normal_distribution<double> nz(0.0, sigma);
            MetricRef ref = makeRef(c, T);
            for (Vec3& p : ref.targets) p = p + Vec3{nz(rng), nz(rng), nz(rng)};
            return fitMetricGauge(ref, 10.0);
        };
        MetricFit loud = run(thin, 0.08);
        MetricFit quiet = run(thin, 0.0005);
        printf("  T9: thin arc perp frac %.5f (loud) / %.5f (quiet); rot unc %.3f / %.4f deg\n",
               loud.perp_frac, quiet.perp_frac, loud.rot_unc_deg, quiet.rot_unc_deg);
        check(!loud.ok && loud.reason == MetricFail::Collinear, "T9: thin arc -> Collinear");
        check(!quiet.ok && quiet.reason == MetricFail::Collinear,
              "T9: and 160x quieter is still Collinear -- geometry, not noise");
        check(quiet.rot_unc_deg < 5.0 && loud.rot_unc_deg > 5.0,
              "T9: the reported uncertainty WOULD have split them, which is why it cannot gate");
        check(loud.perp_frac < 0.05 && quiet.perp_frac < 0.05, "T9: both below the floor");
        // 相同六十相机增加横向跨度后应通过。
        std::vector<Vec3> fat(60);
        for (int i = 0; i < 60; i++)
            fat[i] = {0.5 * i, 4.0 * std::sin(0.3 * i), 3.0 * std::cos(0.21 * i)};
        MetricFit ok = run(fat, 0.08);
        printf("  T9: spread arc perp frac %.4f\n", ok.perp_frac);
        check(ok.ok, "T9: the same cameras spread off the axis pass");
    }

    // ---------------- 重复输入须逐位一致 ----------------
    // 两组各三十图的精确共识在 MSAC 上平局，固定随机序列必须稳定选择。
    {
        Sim3 A, B;
        A.scale = 3.3;
        A.R = rotFromAxisAngle({0.3, 0.3, 0.9}, 2.0);
        A.t = {-7.0, 1.0, 4.0};
        B.scale = 1.7;
        B.R = rotFromAxisAngle({0.9, -0.2, 0.1}, 1.1);
        B.t = {50.0, -30.0, 12.0};
        std::vector<Vec3> centres = arcCentres(60);
        MetricRef ref;
        ref.centres = centres;
        for (int i = 0; i < 60; i++)
            ref.targets.push_back(transformPoint(i < 30 ? A : B, centres[i]));
        // 重复八次，避免仅两次偶然选择相同共识而误判确定性。
        MetricFit a = fitMetricGauge(ref, 0.1);
        bool same = true;
        for (int rep = 0; rep < 7; rep++) {
            MetricFit b = fitMetricGauge(ref, 0.1);
            same = same && a.ok == b.ok && a.reason == b.reason && a.T.scale == b.T.scale &&
                   a.T.t.x == b.T.t.x && a.T.t.y == b.T.t.y && a.T.t.z == b.T.t.z &&
                   a.inliers == b.inliers && a.inlier_mask == b.inlier_mask &&
                   a.rms == b.rms && a.scale_unc == b.scale_unc && a.perp_frac == b.perp_frac;
            for (int i = 0; i < 9; i++) same = same && a.T.R[i] == b.T.R[i];
        }
        printf("  T10: locked onto scale %.4f with %d/%d inliers\n", a.T.scale, a.inliers, a.n);
        check(same, "T10: eight runs agree bit for bit");
        check(a.inliers == 30, "T10: exactly one of the two consensus sets is found");
        const bool bimodal = std::fabs(a.T.scale / A.scale - 1.0) < 1e-9 ||
                             std::fabs(a.T.scale / B.scale - 1.0) < 1e-9;
        check(bimodal, "T10: the fixture really has two answers to choose between");
    }

    // ---------------- 参考位置与图像名配对 ----------------
    // 覆盖不同目录同名、无扩展名和模型缺失图像。
    {
        namespace fs = std::filesystem;
        const fs::path dir = fs::temp_directory_path() / "sfm_metric_t5";
        fs::remove_all(dir);
        fs::create_directories(dir);
        const fs::path pf = dir / "positions.txt";
        {
            std::ofstream f(pf.string());
            f << "# a comment, and the blank line below\n\n";
            f << "cam1/a.jpg 1 2 3\n";
            f << "cam2/a.jpg 4 5 6\n";
            f << "b 7 8 9\n";              // 按主干匹配 b.png
            f << "zzz.jpg 10 11 12\n";     // 模型中不存在
            f << "e.jpg 20 21 22\n";       // 模型存在但未配准
            f << "v1 99 99 99\n";          // 文件夹中的点号不属于扩展名
        }
        std::map<std::string, Vec3> pos;
        std::string err;
        check(readMetricPositions(pf.string(), pos, err), "T5: the file parses");
        check(pos.size() == 6, "T5: six entries read");

        Reconstruction rec;
        const char* names[6] = {"cam1/a.jpg", "cam2/a.jpg", "b.png", "d.jpg", "e.jpg",
                                "v1.0/frame"};
        for (int i = 0; i < 6; i++) {
            Image im;
            im.id = (uint32_t)(i + 1);
            im.name = names[i];
            im.registered = i != 4;  // e.jpg 在模型中但未求解位姿
            im.pose = {mat3Identity(), {(double)-i, 0, 0}};
            rec.images[im.id] = im;
        }
        MetricRef ref;
        MetricPairCounts pc = pairMetricRef(rec, pos, ref);
        printf("  T5: matched %d, file-only %d, model-only %d\n",
               pc.matched, pc.unmatched_file, pc.unmatched_model);
        check(pc.matched == 3 && (int)ref.targets.size() == 3, "T5: three matched");
        check(pc.unmatched_file == 3, "T5: an unregistered image leaves its entry unused");
        check(pc.unmatched_model == 2, "T5: a dotted folder is not an extension");
        std::map<uint32_t, Vec3> got;
        for (size_t i = 0; i < ref.image_ids.size(); i++) got[ref.image_ids[i]] = ref.targets[i];
        check(got.count(1) && got.count(2) && got.count(3), "T5: the expected three images");
        check(got.count(1) && got[1].x == 1 && got[1].y == 2 && got[1].z == 3 &&
              got.count(2) && got[2].x == 4 && got[2].y == 5 && got[2].z == 6,
              "T5: same basename in two folders stays two cameras");
        check(got.count(3) && got[3].x == 7, "T5: an extensionless entry matches by stem");

        {   // 畸形行须拒绝且错误包含行号
            const fs::path bad = dir / "bad.txt";
            std::ofstream f(bad.string());
            f << "ok.jpg 1 2 3\n\nbroken.jpg 1 2\n";
            f.close();
            std::map<std::string, Vec3> p2;
            std::string e2;
            check(!readMetricPositions(bad.string(), p2, e2), "T5: a short line is refused");
            check(e2.find("3") != std::string::npos, "T5: the error names line 3");
        }
        {   // 第四个数字不是注释，必须拒绝
            const fs::path junk = dir / "junk.txt";
            std::ofstream f(junk.string());
            f << "ok.jpg 1 2 3 4\n";
            f.close();
            std::map<std::string, Vec3> pj;
            std::string ej;
            check(!readMetricPositions(junk.string(), pj, ej),
                  "T5: a trailing field is refused");
        }
        {   // 重复图像条目是错误，不能后值覆盖前值
            const fs::path dup = dir / "dup.txt";
            std::ofstream f(dup.string());
            f << "a.jpg 1 2 3\na.jpg 4 5 6\n";
            f.close();
            std::map<std::string, Vec3> p3;
            std::string e3;
            check(!readMetricPositions(dup.string(), p3, e3), "T5: a repeated name is refused");
        }
        {   // 路径不存在须明确报告，不能视为空输入
            std::map<std::string, Vec3> p4;
            std::string e4;
            check(!readMetricPositions((dir / "nope.txt").string(), p4, e4),
                  "T5: a missing file is refused");
        }
        fs::remove_all(dir);
    }

    // ---------------- 两种字节序的 GPS IFD ----------------
    // IFD0 放入与 GPS 标签同号的干扰项，验证不能混用分派表。
    struct Tiff {
        std::vector<uint8_t> b;
        bool le;
        void u8(uint8_t v) { b.push_back(v); }
        void u16(uint16_t v) {
            if (le) { u8((uint8_t)v); u8((uint8_t)(v >> 8)); }
            else { u8((uint8_t)(v >> 8)); u8((uint8_t)v); }
        }
        void u32(uint32_t v) {
            if (le) { u16((uint16_t)v); u16((uint16_t)(v >> 16)); }
            else { u16((uint16_t)(v >> 16)); u16((uint16_t)v); }
        }
        void ent(uint16_t tag, uint16_t type, uint32_t count, uint32_t val) {
            u16(tag); u16(type); u32(count); u32(val);
        }
        // 不超过四字节的值左对齐存于条目内部。
        void entIn(uint16_t tag, uint16_t type, uint32_t count,
                   const std::vector<uint8_t>& raw) {
            u16(tag); u16(type); u32(count);
            for (int i = 0; i < 4; i++) u8(i < (int)raw.size() ? raw[i] : 0);
        }
    };
    auto build = [](bool le, char latref, char lonref, uint8_t altref, bool with_gps,
                    uint32_t sec100 = 355, uint32_t lonsec100 = 1786,
                    bool with_alt = true) {
        Tiff t;
        t.le = le;
        t.u8(le ? 'I' : 'M'); t.u8(le ? 'I' : 'M');
        t.u16(42);
        t.u32(8);
        const uint16_t n0 = with_gps ? 7 : 6;
        t.u16(n0);
        for (uint16_t tag = 1; tag <= 6; tag++) t.ent(tag, 3, 1, 0xDEAD);
        const uint32_t gps = 8 + 2 + (uint32_t)n0 * 12 + 4;
        if (with_gps) t.ent(0x8825, 4, 1, gps);
        t.u32(0);
        if (!with_gps) return t.b;
        const uint32_t nv = with_alt ? 6 : 5;
        t.u16((uint16_t)nv);
        const uint32_t vals = gps + 2 + nv * 12 + 4;
        t.entIn(1, 2, 2, {(uint8_t)latref, 0});
        t.ent(2, 5, 3, vals);
        t.entIn(3, 2, 2, {(uint8_t)lonref, 0});
        t.ent(4, 5, 3, vals + 24);
        t.entIn(5, 1, 1, {altref});
        if (with_alt) t.ent(6, 5, 1, vals + 48);
        t.u32(0);
        const uint32_t r[] = {42, 1, 12, 1, sec100, 100,
                              83, 1, 40, 1, lonsec100, 100,
                              25366, 100};
        for (uint32_t v : r) t.u32(v);
        return t.b;
    };
    {
        const double lat = 42.0 + 12.0 / 60.0 + 3.55 / 3600.0;
        const double lon = 83.0 + 40.0 / 60.0 + 17.86 / 3600.0;
        for (bool le : {true, false}) {
            std::vector<uint8_t> blk = build(le, 'N', 'E', 0, true);
            ExifData e = parseExifTiff(blk.data(), blk.size());
            const char* w = le ? "T6 LE" : "T6 BE";
            check(e.valid && e.has_gps, w);
            check(std::fabs(e.lat_deg - lat) <= 1e-9, "T6: latitude, both byte orders");
            check(std::fabs(e.lon_deg - lon) <= 1e-9, "T6: longitude, both byte orders");
            check(e.has_alt && std::fabs(e.alt_m - 253.66) <= 1e-9, "T6: altitude");
        }
        {
            std::vector<uint8_t> blk = build(true, 'S', 'W', 0, true);
            ExifData e = parseExifTiff(blk.data(), blk.size());
            check(std::fabs(e.lat_deg + lat) <= 1e-9, "T6: S is negative");
            check(std::fabs(e.lon_deg + lon) <= 1e-9, "T6: W is negative");
        }
        {   // GPSAltitudeRef=1 表示海平面以下
            std::vector<uint8_t> blk = build(true, 'N', 'E', 1, true);
            ExifData e = parseExifTiff(blk.data(), blk.size());
            check(std::fabs(e.alt_m + 253.66) <= 1e-9, "T6: altitude ref 1 is below sea level");
        }
        {
            std::vector<uint8_t> blk = build(true, 'N', 'E', 0, false);
            ExifData e = parseExifTiff(blk.data(), blk.size());
            check(e.valid && !e.has_gps, "T6: an IFD0 of decoys is not a GPS fix");
            check(e.lat_deg == 0 && e.lon_deg == 0, "T6: and leaves no position behind");
        }
    }

    // ---------------- 经 JPEG 遍历读取 GPS ----------------
    // 三份最小 JPEG 包含无 EXIF 情况，明确验证无 GPS 计数。
    {
        namespace fs = std::filesystem;
        const fs::path dir = fs::temp_directory_path() / "sfm_metric_t6b";
        fs::remove_all(dir);
        fs::create_directories(dir / "sub");
        auto write_jpeg = [&](const fs::path& path, const std::vector<uint8_t>& tiff) {
            std::vector<uint8_t> j = {0xFF, 0xD8};
            if (!tiff.empty()) {
                const uint16_t seg = (uint16_t)(tiff.size() + 8);
                j.push_back(0xFF);
                j.push_back(0xE1);
                j.push_back((uint8_t)(seg >> 8));
                j.push_back((uint8_t)seg);
                for (const char* p2 = "Exif"; *p2; p2++) j.push_back((uint8_t)*p2);
                j.push_back(0);
                j.push_back(0);
                j.insert(j.end(), tiff.begin(), tiff.end());
            }
            j.push_back(0xFF);
            j.push_back(0xD9);
            std::ofstream f(path.string(), std::ios::binary);
            f.write((const char*)j.data(), (std::streamsize)j.size());
        };
        write_jpeg(dir / "a.jpg", build(true, 'N', 'W', 0, true, 355, 1786));
        write_jpeg(dir / "sub" / "b.jpg", build(true, 'N', 'W', 0, true, 1055, 1786));
        write_jpeg(dir / "c.jpg", {});
        write_jpeg(dir / "d.jpg", build(true, 'N', 'W', 0, true, 705, 1786, false));
        write_jpeg(dir / "e.jpg", build(true, 'N', 'W', 0, true, 355, 1786));
        Reconstruction rec;
        const char* names[5] = {"a.jpg", "sub/b.jpg", "c.jpg", "d.jpg", "e.jpg"};
        for (int i = 0; i < 5; i++) {
            Image im;
            im.id = (uint32_t)(i + 1);
            im.name = names[i];
            im.registered = i != 4;   // e.jpg 有定位但未配准
            im.pose = {mat3Identity(), {(double)-i, 0, 0}};
            rec.images[im.id] = im;
        }
        MetricRef ref;
        MetricGpsCounts gc = metricRefFromGps(rec, dir.string(), ref);
        printf("  T6b: gps %d, no-gps %d, no-alt %d\n", gc.matched, gc.no_gps, gc.no_alt);
        check(gc.matched == 3 && gc.no_gps == 1,
              "T6b: three fixes, one file without EXIF, one unregistered");
        check(gc.no_alt == 1, "T6b: the fix with no altitude tag is counted");
        check(ref.targets.size() == 3 && ref.image_ids.size() == 3,
              "T6b: the unpositioned and unregistered images are left out");
        // 仅纬度相差 7 角秒应为 215.99 m；简单球体为 216.43 m，0.05 m 容差可检出错误 ENU 近似。
        const Vec3 d = ref.targets[1] - ref.targets[0];
        printf("  T6b: b - a = (%.4f, %.4f, %.4f) m\n", d.x, d.y, d.z);
        check(std::fabs(d.y - 215.99) < 0.05 && std::fabs(d.x) < 0.01,
              "T6b: 216 m due north, by the meridional radius not by a");
        fs::remove_all(dir);
    }

    // ---------------- WGS-84 与外部计算参考 ----------------
    // 赤道和极点半轴可精确检查，四个样本均要求 float ECEF 无法达到的 1e-6 m 精度。
    {
        struct Case { double lat, lon, h, X, Y, Z; };
        const Case cs[4] = {
            {0, 0, 0, 6378137.000000, 0.0, 0.0},
            {90, 0, 0, 0.0, 0.0, 6356752.314245},
            {45, 0, 0, 4517590.878849, 0.0, 4487348.408866},
            {42.216, -83.664, 253.66, 522118.504341, -4702200.844425, 4263573.714297},
        };
        double worst = 0;
        for (const Case& c : cs) {
            const Vec3 p = ecefFromGeodetic(c.lat, c.lon, c.h);
            worst = std::max(worst, std::max(std::fabs(p.x - c.X),
                                             std::max(std::fabs(p.y - c.Y),
                                                      std::fabs(p.z - c.Z))));
        }
        printf("  T7: worst ECEF error %.3e m\n", worst);
        check(worst <= 1e-6, "T7: the four pinned ECEF vectors");

        const double lat0 = 42.216, lon0 = -83.664, h0 = 253.66;
        std::vector<Geodetic> g = {{lat0, lon0, h0},
                                   {lat0 + 0.001, lon0, h0},
                                   {lat0, lon0 + 0.001, h0},
                                   {lat0, lon0, h0 + 10.0}};
        std::vector<Vec3> enu = enuFromGeodetic(g, Geodetic{lat0, lon0, h0});
        const Vec3 o = enu[0];
        const double want[3][3] = {{-0.000000, 111.081917, -0.000969},
                                   {82.573260, 0.000484, -0.000534},
                                   {0.0, 0.0, 10.000000}};
        double we = 0;
        for (int i = 0; i < 3; i++) {
            const Vec3 d = enu[i + 1] - o;
            we = std::max(we, std::max(std::fabs(d.x - want[i][0]),
                                       std::max(std::fabs(d.y - want[i][1]),
                                                std::fabs(d.z - want[i][2]))));
        }
        printf("  T7: worst ENU displacement error %.3e m\n", we);
        check(we <= 1e-6, "T7: the three pinned ENU displacements");
        // 轴置换或镜像可被 Sim3 吸收而不改 RMS，必须逐命名轴检查。
        const Vec3 north = enu[1] - o, east = enu[2] - o, up = enu[3] - o;
        check(north.y > 100.0 && std::fabs(north.x) < 1.0, "T7: +latitude is +N");
        check(east.x > 80.0 && std::fabs(east.y) < 1.0, "T7: +longitude is +E");
        check(up.z > 9.9, "T7: +height is +U");
        check(east.normalized().cross(north.normalized()).dot(up.normalized()) > 0.999,
              "T7: E x N = U, from the fitted axes");
        // GPS 原点采用定位经纬度算术均值而非 ECEF 质心，后者在此有 3e-4 m 偏移，可被平移掩盖。
        Geodetic mo;
        for (const Geodetic& q : g) {
            mo.lat_deg += q.lat_deg / g.size();
            mo.lon_deg += q.lon_deg / g.size();
            mo.alt_m += q.alt_m / g.size();
        }
        std::vector<Vec3> by_mean = enuFromGeodetic(g);
        std::vector<Vec3> by_hand = enuFromGeodetic(g, mo);
        double dm = 0;
        for (size_t i = 0; i < g.size(); i++) dm = std::max(dm, (by_mean[i] - by_hand[i]).norm());
        check(dm == 0.0, "T7: the default origin is the mean of the fixes");
        check((by_mean[0] - by_hand[0]).norm() == 0.0 &&
                  (by_mean[0] - enuFromGeodetic(g, g[0])[0]).norm() > 1e-3,
              "T7: and not the first fix");
    }

    // ---------------- 条件数门限两侧测试 ----------------
    // 横向比例 0.045/0.055 夹住阈值，移动超过 10% 应使至少一项失败。
    {
        Sim3 T;
        T.scale = 2.5;
        T.R = rotFromAxisAngle({0.2, 0.5, 0.84}, 0.7);
        T.t = {3.0, -1.0, 8.0};
        // 沿 x 的 lambda1=(n²-1)/12，lambda2=a²，perp_frac=sqrt(a²/(lambda1+a²))；+--+ 符号模式使交叉协方差严格为零。
        auto line = [](double a) {
            std::vector<Vec3> c(60);
            for (int i = 0; i < 60; i++) {
                const int j = i % 4;
                c[i] = {(double)i, (j == 0 || j == 3) ? a : -a, 0.0};
            }
            return c;
        };
        MetricFit below = fitMetricGauge(makeRef(line(0.780105), T), 0.5);
        MetricFit above = fitMetricGauge(makeRef(line(0.953940), T), 0.5);
        printf("  T11: perp frac %.6f (below the floor) / %.6f (above)\n",
               below.perp_frac, above.perp_frac);
        check(std::fabs(below.perp_frac - 0.045) < 1e-6, "T11: the low fixture really is 0.045");
        check(std::fabs(above.perp_frac - 0.055) < 1e-6, "T11: the high fixture really is 0.055");
        check(!below.ok && below.reason == MetricFail::Collinear,
              "T11: 0.045 is refused, so the floor is not below it");
        check(above.ok, "T11: 0.055 is accepted, so the floor is not above it");
    }

    // ---------------- 高度漂移下的水平拟合 ----------------
    // 水平 30 m 环形参考高度随向东每米增加 0.1 m，完整三维拟合会把漂移吸收为倾斜。
    {
        Sim3 T;
        T.scale = 4.0;
        T.R = rotFromAxisAngle({0, 0, 1}, 0.7);
        T.t = {12.0, -3.0, 5.0};
        std::vector<Vec3> c(40);
        for (int i = 0; i < 40; i++) {
            const double a = 2.0 * M_PI * i / 40;
            c[i] = {7.5 * std::cos(a), 7.5 * std::sin(a), 0.08 * std::sin(3.0 * a)};
        }
        MetricRef ref = makeRef(c, T);
        for (Vec3& p : ref.targets) p.z += 0.1 * p.x;
        MetricFit flat = fitMetricGauge(ref, 3.0, MetricAxes::Horizontal);
        MetricFit full = fitMetricGauge(ref, 3.0);
        const double flat_tilt = std::acos(std::min(1.0, flat.T.R[8])) * 180.0 / M_PI;
        const double full_tilt = std::acos(std::min(1.0, full.T.R[8])) * 180.0 / M_PI;
        printf("  T12: tilt %.4f deg (horizontal) / %.4f deg (full); scale error "
               "%.2e / %.2e\n", flat_tilt, full_tilt,
               std::fabs(flat.T.scale / T.scale - 1.0),
               std::fabs(full.T.scale / T.scale - 1.0));
        check(flat.ok && full.ok, "T12: both fits are accepted");
        check(std::fabs(flat.T.scale / T.scale - 1.0) <= 1e-12,
              "T12: the horizontal fit recovers the scale exactly");
        check(chordal(flat.T.R, T.R) <= 1e-12,
              "T12: and the heading, with no tilt of its own");
        check(flat.T.R[2] == 0.0 && flat.T.R[5] == 0.0 && flat.T.R[6] == 0.0 &&
                  flat.T.R[7] == 0.0 && flat.T.R[8] == 1.0,
              "T12: the horizontal rotation is a turn about +Z, bit for bit");
        check(full_tilt > 2.0, "T12: the full fit tips the scene by degrees");
        check(flat.rms <= 1e-12, "T12: the level residuals are the fit's own");
        // 水平拟合虽不使用高度，仍须报告高度误差。
        double u = 0;
        for (size_t k = 0; k < ref.centres.size(); k++) {
            const Vec3 r = ref.targets[k] - transformPoint(flat.T, ref.centres[k]);
            u += r.z * r.z;
        }
        check(std::sqrt(u / ref.centres.size()) > 1.0,
              "T12: and the altitude it refused is still there to be printed");
    }

    // ---------------- 共线街道可接受水平拟合 ----------------
    // 横向比例 0.045 时绕轨迹滚转不可观，但绕竖轴仍有充分约束。
    {
        Sim3 T;
        T.scale = 2.5;
        T.R = rotFromAxisAngle({0, 0, 1}, 0.4);
        T.t = {3.0, -1.0, 8.0};
        std::vector<Vec3> c(60);
        for (int i = 0; i < 60; i++) {
            const int j = i % 4;
            c[i] = {(double)i, (j == 0 || j == 3) ? 0.780105 : -0.780105, 0.0};
        }
        MetricRef ref = makeRef(c, T);
        MetricFit full = fitMetricGauge(ref, 0.5);
        MetricFit flat = fitMetricGauge(ref, 0.5, MetricAxes::Horizontal);
        printf("  T13: perp frac %.4f; scale %.6f (horizontal), refused: %d (full)\n",
               full.perp_frac, flat.T.scale, (int)full.reason);
        check(!full.ok && full.reason == MetricFail::Collinear,
              "T13: the full fit refuses the line");
        check(flat.ok && std::fabs(flat.T.scale / T.scale - 1.0) <= 1e-12,
              "T13: the horizontal fit takes it and recovers the scale");
        check(flat.inliers == 60, "T13: with every camera an inlier");
    }

    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

int main(int argc, char** argv) { return sfmTestMain(argc, argv, cmdMetricSelftest); }
