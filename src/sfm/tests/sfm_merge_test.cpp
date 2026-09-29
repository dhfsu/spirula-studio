// 纯主机模型合并测试：Sim(3)、对齐与轨迹拼接。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "sfm/core/Model.h"
#include "sfm/map/Assemble.h"
#include "sfm/map/Merge.h"
#include "sfm/tests/TestMain.h"

namespace fs = std::filesystem;
using namespace sfm;

// 相机位于 C 看向 target，y 向上；简短场景初始化在测试内局部定义。
static Pose lookAt(const Vec3& C, const Vec3& target) {
    Vec3 f = (target - C).normalized();
    Vec3 up0 = {0, 1, 0};
    Vec3 r = up0.cross(f).normalized();
    Vec3 u = f.cross(r);
    Mat3 R = {r.x, r.y, r.z, u.x, u.y, u.z, f.x, f.y, f.z};
    Vec3 t = mul(R, C);
    return {R, {-t.x, -t.y, -t.z}};
}

// ---------------- 模型合并测试 ----------------
// 圆弧相机分成有四张共享图像的两半，一半施加已知相似变换；合并须恢复规范并拒绝不满足条件的情况。
int cmdMergeSelftest(int, char**) {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { printf("  FAIL: %s\n", what); fails++; }
        return ok;
    };

    const int W = 1280, H = 960, M = 12, N = 240;
    Camera K = Camera::defaultFor(1, W, H, 1200);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> ub(-2.5, 2.5);
    std::normal_distribution<double> noise(0.0, 0.25);

    std::vector<Vec3> pts(N);
    for (Vec3& p : pts) p = {ub(rng), ub(rng), ub(rng)};
    std::vector<Pose> gt(M);
    for (int c = 0; c < M; c++) {
        double ang = -1.0 + 2.0 * c / (M - 1);
        gt[c] = lookAt({9 * std::sin(ang), 1.5 * std::sin(0.6 * c), 9 * std::cos(ang)}, {0, 0, 0});
    }
    std::vector<std::vector<Vec2>> obs(M, std::vector<Vec2>(N, {-1000, -1000}));
    std::vector<std::vector<char>> vis(M, std::vector<char>(N, 0));
    for (int c = 0; c < M; c++)
        for (int p = 0; p < N; p++) {
            Vec3 pc = mul(gt[c].R, pts[p]) + gt[c].t;
            Vec2 px = K.project(pc);
            if (pc.z > 0.1 && px.x > 0 && px.x < W && px.y > 0 && px.y < H) {
                obs[c][p] = {px.x + noise(rng), px.y + noise(rng)};
                vis[c][p] = 1;
            }
        }

    // 各图保留全部 N 个特征槽，使两半的 point2D_idx 含义一致；至少双视图的点参与三角化。
    auto buildModel = [&](int first, int last) {
        Reconstruction r;
        r.cameras[1] = K;
        for (int c = first; c <= last; c++) {
            Image im;
            im.id = (uint32_t)c;
            im.camera_id = 1;
            im.name = "cam" + std::to_string(c) + ".jpg";
            im.registered = true;
            im.pose = gt[c];
            im.points2D = obs[c];
            im.point3D_ids.assign(N, kInvalidPoint3D);
            r.images[(uint32_t)c] = std::move(im);
        }
        for (int p = 0; p < N; p++) {
            std::vector<TrackElement> tr;
            for (int c = first; c <= last; c++)
                if (vis[c][p]) tr.push_back({(uint32_t)c, (uint32_t)p});
            if (tr.size() >= 2) r.addPoint3D(pts[p], tr);
        }
        return r;
    };
    auto applySim3 = [](Reconstruction& r, const Sim3& S) {
        for (auto& kv : r.images) kv.second.pose = transformPose(S, kv.second.pose);
        for (auto& kv : r.points3D) kv.second.xyz = transformPoint(S, kv.second.xyz);
    };
    auto meanReproj = [](const Reconstruction& r) {
        double sum = 0;
        size_t n = 0;
        for (const auto& kv : r.points3D)
            for (const TrackElement& e : kv.second.track) {
                const Image& im = r.images.at(e.image_id);
                double v = reprojErrorAt(r.cameras.at(im.camera_id), im.pose,
                                         im.points2D[e.point2D_idx], kv.second.xyz);
                if (v < 1e29) { sum += v; n++; }
            }
        return n ? sum / (double)n : 0.0;
    };

    // ---------------- Sim3 代数 ----------------
    // 世界与相机同时变换后投影必须不变，验证 transformPose 与 transformPoint 一致。
    Sim3 S;
    S.scale = 2.7;
    S.R = angleAxisToRotation({0.3, -0.7, 0.2});
    S.t = {5, -3, 11};
    {
        double maxPx = 0, maxRt = 0;
        for (int c = 0; c < M; c++) {
            Pose tp = transformPose(S, gt[c]);
            for (int p = 0; p < N; p += 17) {
                Vec2 a = K.project(mul(gt[c].R, pts[p]) + gt[c].t);
                Vec3 X = transformPoint(S, pts[p]);
                Vec2 b = K.project(mul(tp.R, X) + tp.t);
                maxPx = std::max(maxPx, std::hypot(a.x - b.x, a.y - b.y));
                Vec3 back = transformPoint(invertSim3(S), X);
                maxRt = std::max(maxRt, (back - pts[p]).norm());
            }
        }
        printf("merge-selftest: Sim3 invariance %.2e px, inverse round-trip %.2e\n", maxPx, maxRt);
        check(maxPx < 1e-6, "transformPose/transformPoint disagree on projection");
        check(maxRt < 1e-9, "invertSim3 round-trip");

        Sim3 T2;
        T2.scale = 0.4;
        T2.R = angleAxisToRotation({-0.2, 0.5, 0.9});
        T2.t = {-2, 7, 1};
        double maxC = 0;
        for (int p = 0; p < N; p += 17) {
            const Vec3 a = transformPoint(T2, transformPoint(S, pts[p]));
            maxC = std::max(maxC, (a - transformPoint(composeSim3(T2, S), pts[p])).norm());
        }
        check(maxC < 1e-9, "composeSim3 is the two applied in order");
    }

    // ---------------- 精确对应的 Umeyama ----------------
    {
        std::vector<Vec3> a, b;
        for (int i = 0; i < 8; i++) {
            Vec3 x = {ub(rng), ub(rng), ub(rng)};
            a.push_back(x);
            b.push_back(transformPoint(S, x));
        }
        Sim3 T;
        bool ok = estimateSim3(a, b, T);
        double err = 0;
        for (size_t i = 0; i < a.size(); i++)
            err = std::max(err, (transformPoint(T, a[i]) - b[i]).norm());
        printf("  estimateSim3: scale %.6f (want %.6f), max residual %.2e\n", T.scale, S.scale,
               err);
        check(ok && err < 1e-9, "estimateSim3 does not reproduce an exact similarity");
        // 全部点重合的退化输入必须报告失败。
        std::vector<Vec3> same(4, Vec3{1, 2, 3});
        Sim3 bad;
        check(!estimateSim3(same, same, bad), "estimateSim3 accepted a degenerate sample");
    }

    // ---------------- 共线相机的位姿对齐 ----------------
    // 三个中心共线不能确定绕线旋转，完整位姿仍可确定 Sim3。
    {
        std::vector<Pose> src, dst;
        for (int i = 0; i < 3; i++) {
            Pose p = lookAt({-2.0 + 2.0 * i, 0, 0}, {-2.0 + 2.0 * i, 0, 6});
            src.push_back(p);
            dst.push_back(transformPose(S, p));
        }
        Sim3 T;
        bool ok = estimateSim3FromPoses(src, dst, T);
        double err = 0;
        for (int p = 0; p < N; p += 13)
            err = std::max(err, (transformPoint(T, pts[p]) - transformPoint(S, pts[p])).norm());
        // 同样中心仅用 Umeyama 可能表面成功但结果错误。
        std::vector<Vec3> cs, cd;
        for (int i = 0; i < 3; i++) {
            cs.push_back(cameraCenter(src[i]));
            cd.push_back(cameraCenter(dst[i]));
        }
        Sim3 U;
        double uerr = 1e30;
        if (estimateSim3(cs, cd, U)) {
            uerr = 0;
            for (int p = 0; p < N; p += 13)
                uerr = std::max(uerr,
                                (transformPoint(U, pts[p]) - transformPoint(S, pts[p])).norm());
        }
        printf("  collinear shared cameras: pose fit off by %.2e, center fit off by %.2e\n", err,
               uerr);
        check(ok && err < 1e-9, "estimateSim3FromPoses failed on collinear centers");
        check(uerr > 1e-3, "the center-only fit was expected to be degenerate here");
    }

    // ---------------- 共享图像对齐 ----------------
    Reconstruction A = buildModel(0, 7);
    Reconstruction B = buildModel(4, 11);
    applySim3(B, S);  // 第二半位于独立坐标规范
    {
        AlignmentResult r = alignReconstructions(B, A, MergeOptions());
        double err = 0;
        for (int p = 0; p < N; p += 13)
            err = std::max(err, (transformPoint(r.transform, transformPoint(S, pts[p])) - pts[p])
                                    .norm());
        printf("  alignment: %s, %zu/%zu shared images agree, %.3f px, recovers the gauge to %.2e\n",
               r.success ? "ok" : r.reason.c_str(), r.inliers, r.common_images, r.mean_error, err);
        check(r.success, "alignment failed on four exact shared images");
        check(r.common_images == 4 && r.inliers == 4, "not all shared images were inliers");
        check(err < 1e-2, "recovered similarity is wrong");
    }

    // ---------------- 执行合并 ----------------
    {
        MergeOptions mo;
        mo.verbose = false;
        MergeSession s({A, B}, mo);
        size_t merges = s.runAuto();
        std::vector<Reconstruction> out = s.take();
        const size_t spliced = s.log().empty() ? 0 : s.log().front().counts.points_spliced;
        double mr = out.empty() ? 0 : meanReproj(out.front());
        printf("  merge: %zu merge(s) -> %zu model(s), %u images, %zu points (%zu spliced), "
               "%.3f px mean\n",
               merges, out.size(), out.empty() ? 0 : out.front().numRegistered(),
               out.empty() ? 0 : out.front().points3D.size(), spliced, mr);
        check(merges == 1 && out.size() == 1, "the two halves did not merge into one model");
        if (!out.empty()) {
            check(out.front().numRegistered() == (uint32_t)M, "merged model lost images");
            // 共享特征的轨迹必须拼接，不能只连接数组而生成重复三维点。
            check(out.front().points3D.size() <= (size_t)(1.2 * N),
                  "merged model has duplicate points");
            check(spliced > (size_t)(N / 2), "hardly any tracks were spliced");
            check(mr < 1.0, "merged model reprojects badly");
        }
    }

    // ---------------- 重叠不足时拒绝且保持模型 ----------------
    {
        Reconstruction a2 = buildModel(0, 5), b2 = buildModel(5, 11);
        applySim3(b2, S);
        MergeOptions mo;
        mo.verbose = false;
        MergeSession s({a2, b2}, mo);
        size_t merges = s.runAuto();
        printf("  one shared image: %zu merge(s), %zu candidate(s), models %s\n", merges,
               s.candidates().size(),
               s.alive(0) && s.alive(1) ? "both intact" : "MODIFIED");
        check(merges == 0 && s.candidates().empty(), "merged on a single shared image");
        check(s.alive(0) && s.alive(1) && s.model(0).numRegistered() == 6,
              "a refused merge modified the models");
    }

    // ---------------- 错误外部对齐须检测并回滚 ----------------
    // 测试界面可提供的看似合理但错误变换，新增图像缺少观测支持时必须拒绝。
    {
        MergeOptions mo;
        mo.verbose = false;
        MergeSession s({A, B}, mo);
        Sim3 bad = invertSim3(S);
        bad.R = mul(angleAxisToRotation({0, 0.5, 0}), bad.R);  // 旋转偏差 29 度
        bad.scale *= 1.3;
        const size_t points_before = s.model(0).points3D.size();
        const uint32_t imgs_before = s.model(0).numRegistered();
        MergeAttempt a = s.tryMerge(0, 1, &bad);
        printf("  wrong transform: %s (%s), anchor %u images / %zu points -> %u / %zu\n",
               a.merged ? "MERGED" : "refused", a.reason.c_str(), imgs_before, points_before,
               s.model(0).numRegistered(), s.model(0).points3D.size());
        check(!a.merged, "a badly aligned merge was accepted");
        check(s.model(0).numRegistered() == imgs_before &&
                  s.model(0).points3D.size() == points_before,
              "a refused merge left the anchor modified");
        check(s.alive(1), "a refused merge consumed the source model");
    }

    // ---------------- 拒绝不兼容模型 ----------------
    // 同图像 ID 但关键点不同，意味着 point2D_idx 不再指向同一特征。
    {
        Reconstruction b3 = buildModel(4, 11);
        applySim3(b3, S);
        for (auto& kv : b3.images) kv.second.points2D.resize(N - 1);
        AlignmentResult r = alignReconstructions(b3, A, MergeOptions());
        printf("  mismatched features: %s (%s)\n", r.success ? "ALIGNED" : "refused",
               r.reason.c_str());
        check(!r.success, "aligned two models built from different features");
    }

    // ---------------- 过滤不应保留的点 ----------------
    {
        Reconstruction f = buildModel(0, 7);
        const size_t before = f.points3D.size();
        f.points3D.begin()->second.xyz = {1000, 1000, 1000};  // 点位置远离轨迹支持
        size_t robs = 0, rpts = 0;
        filterModel(f, 4.0, 1.5, robs, rpts);
        printf("  filterModel: %zu points -> %zu (%zu obs dropped)\n", before, f.points3D.size(),
               robs);
        check(rpts == 1 && f.points3D.size() == before - 1,
              "filterModel did not drop a point that reprojects nowhere");
    }

    // ---------------- 两处场景错误叠合的折叠（D45）----------------
    // 复制相机和点位置但使用互斥点 ID，形成内部自洽副本；检查器应找到同位同向却无共同结构的图像对，并拆成两个模型。
    {
        Reconstruction sound = buildModel(0, 11);
        DuplicateOptions dopt;
        DuplicateReport clean = findDuplicateStructure(sound, dopt);
        printf("  fold detector on a sound model: %zu of %zu co-located pairs share nothing\n",
               clean.conflicts, clean.colocated);
        check(!clean.duplicated(dopt), "the fold detector fired on a sound model");

        // 第二副本使用偏移图像 ID、第一副本相同位姿和独立轨迹，双方不共享点。
        Reconstruction folded = sound;
        const uint32_t off = 100;
        const uint64_t pid_off = 100000;
        for (int c = 0; c <= 11; c++) {
            Image im = sound.images.at((uint32_t)c);
            im.id = (uint32_t)c + off;
            im.name = "dup" + std::to_string(c) + ".jpg";
            for (uint64_t& p : im.point3D_ids)
                if (p != kInvalidPoint3D) p += pid_off;
            folded.images[im.id] = std::move(im);
        }
        for (const auto& kv : sound.points3D) {
            Point3D p = kv.second;
            for (TrackElement& e : p.track) e.image_id += off;
            folded.points3D[kv.first + pid_off] = std::move(p);
        }
        DuplicateReport rep = findDuplicateStructure(folded, dopt);
        size_t dropped = 0;
        DuplicateCut cut;
        std::vector<Reconstruction> parts =
            splitDuplicateStructure(folded, rep, 4, &dropped, &cut);
        std::vector<uint32_t> sizes;
        for (const Reconstruction& p : parts) sizes.push_back(p.numRegistered());
        printf("  fold detector on two copies at the same poses: %zu of %zu co-located pairs "
               "share nothing -> %zu part(s)", rep.conflicts, rep.colocated, parts.size());
        for (uint32_t n : sizes) printf(" %u", n);
        printf("\n");
        check(rep.duplicated(dopt), "the fold detector missed two copies at identical poses");
        check(parts.size() == 2 && sizes.size() == 2 && sizes[0] == 12 && sizes[1] == 12,
              "the fold split did not separate the two copies");
        // 对应图若确认两副本图像曾真实匹配，应否决折叠判定，因为可能只是轨迹未合并。
        DuplicateReport vetoed = findDuplicateStructure(
            folded, dopt, [](uint32_t, uint32_t) { return true; });
        printf("  ... with every pair reported as matched: %zu conflicts (%zu vetoed)\n",
               vetoed.conflicts, vetoed.unmatched_but_seen);
        check(vetoed.conflicts == 0 && vetoed.unmatched_but_seen == rep.conflicts,
              "the matched-pair veto did not suppress the conflicts");

        // 按切割代价而非冲突数判断；真正独立副本分开不损失共视（D46）。
        printf("  ... the split severs %.2f%% of co-visibility (%llu of %llu) -> %s\n",
               100.0 * cut.fraction(), (unsigned long long)cut.severed,
               (unsigned long long)(cut.severed + cut.kept),
               foldSplitAccepted(rep, cut, dopt) ? "folded" : "kept whole");
        check(foldSplitAccepted(rep, cut, dopt), "the fold's split was not accepted");

        // 正确密集共视模型即使人为声明冲突，切分也会切断真实结构，必须因高代价拒绝，不能仅看冲突率。
        DuplicateReport bogus;
        bogus.colocated = 20;
        bogus.conflicts = 8;
        for (uint32_t i = 0; i + 1 < 12; i += 3) bogus.pairs.push_back({i, i + 1});
        DuplicateCut bcut;
        size_t bdropped = 0;
        std::vector<Reconstruction> bparts =
            splitDuplicateStructure(sound, bogus, 2, &bdropped, &bcut);
        printf("  ... false conflicts inside a sound model: split into %zu, severing %.1f%% "
               "-> %s\n",
               bparts.size(), 100.0 * bcut.fraction(),
               foldSplitAccepted(bogus, bcut, dopt) ? "folded" : "kept whole");
        check(bcut.groups > 1 && !foldSplitAccepted(bogus, bcut, dopt),
              "a split through a sound model's own co-visibility was accepted");

        // 切出组还必须与其他部分空间重叠；将第二副本除三图外移远，仍有冲突和零共视损失，但不是真正整体折叠（D67）。
        {
            Reconstruction partial = folded;
            for (int c = 3; c <= 11; c++) {
                Pose& p = partial.images.at((uint32_t)c + off).pose;
                p.t = p.t - mul(p.R, Vec3{100, 0, 0});  // 相机中心增加 (100,0,0)
            }
            DuplicateReport prep = findDuplicateStructure(partial, dopt);
            size_t pdropped = 0;
            DuplicateCut pcut;
            std::vector<Reconstruction> pparts = splitDuplicateStructure(
                partial, prep, 4, &pdropped, &pcut, dopt.min_fold_overlap);
            printf("  ... only 3 of 12 images of the second copy left on top of the first: "
                   "%zu conflicts, %zu part(s), %zu re-attached\n",
                   prep.conflicts, pparts.size(), pcut.reattached);
            check(prep.duplicated(dopt), "the moved copy stopped conflicting at all");
            check(pparts.size() == 1 && pcut.reattached == 1,
                  "a piece standing on top of nothing was written out as a duplicate");

            // 相同门限下真正折叠仍应成功拆开。
            DuplicateCut fcut;
            size_t fdropped = 0;
            std::vector<Reconstruction> fparts = splitDuplicateStructure(
                folded, rep, 4, &fdropped, &fcut, dopt.min_fold_overlap);
            printf("  ... the real fold under the same bar: %zu part(s), overlap %.0f%%\n",
                   fparts.size(), 100.0 * fcut.min_overlap);
            check(fparts.size() == 2 && fcut.reattached == 0,
                  "the overlap bar rejected a genuine fold");
        }
    }

    printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

int main() { return sfmTestMain(0, nullptr, cmdMergeSelftest); }
