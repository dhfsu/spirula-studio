// 使用 GPU BA 的合成完整重建测试，逐项输出 PASS/FAIL。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "sfm/core/FeatureCompaction.h"
#include "sfm/core/Model.h"
#include "sfm/map/Bundle.h"
#include "sfm/map/Assemble.h"
#include "sfm/map/Mapper.h"
#include "sfm/tests/TestMain.h"

namespace fs = std::filesystem;
using namespace sfm;

// ---------------- 合成端到端 GPU 重建 ----------------
static Pose lookAt(const Vec3& C, const Vec3& target) {
    Vec3 f = (target - C).normalized();
    Vec3 up0 = {0, 1, 0};
    Vec3 r = up0.cross(f).normalized();
    Vec3 u = f.cross(r);
    Mat3 R = {r.x, r.y, r.z, u.x, u.y, u.z, f.x, f.y, f.z};
    Vec3 t = mul(R, C);
    return {R, {-t.x, -t.y, -t.z}};
}

int cmdMapSelftest(int argc, char** argv) {
    MapperOptions opt;
    opt.verbose = false;
    opt.focal = 1200;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) opt.device = std::stoi(argv[++i]);
        else if (a == "--verbose") opt.verbose = true;
    }
    // 两项刻意病态的焦距/主点收敛断言需要 fp64，约 48 位 df 可能达不到；真实数据重投影通常仍相差不超过 0.01 px，其他检查必须全部通过。
    const bool fp64_ba =
        realSupportedByDevice(RealCfg::F64, VkContext::probeCaps(opt.device));
    if (!fp64_ba)
        printf("  note: no fp64 BA on this device -- the two ill-conditioned "
               "convergence checks are reported, not asserted\n");

    const int W = 1280, H = 960, M = 8, N = 160;
    Camera K = Camera::defaultFor(1, W, H, 1200);

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> ub(-2.5, 2.5);
    std::normal_distribution<double> noise(0.0, 0.4);

    std::vector<Vec3> pts(N);
    for (auto& p : pts) p = {ub(rng), ub(rng), ub(rng)};

    // 相机沿半径 9、跨度 120 度的圆弧看向原点。
    std::vector<Pose> gt(M);
    for (int c = 0; c < M; c++) {
        double ang = -1.05 + 2.1 * c / (M - 1);
        gt[c] = lookAt({9 * std::sin(ang), 1.5 * std::sin(0.7 * c), 9 * std::cos(ang)}, {0, 0, 0});
    }

    // 投影生成特征与真值匹配。
    std::vector<FeatureSet> feats(M);
    std::vector<std::vector<char>> vis(M, std::vector<char>(N, 0));
    for (int c = 0; c < M; c++) {
        feats[c].width = W;
        feats[c].height = H;
        feats[c].keypoints.resize(N);
        for (int p = 0; p < N; p++) {
            Vec3 pc = mul(gt[c].R, pts[p]) + gt[c].t;
            Vec2 px = K.project(pc);
            if (pc.z > 0.1 && px.x > 0 && px.x < W && px.y > 0 && px.y < H) {
                feats[c].keypoints[p] = {(float)(px.x + noise(rng)), (float)(px.y + noise(rng)), 2, 0, 0};
                vis[c][p] = 1;
            } else {
                feats[c].keypoints[p] = {-1000, -1000, 2, 0, 0};
            }
        }
    }
    MatchesDatabase db;
    db.images.resize(M);
    for (int c = 0; c < M; c++) db.images[c] = {"cam" + std::to_string(c), (uint32_t)N};
    for (int i = 0; i < M; i++)
        for (int j = i + 1; j < M; j++) {
            TwoViewMatches tv;
            tv.image1 = i; tv.image2 = j; tv.config = (int)TwoViewConfig::Uncalibrated;
            for (int p = 0; p < N; p++)
                if (vis[i][p] && vis[j][p]) tv.matches.push_back({(uint32_t)p, (uint32_t)p, 0});
            if (tv.matches.size() >= 15) db.pairs.push_back(std::move(tv));
        }

    MatchesDatabase compact_db = db;
    std::vector<FeatureSet> compact_feats(feats.size());
    {
        FeatureCompactionPlan plan = buildFeatureCompactionPlan(compact_db);
        for (size_t i = 0; i < feats.size(); i++)
            compact_feats[i] =
                compactFeatureSet(feats[i], plan.old_to_new[i], plan.compact_counts[i]);
        remapMatches(compact_db, plan, compact_feats);
    }

    Mapper mapper(db, feats, opt);
    // 合成图为单连通分量，必须输出单模型（D41）。
    std::vector<Reconstruction> models = mapper.run();
    const Reconstruction& rec = models.front();

    int fails = 0;
    uint32_t reg = rec.numRegistered();
    printf("map-selftest: %u/%d images registered, %zu points, %zu model(s)\n", reg, M,
           rec.points3D.size(), models.size());
    if (reg < (uint32_t)M) { printf("  FAIL: not all images registered\n"); fails++; }
    if (models.size() != 1) { printf("  FAIL: connected scene split into models\n"); fails++; }

    Mapper compact_mapper(compact_db, compact_feats, opt);
    std::vector<Reconstruction> compact_models = compact_mapper.run();
    bool compact_equivalent = !compact_models.empty() && compact_models.size() == models.size();
    const Reconstruction* compact_rec = compact_models.empty() ? nullptr : &compact_models.front();
    if (compact_equivalent)
        compact_equivalent = compact_rec->numRegistered() == rec.numRegistered() &&
                             compact_rec->points3D.size() == rec.points3D.size() &&
                             countObservations(*compact_rec) == countObservations(rec);
    if (compact_equivalent) {
        for (const auto& image : rec.images) {
            auto found = compact_rec->images.find(image.first);
            if (found == compact_rec->images.end() ||
                found->second.registered != image.second.registered) {
                compact_equivalent = false;
                break;
            }
        }
    }
    if (compact_equivalent) {
        for (const auto& camera : rec.cameras) {
            auto found = compact_rec->cameras.find(camera.first);
            if (found == compact_rec->cameras.end() ||
                std::fabs(found->second.fx - camera.second.fx) > 1e-9 ||
                std::fabs(found->second.fy - camera.second.fy) > 1e-9 ||
                std::fabs(found->second.cx - camera.second.cx) > 1e-9 ||
                std::fabs(found->second.cy - camera.second.cy) > 1e-9) {
                compact_equivalent = false;
                break;
            }
        }
    }
    printf("  unused-feature compaction A/B: %s (%u images, %zu points, %zu obs)\n",
           compact_equivalent ? "equivalent" : "BAD",
           compact_rec ? compact_rec->numRegistered() : 0,
           compact_rec ? compact_rec->points3D.size() : 0,
           compact_rec ? countObservations(*compact_rec) : 0);
    if (!compact_equivalent) fails++;

    // 相对旋转不受全局相似规范影响。
    double maxRelErr = 0;
    int cmpCount = 0;
    for (auto& a : rec.images)
        for (auto& b : rec.images) {
            if (a.first >= b.first || !a.second.registered || !b.second.registered) continue;
            Mat3 relRec = mul(b.second.pose.R, transpose(a.second.pose.R));
            Mat3 relGt = mul(gt[b.first].R, transpose(gt[a.first].R));
            Mat3 D = mul(relRec, transpose(relGt));
            double tr = (D[0] + D[4] + D[8] - 1) * 0.5;
            maxRelErr = std::max(maxRelErr,
                                 std::acos(std::max(-1.0, std::min(1.0, tr))) * 180.0 / M_PI);
            cmpCount++;
        }
    printf("  max relative-rotation error: %.2f deg over %d pairs\n", maxRelErr, cmpCount);
    if (reg >= 2 && maxRelErr > 1.5) { printf("  FAIL: pose accuracy\n"); fails++; }

    // 全部观测的平均重投影误差。
    double sum = 0;
    int nobs = 0;
    for (const auto& kv : rec.points3D)
        for (const TrackElement& e : kv.second.track) {
            const Pose& p = rec.images.at(e.image_id).pose;
            Vec3 pc = mul(p.R, kv.second.xyz) + p.t;
            Vec2 px = rec.cameras.at(1).project(pc);
            Vec2 o = {feats[e.image_id].keypoints[e.point2D_idx].x,
                      feats[e.image_id].keypoints[e.point2D_idx].y};
            sum += std::hypot(px.x - o.x, px.y - o.y);
            nobs++;
        }
    double meanReproj = nobs ? sum / nobs : 0;
    printf("  mean reprojection error: %.3f px over %d obs\n", meanReproj, nobs);
    if (nobs > 0 && meanReproj > 1.5) { printf("  FAIL: reprojection error\n"); fails++; }

    // COLMAP 模型读写往返。
    std::string dir = "/tmp/spirula_sfm_map_selftest";
    fs::create_directories(dir);
    rec.writeBinary(dir);
    Reconstruction rd = Reconstruction::readBinary(dir);
    bool rt = rd.images.size() == rec.numRegistered() && rd.points3D.size() == rec.points3D.size() &&
              rd.cameras.size() == rec.cameras.size();
    printf("  COLMAP model round-trip: %s\n", rt ? "ok" : "BAD");
    if (!rt) fails++;

    // ---------------- 无共同图像的结构对齐（D70）----------------
    // 两段独立重建虽无共同帧，但对应图关联双方三角化的同一结构，应能恢复相似变换。
    {
        auto half = [&](int first, int last) {
            Reconstruction r;
            r.cameras[1] = K;
            for (int c = first; c <= last; c++) {
                Image im;
                im.id = (uint32_t)c;
                im.camera_id = 1;
                im.name = "cam" + std::to_string(c);
                im.registered = true;
                im.pose = gt[c];
                im.points2D.resize(N);
                for (int p = 0; p < N; p++)
                    im.points2D[p] = {feats[c].keypoints[p].x, feats[c].keypoints[p].y};
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
        Reconstruction A = half(0, M / 2 - 1), B = half(M / 2, M - 1);
        // B 采用独立规范：1.7 倍尺度、25 度偏航及平移。
        Sim3 S;
        S.scale = 1.7;
        const double a = 25.0 * M_PI / 180.0;
        S.R = {std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a)};
        S.t = {3.0, -1.5, 0.5};
        for (auto& kv : B.images) kv.second.pose = transformPose(S, kv.second.pose);
        for (auto& kv : B.points3D) kv.second.xyz = transformPoint(S, kv.second.xyz);

        MergeOptions mopt;
        AlignmentResult al = mapper.alignByStructure(A, B, mopt);
        // 使用未参与三维点拟合的相机中心验证恢复变换，应落到 A 的真值规范。
        double worst = 0;
        for (const auto& kv : B.images) {
            const Vec3 got = transformPoint(al.transform, cameraCenter(kv.second.pose));
            const Vec3 want = cameraCenter(gt[kv.first]);
            worst = std::max(worst, (got - want).norm());
        }
        printf("  align on shared structure: %zu shared image(s), %zu point pair(s), "
               "%zu inlier(s), %.2f px, camera centres off by %.4f\n",
               al.common_images, al.structure_pairs, al.inliers, al.mean_error, worst);
        if (!al.success || al.common_images != 0 || worst > 0.05) {
            printf("  FAIL: the similarity was not recovered from shared structure alone\n");
            fails++;
        }
    }

    // ---------------- 两个不连通分量应输出两个模型（D41）----------------
    // 两份场景之间无验证边，结果应完整保留、按大小排序且图像不重叠。
    {
        MapperOptions opt2 = opt;
        opt2.min_model_size = 5;  // 每分量八张图像
        MatchesDatabase db2;
        std::vector<FeatureSet> feats2;
        // 复制两份场景，仅生成各自内部匹配，编号为 k*M+c。
        for (int k = 0; k < 2; k++) {
            for (int c = 0; c < M; c++) {
                feats2.push_back(feats[c]);
                db2.images.push_back({"comp" + std::to_string(k) + "/cam" + std::to_string(c),
                                      (uint32_t)N});
            }
            for (int i = 0; i < M; i++)
                for (int j = i + 1; j < M; j++) {
                    TwoViewMatches tv;
                    tv.image1 = (uint32_t)(k * M + i);
                    tv.image2 = (uint32_t)(k * M + j);
                    tv.config = (int)TwoViewConfig::Uncalibrated;
                    for (int p = 0; p < N; p++)
                        if (vis[i][p] && vis[j][p]) tv.matches.push_back({(uint32_t)p, (uint32_t)p, 0});
                    if (tv.matches.size() >= 15) db2.pairs.push_back(std::move(tv));
                }
        }
        Mapper mapper2(db2, feats2, opt2);
        std::vector<Reconstruction> ms = mapper2.run();
        uint32_t total = 0;
        for (const Reconstruction& m : ms) total += m.numRegistered();
        printf("  two disconnected components: %zu model(s), %u/%d images registered in total\n",
               ms.size(), total, 2 * M);
        if (ms.size() != 2 || total != (uint32_t)(2 * M)) {
            printf("  FAIL: expected 2 models covering all %d images\n", 2 * M);
            fails++;
        }
        // 每模型完整对应一个分量，不能凭空跨分量建立约束。
        std::set<uint32_t> seen;
        bool disjoint = true, whole = true, sorted = true;
        for (size_t i = 0; i < ms.size(); i++) {
            if (i && ms[i].points3D.size() > ms[i - 1].points3D.size()) sorted = false;
            std::set<int> comps;
            for (const auto& kv : ms[i].images) {
                if (!seen.insert(kv.first).second) disjoint = false;
                comps.insert(kv.first / M);
            }
            if (comps.size() != 1 || ms[i].images.size() != (size_t)M) whole = false;
        }
        if (!disjoint) { printf("  FAIL: models share images\n"); fails++; }
        if (!whole) { printf("  FAIL: a model straddles the two components\n"); fails++; }
        if (!sorted) { printf("  FAIL: models not ordered by point count\n"); fails++; }

        // 装配不能合并或跨越真正独立的分量（D44）。
        {
            ManagerOptions mo;
            mo.verbose = false;
            AssembleOptions ao;
            ao.verbose = false;
            ao.max_rounds = 2;
            AssembleStats ast;
            std::vector<Reconstruction> done = assembleModels(mapper2, ms, mo, ao, ast);
            uint32_t tot = 0;
            for (const Reconstruction& m : done) tot += m.numRegistered();
            printf("  assembly of two separate components: %zu model(s), %u images, %zu merge(s)\n",
                   done.size(), tot, ast.merges);
            if (done.size() != 2 || tot != (uint32_t)(2 * M) || ast.merges != 0) {
                printf("  FAIL: assembly should have left two separate components alone\n");
                fails++;
            }
        }

        // 共享相机组的两分量中故意扰乱一个焦距，联合 BA 应借另一分量证据恢复一致（D45）。
        {
            std::vector<Reconstruction> two = ms;
            const double good = two[0].cameras.at(1).focal();
            two[1].cameras[1].setFocal(good * 0.7);
            const double before = two[1].cameras.at(1).focal();
            mapper2.jointRefine(two);
            const double a = two[0].cameras.at(1).focal(), b = two[1].cameras.at(1).focal();
            printf("  joint BA with shared intrinsics: focals %.1f / %.1f -> %.1f / %.1f "
                   "(truth %.1f)\n", good, before, a, b, good);
            if (std::fabs(a - b) > 1e-6 * good || std::fabs(a - good) > 0.05 * good) {
                printf("  FAIL: joint BA did not share one focal across the components\n");
                fails++;
            }
        }

        // 将正确模型半数相机整体旋转，内部两半各自一致但跨缝验证边失效，拆分应仅凭独立对应图识别（D45）。
        {
            Reconstruction broken = ms[0];
            uint32_t moved = 0, kept = 0;
            std::vector<uint32_t> ids;
            for (const auto& kv : broken.images)
                if (kv.second.registered) ids.push_back(kv.first);
            const double a35 = 35.0 * M_PI / 180.0;
            Mat3 spin = {std::cos(a35), 0, std::sin(a35), 0, 1, 0,
                         -std::sin(a35), 0, std::cos(a35)};
            for (size_t i = 0; i < ids.size(); i++) {
                if (i * 2 < ids.size()) { kept++; continue; }
                // 仅对一半施加世界旋转，保留半内一致性并破坏跨半关系。
                Image& im = broken.images[ids[i]];
                im.pose.R = mul(im.pose.R, transpose(spin));
                moved++;
            }
            Mapper::SplitStats ss;
            std::vector<Reconstruction> parts =
                mapper2.splitInconsistent(broken, 8.0, 0.5, 15, 2, &ss);
            std::vector<uint32_t> sizes;
            for (const Reconstruction& p : parts) sizes.push_back(p.numRegistered());
            std::sort(sizes.begin(), sizes.end(), std::greater<uint32_t>());
            printf("  split a model with %u of %zu images rotated 35 deg: %zu/%zu inner pairs "
                   "hold -> %zu part(s)", moved, ids.size(), ss.pairs_agree, ss.pairs_tested,
                   parts.size());
            for (uint32_t s : sizes) printf(" %u", s);
            printf("\n");
            bool ok = parts.size() == 2 && sizes.size() == 2 &&
                      sizes[0] == std::max(moved, kept) && sizes[1] == std::min(moved, kept);
            if (!ok) { printf("  FAIL: split did not separate the rotated half\n"); fails++; }
            // 正确模型必须保持不变。
            Mapper::SplitStats ok_ss;
            std::vector<Reconstruction> whole_parts =
                mapper2.splitInconsistent(ms[0], 8.0, 0.5, 15, 2, &ok_ss);
            printf("  split a sound model: %zu/%zu inner pairs hold -> %zu part(s)\n",
                   ok_ss.pairs_agree, ok_ss.pairs_tested, whole_parts.size());
            if (whole_parts.size() != 1) {
                printf("  FAIL: split broke up a model that agrees with its own pairs\n");
                fails++;
            }
        }

        // max-models=1 应保持单模型行为。
        MapperOptions opt1 = opt2;
        opt1.max_num_models = 1;
        std::vector<Reconstruction> one = Mapper(db2, feats2, opt1).run();
        printf("  --max-models 1 on the same input: %zu model(s), %u images\n", one.size(),
               one.empty() ? 0 : one.front().numRegistered());
        if (one.size() != 1 || one.front().numRegistered() != (uint32_t)M) {
            printf("  FAIL: max_num_models 1 did not collapse to one component\n");
            fails++;
        }
    }

    // ---------------- 外部驱动的继续增长与审查（D44）----------------
    // 用真值已知模型测试 continueFrom 和 audit 对外部模型的处理。
    {
        const uint32_t victim = 3;
        auto relRot = [](const Reconstruction& r, uint32_t a, uint32_t b) {
            return mul(r.images.at(b).pose.R, transpose(r.images.at(a).pose.R));
        };
        auto rotDiffDeg = [](const Mat3& A, const Mat3& B) {
            Mat3 D = mul(A, transpose(B));
            double tr = std::max(-1.0, std::min(1.0, (D[0] + D[4] + D[8] - 1) * 0.5));
            return std::acos(tr) * 180.0 / M_PI;
        };
        const Mat3 truth = relRot(rec, 0, victim);

        // 完全移除一张图像，再由 continueFrom 恢复配准。
        Reconstruction partial = rec;
        partial.images.erase(victim);
        for (auto it = partial.points3D.begin(); it != partial.points3D.end();) {
            auto& tr = it->second.track;
            tr.erase(std::remove_if(tr.begin(), tr.end(),
                                    [&](const TrackElement& e) { return e.image_id == victim; }),
                     tr.end());
            it = tr.size() < 2 ? partial.points3D.erase(it) : std::next(it);
        }
        Mapper::GrowStats gs;
        Reconstruction back = Mapper(db, feats, opt).continueFrom(partial, &gs);
        const bool got = back.images.count(victim) && back.images.at(victim).registered;
        double err = got ? rotDiffDeg(relRot(back, 0, victim), truth) : 1e9;
        printf("  continueFrom: %u/%u images -> %u (%u registered), pose error %.3f deg\n",
               partial.numRegistered(), M, back.numRegistered(), gs.registered, err);
        if (!got || gs.registered != 1 || err > 1.0) {
            printf("  FAIL: continueFrom did not put the missing image back correctly\n");
            fails++;
        }

        // 正确模型审查后必须不变，防止假阳性损失覆盖。
        Mapper::AuditStats clean;
        Mapper(db, feats, opt).audit(rec, &clean);
        printf("  audit of a sound model: %u checked, %u contradicted\n", clean.checked,
               clean.unsupported);
        if (clean.unsupported != 0) {
            printf("  FAIL: audit contradicted an image of a model that is correct\n");
            fails++;
        }

        // 错误位姿图像失去自身观测时，应由其余模型结构发现不一致。
        Reconstruction bad = partial;
        Image moved = rec.images.at(victim);
        moved.pose.R = mul(angleAxisToRotation({0.0, 0.6, 0.0}), moved.pose.R);
        moved.point3D_ids.assign(moved.points2D.size(), kInvalidPoint3D);
        bad.images[victim] = moved;
        Mapper::AuditStats as;
        Reconstruction fixed = Mapper(db, feats, opt).audit(bad, &as);
        const bool kept = fixed.images.count(victim) && fixed.images.at(victim).registered;
        double ferr = kept ? rotDiffDeg(relRot(fixed, 0, victim), truth) : 1e9;
        printf("  audit of a misplaced image: %u contradicted, %u repaired, pose error "
               "%.3f deg (was %.1f)\n", as.unsupported, as.reregistered, ferr,
               rotDiffDeg(relRot(bad, 0, victim), truth));
        if (as.unsupported != 1 || !kept || ferr > 1.0) {
            printf("  FAIL: audit did not detect and repair the misplaced image\n");
            fails++;
        }
    }

    // ---------------- 前向运动（D48）----------------
    // 模拟沿光轴行驶的相机，种子需最终允许前向基线；带真实畸变的合成镜头还要求从偏长 71% 的焦距猜测接近真值。
    {
        const int Wd = 1280, Hd = 400, Md = 14, Nd = 900;
        Camera Kd = Camera::defaultFor(1, Wd, Hd, 900, CamModel::OpenCV);
        Kd.k1 = -0.28;
        Kd.k2 = 0.07;

        std::mt19937 r2(7);
        std::uniform_real_distribution<double> ux(-9.0, 9.0), uy(-3.0, 3.0), uz(3.0, 42.0);
        std::normal_distribution<double> jit(0.0, 0.02), noise2(0.0, 0.3);
        std::vector<Vec3> pts2(Nd);
        for (Vec3& p : pts2) p = {ux(r2), uy(r2), uz(r2)};

        std::vector<Pose> gtd(Md);
        for (int c = 0; c < Md; c++) {
            // 沿 +Z 每帧前进 1.1 单位，并加入车辆姿态抖动。
            Vec3 C = {0, 0, 1.1 * c};
            Mat3 R = angleAxisToRotation({jit(r2), jit(r2), jit(r2) * 0.3});
            Vec3 t = mul(R, C);
            gtd[c] = {R, {-t.x, -t.y, -t.z}};
        }

        std::vector<FeatureSet> fd(Md);
        std::vector<std::vector<char>> vd(Md, std::vector<char>(Nd, 0));
        for (int c = 0; c < Md; c++) {
            fd[c].width = Wd;
            fd[c].height = Hd;
            fd[c].keypoints.resize(Nd);
            for (int p = 0; p < Nd; p++) {
                Vec3 pc = mul(gtd[c].R, pts2[p]) + gtd[c].t;
                Vec2 px = Kd.project(pc);
                if (pc.z > 0.5 && px.x > 0 && px.x < Wd && px.y > 0 && px.y < Hd) {
                    fd[c].keypoints[p] = {(float)(px.x + noise2(r2)), (float)(px.y + noise2(r2)),
                                          2, 0, 0};
                    vd[c][p] = 1;
                } else {
                    fd[c].keypoints[p] = {-1000, -1000, 2, 0, 0};
                }
            }
        }
        MatchesDatabase dbd;
        dbd.images.resize(Md);
        for (int c = 0; c < Md; c++) dbd.images[c] = {"f" + std::to_string(c), (uint32_t)Nd};
        for (int i = 0; i < Md; i++)
            for (int j = i + 1; j < Md; j++) {
                TwoViewMatches tv;
                tv.image1 = i;
                tv.image2 = j;
                tv.config = (int)TwoViewConfig::Uncalibrated;
                for (int p = 0; p < Nd; p++)
                    if (vd[i][p] && vd[j][p]) tv.matches.push_back({(uint32_t)p, (uint32_t)p, 0});
                if (tv.matches.size() >= 15) dbd.pairs.push_back(std::move(tv));
            }

        MapperOptions od;
        od.verbose = opt.verbose;
        od.device = opt.device;
        od.camera_model = CamModel::OpenCV;
        od.focal = 0;  // 默认猜测 1.2*最大尺寸为 1536，真实焦距 900

        Mapper md(dbd, fd, od);
        std::vector<Reconstruction> mds = md.run();
        const Reconstruction& rd2 = mds.front();
        const uint32_t regd = rd2.numRegistered();
        const double fd_est = rd2.cameras.count(1) ? rd2.cameras.at(1).focal() : 0;
        const double guess = 1.2 * Wd;
        printf("  forward-motion capture: %u/%d images, focal %.0f (guess %.0f, truth 900)\n",
               regd, Md, fd_est, guess);
        if (regd < (uint32_t)Md) {
            printf("  FAIL: a forward-motion capture must still seed and grow\n");
            fails++;
        }
        // 误差应显著小于初值的 71%，且朝正确方向改善。
        if (std::fabs(fd_est - 900.0) > 0.20 * 900.0) {
            printf("  %s: focal not recovered from a rotation-degenerate capture\n",
                   fp64_ba ? "FAIL" : "df-limited");
            if (fp64_ba) fails++;
        }
        // 焦距初始化不能比禁用更差；本合成场景 900 点、0.3 px 噪声及强 k1，BA 自身可恢复，作为对照。
        // 真实行车数据中搜索将偏长 54% 改善到 1.5%，无需把合成测试调到不稳定恢复边界。
        MapperOptions oo = od;
        oo.focal_trials = 0;
        Mapper mo(dbd, fd, oo);
        std::vector<Reconstruction> mos = mo.run();
        const double f_off = mos.front().cameras.count(1) ? mos.front().cameras.at(1).focal() : 0;
        printf("  ... with the search off: focal %.0f over %u image(s)\n", f_off,
               mos.front().numRegistered());
        if (std::fabs(fd_est - 900.0) > std::fabs(f_off - 900.0) + 0.01 * 900.0) {
            printf("  FAIL: the focal bootstrap made a recoverable focal worse\n");
            fails++;
        }
    }

    // ---------------- 仅固定主点（D50）----------------
    // 主点故意偏 30 px，默认应逐位固定但仍优化焦距；开启主点优化后应向真值恢复，验证自由参数前缀机制。
    {
        Camera c0 = Camera::defaultFor(1, W, H, 1150.0, CamModel::OpenCV);
        c0.cx = W * 0.5 + 30;
        c0.cy = H * 0.5 - 30;

        MapperOptions op = opt;
        op.camera_model = CamModel::OpenCV;
        op.initial_cameras[1] = c0;
        op.focal_trials = 0;  // 隔离 BA 行为，不让焦距搜索影响测试

        Mapper mp(db, feats, op);
        std::vector<Reconstruction> mps = mp.run();
        const Camera& cp = mps.front().cameras.at(1);

        MapperOptions oq = op;
        oq.refine_principal_point = true;
        oq.pp_min_images = 4;   // 本组八张图，默认主点优化要求二十张
        Mapper mq(db, feats, oq);
        std::vector<Reconstruction> mqs = mq.run();
        const Camera& cq = mqs.front().cameras.at(1);

        // 降低共享图像门槛后，完整模型的主点释放精化应恢复真值（D51）。
        MapperOptions ol = op;
        ol.pp_min_images = 4;
        Mapper ml(db, feats, ol);
        std::vector<Reconstruction> mls = ml.run();
        Reconstruction polished = ml.polish(mls.front());
        const Camera& cl = polished.cameras.at(1);

        const bool held = cp.cx == c0.cx && cp.cy == c0.cy;
        const bool moved = cq.cx != c0.cx || cq.cy != c0.cy;
        // 固定主点不能同时冻结焦距。
        const bool focal_ok = std::fabs(cp.focal() - 1200.0) < 0.05 * 1200.0;
        // 最终优化须接近图像中心真值，而非仅产生任意移动。
        const double d0 = std::hypot(c0.cx - W * 0.5, c0.cy - H * 0.5);
        const double dl = std::hypot(cl.cx - W * 0.5, cl.cy - H * 0.5);
        printf("  principal point: held (%.1f,%.1f) vs start (%.1f,%.1f), focal %.0f; "
               "refined -> (%.1f,%.1f); final pass -> (%.1f,%.1f), %.1f px from truth "
               "(started %.1f)\n",
               cp.cx, cp.cy, c0.cx, c0.cy, cp.focal(), cq.cx, cq.cy, cl.cx, cl.cy, dl, d0);
        if (!held) { printf("  FAIL: BA moved a held principal point\n"); fails++; }
        if (!moved) { printf("  FAIL: --refine-principal-point did not free it\n"); fails++; }
        if (dl > 0.5 * d0) {
            printf("  %s: the final principal-point pass did not recover it\n",
                   fp64_ba ? "FAIL" : "df-limited");
            if (fp64_ba) fails++;
        }
        // 图像数不足的共享组保持原样。
        Reconstruction unpolished = mp.polish(mps.front());
        if (unpolished.cameras.at(1).cx != c0.cx) {
            printf("  FAIL: polished a camera group below pp_min_images\n");
            fails++;
        }
        // 多相机组模型不执行主点收尾，避免 rig 相对朝向漂移（D51）。
        {
            MapperOptions orig = ol;
            Reconstruction two = mls.front();
            Camera c2 = two.cameras.at(1);
            c2.id = 2;
            two.cameras[2] = c2;
            uint32_t n = 0;
            for (auto& kv : two.images)
                if (kv.second.registered && n++ % 2) kv.second.camera_id = 2;
            Mapper mr(db, feats, orig);
            Reconstruction rr2 = mr.polish(two);
            if (rr2.cameras.at(1).cx != two.cameras.at(1).cx ||
                rr2.cameras.at(2).cx != two.cameras.at(2).cx) {
                printf("  FAIL: the final pass touched a multi-group model\n");
                fails++;
            }
        }
        if (!focal_ok) { printf("  FAIL: focal did not converge with the PP held\n"); fails++; }
    }

    // ---------------- 建图固定畸变，收尾恢复（D72）----------------
    // 同一弧形场景也测试逐图独立内参，将共享八图相机拆为八个（D73）。
    {
        Camera Kx = Camera::defaultFor(1, W, H, 1200, CamModel::OpenCV);
        Kx.k1 = -0.06;
        Kx.k2 = 0.01;

        std::mt19937 r4(29);
        std::normal_distribution<double> n4(0.0, 0.3);
        std::vector<FeatureSet> fx(M);
        std::vector<std::vector<char>> vx(M, std::vector<char>(N, 0));
        for (int c = 0; c < M; c++) {
            fx[c].width = W;
            fx[c].height = H;
            fx[c].keypoints.resize(N);
            for (int p = 0; p < N; p++) {
                Vec3 pc = mul(gt[c].R, pts[p]) + gt[c].t;
                Vec2 px = Kx.project(pc);
                if (pc.z > 0.1 && px.x > 0 && px.x < W && px.y > 0 && px.y < H) {
                    fx[c].keypoints[p] = {(float)(px.x + n4(r4)), (float)(px.y + n4(r4)), 2, 0, 0};
                    vx[c][p] = 1;
                } else {
                    fx[c].keypoints[p] = {-1000, -1000, 2, 0, 0};
                }
            }
        }
        MatchesDatabase dbx;
        dbx.images.resize(M);
        for (int c = 0; c < M; c++) dbx.images[c] = {"dis" + std::to_string(c), (uint32_t)N};
        for (int i = 0; i < M; i++)
            for (int j = i + 1; j < M; j++) {
                TwoViewMatches tv;
                tv.image1 = i;
                tv.image2 = j;
                tv.config = (int)TwoViewConfig::Uncalibrated;
                for (int p = 0; p < N; p++)
                    if (vx[i][p] && vx[j][p]) tv.matches.push_back({(uint32_t)p, (uint32_t)p, 0});
                if (tv.matches.size() >= 15) dbx.pairs.push_back(std::move(tv));
            }

        MapperOptions ox = opt;
        ox.camera_model = CamModel::OpenCV;
        ox.focal_trials = 0;
        ox.refine_extra_params = false;
        Mapper mx(dbx, fx, ox);
        std::vector<Reconstruction> mxs = mx.run();
        const Camera& ch = mxs.front().cameras.at(1);
        const bool held = ch.k1 == 0 && ch.k2 == 0 && ch.p1 == 0 && ch.p2 == 0;

        Reconstruction fin = mx.polish(mxs.front(), /*free_pp=*/false, /*free_extra=*/true);
        const Camera& cf = fin.cameras.at(1);
        printf("  distortion: held (%.4f, %.4f) over %u image(s); finishing pass -> "
               "(%.4f, %.4f), truth (%.4f, %.4f)\n",
               ch.k1, ch.k2, mxs.front().numRegistered(), cf.k1, cf.k2, Kx.k1, Kx.k2);
        if (!held) { printf("  FAIL: BA moved a held distortion coefficient\n"); fails++; }
        // k1 至少恢复一半，它不是可由旋转规范免费吸收的参数。
        if (std::fabs(cf.k1 - Kx.k1) > 0.5 * std::fabs(Kx.k1)) {
            printf("  %s: the finishing pass did not recover the distortion\n",
                   fp64_ba ? "FAIL" : "df-limited");
            if (fp64_ba) fails++;
        }
        // 固定畸变不能冻结焦距。
        if (std::fabs(ch.focal() - 1200.0) > 0.10 * 1200.0) {
            printf("  FAIL: focal did not converge with the distortion held\n");
            fails++;
        }

        Reconstruction per = mx.perImageIntrinsics(fin);
        std::set<uint32_t> percam;
        for (const auto& kv : per.images)
            if (kv.second.registered) percam.insert(kv.second.camera_id);
        double spread = 0;
        for (const auto& kv : per.cameras)
            spread = std::max(spread, std::fabs(kv.second.focal() - cf.focal()));
        printf("  per-image intrinsics: %zu camera(s) over %u image(s), focal spread %.2f px\n",
               percam.size(), per.numRegistered(), spread);
        if (percam.size() != per.numRegistered() || per.cameras.size() != percam.size()) {
            printf("  FAIL: the per-image pass did not give every image its own camera\n");
            fails++;
        }
        if (per.numRegistered() != fin.numRegistered()) {
            printf("  FAIL: the per-image pass lost images\n");
            fails++;
        }
        // 逐图自由内参应改善拟合，但共享真值场景仍应保持参数接近，避免发散。
        if (!(spread < 0.05 * cf.focal())) {
            printf("  %s: per-image focals ran away from the shared solution\n",
                   fp64_ba ? "FAIL" : "df-limited");
            if (fp64_ba) fails++;
        }
    }

    // ---------------- 等距柱状全景（D49）----------------
    // 直线运动中覆盖后方与侧向点，验证全向可见性及前向运动非退化；图像宽高仅为固定元数据，BA 必须逐位保持。
    {
        const int We = 2048, He = 1024, Me = 10, Ne = 500;
        Camera Ke = Camera::defaultFor(1, We, He, 0, CamModel::Equirect);

        std::mt19937 r3(23);
        std::normal_distribution<double> gs(0.0, 1.0), noise3(0.0, 0.4);
        std::uniform_real_distribution<double> ur(4.0, 14.0), uy3(-2.0, 2.0);

        // 点分布于整条轨迹外围壳层，每站均可看到所有方向。
        std::vector<Vec3> pts3(Ne);
        for (Vec3& p : pts3) {
            Vec3 d = {gs(r3), 0.35 * gs(r3), gs(r3)};
            double n = std::max(1e-9, std::sqrt(d.dot(d)));
            double r = ur(r3);
            p = {d.x / n * r, d.y / n * r + uy3(r3), d.z / n * r + 4.5};
        }

        std::vector<Pose> gte(Me);
        for (int c = 0; c < Me; c++) {
            Vec3 C = {0.15 * c, 0.05 * std::sin(0.9 * c), 1.0 * c};
            // 加入全景相机携带时的偏航漂移，同时验证位置和方向。
            Mat3 R = angleAxisToRotation({0.03 * std::sin(0.7 * c), 0.25 * c, 0.02 * c});
            Vec3 t = mul(R, C);
            gte[c] = {R, {-t.x, -t.y, -t.z}};
        }

        std::vector<FeatureSet> fe3(Me);
        std::vector<std::vector<char>> ve(Me, std::vector<char>(Ne, 0));
        int behind = 0;
        for (int c = 0; c < Me; c++) {
            fe3[c].width = We;
            fe3[c].height = He;
            fe3[c].keypoints.resize(Ne);
            for (int p = 0; p < Ne; p++) {
                Vec3 pc = mul(gte[c].R, pts3[p]) + gte[c].t;
                Vec2 px = Ke.project(pc);
                fe3[c].keypoints[p] = {(float)(px.x + noise3(r3)), (float)(px.y + noise3(r3)),
                                       2, 0, 0};
                ve[c][p] = 1;
                if (pc.z <= 0) behind++;
            }
        }
        MatchesDatabase dbe;
        dbe.images.resize(Me);
        for (int c = 0; c < Me; c++) dbe.images[c] = {"s" + std::to_string(c), (uint32_t)Ne};
        for (int i = 0; i < Me; i++)
            for (int j = i + 1; j < Me; j++) {
                TwoViewMatches tv;
                tv.image1 = i;
                tv.image2 = j;
                tv.config = (int)TwoViewConfig::Uncalibrated;
                for (int p = 0; p < Ne; p++)
                    if (ve[i][p] && ve[j][p]) tv.matches.push_back({(uint32_t)p, (uint32_t)p, 0});
                if (tv.matches.size() >= 15) dbe.pairs.push_back(std::move(tv));
            }

        MapperOptions oe;
        oe.verbose = opt.verbose;
        oe.device = opt.device;
        oe.camera_model = CamModel::Equirect;
        oe.focal = 0;

        Mapper me(dbe, fe3, oe);
        std::vector<Reconstruction> mes = me.run();
        const Reconstruction& re = mes.front();
        const uint32_t rege = re.numRegistered();

        double maxRel = 0;
        for (const auto& a : re.images)
            for (const auto& b : re.images) {
                if (a.first >= b.first || !a.second.registered || !b.second.registered) continue;
                Mat3 D = mul(mul(b.second.pose.R, transpose(a.second.pose.R)),
                             transpose(mul(gte[b.first].R, transpose(gte[a.first].R))));
                double tr = (D[0] + D[4] + D[8] - 1) * 0.5;
                maxRel = std::max(maxRel,
                                  std::acos(std::max(-1.0, std::min(1.0, tr))) * 180.0 / M_PI);
            }
        const Camera& ce = re.cameras.at(1);
        const bool intr_fixed = ce.model == CamModel::Equirect &&
                                ce.fx == Ke.fx && ce.fy == Ke.fy &&
                                ce.cx == Ke.cx && ce.cy == Ke.cy;
        printf("  equirect capture: %u/%d images, max rel-rot %.3f deg, %d obs behind the "
               "camera, intrinsics %s\n",
               rege, Me, maxRel, behind, intr_fixed ? "unchanged" : "MOVED");
        if (rege < (uint32_t)Me) {
            printf("  FAIL: a spherical capture must register every station\n");
            fails++;
        }
        if (rege >= 2 && maxRel > 1.0) {
            printf("  FAIL: spherical pose accuracy\n");
            fails++;
        }
        if (!intr_fixed) {
            printf("  FAIL: bundle adjustment moved held-constant intrinsics\n");
            fails++;
        }
        if (behind == 0) {
            printf("  FAIL: the scene did not exercise the rear hemisphere\n");
            fails++;
        }
    }

    printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    return sfmTestMain(argc - 1, argv + 1, cmdMapSelftest);
}
