// GPU 匹配作为单线程生产者，主机工作池并行执行双视图 RANSAC，与匹配重叠；验证仍占匹配阶段约三分之一至三分之二。
// 结果按固定图像对索引收集，估计器可重入且逐调用固定随机种子，因此线程数量与调度不改变输出顺序或几何。
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "core/Env.h"
#include "sfm/core/Progress.h"
#include "sfm/core/Camera.h"
#include "sfm/core/Features.h"
#include "sfm/core/Matches.h"
#include "sfm/core/Cancel.h"
#include "sfm/core/Events.h"
#include "sfm/core/Log.h"
#include "sfm/core/Resume.h"
#include "sfm/core/PriorSource.h"
#include "sfm/geometry/KnownRotation.h"
#include "sfm/geometry/TwoView.h"
#include "i18n/catalog/Sfm.h"

namespace sfm {

// 缓存逐图像、逐关键点单位视线及生成时的相机，避免每图参与多对时重复 Newton 反解。
// 鱼眼必须在视线上验证，不能用针孔像素解释宽角方向；空图像条目表示沿用像素路径。
struct BearingCache {
    std::vector<Camera> cameras;              // 逐图像
    std::vector<std::vector<Vec3>> bearings;  // 逐图像、逐特征
    bool has(uint32_t i) const { return i < bearings.size() && !bearings[i].empty(); }
    size_t bytes() const {
        size_t n = 0;
        for (const auto& b : bearings) n += b.size() * sizeof(Vec3);
        return n;
    }
};

// 为需要宽角表示的相机预计算视线，普通镜头保留已有精确像素路径。
inline BearingCache precomputeBearings(const std::vector<FeatureSet>& feats,
                                       const std::vector<Camera>& cams,
                                       bool wide_only = true, int threads = 0) {
    BearingCache bc;
    bc.cameras = cams;
    bc.bearings.resize(feats.size());
    const size_t n = std::min(feats.size(), cams.size());
    // 逐图像写入独立槽位，并行计算数百万个关键点的 Newton 反解。
    const unsigned hc = std::thread::hardware_concurrency();
    int nt = threads > 0 ? threads : (hc > 0 ? (int)hc : 1);
    nt = std::max(1, std::min<int>(nt, (int)std::max<size_t>(n, 1)));
    std::atomic<size_t> next{0};
    auto worker = [&] {
        for (size_t i = next++; i < n; i = next++) {
            if (wide_only && !cams[i].wideFov()) continue;
            std::vector<Vec3>& out = bc.bearings[i];
            out.resize(feats[i].keypoints.size());
            for (size_t k = 0; k < out.size(); k++)
                out[k] = cams[i].bearing({feats[i].keypoints[k].x, feats[i].keypoints[k].y});
        }
    };
    if (nt == 1) {
        worker();
        return bc;
    }
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (int t = 0; t < nt; t++) pool.emplace_back(worker);
    for (std::thread& t : pool) t.join();
    return bc;
}

// 仅对样本匹配运行极线估计，统计焦距假设能解释的内点；不运行回答不同问题的单应性检查。
// 光轴附近视线对焦距不敏感，最终最大化两侧均位于半对角线 kPeripheralFrac 之外的周边内点数。
struct FocalScore {
    int total = 0;
    int peripheral = 0;
};
inline constexpr double kPeripheralFrac = 0.45;

inline FocalScore focalInliers(const std::vector<FeatureSet>& feats,
                               const std::vector<std::pair<uint32_t, uint32_t>>& sample,
                               const std::vector<std::vector<FeatureMatch>>& matches,
                               const std::vector<Camera>& cams, double focal,
                               const TwoViewOptions& tvopt, int threads) {
    std::atomic<int> total{0}, periph{0};
    std::atomic<size_t> next{0};
    auto worker = [&] {
        for (;;) {
            size_t s = next++;
            if (s >= sample.size()) return;
            const std::vector<FeatureMatch>& m = matches[s];
            if ((int)m.size() < tvopt.min_num_inliers) continue;
            uint32_t i = sample[s].first, j = sample[s].second;
            Camera c1 = cams[i], c2 = cams[j];
            c1.setFocal(focal);
            c2.setFocal(focal);
            const double r1 = kPeripheralFrac * 0.5 * std::hypot((double)feats[i].width,
                                                                 (double)feats[i].height);
            const double r2 = kPeripheralFrac * 0.5 * std::hypot((double)feats[j].width,
                                                                 (double)feats[j].height);
            int n = (int)m.size();
            std::vector<Vec3> b1(n), b2(n);
            std::vector<char> wide(n, 0);
            for (int k = 0; k < n; k++) {
                const Keypoint& a = feats[i].keypoints[m[k].idx1];
                const Keypoint& b = feats[j].keypoints[m[k].idx2];
                b1[k] = c1.bearing({a.x, a.y});
                b2[k] = c2.bearing({b.x, b.y});
                wide[k] = std::hypot(a.x - c1.cx, a.y - c1.cy) > r1 &&
                          std::hypot(b.x - c2.cx, b.y - c2.cy) > r2;
            }
            RansacOptions ro = tvopt.ransac;
            ro.max_error = tvopt.ransac.max_error *
                           (0.5 * (feats[i].pixelScale() + feats[j].pixelScale())) / focal;
            ro.max_num_trials = std::min(ro.max_num_trials, 2000);
            auto fit = [&](const std::vector<int>& s2) {
                return estimateEpipolar7Bearing(b1, b2, s2);
            };
            auto refit = [&](const std::vector<int>& s2) {
                return estimateEpipolar8Bearing(b1, b2, s2);
            };
            auto res = [&](const Mat3& E, int k) {
                return sampsonSqBearing(E, b1[k], b2[k]);
            };
            RansacReport<Mat3> rep = loransac<Mat3>(n, 7, fit, refit, res, ro);
            if (rep.num_inliers < tvopt.min_num_inliers) continue;
            int w = 0;
            for (int k = 0; k < n; k++)
                if (rep.inlier_mask[k] && wide[k]) w++;
            total += rep.num_inliers;
            periph += w;
        }
    };
    int nt = threads > 0 ? threads : (int)std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; t++) pool.emplace_back(worker);
    for (std::thread& t : pool) t.join();
    return {total, periph};
}

// 在半对角视场 55–175 度网格搜索鱼眼焦距，最大化周边极线内点；过长压平、过短扩散视线，真实值应位于窗口内部。
inline double bootstrapFocal(const std::vector<FeatureSet>& feats,
                             const std::vector<std::pair<uint32_t, uint32_t>>& sample,
                             const std::vector<std::vector<FeatureMatch>>& matches,
                             const std::vector<Camera>& cams, const TwoViewOptions& tvopt,
                             int threads, bool verbose) {
    if (sample.empty()) return 0;
    double diag = std::hypot((double)feats[sample[0].first].width,
                             (double)feats[sample[0].first].height);
    double best_f = 0;
    int best_n = -1;
    std::vector<std::pair<double, int>> curve;
    std::vector<int> totals;
    for (int step = 0; step <= 24; step++) {
        double half_fov = (55.0 + step * 5.0) * M_PI / 180.0;
        double f = 0.5 * diag / half_fov;  // 等距投影：r = f * theta
        FocalScore sc = focalInliers(feats, sample, matches, cams, f, tvopt, threads);
        curve.push_back({f, sc.peripheral});
        totals.push_back(sc.total);
        if (sc.peripheral > best_n) { best_n = sc.peripheral; best_f = f; }
    }
    // 在峰值与邻居间做抛物线插值，避免初始焦距受网格量化影响。
    for (size_t i = 1; i + 1 < curve.size(); i++) {
        if (curve[i].first != best_f) continue;
        double y0 = curve[i - 1].second, y1 = curve[i].second, y2 = curve[i + 1].second;
        double den = y0 - 2 * y1 + y2;
        if (den < 0) {
            double d = 0.5 * (y0 - y2) / den;
            if (std::fabs(d) < 1.0) {
                double lf = std::log(curve[i - 1].first), rf = std::log(curve[i + 1].first);
                best_f = std::exp(std::log(best_f) + d * 0.5 * (rf - lf));
            }
        }
        break;
    }
    if (verbose) {
        slog::out(slog::Tag::Match, spirula::i18n::msg::sfm::focal_search,
                 {(long long)sample.size(), slog::num(best_f, 1),
                  (long long)((0.5 * diag / best_f) * 180.0 / M_PI)});
        // 搜索曲线作为开发调试直方图输出。
        std::string line = "[focal] curve (f:peripheral/total):";
        for (size_t i = 0; i < curve.size(); i++) {
            char buf[64];
            snprintf(buf, sizeof buf, " %.0f:%d/%d", curve[i].first,
                     curve[i].second, totals[i]);
            line += buf;
        }
        slog::diag(slog::Tag::Match, "%s", line.c_str());
    }
    return best_f;
}

// 仅使用同组内图像对估计无给定先验的鱼眼焦距，避免跨组无法判断哪个镜头出错。
// 普通针孔错误焦距仅线性变换视线，仍保持秩二极线关系，不能通过内点数量搜索；鱼眼非线性扭曲才提供此信号。
inline void bootstrapGroupFocals(const std::vector<FeatureSet>& feats,
                                 const std::vector<uint32_t>& cam_ids,
                                 const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                                 const std::vector<std::vector<FeatureMatch>>& matches,
                                 std::map<uint32_t, Camera>& cams,
                                 const std::set<uint32_t>& given, std::set<uint32_t>& measured,
                                 const TwoViewOptions& tvopt,
                                 size_t max_sample, int threads, bool verbose) {
    std::map<uint32_t, std::vector<size_t>> by_group;
    for (size_t p = 0; p < pairs.size() && p < matches.size(); p++) {
        uint32_t a = pairs[p].first, b = pairs[p].second;
        if (a >= cam_ids.size() || b >= cam_ids.size() || cam_ids[a] != cam_ids[b]) continue;
        by_group[cam_ids[a]].push_back(p);
    }
    // 将组相机复制到逐图像视图，满足 focalInliers 的输入要求。
    std::vector<Camera> percam(feats.size());
    for (size_t i = 0; i < feats.size() && i < cam_ids.size(); i++) {
        auto it = cams.find(cam_ids[i]);
        if (it != cams.end()) percam[i] = it->second;
    }
    for (auto& kv : cams) {
        const uint32_t id = kv.first;
        if (!kv.second.isFisheye() || given.count(id)) continue;
        auto g = by_group.find(id);
        if (g == by_group.end() || g->second.size() < 8) {
            if (verbose)
                slog::diag(slog::Tag::Match,
                           "[focal] camera %u: %zu in-group pair(s), too few to search; "
                           "keeping f=%.1f",
                           id, g == by_group.end() ? 0 : g->second.size(), kv.second.focal());
            continue;
        }
        // 样本遍布组内列表，避免排序前缀仅覆盖局部场景。
        std::vector<std::pair<uint32_t, uint32_t>> sp;
        std::vector<std::vector<FeatureMatch>> sm;
        const size_t stride = std::max<size_t>(1, g->second.size() / std::max<size_t>(1, max_sample));
        for (size_t k = 0; k < g->second.size() && sp.size() < max_sample; k += stride) {
            sp.push_back(pairs[g->second[k]]);
            sm.push_back(matches[g->second[k]]);
        }
        if (verbose) slog::diag(slog::Tag::Match, "[focal] camera %u:", id);
        double f = bootstrapFocal(feats, sp, sm, percam, tvopt, threads, verbose);
        if (f > 0) {
            kv.second.setFocal(f);
            measured.insert(id);
            for (size_t i = 0; i < percam.size() && i < cam_ids.size(); i++)
                if (cam_ids[i] == id) percam[i].setFocal(f);
        }
    }
}

// ---------------- 普通镜头的基础矩阵焦距估计（D53）----------------
// 共享内参满足 E=K^T F K，本质矩阵奇异值为 (s,s,0)；搜索使前两奇异值接近的焦距，仅接受曲线有明确谷值且组内一致的投票。
// 沿光轴运动或平面场景退化时保留几何猜测；实测距最终 BA 焦距约 1.5–6%，略偏小，因此仅作可继续优化的初值，不当作固定先验。
struct FocalVoteOptions {
    double min_ratio = 0.25, max_ratio = 3.0;  // 围绕初始焦距的搜索窗口
    int samples = 41;                          // 窗口内对数均匀采样数
    // 谷底需比窗口不对称度中位数低足够比例，平坦曲线不参与投票。
    double min_contrast = 0.35;
    int min_votes = 8;
    // 投票一致性：绝对偏差中位数除以中位数
    double max_spread = 0.25;
};

struct FocalVoteReport {
    double focal = 0;     // 0 表示没有可信估计
    int votes = 0, pairs = 0;
    double spread = 1.0;  // 接受投票的 MAD/中位数
};

// K^T F K 的 |s1-s2|/(s1+s2)，为零时满足前两奇异值相等，无尺度量可跨焦距比较。
inline double essentialAsymmetry(const Mat3& F, double f, double cx, double cy) {
    // K=[[f,0,cx],[0,f,cy],[0,0,1]]，展开计算 K^T F K。
    Mat3 K = {f, 0, cx, 0, f, cy, 0, 0, 1};
    Svd3 s = svd3(mul(transpose(K), mul(F, K)));
    const double d = s.s.x + s.s.y;
    return d > 0 ? (s.s.x - s.s.y) / d : 1.0;
}

// 在对数焦距坐标中对谷底三点拟合抛物线，返回分数索引偏移；否则十二倍范围仅 41 点会产生约 6% 网格量化误差。
inline bool refineMinimum(const std::vector<double>& v, int& best_i, double& frac) {
    best_i = -1;
    double best = 1e30;
    for (int i = 0; i < (int)v.size(); i++)
        if (v[i] < best) { best = v[i]; best_i = i; }
    // 极小值位于窗口端点属于外推，不能视为可靠谷值。
    if (best_i <= 0 || best_i + 1 >= (int)v.size()) return false;
    const double y0 = v[best_i - 1], y1 = v[best_i], y2 = v[best_i + 1];
    const double den = y0 - 2 * y1 + y2;
    frac = 0.0;
    if (den > 0) {
        const double d = 0.5 * (y0 - y2) / den;
        if (std::fabs(d) < 1.0) frac = d;
    }
    return true;
}

// 由像素基础矩阵 F 得到单对在对数焦距窗口中的分数索引投票，曲线太平时返回 -1，guess 为窗口中心。
inline double focalVote(const Mat3& F, double guess, double cx, double cy,
                        const FocalVoteOptions& o) {
    const double lo = std::log(guess * o.min_ratio), hi = std::log(guess * o.max_ratio);
    const double step = (hi - lo) / (o.samples - 1);
    std::vector<double> vals;
    vals.reserve(o.samples);
    for (int i = 0; i < o.samples; i++)
        vals.push_back(essentialAsymmetry(F, std::exp(lo + step * i), cx, cy));
    int best_i = 0;
    double frac = 0;
    if (!refineMinimum(vals, best_i, frac)) return -1;
    std::vector<double> sorted(vals);
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const double med = sorted[sorted.size() / 2];
    if (med <= 0 || vals[best_i] > (1.0 - o.min_contrast) * med) return -1;
    return best_i + frac;
}

// 使用同组普通镜头的图像对估计焦距，pairs 与 matches 一一对应。
inline FocalVoteReport focalFromEpipolar(const std::vector<FeatureSet>& feats,
                                         const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                                         const std::vector<std::vector<FeatureMatch>>& matches,
                                         const Camera& cam, const TwoViewOptions& tvopt,
                                         const FocalVoteOptions& fo, int threads) {
    FocalVoteReport rep;
    rep.pairs = (int)pairs.size();
    std::mutex mtx;
    std::vector<double> votes;
    std::atomic<size_t> next{0};
    auto worker = [&] {
        std::vector<double> local;
        for (;;) {
            const size_t s = next++;
            if (s >= pairs.size()) break;
            const std::vector<FeatureMatch>& m = matches[s];
            if ((int)m.size() < tvopt.min_num_inliers) continue;
            const uint32_t i = pairs[s].first, j = pairs[s].second;
            const int n = (int)m.size();
            std::vector<Vec2> x1(n), x2(n);
            for (int k = 0; k < n; k++) {
                const Keypoint& a = feats[i].keypoints[m[k].idx1];
                const Keypoint& b = feats[j].keypoints[m[k].idx2];
                x1[k] = {a.x, a.y};
                x2[k] = {b.x, b.y};
            }
            RansacOptions ro = tvopt.ransac;
            ro.max_error = tvopt.ransac.max_error *
                           0.5 * (feats[i].pixelScale() + feats[j].pixelScale());
            ro.max_num_trials = std::min(ro.max_num_trials, 2000);
            auto fit = [&](const std::vector<int>& s2) { return estimateFundamental7(x1, x2, s2); };
            auto refit = [&](const std::vector<int>& s2) {
                return estimateFundamental8(x1, x2, s2);
            };
            auto res = [&](const Mat3& F, int k) { return sampsonSq(F, x1[k], x2[k]); };
            RansacReport<Mat3> r = loransac<Mat3>(n, 7, fit, refit, res, ro);
            if (r.num_inliers < std::max(30, tvopt.min_num_inliers)) continue;
            const double v = focalVote(r.model, cam.focal(), cam.cx, cam.cy, fo);
            if (v >= 0) local.push_back(v);
        }
        std::lock_guard<std::mutex> lk(mtx);
        votes.insert(votes.end(), local.begin(), local.end());
    };
    const int nt = threads > 0 ? threads : (int)std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (int t = 0; t < std::max(1, nt); t++) pool.emplace_back(worker);
    for (std::thread& t : pool) t.join();

    rep.votes = (int)votes.size();
    if (rep.votes < fo.min_votes) return rep;
    // 在对数焦距网格索引上计算投票中位数与 MAD，偏差直接表示相对离散度。
    std::nth_element(votes.begin(), votes.begin() + votes.size() / 2, votes.end());
    const double med = votes[votes.size() / 2];
    std::vector<double> dev;
    dev.reserve(votes.size());
    for (double v : votes) dev.push_back(std::fabs(v - med));
    std::nth_element(dev.begin(), dev.begin() + dev.size() / 2, dev.end());
    const double lo = std::log(cam.focal() * fo.min_ratio);
    const double step = (std::log(cam.focal() * fo.max_ratio) - lo) / (fo.samples - 1);
    rep.spread = std::exp(dev[dev.size() / 2] * step) - 1.0;
    if (rep.spread > fo.max_spread) return rep;
    rep.focal = std::exp(lo + step * med);
    return rep;
}

// 为各无焦距先验的普通镜头组估计，将可信结果加入 measured。
inline void bootstrapRectilinearFocals(const std::vector<FeatureSet>& feats,
                                       const std::vector<uint32_t>& cam_ids,
                                       const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                                       const std::vector<std::vector<FeatureMatch>>& matches,
                                       std::map<uint32_t, Camera>& cams,
                                       const std::set<uint32_t>& given,
                                       std::set<uint32_t>& measured,
                                       const TwoViewOptions& tvopt, size_t max_sample,
                                       int threads, bool verbose) {
    std::map<uint32_t, std::vector<size_t>> by_group;
    for (size_t p = 0; p < pairs.size() && p < matches.size(); p++) {
        const uint32_t a = pairs[p].first, b = pairs[p].second;
        if (a >= cam_ids.size() || b >= cam_ids.size() || cam_ids[a] != cam_ids[b]) continue;
        by_group[cam_ids[a]].push_back(p);
    }
    const FocalVoteOptions fo;
    for (auto& kv : cams) {
        const uint32_t id = kv.first;
        if (kv.second.isFisheye() || kv.second.isSpherical() || given.count(id)) continue;
        auto g = by_group.find(id);
        if (g == by_group.end() || (int)g->second.size() < fo.min_votes) continue;
        std::vector<std::pair<uint32_t, uint32_t>> sp;
        std::vector<std::vector<FeatureMatch>> sm;
        const size_t stride =
            std::max<size_t>(1, g->second.size() / std::max<size_t>(1, max_sample));
        for (size_t k = 0; k < g->second.size() && sp.size() < max_sample; k += stride) {
            sp.push_back(pairs[g->second[k]]);
            sm.push_back(matches[g->second[k]]);
        }
        FocalVoteReport r =
            focalFromEpipolar(feats, sp, sm, kv.second, tvopt, fo, threads);
        if (r.focal > 0) {
            if (verbose)
                slog::diag(slog::Tag::Match,
                           "[focal] camera %u: %.0f -> %.0f px from %d/%d epipolar vote(s), "
                           "spread %.0f%%",
                           id, kv.second.focal(), r.focal, r.votes, r.pairs, 100.0 * r.spread);
            kv.second.setFocal(r.focal);
            measured.insert(id);
        } else if (verbose) {
            slog::diag(slog::Tag::Match,
                       "[focal] camera %u: %d/%d epipolar vote(s)%s; keeping the f=%.0f guess",
                       id, r.votes, r.pairs,
                       r.votes >= fo.min_votes ? " disagree" : " -- too few", kv.second.focal());
        }
    }
}

struct VerificationOptions {
    TwoViewOptions two_view;
    // 两图均有视线时在球面验证；max_error 保持像素单位，再按各相机焦距换算。
    const BearingCache* bearings = nullptr;
    // RANSAC 线程数，0 使用全部核心，1 在调用线程直接验证。
    int num_threads = 0;
    // 候选队列上限为线程数乘此系数，防止慢验证器使匹配器缓存全部数据。
    int queue_depth_per_thread = 4;
    // 每次请求的匹配图像对数，应与匹配器批量一致以摊薄提交开销。
    int match_batch_pairs = 64;
    // 保存完成验证的恢复日志，空指针不写入。
    resume::MatchJournal* journal = nullptr;
    // 本调用在整个任务列表中的偏移，使恢复运行只处理缺失图像对但进度仍按完整阶段统计。
    size_t progress_done_base = 0;
    size_t progress_total = 0;   // 0 使用 pairs.size()
    // 有旋转先验时还执行固定旋转验证，解释自由估计内点的 prior_agree 比例时采用先验结果，需逐图像 cameras。
    const PriorSource* priors = nullptr;
    const std::vector<Camera>* cameras = nullptr;
    double prior_agree = 0.7;
};

// 本次验证的旋转先验统计。
struct VerifyPriorStats {
    std::atomic<uint64_t> pairs{0};      // 先验覆盖的图像对数
    std::atomic<uint64_t> kept{0};       // 采用先验内点集通过验证的对数
    std::atomic<uint64_t> disagreed{0};  // 先验解释不足的对数
    std::atomic<uint64_t> dropped{0};    // 自由估计保留而先验拒绝的匹配数
    std::atomic<uint64_t> rescued{0};    // 仅凭先验才能通过验证的对数
    std::atomic<uint64_t> contradicted{0};  // 旋转远离陀螺且缺少支持而丢弃的对数
};

inline int verificationThreadCount(const VerificationOptions& opt) {
    if (opt.num_threads > 0) return opt.num_threads;
    unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? (int)hc : 1;
}

// 匹配与进度回调仅在调用线程执行，工作池逐对验证并施加背压；GPU 批量由匹配器负责。
// 返回结果保持输入对顺序，只含满足最少内点和有效几何的图像对，putative_out 累计验证前匹配数。
inline std::vector<TwoViewMatches> verifyPairs(
    const std::vector<FeatureSet>& feats,
    const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
    const std::function<void(size_t, size_t, std::vector<std::vector<FeatureMatch>>&)>& matchFn,
    const VerificationOptions& opt, uint64_t* putative_out = nullptr,
    const std::function<void(size_t, size_t)>& progress = nullptr,
    VerifyPriorStats* prior_stats = nullptr) {

    const int nthreads = verificationThreadCount(opt);
    const TwoViewOptions& tv = opt.two_view;
    const bool priors = opt.priors && opt.cameras && opt.cameras->size() >= feats.size();
    // SS_SFM_PRIOR_DUMP=1 逐对输出先验与自由估计旋转，供比较诊断。
    const bool prior_dump = priors && spirula::env("SFM_PRIOR_DUMP") != nullptr;
    auto bearingOf = [&](uint32_t img, uint32_t k) {
        if (opt.bearings && opt.bearings->has(img)) return opt.bearings->bearings[img][k];
        return (*opt.cameras)[img].bearing({feats[img].keypoints[k].x, feats[img].keypoints[k].y});
    };

    // 每对独立结果槽，由领取任务的线程填写。
    std::vector<TwoViewMatches> results(pairs.size());
    std::vector<char> kept(pairs.size(), 0);
    uint64_t putative = 0;

    // 单个工作任务原地验证一对并返回保留内点。
    auto verifyBody = [&](size_t p, std::vector<FeatureMatch>& m,
                          uint32_t i, uint32_t j) -> uint32_t {
        if ((int)m.size() < tv.min_num_inliers) return 0;  // 少于内点下限
        // 阈值以提取像素定义，按各图缩放比例分别换到源像素（D47）。
        const double sc = 0.5 * (feats[i].pixelScale() + feats[j].pixelScale());
        TwoViewGeometry g;
        Mat3 R_prior;
        double prior_sigma = 0;
        if (priors && opt.priors->relativeRotation(i, j, R_prior, prior_sigma)) {
            // 自由估计提供基准和回退，固定旋转先验可排除重复场景或相机自身设备造成的伪匹配。
            std::vector<Vec3> b1(m.size()), b2(m.size());
            for (size_t k = 0; k < m.size(); k++) {
                b1[k] = bearingOf(i, m[k].idx1);
                b2[k] = bearingOf(j, m[k].idx2);
            }
            TwoViewOptions tvb = tv;
            const double f = 0.5 * ((*opt.cameras)[i].focal() + (*opt.cameras)[j].focal());
            tvb.ransac.max_error = tv.ransac.max_error * sc / std::max(1.0, f);
            tvb.recover_pose = true;
            g = estimateTwoViewBearing(b1, b2, tvb);
            KnownRotationOptions ko;
            ko.ransac = tvb.ransac;
            ko.min_num_inliers = tv.min_num_inliers;
            ko.max_rotation_only_ratio = tv.max_H_inlier_ratio;
            ko.rot_sigma = prior_sigma;
            ko.start = g.has_pose ? &g.pose : nullptr;
            const KnownRotationGeometry k = estimateTwoViewKnownRotation(b1, b2, R_prior, ko);
            const int free_inl = g.config == TwoViewConfig::Degenerate ||
                                         g.config == TwoViewConfig::Undefined
                                     ? 0
                                     : g.num_inliers;
            // 自由几何旋转明显偏离陀螺且固定估计无法重现时，说明匹配指向相机未朝向的位置，丢弃该对。
            double free_deg = -1;
            if (g.has_pose) {
                const Mat3 D = mul(g.pose.R, transpose(R_prior));
                const double tr = std::max(-1.0, std::min(1.0, (D[0] + D[4] + D[8] - 1.0) * 0.5));
                free_deg = std::acos(tr) * 180.0 / M_PI;
            }
            const double contradict_deg = std::max(10.0, 5.0 * prior_sigma * 180.0 / M_PI);
            const bool contradicted = free_inl && free_deg > contradict_deg &&
                                      (!k.ok || k.num_inliers < opt.prior_agree * (double)free_inl);
            if (prior_stats) prior_stats->pairs++;
            if (prior_dump)
                slog::diag(slog::Tag::Match,
                           "[prior] pair %u-%u: %zu matches, free %s %d inliers (rotation %.2f deg "
                           "from the prior), held: loose %d, strict %d, moved %.2f deg, sigma %.2f",
                           i, j, m.size(), twoViewConfigName(g.config), free_inl, free_deg,
                           k.loose_inliers, k.num_inliers, k.moved_deg,
                           prior_sigma * 180.0 / M_PI);
            if (contradicted) {
                if (prior_stats) prior_stats->contradicted++;
                return 0;
            }
            if (k.ok && k.num_inliers >= opt.prior_agree * (double)free_inl) {
                if (prior_stats) {
                    prior_stats->kept++;
                    if (free_inl > k.num_inliers) prior_stats->dropped += (uint64_t)(free_inl - k.num_inliers);
                    if (!free_inl) prior_stats->rescued++;
                }
                g.config = k.panoramic ? TwoViewConfig::PlanarOrPanoramic : TwoViewConfig::Uncalibrated;
                g.inlier_mask = k.inlier_mask;
                g.num_inliers = k.num_inliers;
            } else if (prior_stats && free_inl) {
                prior_stats->disagreed++;
            }
        } else if (opt.bearings && opt.bearings->has(i) && opt.bearings->has(j)) {
            const BearingCache& bc = *opt.bearings;
            std::vector<Vec3> b1(m.size()), b2(m.size());
            for (size_t k = 0; k < m.size(); k++) {
                b1[k] = bc.bearings[i][m[k].idx1];
                b2[k] = bc.bearings[j][m[k].idx2];
            }
            TwoViewOptions tvb = tv;
            double f = 0.5 * (bc.cameras[i].focal() + bc.cameras[j].focal());
            tvb.ransac.max_error = tv.ransac.max_error * sc / std::max(1.0, f);  // 像素 -> 弧度
            g = estimateTwoViewBearing(b1, b2, tvb);
        } else {
            std::vector<Vec2> q1(m.size()), q2(m.size());
            for (size_t k = 0; k < m.size(); k++) {
                const Keypoint& a = feats[i].keypoints[m[k].idx1];
                const Keypoint& b = feats[j].keypoints[m[k].idx2];
                q1[k] = {a.x, a.y};
                q2[k] = {b.x, b.y};
            }
            TwoViewOptions tvp = tv;
            tvp.ransac.max_error = tv.ransac.max_error * sc;  // 提取像素 -> 源图像素
            g = estimateTwoView(q1, q2, tvp);
        }
        if (g.config == TwoViewConfig::Degenerate || g.config == TwoViewConfig::Undefined) return 0;
        std::vector<FeatureMatch> inl;
        inl.reserve(g.num_inliers);
        for (size_t k = 0; k < m.size(); k++)
            if (g.inlier_mask[k]) inl.push_back(m[k]);
        if (inl.empty()) return 0;
        const uint32_t n_inl = (uint32_t)inl.size();
        results[p] = {i, j, (int)g.config, std::move(inl)};
        kept[p] = 1;
        return n_inl;
    };

    auto verifyOne = [&](size_t p, std::vector<FeatureMatch>& m) {
        const uint32_t i = pairs[p].first, j = pairs[p].second;
        // 成功和失败图像对均报告，使预览能区分验证失败与尚未处理；进度模块内部锁支持工作线程调用。
        const uint32_t inl = verifyBody(p, m, i, j);
        progress::pair(i, j, inl);
        if (inl) {
            const std::vector<FeatureMatch>& kept = results[p].matches;
            progress::live_pair(i, j, results[p].config,
                                &kept[0].idx1, &kept[0].idx2,
                                sizeof(FeatureMatch), (uint32_t)kept.size());
            if (opt.journal)
                opt.journal->record(i, j, results[p].config, (uint32_t)m.size(),
                                    &kept[0].idx1, &kept[0].idx2,
                                    sizeof(FeatureMatch), (uint32_t)kept.size());
        } else if (opt.journal) {
            // 失败对也写入恢复日志，避免下次重复验证，失败对往往更昂贵。
            opt.journal->record(i, j, 0, (uint32_t)m.size(), nullptr, nullptr, 4, 0);
        }
        Event e;
        e.kind = Event::Kind::PairVerified;
        e.stage = Stage::Match;
        e.image_a = i;
        e.image_b = j;
        e.inliers = inl;
        events::emit(e);
    };

    // 匹配批量独立于工作线程数，只限制 GPU 相对验证器可领先的程度。
    const size_t batch = std::max<size_t>(1, (size_t)opt.match_batch_pairs);
    std::vector<std::vector<FeatureMatch>> batch_out;

    // 整阶段只报告数百次进度，避免逐对争用全局锁，屏幕也无法展示数十万个独立更新。
    const size_t step = std::max<size_t>(1, pairs.size() / 400);
    const size_t ptotal = opt.progress_total ? opt.progress_total : pairs.size();
    auto tick = [&](size_t p) {
        if ((p + 1) % step == 0 || p + 1 == pairs.size())
            events::progress(Stage::Match, (int64_t)(opt.progress_done_base + p + 1),
                             (int64_t)ptotal);
    };

    if (nthreads <= 1) {
        for (size_t b = 0; b < pairs.size(); b += batch) {
            size_t e = std::min(b + batch, pairs.size());
            matchFn(b, e, batch_out);
            for (size_t p = b; p < e; p++) {
                std::vector<FeatureMatch>& m = batch_out[p - b];
                putative += m.size();
                verifyOne(p, m);
                if (progress) progress(p + 1, pairs.size());
                tick(p);
                cancel::check();
            }
        }
    } else {
        struct Job {
            size_t index;
            std::vector<FeatureMatch> matches;
        };
        std::queue<Job> queue;
        std::mutex mtx;
        std::condition_variable cv_job, cv_space;
        bool done_producing = false;
        const size_t max_queued = (size_t)nthreads * std::max(1, opt.queue_depth_per_thread);

        // 匹配器或工作线程抛异常后必须先汇合所有线程，再向调用方传播，避免 std::thread 析构触发终止。
        std::exception_ptr error;
        std::atomic<bool> failed{false};
        auto fail = [&](std::exception_ptr e) {
            std::lock_guard<std::mutex> lk(mtx);
            if (!error) error = e;
            failed = true;
        };

        std::vector<std::thread> workers;
        workers.reserve(nthreads);
        for (int t = 0; t < nthreads; t++) {
            workers.emplace_back([&] {
                for (;;) {
                    Job job;
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv_job.wait(lk, [&] { return !queue.empty() || done_producing; });
                        if (queue.empty()) return;  // 队列已清空且生产者已结束
                        job = std::move(queue.front());
                        queue.pop();
                    }
                    cv_space.notify_one();
                    if (failed) continue;  // 继续清空队列，避免生产者阻塞
                    try {
                        verifyOne(job.index, job.matches);
                    } catch (...) {
                        fail(std::current_exception());
                    }
                }
            });
        }

        try {
            for (size_t b = 0; b < pairs.size() && !failed; b += batch) {
                size_t e = std::min(b + batch, pairs.size());
                matchFn(b, e, batch_out);
                for (size_t p = b; p < e; p++) {
                    std::vector<FeatureMatch>& m = batch_out[p - b];
                    putative += m.size();
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv_space.wait(lk, [&] { return queue.size() < max_queued; });
                        queue.push({p, std::move(m)});
                    }
                    cv_job.notify_one();
                    if (progress) progress(p + 1, pairs.size());
                    tick(p);
                    if (cancel::requested() || failed) break;
                }
                if (cancel::requested()) break;
            }
        } catch (...) {
            fail(std::current_exception());
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            done_producing = true;
        }
        cv_job.notify_all();
        for (std::thread& w : workers) w.join();
        if (error) std::rethrow_exception(error);
    }
    cancel::check();   // 所有工作线程已汇合，可以安全传播异常

    if (putative_out) *putative_out = putative;

    std::vector<TwoViewMatches> out;
    for (size_t p = 0; p < pairs.size(); p++)
        if (kept[p]) out.push_back(std::move(results[p]));
    return out;
}

}  // 命名空间 sfm
