// SfM 各阶段的实现与 auto 调度顺序；CLI 与进程内前端共用同一流水线。
// 阶段均通过相同磁盘格式交互，可替换为 COLMAP 对应阶段以定位故障。

#include "sfm/Pipeline.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/ColorSpace.h"
#include "core/Env.h"
#include "core/ExrImage.h"
#include "sfm/core/Cancel.h"
#include "sfm/core/Events.h"
#include "sfm/core/Progress.h"
#include "sfm/core/CameraSetup.h"
#include "sfm/core/FeatureCompaction.h"
#include "sfm/core/Log.h"
#include "sfm/core/Features.h"
#include "sfm/core/Image.h"
#include "sfm/core/ImageLoader.h"
#include "sfm/core/Manifest.h"
#include "sfm/core/Mask.h"
#include "sfm/core/Matches.h"
#include "sfm/feature/Matcher.h"
#include "sfm/feature/PairSelection.h"
#include "sfm/feature/Pairing.h"
#include "sfm/feature/RigPairs.h"
#include "sfm/feature/GpsPairs.h"
#include "sfm/feature/Sift.h"
#include "sfm/feature/Verification.h"
#include "sfm/geometry/TwoView.h"
#include "sfm/map/Assemble.h"
#include "sfm/map/AttitudeGauge.h"
#include "sfm/map/Mapper.h"
#include "sfm/map/MetricGauge.h"
#include "sfm/map/Orient.h"
#include "sfm/map/SensorGauge.h"
#include "sfm/map/Merge.h"

#include "i18n/TimeFormat.h"
#include "i18n/catalog/Sfm.h"

namespace fs = std::filesystem;

namespace sfm {

namespace L = sfm::slog;
namespace M = spirula::i18n::msg::sfm;
using sfm::slog::Tag;
using spirula::i18n::format_duration;

bool isImageExt(const std::string& e) {
    std::string s;
    for (char c : e) s += (char)std::tolower((unsigned char)c);
    return s == ".jpg" || s == ".jpeg" || s == ".png" || s == ".bmp" || s == ".tga" ||
           s == ".ppm" || s == ".pgm" || s == ".exr";
}

// 忽略 macOS 在 exFAT/NTFS/SMB 上写入的 ._<name> AppleDouble 附属文件；它们沿用图像或 .bin 扩展名，但内容并非对应格式。
bool isSidecar(const fs::path& p) {
    const std::string n = p.filename().string();
    return n.size() > 2 && n[0] == '.' && n[1] == '_';
}

// 只有 root 在 nested 之外没有图像时才将 DATASET 自动解释为 DATASET/images，避免丢失其他文件夹中的输入。
bool holdsImagesOutside(const fs::path& root, const fs::path& nested) {
    std::error_code walk, ec;
    for (auto it = fs::recursive_directory_iterator(
             root, fs::directory_options::follow_directory_symlink, walk);
         !walk && it != fs::recursive_directory_iterator(); it.increment(walk)) {
        if (it->is_directory(ec)) {
            if (fs::equivalent(it->path(), nested, ec)) it.disable_recursion_pending();
            continue;
        }
        if (it->is_regular_file(ec) && isImageExt(it->path().extension().string()) &&
            !isSidecar(it->path()))
            return true;
    }
    return false;
}

// 按枚举目录计算纯词法相对路径；fs::relative 会解析符号链接，可能得到 ../.. 并使特征写到输出目录之外。
fs::path relativeTo(const fs::path& p, const fs::path& root) {
    fs::path r = root;
    // 移除 dir/ 末尾空分量，避免 lexically_relative 将 dir/x.jpg 错算成 ../x.jpg。
    if (!r.empty() && r.filename().empty()) r = r.parent_path();
    return p.lexically_relative(r);
}

// 任意起点的实时时间秒数，用于 auto 阶段计时。
double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// 统计重建观测的平均与中位重投影误差，供 map 与 auto 共同报告。
void reprojStats(const Reconstruction& rec, const std::vector<FeatureSet>& feats,
                        double& mean, double& median, size_t& nobs) {
    std::vector<double> e;
    for (const auto& kv : rec.points3D)
        for (const TrackElement& t : kv.second.track) {
            auto img = rec.images.find(t.image_id);
            if (img == rec.images.end() || !img->second.registered) continue;
            auto cam = rec.cameras.find(img->second.camera_id);
            if (cam == rec.cameras.end()) continue;
            Vec3 pc = mul(img->second.pose.R, kv.second.xyz) + img->second.pose.t;
            if (pc.z <= 0) continue;
            Vec2 px = cam->second.project(pc);
            const Keypoint& k = feats[t.image_id].keypoints[t.point2D_idx];
            e.push_back(std::hypot(px.x - k.x, px.y - k.y));
        }
    nobs = e.size();
    mean = median = 0;
    if (e.empty()) return;
    double s = 0;
    for (double v : e) s += v;
    mean = s / e.size();
    std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
    median = e[e.size() / 2];
}

// 从已解码源图为各关键点采样 RGB，避免点云着色时再次解码；没有颜色缓冲时跳过。
void sampleFeatureColors(FeatureSet& fs, const GrayImage& img) {
    if (!img.hasColor()) return;
    fs.colors.resize((size_t)fs.count() * 3);
    for (uint32_t i = 0; i < fs.count(); i++)
        sampleColor(img, fs.keypoints[i].x, fs.keypoints[i].y, &fs.colors[(size_t)i * 3]);
}

// 颜色、掩码等解码坐标操作完成后，将特征映射到原图坐标并附加 EXIF 相机信息。
void finishFeatures(FeatureSet& fs, const GrayImage& img) {
    scaleKeypoints(fs, img.orig_width, img.orig_height);
    fs.exif_focal = exifFocalPx(img.exif, fs.width, fs.height);
    fs.exif_camera = exifCameraKey(img.exif, fs.width, fs.height);
    fs.exif_orientation = (uint8_t)img.exif.orientation;
}

// 公制拟合拒绝原因及相关数值，便于仅凭一行日志诊断。
std::string metricReason(const MetricFit& f) {
    switch (f.reason) {
        case MetricFail::Pairs:
            return spirula::i18n::format(M::metric_fail_pairs, {(long long)f.n});
        case MetricFail::Spread:
            return spirula::i18n::format(M::metric_fail_spread, {L::num(f.spread, 3)});
        case MetricFail::Inliers:
            return spirula::i18n::format(M::metric_fail_inliers,
                                {L::num(f.max_error, 3), (long long)f.inliers,
                                 (long long)f.n});
        case MetricFail::Collinear:
            return spirula::i18n::format(
                M::metric_fail_collinear,
                {L::num(100.0 * f.perp_frac, 2), L::num(100.0 * kMetricMinPerpFraction, 1),
                 L::num(f.perp_frac > 0 ? 1.0 / f.perp_frac : 0.0, 0)});
        case MetricFail::None: break;
    }
    return {};
}


SensorCaptures loadSensorCaptures(const SfmConfig& cfg, bool verbose) {
    SensorCaptures out;
    if (cfg.sensor_gauge == "none") return out;
    for (const TelemetryInput& in : cfg.telemetry_inputs) {
        Telemetry t;
        std::string err;
        if (!telemetry_read(in.path, t, err) || t.empty()) {
            L::warn(Tag::Orient, M::sensor_file_bad, {in.path, err.empty() ? "-" : err});
            continue;
        }
        const TelemetryCheck c = telemetry_check(t);
        auto lc = std::make_unique<LoadedCapture>();
        if (!lc->timeline.init(t, c, err)) {
            L::warn(Tag::Orient, M::sensor_file_bad, {in.path, err});
            continue;
        }
        lc->cap.prefix = in.prefix;
        lc->cap.path = in.path;
        lc->cap.camera = t.camera;
        lc->cap.fps = in.fps > 0 ? in.fps : t.video_fps;
        lc->cap.time_offset = in.time_offset;
        lc->cap.readout = t.frame_readout;
        lc->cap.timeline = &lc->timeline;
        L::out(Tag::Orient, M::sensor_file,
               {in.path, t.camera.empty() ? "?" : t.camera, L::num(c.gyro.rate_hz, 0),
                L::num(c.accel.rate_hz, 0), L::num(c.orientation.rate_hz, 0),
                (long long)c.gps_distinct, L::num(c.gps_path_m, 0)});
        if (!(lc->cap.fps > 0)) {
            L::warn(Tag::Orient, M::sensor_file_no_fps, {in.path});
            continue;
        }
        if (verbose)
            for (const std::string& w : c.warnings) L::err_raw(Tag::Orient, w);
        out.loaded.push_back(std::move(lc));
    }
    return out;
}


const spirula::i18n::Msg& extrinsicReason(ExtrinsicFail f) {
    switch (f) {
        case ExtrinsicFail::Frames: return M::sensor_calib_fail_frames;
        case ExtrinsicFail::Pairs: return M::sensor_calib_fail_pairs;
        case ExtrinsicFail::Disagree: return M::sensor_calib_fail_disagree;
        case ExtrinsicFail::NoStream: return M::sensor_calib_fail_nostream;
        case ExtrinsicFail::Degenerate: return M::sensor_calib_fail_degenerate;
        case ExtrinsicFail::None: break;
    }
    return M::sensor_calib_fail_pairs;
}

const spirula::i18n::Msg& noScaleReason(SensorNoScale r) {
    switch (r) {
        case SensorNoScale::NotAsked: return M::sensor_no_scale_not_asked;
        case SensorNoScale::ImuWeak: return M::sensor_no_scale_imu_weak;
        case SensorNoScale::GpsRefused: return M::sensor_no_scale_gps_refused;
        case SensorNoScale::Disagree: return M::sensor_no_scale_disagree;
        case SensorNoScale::NoSource:
        case SensorNoScale::None: break;
    }
    return M::sensor_no_scale_no_source;
}

void reportSensorGauge(size_t i, const SensorGaugeResult& r,
                       const std::vector<std::unique_ptr<LoadedCapture>>& caps, bool verbose) {
    const long long model = (long long)i;
    if (r.fail == SensorFail::NoFrames || r.fail == SensorFail::NoTelemetry) {
        L::err(Tag::Orient, M::sensor_declined, {model, M::sensor_fail_frames.get()});
        return;
    }
    if (r.frames_untimed > 0)
        L::err(Tag::Orient, M::sensor_untimed,
               {model, (long long)r.frames_untimed, (long long)(r.frames_untimed + r.frames_timed)});
    for (size_t c = 0; c < r.time_offsets.size() && c < caps.size(); c++)
        if (r.time_offsets[c].found)
            L::err(Tag::Orient, M::sensor_time_offset,
                   {caps[c]->cap.path, L::num(1000.0 * r.time_offsets[c].offset, 1),
                    (long long)r.time_offsets[c].pairs});
    for (const SensorGroupReport& g : r.groups) {
        const std::string name = g.name.empty() ? std::string(".") : g.name;
        if (!g.fit.ok) {
            L::err(Tag::Orient, M::sensor_calib_failed, {name, extrinsicReason(g.fit.reason).get()});
            continue;
        }
        if (verbose) {
            L::err(Tag::Orient, M::sensor_calib,
                   {name, (long long)g.fit.frames, L::num(g.fit.sig_rot_deg, 2),
                    L::num(g.fit.sig_grav_deg, 2)});
            if (g.triples > 0)
                L::err(Tag::Orient, M::sensor_calib_scale,
                       {name, L::num(g.scale, 6), L::num(100.0 * g.scale_sigma, 2),
                        (long long)g.triples, L::num(g.g_norm, 2), L::num(g.g_angle_deg, 1)});
        }
        if (g.fit.mirrored && !g.fit.degenerate)
            L::err(Tag::Orient, M::sensor_calib_mirrored, {name});
        if (g.fit.degenerate) L::err(Tag::Orient, M::sensor_calib_yaw_free, {name});
    }
    if (r.up_from_imu)
        L::out(Tag::Orient, M::sensor_up,
               {model, (long long)r.up.votes, L::num(r.up.spread_deg, 2), (long long)r.up.outliers});
    if (r.fail == SensorFail::NoUp) {
        L::err(Tag::Orient, M::sensor_declined, {model, M::sensor_fail_noup.get()});
        return;
    }
    if (r.scale_imu_sigma > 0) {
        int triples = 0;
        double g_norm = 0, g_angle = 0;
        for (const SensorGroupReport& g : r.groups) {
            triples += g.triples;
            if (g.triples > 0 && g.g_norm > 0) { g_norm = g.g_norm; g_angle = g.g_angle_deg; }
        }
        if (r.scale_from_imu || r.disagree)
            L::out(Tag::Orient, M::sensor_scale_imu,
                   {model, L::num(r.scale_imu, 6), L::num(100.0 * r.scale_imu_sigma, 2),
                    (long long)triples, L::num(g_norm, 2), L::num(g_angle, 1)});
        else
            L::err(Tag::Orient, M::sensor_scale_imu_weak,
                   {model, L::num(100.0 * r.scale_imu_sigma, 1), (long long)triples});
    }
    if (r.gps_frames > 0) {
        if (r.gps.ok)
            L::out(Tag::Orient, M::sensor_scale_gps,
                   {model, L::num(r.scale_gps, 6), L::num(100.0 * r.scale_gps_sigma, 2),
                    (long long)r.gps_frames, L::num(r.gps.max_error, 1), (long long)r.gps.inliers,
                    (long long)r.gps.n, L::num(r.gps.rms, 2)});
        else
            L::err(Tag::Orient, M::sensor_scale_gps_failed, {model, metricReason(r.gps)});
    }
    if (r.disagree)
        L::warn(Tag::Orient, M::sensor_disagree, {model, L::num(r.scale_imu, 6), L::num(r.scale_gps, 6)});
    if (!r.applied) return;
    if (!r.metric) {
        L::out(Tag::Orient, M::sensor_up_only,
               {model, L::num(r.T.scale, 4), noScaleReason(r.no_scale).get()});
        return;
    }
    const spirula::i18n::Msg& src = r.scale_from_imu && r.scale_from_gps ? M::sensor_src_both
                                    : r.scale_from_imu                    ? M::sensor_src_imu
                                                                          : M::sensor_src_gps;
    L::out(Tag::Orient, M::sensor_done,
           {model, src.get(), L::num(r.T.scale, 6), L::num(100.0 * r.scale_sigma, 2),
            L::num(r.tilt_sigma_deg, 2)});
}

// 依次尝试视频传感器、图像姿态、公制参考和默认朝向，为各模型确定坐标规范；请求公制坐标却未拟合成功时返回 false。

// ---------------- 传感器先验，见 sfm/map/SensorPriors.h ----------------

std::vector<Camera> perImageCameras(const CameraSetup& cs, size_t num_images) {
    std::vector<Camera> percam(num_images);
    for (size_t i = 0; i < num_images && i < cs.ids.size(); i++) {
        auto it = cs.cameras.find(cs.ids[i]);
        if (it != cs.cameras.end()) percam[i] = it->second;
    }
    return percam;
}

std::unique_ptr<TelemetryPriors> makeSensorPriors(const SfmConfig& cfg,
                                                  const SensorCaptures& sensors,
                                                  const MatchesDatabase& db,
                                                  const std::vector<uint32_t>& cam_ids) {
    if (sensors.empty() || !(cfg.sensor_verify || cfg.sensor_map || cfg.sensor_pairs))
        return nullptr;
    std::vector<std::string> names;
    names.reserve(db.images.size());
    for (const ImageEntry& im : db.images) names.push_back(im.name);
    SensorPriorOptions po;
    po.max_dt = cfg.sensor_max_dt;
    po.gps_max_error = cfg.metric_gps != "none" && cfg.metric_max_error > 0 ? cfg.metric_max_error : 5.0;
    po.gps_max_error_frac = cfg.metric_max_error_frac;
    po.verbose = !cfg.quiet;
    auto priors = std::make_unique<TelemetryPriors>(sensors.caps(), names, cam_ids, po);
    return priors->timedImages() ? std::move(priors) : nullptr;
}


// 先用可标定图像对的单位视线双视图几何估计相对旋转，再执行传感器源的手眼拟合。
void calibrateSensorPriorsFrom(TelemetryPriors& priors, const std::vector<FeatureSet>& feats,
                               const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                               const std::vector<std::vector<FeatureMatch>>& matches,
                               const std::vector<Camera>& cams, const TwoViewOptions& tvopt,
                               int threads, bool verbose, bool inliers_only) {
    std::vector<size_t> use;
    for (size_t p = 0; p < pairs.size() && p < matches.size(); p++)
        if (priors.calibrationPair(pairs[p].first, pairs[p].second) &&
            (int)matches[p].size() >= std::max(30, tvopt.min_num_inliers))
            use.push_back(p);
    // 在整段采集中分散选取 600 对，覆盖各轴旋转，足以达到约十分之一度精度；仅采开头一段会缺少运动激励。
    const size_t kMax = 600;
    if (use.size() > kMax) {
        std::vector<size_t> thin;
        const double step = (double)use.size() / (double)kMax;
        for (size_t k = 0; k < kMax; k++) thin.push_back(use[(size_t)(k * step)]);
        use.swap(thin);
    }
    std::vector<PairRotationObs> obs(use.size());
    std::vector<char> ok(use.size(), 0);
    std::atomic<size_t> next{0};
    auto worker = [&] {
        for (size_t k = next++; k < use.size(); k = next++) {
            const size_t p = use[k];
            const uint32_t i = pairs[p].first, j = pairs[p].second;
            const std::vector<FeatureMatch>& m = matches[p];
            std::vector<Vec3> b1(m.size()), b2(m.size());
            for (size_t q = 0; q < m.size(); q++) {
                b1[q] = cams[i].bearing({feats[i].keypoints[m[q].idx1].x, feats[i].keypoints[m[q].idx1].y});
                b2[q] = cams[j].bearing({feats[j].keypoints[m[q].idx2].x, feats[j].keypoints[m[q].idx2].y});
            }
            TwoViewOptions tvo = tvopt;
            tvo.recover_pose = true;
            // 已验证内点无需再次排除离群匹配，但仍须用单应性检查是否存在平移。
            const double sc = 0.5 * (feats[i].pixelScale() + feats[j].pixelScale());
            tvo.ransac.max_error = tvopt.ransac.max_error * sc /
                                   std::max(1.0, 0.5 * (cams[i].focal() + cams[j].focal()));
            if (inliers_only) tvo.ransac.max_num_trials = std::min(tvo.ransac.max_num_trials, 500);
            const TwoViewGeometry g = estimateTwoViewBearing(b1, b2, tvo);
            if (g.config != TwoViewConfig::Uncalibrated || !g.has_pose || g.num_inliers < 30) continue;
            obs[k] = {i, j, g.pose.R};
            ok[k] = 1;
        }
    };
    const unsigned hc = std::thread::hardware_concurrency();
    const int nt = std::max(1, std::min<int>(threads > 0 ? threads : (hc ? (int)hc : 1),
                                             (int)std::max<size_t>(use.size(), 1)));
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; t++) pool.emplace_back(worker);
    for (std::thread& t : pool) t.join();
    std::vector<PairRotationObs> kept;
    for (size_t k = 0; k < obs.size(); k++)
        if (ok[k]) kept.push_back(obs[k]);
    priors.calibrateFromPairs(kept);
    if (!verbose) return;
    for (size_t c = 0; c < priors.timeOffsets().size(); c++)
        if (priors.timeOffsets()[c].found)
            L::err(Tag::Match, M::sensor_time_offset,
                   {priors.groups().empty() ? std::string("?") : std::to_string(c),
                    L::num(1000.0 * priors.timeOffsets()[c].offset, 1),
                    (long long)priors.timeOffsets()[c].pairs});
    for (const SensorGroupState& g : priors.groups()) {
        if (g.ok)
            L::out(Tag::Match, M::sensor_prior_calib,
                   {g.name, (long long)g.pairs, L::num(g.fit.sig_rot_deg, 2)});
        else
            L::err(Tag::Match, M::sensor_prior_calib_failed,
                   {g.name, (long long)g.pairs, extrinsicReason(g.fit.reason).get()});
    }
}


void calibrateSensorPriors(TelemetryPriors& priors, const std::vector<FeatureSet>& feats,
                           const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                           const std::vector<std::vector<FeatureMatch>>& matches,
                           const std::vector<Camera>& cams, const TwoViewOptions& tvopt,
                           int threads, bool verbose) {
    calibrateSensorPriorsFrom(priors, feats, pairs, matches, cams, tvopt, threads, verbose, false);
}

void calibrateSensorPriorsFromDatabase(TelemetryPriors& priors, const MatchesDatabase& db,
                                       const std::vector<FeatureSet>& feats,
                                       const std::vector<Camera>& cams,
                                       const TwoViewOptions& tvopt, int threads, bool verbose) {
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    std::vector<std::vector<FeatureMatch>> matches;
    for (const TwoViewMatches& p : db.pairs) {
        if (p.config != (int)TwoViewConfig::Uncalibrated) continue;
        if (!priors.calibrationPair(p.image1, p.image2)) continue;
        pairs.emplace_back(p.image1, p.image2);
        matches.push_back(p.matches);
    }
    calibrateSensorPriorsFrom(priors, feats, pairs, matches, cams, tvopt, threads, verbose, true);
}


bool fixGauge(std::vector<Reconstruction>& models, const SfmConfig& cfg,
              const std::string& imagedir, bool verbose,
              std::vector<ModelGauge>& gauge, const SensorCaptures* sensors) {
    const bool gps = cfg.metric_gps != "none";
    const bool flat = cfg.metric_gps == "horizontal";
    const bool file = !cfg.metric_positions.empty();
    // 竖拍的向上方向与图像相差 90 度；apply 已旋转像素，仅 orient 需修正姿态。磁盘模型由 fillExifOrientations 补齐标签。
    const bool exif_up = cfg.exif_orientation == "orient";
    const bool on_ground = cfg.level == "ground";
    // 没有测量约束的模型先按相机朝向立正，再按可检测到的地面调平。
    auto unmeasured = [&](size_t i, GroundFit& g) {
        const Sim3 T = uprightTransform(models[i], exif_up);
        g = on_ground ? groundTransform(models[i], true, T) : GroundFit{};
        return g.found ? composeSim3(g.T, T) : T;
    };
    std::vector<GroundFit> levelled(models.size());

    // gauge[i] 同时记录有效状态；后续来源须先检查 oriented/metric，不能覆盖先前来源已确定的结果。
    gauge.assign(models.size(), ModelGauge());

    // ---------------- 视频自身的传感器 ----------------
    SensorCaptures own;
    if (!sensors) {
        own = loadSensorCaptures(cfg, verbose);
        sensors = &own;
    }
    const std::vector<std::unique_ptr<LoadedCapture>>& loaded = sensors->loaded;
    if (!loaded.empty()) {
        std::vector<SensorCapture> caps = sensors->caps();
        SensorGaugeOptions opt;
        opt.mode = cfg.sensor_gauge == "up" ? SensorMode::Up : SensorMode::Auto;
        opt.gps_full = cfg.metric_gps == "full";
        opt.gps_max_error = gps ? cfg.metric_max_error : 5.0;
        opt.gps_max_error_frac = cfg.metric_max_error_frac;
        opt.verbose = verbose;
        for (size_t i = 0; i < models.size(); i++) {
            const SensorGaugeResult r = fitSensorGauge(models[i], caps, opt);
            reportSensorGauge(i, r, loaded, verbose);
            if (!r.applied) continue;
            applySim3(models[i], r.T);
            gauge[i].oriented = true;
            gauge[i].up = "sensors";
            if (r.metric) {
                gauge[i].metric = true;
                gauge[i].scale = r.scale_from_imu && r.scale_from_gps ? "imu+gps"
                                 : r.scale_from_imu                   ? "imu"
                                                                     : "gps";
                gauge[i].scale_sigma = r.scale_sigma;
            }
        }
    }

    // ---------------- 图像记录的姿态 ----------------
    // 已由视频传感器调平的模型保持原结果；north 用于检查水平拟合的航向。
    std::vector<char> north(models.size(), 0);
    const bool attitude = cfg.orient && cfg.exif_attitude != "none" && !imagedir.empty();
    for (size_t i = 0; i < models.size() && attitude; i++) {
        if (gauge[i].oriented) continue;
        const AttitudeRef ref =
            attitudeRefFromImages(models[i], imagedir, cfg.exif_orientation == "apply");
        if (ref.image_ids.empty()) continue;
        const AttitudeFit fit = fitAttitudeGauge(models[i], ref, cfg.exif_attitude == "auto");
        const long long model = (long long)i;
        if (!fit.ok) {
            if (fit.reason == AttitudeFail::Disagree)
                L::warn(Tag::Orient, M::attitude_declined,
                        {model, (long long)fit.up.outliers, (long long)fit.up.votes});
            continue;
        }
        const double guess_deg = extrinsic_detail::angleDeg(
            fit.up.up, meanCameraUp(models[i], exif_up));
        L::out(Tag::Orient, M::attitude_up,
               {model, (long long)ref.image_ids.size(), (long long)ref.registered,
                L::num(fit.up.spread_deg, 2), (long long)fit.up.outliers, L::num(guess_deg, 1)});
        if (fit.north)
            L::out(Tag::Orient, M::attitude_north,
                   {model, (long long)fit.heading.votes, L::num(fit.heading.spread_deg, 2),
                    (long long)fit.heading.outliers});
        else if (fit.north_reason == AttitudeFail::Disagree)
            L::err(Tag::Orient, M::attitude_north_declined,
                   {model, (long long)fit.heading.outliers, (long long)fit.heading.votes});
        applySim3(models[i], fit.T);
        gauge[i].oriented = true;
        gauge[i].up = "attitude";
        north[i] = fit.north;
    }

    // ---------------- 外部公制参考 ----------------
    bool all = true;
    std::map<std::string, Vec3> positions;
    if (file) {
        std::string err;
        if (!readMetricPositions(cfg.metric_positions, positions, err)) {
            L::fail(Tag::Orient, M::metric_positions_bad, {cfg.metric_positions, err});
            all = false;
        }
    }
    for (size_t i = 0; i < models.size() && all && (gps || file); i++) {
        // 已由传感器确定公制坐标时不再应用第二个参考，避免相互矛盾。
        if (gauge[i].metric) continue;
        MetricRef ref;
        if (file) {
            const MetricPairCounts pc = pairMetricRef(models[i], positions, ref);
            L::out(Tag::Orient, M::metric_matched,
                   {(long long)pc.matched, (long long)(pc.matched + pc.unmatched_model),
                    (long long)pc.unmatched_file});
        } else {
            const MetricGpsCounts gc = metricRefFromGps(models[i], imagedir, ref);
            L::out(Tag::Orient, M::metric_gps_read,
                   {(long long)gc.matched, (long long)(gc.matched + gc.no_gps),
                    (long long)gc.no_alt});
        }
        // 水平模式沿用调用方的向上方向，只拟合尺度、航向和位置；传感器已调平时直接使用当前坐标系。
        const Sim3 pre = flat && !gauge[i].oriented ? unmeasured(i, levelled[i]) : Sim3{};
        for (Vec3& c : ref.centres) c = transformPoint(pre, c);
        const MetricFit fit =
            fitMetricGauge(ref, cfg.metric_max_error,
                           flat ? MetricAxes::Horizontal : MetricAxes::Full,
                           cfg.metric_max_error_frac);
        // 拟合被拒绝时保留输入坐标系；即使返回恒等变换，也不能据此宣称单位为米。
        if (!fit.ok) {
            all = false;
            L::warn(Tag::Orient, M::metric_failed, {(long long)i, metricReason(fit)});
            continue;
        }
        applySim3(models[i], composeSim3(fit.T, pre));
        gauge[i].metric = true;
        gauge[i].scale = file ? "positions" : "gps";
        gauge[i].scale_sigma = fit.scale_unc;
        // 水平拟合不改变倾斜，向上方向仍由 pre 中的传感器或相机结果决定。
        if (!flat) {
            gauge[i].oriented = true;
            gauge[i].up = file ? "positions" : "gps";
        } else if (levelled[i].found) {
            gauge[i].oriented = true;
            gauge[i].up = "ground";
        } else if (!gauge[i].oriented) {
            gauge[i].up = "cameras";
        }
        L::out(Tag::Orient, M::metric_done,
               {(long long)i,
                spirula::i18n::format(!gps    ? M::metric_source_positions
                                      : flat  ? M::metric_source_gps_flat
                                              : M::metric_source_gps,
                                      {}),
                L::num(fit.T.scale, 6), L::num(fit.max_error, 3), (long long)fit.inliers,
                (long long)fit.n, L::num(fit.rms, 4), L::num(fit.scale_unc, 3),
                L::num(fit.rot_unc_deg, 3)});
        if (!verbose) continue;
        if (flat && north[i])
            L::err(Tag::Orient, M::attitude_vs_gps,
                   {L::num(std::atan2(fit.T.R[3], fit.T.R[0]) * 180.0 / M_PI, 2)});
        // 漂移的高度偏差可能被旋转吸收为倾斜而保持较低 RMS，这两个量用于揭示该问题。
        double e = 0, n = 0, u = 0;
        for (size_t k = 0; k < ref.centres.size(); k++) {
            if (!fit.inlier_mask[k]) continue;
            const Vec3 r = ref.targets[k] - transformPoint(fit.T, ref.centres[k]);
            e += r.x * r.x;
            n += r.y * r.y;
            u += r.z * r.z;
        }
        const double m = std::max(fit.inliers, 1);
        L::err(Tag::Orient, M::metric_axes,
               {L::num(std::sqrt(e / m), 4), L::num(std::sqrt(n / m), 4),
                L::num(std::sqrt(u / m), 4),
                L::num(metricUpDisagreementDeg(models[i]), 2)});
    }

    // ---------------- 未被任何来源确定的规范 ----------------
    // 仅在此处回退到相机平均向上轴，避免用猜测覆盖有效测量。
    std::vector<char> placed(models.size(), 0);
    if (cfg.orient)
        for (size_t i = 0; i < models.size(); i++) {
            if (gauge[i].oriented || gauge[i].metric) continue;
            GroundFit g;
            const Sim3 T = unmeasured(i, g);
            applySim3(models[i], T);
            placed[i] = 1;
            const long long model = (long long)i;
            if (g.found) {
                gauge[i].oriented = true;
                gauge[i].up = "ground";
                L::out(Tag::Orient, M::orient_ground,
                       {model, (long long)std::lround(g.share * 100.0), L::num(T.scale, 4)});
                continue;
            }
            gauge[i].up = exif_up ? "cameras+exif" : "cameras";
            if (on_ground) L::out(Tag::Orient, M::orient_ground_missed, {model});
            else if (verbose) L::err(Tag::Orient, M::orient_done, {model, L::num(T.scale, 4)});
        }

    // 已测得向上方向时保持倾斜，仅允许调整高度；参考位置或 GPS 已测得高度时也保持高度。
    const bool height_measured = file || cfg.metric_gps == "full";
    if (cfg.orient && on_ground && !height_measured)
        for (size_t i = 0; i < models.size(); i++) {
            if (placed[i] || !gauge[i].oriented) continue;
            const GroundFit g = groundTransform(models[i], false);
            if (!g.found) continue;
            applySim3(models[i], g.T);
            L::out(Tag::Orient, M::orient_ground_height,
                   {(long long)i, (long long)std::lround(g.share * 100.0)});
        }
    return true;
}

// EXR 解码可沿用文件色彩空间，但 point-color 与日志需要显式有效值，因此在运行任何阶段前读取并补齐。
void adoptExrColorSpace(SfmConfig& cfg, const std::string& imagedir,
                        const std::set<std::string>& seen) {
    const bool take_gamut = !seen.count("image-gamut");
    const bool take_linear = !seen.count("image-linear");
    if (!take_gamut && !take_linear) return;
    std::error_code ec, walk;
    for (auto it = fs::recursive_directory_iterator(
             imagedir, fs::directory_options::follow_directory_symlink, walk);
         !walk && it != fs::recursive_directory_iterator(); it.increment(walk)) {
        if (!it->is_regular_file(ec)) continue;
        if (!isImageExt(it->path().extension().string()) || isSidecar(it->path()))
            continue;
        exr::Info info;
        if (!exr::declared_color_space(it->path().string(), info)) return;
        if (take_gamut) cfg.image_gamut = info.gamut;
        if (take_linear) cfg.image_is_linear = info.is_linear;
        const std::string name =
            cfg.image_gamut.empty() ? "Rec.709" : cfg.image_gamut;
        if (take_linear) L::out(Tag::Run, M::run_exr_color, {name});
        else             L::out(Tag::Run, M::run_exr_gamut_from_file, {name});
        if (take_gamut && !info.gamut_known)
            L::warn(Tag::Run, M::run_exr_gamut_unknown, {});
        return;
    }
}

void reportFeatureCompaction(const FeatureCompactionStats& stats) {
    const double removed_pct =
        stats.original_features ? 100.0 * stats.removedFeatures() / stats.original_features : 0.0;
    L::out(Tag::Map, M::map_feature_compaction,
           {(long long)stats.original_features, (long long)stats.compact_features,
            (long long)stats.removedFeatures(), L::num(removed_pct, 2), (long long)stats.images,
            (long long)stats.zero_feature_images, (long long)stats.pairs,
            (long long)stats.correspondences});
}

// 加载器以 sRGB 采样点颜色；srgb 保持原值，image 则转回照片空间，以匹配训练器 point_color_* 的默认约定。
void recolorPoints(std::vector<Reconstruction>& models, const SfmConfig& cfg) {
    if (cfg.point_color_space != "image") return;
    if (colorspace::is_identity(cfg.image_gamut, cfg.image_is_linear)) return;
    for (Reconstruction& m : models)
        for (auto& kv : m.points3D)
            colorspace::from_srgb_inplace(kv.second.rgb, 1, cfg.image_gamut,
                                          cfg.image_is_linear);
}

// 相机分组允许尺寸在 2% 内变化，但 COLMAP 每个相机对应固定尺寸；写出前按尺寸拆分相机，同时保持 BA 中共享的参数。
void splitCamerasBySize(std::vector<Reconstruction>& models,
                               const std::vector<FeatureSet>& feats) {
    uint32_t next = 0;
    for (const Reconstruction& rec : models)
        for (const auto& kv : rec.cameras) next = std::max(next, kv.first);
    // 各模型共用相机 ID 空间；模型可能重叠，合并时同 ID 沿用目标模型相机。
    std::map<std::pair<uint32_t, uint64_t>, uint32_t> key2id;
    std::set<uint32_t> kept;  // 首个尺寸已占用其 ID 的分组
    for (Reconstruction& rec : models) {
        std::map<uint32_t, Camera> cams;
        for (auto& kv : rec.images) {
            Image& im = kv.second;
            if (!im.registered) continue;
            auto c = rec.cameras.find(im.camera_id);
            if (c == rec.cameras.end()) continue;
            int w = c->second.width, h = c->second.height;
            if (im.id < feats.size() && feats[im.id].width > 0 && feats[im.id].height > 0) {
                w = feats[im.id].width;
                h = feats[im.id].height;
            }
            const std::pair<uint32_t, uint64_t> key{
                im.camera_id, ((uint64_t)(uint32_t)w << 32) | (uint32_t)h};
            auto it = key2id.find(key);
            if (it == key2id.end())
                // 分组的首个写出尺寸沿用原 ID，使单一尺寸数据保持原编号。
                it = key2id.emplace(
                    key, kept.insert(im.camera_id).second ? im.camera_id : ++next).first;
            if (!cams.count(it->second)) {
                Camera nc = c->second;
                nc.id = it->second;
                nc.width = w;
                nc.height = h;
                cams[nc.id] = nc;
            }
            im.camera_id = it->second;
        }
        if (!cams.empty()) rec.cameras = std::move(cams);
    }
}

// 模型旁的 gauge.txt 明确记录 +Z 是否朝上、单位是否为米，供用户与查看器共同读取。
void writeGauge(const fs::path& dir, const ModelGauge& g) {
    std::ofstream f(dir / "gauge.txt", std::ios::trunc);
    if (!f) return;
    f << "# What this model's frame means, from spirula sfm.\n";
    f << "oriented " << (g.oriented ? 1 : 0) << "\n";
    f << "metric " << (g.metric ? 1 : 0) << "\n";
    f << "up " << g.up << "\n";
    f << "scale " << g.scale << "\n";
    if (g.scale_sigma > 0) f << "scale_sigma " << g.scale_sigma << "\n";
}

// rigs.txt 保存各成员最终的 cam_from_rig，平移使用模型单位，可用于后续清单的 rotation/translation。
void writeRigs(const fs::path& dir, const Reconstruction& m, const RigTable* rigs) {
    if (!rigs || m.rigs.empty()) return;
    std::ofstream f(dir / "rigs.txt", std::ios::trunc);
    if (!f) return;
    f << "# Rig calibration from spirula sfm, cam_from_rig per member as\n"
         "# rig member reference qw qx qy qz tx ty tz frames spread_deg, in this model's units.\n";
    f.precision(12);
    for (size_t r = 0; r < rigs->rigs.size() && r < m.rigs.size(); r++) {
        const RigSpec& spec = rigs->rigs[r];
        const RigCalib& c = m.rigs[r];
        if (c.ref < 0) continue;
        for (size_t k = 0; k < spec.members.size() && k < c.cam_from_rig.size(); k++) {
            if (!c.established[k]) continue;
            const Quat q = rotationToQuaternion(c.cam_from_rig[k].R);
            const Vec3& t = c.cam_from_rig[k].t;
            f << spec.name << ' ' << spec.members[k].prefix << ' '
              << spec.members[(size_t)c.ref].prefix << ' ' << q[0] << ' ' << q[1] << ' ' << q[2]
              << ' ' << q[3] << ' ' << t.x << ' ' << t.y << ' ' << t.z << ' ' << c.support[k]
              << ' ' << c.spread_deg[k] << "\n";
        }
    }
}

RigTable readRigs(const fs::path& dir, Reconstruction& m) {
    std::ifstream f(dir / "rigs.txt");
    if (!f) return {};
    struct Row {
        std::string member, ref;
        Quat q;
        Vec3 t;
        uint32_t frames = 0;
        double spread = 0;
    };
    std::vector<std::string> order;
    std::map<std::string, std::vector<Row>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string rig;
        Row r;
        if (!(ss >> rig >> r.member >> r.ref >> r.q[0] >> r.q[1] >> r.q[2] >> r.q[3] >> r.t.x >>
              r.t.y >> r.t.z))
            throw std::runtime_error((dir / "rigs.txt").string() + ": cannot read '" + line + "'");
        ss >> r.frames >> r.spread;
        if (!rows.count(rig)) order.push_back(rig);
        rows[rig].push_back(r);
    }
    std::vector<RigDef> defs;
    for (const std::string& name : order) {
        RigDef d;
        d.name = name;
        for (const Row& r : rows[name]) d.members.push_back({r.member});
        defs.push_back(std::move(d));
    }
    uint32_t n = 0;
    for (const auto& kv : m.images) n = std::max(n, kv.first + 1);
    std::vector<std::string> names(n);
    for (const auto& kv : m.images) names[kv.first] = kv.second.name;
    RigTable rigs = buildRigTable(names, defs);
    m.rigs.assign(rigs.rigs.size(), RigCalib{});
    for (size_t r = 0; r < order.size(); r++) {
        const std::vector<Row>& rs = rows[order[r]];
        RigCalib& c = m.rigs[r];
        c.resize(rs.size());
        for (size_t k = 0; k < rs.size(); k++) {
            c.cam_from_rig[k] = {quaternionToRotation(rs[k].q), rs[k].t};
            c.established[k] = 1;
            c.support[k] = rs[k].frames;
            c.spread_deg[k] = rs[k].spread;
            if (rs[k].member == rs[k].ref) c.ref = (int)k;
        }
    }
    return rigs;
}

// 各重建按 COLMAP 布局写入 <dir>/0、1 等目录；0 的三维点最多，单模型也写入 sparse/0。
void writeModels(const std::vector<Reconstruction>& models, const fs::path& dir,
                 bool verbose, const std::vector<ModelGauge>& gauge, const RigTable* rigs) {
    for (size_t i = 0; i < models.size(); i++) {
        fs::path p = dir / std::to_string(i);
        fs::create_directories(p);
        models[i].writeBinary(p.string());
        if (i < gauge.size()) writeGauge(p, gauge[i]);
        writeRigs(p, models[i], rigs);
        if (verbose)
            L::err(Tag::Map, M::map_wrote_model,
                   {(long long)i, (long long)models[i].numRegistered(),
                    (long long)models[i].points3D.size(), p.string()});
    }
    // 重跑后模型变少时清除多余编号目录，避免 resume 将旧结果误当作本次重建。
    if (!fs::is_directory(dir)) return;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_directory()) continue;
        const std::string name = e.path().filename().string();
        if (name.find_first_not_of("0123456789") != std::string::npos) continue;
        size_t idx = std::stoull(name);
        if (idx < models.size()) continue;
        if (!fs::exists(e.path() / "images.bin")) continue;
        std::error_code ec;
        fs::remove_all(e.path(), ec);
        if (verbose && !ec)
            L::err(Tag::Map, M::map_removed_stale, {e.path().string()});
    }
}

// 统计所有模型覆盖的不同图像；模型允许重叠以便对齐，直接累加会重复计算。
size_t distinctRegistered(const std::vector<Reconstruction>& models) {
    std::set<uint32_t> ids;
    for (const Reconstruction& m : models)
        for (const auto& kv : m.images)
            if (kv.second.registered) ids.insert(kv.first);
    return ids.size();
}

// 主模型之外每个模型各输出一行，使分散结果在摘要中可见。
void printExtraModels(const std::vector<Reconstruction>& models,
                             const std::vector<FeatureSet>& feats) {
    for (size_t i = 1; i < models.size(); i++) {
        double mn = 0, md = 0;
        size_t nobs = 0;
        reprojStats(models[i], feats, mn, md, nobs);
        L::out(Tag::Run, M::map_model_line_error,
               {(long long)i, (long long)models[i].numRegistered(),
                (long long)models[i].points3D.size(), L::num(mn, 3)});
    }
}

// 多输入文件夹时分别报告配准率，及时显露完全没有参与重建的输入。
void printFolderCoverage(const std::vector<Reconstruction>& models,
                         const MatchesDatabase& db) {
    auto group_of = [](const std::string& name) {
        size_t slash = name.find('/');
        return slash == std::string::npos ? std::string(".") : name.substr(0, slash);
    };
    std::map<std::string, std::pair<size_t, size_t>> per;  // 文件夹 -> {已配准数量，总数}
    for (const auto& im : db.images) per[group_of(im.name)].second++;
    if (per.size() < 2) return;
    std::set<uint32_t> ids;
    for (const Reconstruction& m : models)
        for (const auto& kv : m.images)
            if (kv.second.registered) ids.insert(kv.first);
    for (uint32_t id : ids)
        if (id < db.images.size()) per[group_of(db.images[id].name)].first++;
    L::out(Tag::Run, M::sum_per_folder);
    for (const auto& kv : per)
        L::out(Tag::Run, M::sum_folder_line,
               {kv.first, (long long)kv.second.first, (long long)kv.second.second});
    for (const auto& kv : per)
        if (kv.second.first == 0) L::warn(Tag::Run, M::sum_folder_empty, {kv.first});
}

// 将不含扩展名的相对特征路径映射回图像目录中的真实文件名。
static std::map<std::string, std::string> imageStemMap(const std::string& imagedir) {
    std::map<std::string, std::string> stem2name;
    if (imagedir.empty()) return stem2name;
    for (const auto& e : fs::recursive_directory_iterator(imagedir))
        if (e.is_regular_file() && !isSidecar(e.path())) {
            fs::path rel = relativeTo(e.path(), imagedir);
            fs::path stem = rel;
            stem.replace_extension();
            stem2name[stem.generic_string()] = rel.generic_string();
        }
    return stem2name;
}

// SS_UNREG_LOG 指定文件时，按文件夹列出未被任何模型接纳的图像；数据名称不翻译，全部覆盖时不写内容。
static void writeUnregisteredList(const std::vector<Reconstruction>& models,
                                  const MatchesDatabase& db,
                                  const std::string& imagedir) {
    const char* path = spirula::env("UNREG_LOG");
    if (!path || !*path) return;
    std::set<uint32_t> ids;
    for (const Reconstruction& m : models)
        for (const auto& kv : m.images)
            if (kv.second.registered) ids.insert(kv.first);
    const std::map<std::string, std::string> stem2name = imageStemMap(imagedir);
    std::map<std::string, std::vector<std::string>> missing;
    for (size_t i = 0; i < db.images.size(); i++) {
        if (ids.count(uint32_t(i))) continue;
        std::string name = db.images[i].name;
        const auto stem = stem2name.find(name);
        if (stem != stem2name.end()) name = stem->second;
        const size_t slash = name.find('/');
        missing[slash == std::string::npos ? std::string(".")
                                           : name.substr(0, slash)]
            .push_back(name);
    }
    if (missing.empty()) return;
    std::ofstream f(path, std::ios::trunc);
    if (!f) return;
    size_t unreg = 0;
    for (const auto& kv : missing) unreg += kv.second.size();
    f << "unregistered " << unreg << '/' << db.images.size() << "\n";
    for (const auto& kv : missing) {
        f << '\n' << '[' << (kv.first == "." ? "(root)" : kv.first.c_str())
          << "] " << kv.second.size() << '\n';
        for (const std::string& n : kv.second) f << n << '\n';
    }
}


// 用一行汇总模型装配结果，帮助区分原视图图分散与合并被拒绝。
void printAssembly(const AssembleStats& ast, size_t models, Tag tag) {
    if (!ast.models_in) return;
    const ManagerStats& f = ast.finish;
    L::out(tag, M::map_assembled,
           {format_duration(ast.t_merge + ast.t_ba + ast.t_grow + ast.finishSecs()),
            (long long)ast.models_in,
            (long long)models, (long long)ast.rounds, (long long)ast.merges,
            (long long)ast.merges_refused, (long long)ast.grown_images,
            (long long)f.covered_before, (long long)f.covered_after});
    L::out(tag, M::map_finishing,
           {format_duration(ast.finishSecs()), (long long)f.splits,
            (long long)f.duplicate_splits, (long long)f.reseeded_models,
            (long long)f.dropped_redundant, (long long)f.audited_repaired,
            (long long)f.audited_out});
}

// --mapper 选择 flat 或 bottom-up，默认及基准测量均使用 flat；两者共用 Assemble.h 的装配流程，不设独立管理阶段。
RigTable buildRigs(const MatchesDatabase& db, const SfmConfig& cfg, bool verbose) {
    std::vector<std::string> names;
    names.reserve(db.images.size());
    for (const ImageEntry& im : db.images) names.push_back(im.name);
    RigTable rigs = buildRigTable(names, cfg.rigs);
    if (!verbose) return rigs;
    for (const RigSpec& r : rigs.rigs) {
        size_t full = 0;
        for (const auto& fr : r.frames) {
            bool all = true;
            for (uint32_t img : fr) all = all && img != kNoImage;
            full += all ? 1 : 0;
        }
        std::string members;
        for (const RigMemberDef& m : r.members)
            members += (members.empty() ? "" : ", ") + m.prefix;
        L::out(Tag::Map, M::rig_table,
               {r.name, members, (long long)r.frames.size(), (long long)full,
                r.anyKnownExt() ? M::rig_ext_given.get() : M::rig_ext_estimated.get()});
    }
    return rigs;
}

SequenceTable buildSequences(const MatchesDatabase& db, const SfmConfig& cfg, bool verbose) {
    std::vector<std::string> names;
    names.reserve(db.images.size());
    for (const ImageEntry& im : db.images) names.push_back(im.name);
    SequenceTable seqs = buildSequenceTable(names, cfg.sequences);
    if (!verbose) return seqs;
    for (size_t k = 0; k < seqs.length.size(); k++) {
        size_t images = 0;
        for (int32_t id : seqs.seq) images += id == (int32_t)k ? 1 : 0;
        std::string members;
        for (const std::string& m : seqs.members[k])
            members += (members.empty() ? "" : ", ") + (m.empty() ? std::string(".") : m);
        L::out(Tag::Map, M::sequence_table,
               {(long long)k, members, (long long)images, (long long)seqs.length[k],
                (long long)cfg.overlap});
    }
    return seqs;
}

std::vector<Reconstruction> runMapper(Mapper& mapper, const MatchesDatabase& db,
                                      const std::vector<FeatureSet>& feats, SfmConfig& cfg,
                                      AssembleStats& ast) {
    const bool bup = cfg.mapper_mode == "bottom-up" || cfg.mapper_mode == "hierarchical";
    AssembleOptions ao = cfg.assemble;
    ao.verbose = !cfg.quiet;
    std::vector<Reconstruction> models;
    if (!bup) {
        ao.tag = "map";
        models = assembleModels(mapper, mapper.run(), cfg.manager, ao, ast);
    } else {
        ao.tag = "bup";
        BottomUpStats bs;
        BottomUpOptions bo = cfg.bup;
        bo.verbose = !cfg.quiet;
        models = bottomUpReconstruct(mapper, db, feats, bo, cfg.manager, ao, bs);
        ast = bs.assemble;
    }
    // 装配会改变模型中的图像，最后刷新最大的输出模型，避免显示过期快照。
    if (!models.empty()) sfm::progress::model(models.front(), /*force=*/true);
    return models;
}

// 收尾阶段按模型释放固定参数执行全局 BA，可选再允许逐图像独立内参；原理见 src/sfm/README.md。
std::vector<Reconstruction> finishModels(Mapper& mapper,
                                                std::vector<Reconstruction> models,
                                                const SfmConfig& cfg, bool verbose,
                                                double& secs) {
    const double t0 = now();
    // 没有参数需要释放时跳过无效求解及其日志。
    if (cfg.final_principal_point ||
        (cfg.final_extra_params && !cfg.mapper.refine_extra_params)) {
        for (Reconstruction& m : models)
            if (m.numRegistered() >= 2)
                m = mapper.polish(m, cfg.final_principal_point, cfg.final_extra_params);
        if (verbose)
            L::err(Tag::Map, M::map_final_intrinsics,
                   {(long long)models.size(), format_duration(now() - t0)});
    }
    if (cfg.final_per_image_intrinsics) {
        const double t1 = now();
        for (Reconstruction& m : models)
            m = mapper.perImageIntrinsics(m, cfg.final_extra_params);
        if (verbose)
            L::err(Tag::Map, M::map_per_image_done,
                   {(long long)models.size(), format_duration(now() - t1)});
    }
    if (cfg.final_free_rig && mapper.rigs()) {
        const double t1 = now();
        for (Reconstruction& m : models) m = mapper.releaseRigs(m);
        if (verbose)
            L::err(Tag::Map, M::map_free_rig_done,
                   {(long long)models.size(), format_duration(now() - t1)});
    }
    secs = now() - t0;
    return models;
}

// 统一扫描一次图像目录，将特征文件主干名恢复为带扩展名的图像名，避免对每个子模型重复遍历。
void resolveImageNames(std::vector<Reconstruction>& models, const std::string& imagedir) {
    if (imagedir.empty()) return;
    const std::map<std::string, std::string> stem2name = imageStemMap(imagedir);
    for (Reconstruction& rec : models)
        for (auto& kv : rec.images) {
            auto it = stem2name.find(kv.second.name);
            if (it != stem2name.end()) kv.second.name = it->second;
        }
}

// 逐相机报告分组决策，便于在内参异常前发现 camera-mode 设置错误。
void printCameraSetup(Tag tag, const CameraSetup& cs,
                      const CameraSetupOptions& sopt, size_t nimages) {
    std::map<uint32_t, size_t> counts;
    for (uint32_t id : cs.ids) counts[id]++;
    L::err(tag, M::match_camera_mode,
           {cameraModeName(cs.mode_used), (long long)cs.count(), (long long)nimages});
    if (cs.mode_switched)
        L::err(tag, M::match_camera_mode_switched,
               {(long long)cs.dim_buckets, (long long)nimages});
    if (cs.size_split_groups)
        L::warn(tag, M::match_camera_size_split, {(long long)cs.size_split_groups});
    if (cs.exif_focal_images)
        L::err(tag, sopt.exif_focal ? M::match_exif_focals : M::match_exif_focals_ignored,
               {(long long)cs.exif_focal_images, (long long)nimages});
    // 限制日志数量，避免逐图像相机模式对大型网络图片集输出数千行。
    const size_t kMaxLines = 8;
    size_t shown = 0;
    for (const auto& kv : cs.cameras) {
        if (shown++ >= kMaxLines) {
            L::err(tag, M::match_more_cameras, {(long long)(cs.cameras.size() - kMaxLines)});
            break;
        }
        const Camera& c = kv.second;
        L::err(tag, M::match_camera_line,
               {(long long)kv.first, (long long)counts[kv.first], (long long)c.width,
                (long long)c.height, camInfo(c.model).cli_name, c.focal(),
                (cs.focal_known.count(kv.first)   ? M::focal_prior
                 : cs.focal_given.count(kv.first) ? M::focal_given
                                                  : M::focal_guessed).get()});
    }
}


// 每种不匹配的掩码、图像宽高比组合仅警告一次；尺寸可因 UV 采样而不同，宽高比不同则可能来自错误裁剪。
void checkMaskShape(const std::string& mask_path, const Mask& m,
                    const std::pair<int, int>& img_dims) {
    static std::set<std::array<int, 4>> warned;
    if (m.empty() || img_dims.first <= 0 || img_dims.second <= 0) return;
    const double am = (double)m.width / m.height;
    const double ai = (double)img_dims.first / img_dims.second;
    if (std::fabs(am - ai) <= 0.01 * ai) return;
    if (!warned.insert({m.width, m.height, img_dims.first, img_dims.second}).second) return;
    L::warn(Tag::Extract, M::extract_mask_aspect,
            {mask_path, (long long)m.width, (long long)m.height,
             (long long)img_dims.first, (long long)img_dims.second});
}

// 掩码排除几乎所有关键点时提示可能反向；约定白色保留，但以物体为中心的任务也可能正常排除大部分背景，因此只警告、不自动翻转。
void warnIfMasksLookInverted(const ExtractStats& st) {
    // 只统计本次实际提取的图像，复用文件无法恢复先前掩码排除数。
    const uint64_t before = st.features_new + st.masked_out;
    if (!st.masked_images || before == 0) return;
    const double dropped = (double)st.masked_out / (double)before;
    if (dropped < 0.7) return;
    L::warn(Tag::Extract, M::extract_masks_look_inverted, {(long long)(100.0 * dropped)});
}

namespace {

// 将特征文件的名称、大小与修改时间汇成签名；修改时间能识别尺寸未变的重新提取，保证缓存索引仍对应同一集合。
std::string featureDirDigest(const fs::path& featdir) {
    std::vector<std::string> rows;
    std::error_code walk, ec;
    for (auto it = fs::recursive_directory_iterator(featdir, walk);
         !walk && it != fs::recursive_directory_iterator(); it.increment(walk))
        if (it->is_regular_file(ec))
            rows.push_back(
                it->path().generic_string() + ":" +
                std::to_string((uint64_t)fs::file_size(it->path(), ec)) + ":" +
                std::to_string((int64_t)fs::last_write_time(it->path(), ec)
                                   .time_since_epoch()
                                   .count()));
    std::sort(rows.begin(), rows.end());
    uint64_t h = 1469598103934665603ull;   // FNV-1a
    for (const std::string& s : rows)
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016llx.%zu", (unsigned long long)h, rows.size());
    return buf;
}

// 删除输出目录中不属于 live 的文件；匹配会读取全部 .bin，残留文件会形成幽灵视图。
void sweepStaleFeatures(const fs::path& outdir, const std::set<fs::path>& live) {
    std::error_code walk, ec;
    std::vector<fs::path> dead;
    for (auto it = fs::recursive_directory_iterator(outdir, walk);
         !walk && it != fs::recursive_directory_iterator(); it.increment(walk))
        if (it->is_regular_file(ec) && !live.count(it->path())) dead.push_back(it->path());
    for (const fs::path& p : dead) fs::remove(p, ec);
}

// 检查特征完整性与修改时间，识别设置未变但图像或掩码已重新生成的情况。
bool featuresAreCurrent(const fs::path& feat, const fs::path& img,
                        const std::string& mask, uint32_t& count) {
    std::error_code fe, ie, me;
    const auto t = fs::last_write_time(feat, fe);
    if (fe || t < fs::last_write_time(img, ie) || ie) return false;
    if (!mask.empty() && t < fs::last_write_time(mask, me) && !me) return false;
    return peekFeatures(feat.string(), count);
}

}  // 匿名命名空间

int extractDirectory(const std::string& imagedir, const fs::path& outdir,
                     const SfmConfig& cfg, ExtractStats& stats, bool reuse) {
    const SiftOptions& opt = cfg.sift;
    const std::string& maskdir = cfg.mask_dir;
    // 递归读取图像以保留相机文件夹分组，跳过嵌套掩码目录，避免将 PNG 掩码误作图像。
    std::error_code skip_ec;
    const bool skip_masks = !maskdir.empty() && fs::is_directory(maskdir, skip_ec);
    std::vector<fs::path> found;
    for (auto it = fs::recursive_directory_iterator(imagedir, fs::directory_options::follow_directory_symlink);
         it != fs::recursive_directory_iterator(); ++it) {
        if (skip_masks && it->is_directory() && fs::equivalent(it->path(), maskdir, skip_ec)) {
            it.disable_recursion_pending();
            continue;
        }
        if (it->is_regular_file() && isImageExt(it->path().extension().string()) &&
            !isSidecar(it->path()))
            found.push_back(it->path());
    }
    if (found.empty()) {
        L::fail(Tag::Extract, M::extract_no_images, {imagedir});
        return 1;
    }
    std::sort(found.begin(), found.end());

    // 每个文件头只探测一次，供排序与解码内存预算共用，避免排序比较器重复探测。
    std::vector<fs::path> imgs;
    std::vector<std::pair<int, int>> dims;
    for (const fs::path& p : found) {
        int w = 0, h = 0;
        if (!imageSize(p.string(), w, h) || w <= 0 || h <= 0) {
            L::warn(Tag::Extract, M::extract_skipping_file,
                    {p.filename().string()});
            stats.unreadable++;
            continue;
        }
        imgs.push_back(p);
        dims.emplace_back(w, h);
    }
    if (imgs.empty()) {
        L::fail(Tag::Extract, M::extract_no_decodable, {imagedir});
        return 1;
    }

    // 最大图像优先，使提取器只需一次分配最大设备缓冲。
    std::vector<size_t> order(imgs.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return (int64_t)dims[a].first * dims[a].second > (int64_t)dims[b].first * dims[b].second;
    });
    std::vector<std::string> paths(order.size());
    std::vector<std::pair<int, int>> sorted_dims(order.size());
    for (size_t k = 0; k < order.size(); k++) {
        paths[k] = imgs[order[k]].string();
        sorted_dims[k] = dims[order[k]];
    }

    fs::create_directories(outdir);

    ImageLoadOptions lopt;
    lopt.max_image_size = cfg.max_image_size;
    lopt.num_threads = cfg.decode_threads;
    lopt.want_color = true;  // 趁图像仍在内存中采样关键点颜色
    lopt.gamut = cfg.image_gamut;
    lopt.is_linear = cfg.image_is_linear;
    lopt.flip_mask = cfg.flip_mask;
    lopt.apply_exif_orientation = cfg.exif_orientation == "apply";
    if (cfg.decode_budget_mb > 0)
        lopt.memory_budget_bytes = (size_t)cfg.decode_budget_mb << 20;

    // 按 paths 顺序预先解析掩码，供解码池使用，并在 GPU 提取前报告完全无匹配的掩码目录。
    MaskIndex masks(maskdir);
    if (!maskdir.empty() && !masks.valid())
        L::warn(Tag::Extract, M::extract_mask_dir_missing, {maskdir});
    if (masks.valid()) {
        lopt.mask_paths.assign(paths.size(), std::string());
        for (size_t k = 0; k < paths.size(); k++) {
            std::string rel = relativeTo(paths[k], imagedir).generic_string();
            lopt.mask_paths[k] = masks.find(rel);
            if (lopt.mask_paths[k].empty()) {
                stats.unmasked_images++;
                if (stats.first_unmasked.empty()) stats.first_unmasked = rel;
            } else {
                stats.masked_images++;
            }
        }
        L::out(Tag::Extract, M::extract_masks_matched,
               {(long long)stats.masked_images, (long long)paths.size(), maskdir});
        if (stats.masked_images == 0) {
            L::fail(Tag::Extract, M::extract_no_mask_matches,
                    {maskdir, stats.first_unmasked});
            return 1;
        }
        if (stats.unmasked_images)
            L::warn(Tag::Extract, M::extract_some_unmasked,
                    {(long long)stats.unmasked_images, stats.first_unmasked});
    }

    // 确定特征输出与可复用文件；进度总数表示全部输入图像，而非剩余工作量。
    const size_t n_all = paths.size();
    std::vector<fs::path> outs(n_all);
    std::set<fs::path> live;
    for (size_t k = 0; k < n_all; k++) {
        outs[k] = outdir / relativeTo(paths[k], imagedir);
        outs[k].replace_extension(".bin");
        live.insert(outs[k]);
    }
    sweepStaleFeatures(outdir, live);

    events::stage_begin(Stage::Extract, (int64_t)n_all);
    if (reuse) {
        std::vector<size_t> todo;
        for (size_t k = 0; k < n_all; k++) {
            uint32_t count = 0;
            const std::string mask =
                k < lopt.mask_paths.size() ? lopt.mask_paths[k] : std::string();
            if (!featuresAreCurrent(outs[k], paths[k], mask, count)) {
                todo.push_back(k);
                continue;
            }
            stats.reused++;
            stats.features += count;
            stats.images++;
            Event ev;
            ev.kind = Event::Kind::ImageExtracted;
            ev.stage = Stage::Extract;
            ev.done = (int64_t)stats.images;
            ev.total = (int64_t)n_all;
            ev.name = fs::path(paths[k]).filename().string();
            ev.width = sorted_dims[k].first;
            ev.height = sorted_dims[k].second;
            ev.features = count;
            events::emit(ev);
        }
        if (stats.reused) {
            L::out(Tag::Extract, M::extract_reusing,
                   {(long long)stats.reused, (long long)n_all});
            for (size_t i = 0; i < todo.size(); i++) {
                const size_t k = todo[i];
                if (k == i) continue;   // std::string 自移动可能清空自身
                paths[i] = std::move(paths[k]);
                sorted_dims[i] = sorted_dims[k];
                outs[i] = std::move(outs[k]);
                if (!lopt.mask_paths.empty()) lopt.mask_paths[i] = std::move(lopt.mask_paths[k]);
            }
            paths.resize(todo.size());
            sorted_dims.resize(todo.size());
            outs.resize(todo.size());
            if (!lopt.mask_paths.empty()) lopt.mask_paths.resize(todo.size());
        }
    }
    if (paths.empty()) {
        events::stage_end(Stage::Extract);
        return 0;
    }

    ImageLoadPlan plan = planImageLoad(sorted_dims, lopt);
    if (opt.verbose)
        L::err(Tag::Extract, M::extract_plan,
               {(long long)paths.size(), (long long)plan.num_threads,
                (long long)plan.window,
                (long long)(((size_t)plan.num_threads * plan.decode_peak_bytes +
                             (size_t)plan.window * plan.held_bytes) >> 20)});

    std::unique_ptr<IFeatureExtractor> ext =
        createFeatureExtractor(cfg.features, opt, cfg.aliked, cfg.loma);
    if (opt.verbose) L::err(Tag::Extract, M::extract_frontend, {ext->name()});
    loadImagesInOrder(
        paths, plan, lopt,
        [&](size_t k, GrayImage& img) {
            cancel::check();
            if (img.exif_mirror_dropped && !stats.warned_exif_mirror) {
                stats.warned_exif_mirror = true;
                L::warn(Tag::Extract, M::extract_exif_mirror_dropped,
                        {fs::path(paths[k]).filename().string()});
            }
            FeatureSet f = ext->extract(img);
            sampleFeatureColors(f, img);
            uint32_t dropped = 0;
            if (!lopt.mask_paths.empty() && !lopt.mask_paths[k].empty()) {
                if (img.mask.empty()) {
                    stats.mask_unreadable++;
                    L::warn(Tag::Extract, M::extract_mask_undecodable,
                            {lopt.mask_paths[k],
                             fs::path(paths[k]).filename().string()});
                } else {
                    // 使用 img 实际尺寸，apply 可能已旋转图像，改变探测时的宽高。
                    checkMaskShape(lopt.mask_paths[k], img.mask,
                                   {img.orig_width, img.orig_height});
                    const uint32_t before = f.count();
                    dropped = applyMask(f, img.mask);
                    stats.masked_out += dropped;
                    // 掩码反向或保留值为 0 可能完全清除图像特征；仅警告一次并继续运行。
                    if (before && dropped == before && !stats.warned_empty) {
                        stats.warned_empty = true;
                        L::warn(Tag::Extract, M::extract_mask_empty,
                                {lopt.mask_paths[k],
                                 fs::path(paths[k]).filename().string()});
                    }
                }
            }
            // 所有基于解码图像的关键点操作结束后，恢复原图坐标，使 cameras.bin 描述用户原图而非工作副本（D46）。
            finishFeatures(f, img);
            fs::create_directories(outs[k].parent_path());
            writeFeatures(outs[k].string(), f);
            stats.features += f.count();
            stats.features_new += f.count();
            stats.images++;
            Event ev;
            ev.kind = Event::Kind::ImageExtracted;
            ev.stage = Stage::Extract;
            ev.done = stats.images;
            ev.total = (int64_t)n_all;
            ev.name = fs::path(paths[k]).filename().string();
            ev.width = img.orig_width;
            ev.height = img.orig_height;
            ev.features = f.count();
            ev.masked = dropped;
            events::emit(ev);
            // 用已有图像副本生成预览。
            if (progress::enabled()) {
                fs::path stem = relativeTo(paths[k], imagedir);
                stem.replace_extension();
                progress::thumbnail(stem.generic_string(), img.rgb.data(),
                                    img.width, img.height);
            }
        },
        [&](size_t k, const std::string& err) {
            L::fail(Tag::Extract, M::extract_failed_file,
                    {fs::path(paths[k]).filename().string(), err});
            stats.failed++;
        });
    events::stage_end(Stage::Extract);
    return 0;
}



int loadFeatureDir(const std::string& featdir, const SfmConfig& cfg, bool with_descriptors,
                   std::vector<FeatureSet>& feats, MatchesDatabase& db) {
    // 递归读取与图像树对应的特征文件，按名称排序保证索引稳定；图像内部名称为移除 .bin 的相对路径。
    std::vector<fs::path> files;
    std::error_code walk, ec;
    for (auto it = fs::recursive_directory_iterator(featdir, walk);
         !walk && it != fs::recursive_directory_iterator(); it.increment(walk))
        if (it->is_regular_file(ec) && it->path().extension() == ".bin" &&
            !isSidecar(it->path()))
            files.push_back(it->path());
    std::sort(files.begin(), files.end());
    if (files.size() < 2) {
        L::fail(Tag::Match, M::match_need_two, {featdir});
        return 1;
    }

    // 按文件并行读取大型描述子集合，各任务无共享状态且按固定索引写入，保持排序后的图像顺序。
    feats.assign(files.size(), FeatureSet());
    db.images.resize(files.size());
    events::stage_begin(Stage::Load, (int64_t)files.size());
    {
        const unsigned hc = std::thread::hardware_concurrency();
        int nt = cfg.threads > 0 ? cfg.threads : (hc > 0 ? (int)hc : 1);
        nt = std::max(1, std::min<int>(nt, (int)files.size()));
        std::atomic<size_t> next{0}, done{0};
        std::mutex err_mtx;
        std::string first_error;  // 坏文件须报告错误，不能直接终止进程
        std::vector<std::thread> pool;
        const size_t step = std::max<size_t>(1, files.size() / 200);
        for (int t = 0; t < nt; t++)
            pool.emplace_back([&] {
                for (size_t i = next++; i < files.size(); i = next++) {
                    try {
                        feats[i] = readFeatures(files[i].string(), with_descriptors);
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lk(err_mtx);
                        if (first_error.empty()) first_error = e.what();
                    }
                    const size_t n = ++done;
                    if (n % step == 0 || n == files.size())
                        events::progress(Stage::Load, (int64_t)n, (int64_t)files.size());
                }
            });
        for (std::thread& t : pool) t.join();
        events::stage_end(Stage::Load);
        if (!first_error.empty()) {
            L::err_raw(Tag::Match, first_error);
            return 1;
        }
    }
    for (size_t i = 0; i < files.size(); i++) {
        fs::path rel = relativeTo(files[i], featdir);
        rel.replace_extension();
        db.images[i] = {rel.generic_string(), feats[i].count()};
    }
    return 0;
}

int matchFeatureDir(const std::string& featdir, const SfmConfig& cfg, PairMode mode,
                    bool verify, std::vector<FeatureSet>& feats, MatchesDatabase& db,
                    MatchStats& stats, VerifyCalibration* calib,
                    const MatchResume* res) {
    const MatchOptions& opt = cfg.match;
    const PairSelectionOptions& popt = cfg.prefilter;
    const TwoViewOptions& tvopt = cfg.twoview;
    const bool verbose = !cfg.quiet;
    if (int rc = loadFeatureDir(featdir, cfg, /*with_descriptors=*/true, feats, db)) return rc;
    const size_t n_images = feats.size();
    stats.images = n_images;
    std::vector<std::string> image_names(n_images);
    for (size_t i = 0; i < n_images; i++) image_names[i] = db.images[i].name;
    // 先确定相机分组，传感器以分组为键；GPS 图像对须在列表写出前加入。
    if (calib) {
        calib->cameras = buildCameras(db.images, feats, calib->setup);
        if (verbose) printCameraSetup(Tag::Match, calib->cameras, calib->setup, feats.size());
        if (calib->sensors)
            calib->priors = makeSensorPriors(cfg, *calib->sensors, db, calib->cameras.ids);
    }
    TelemetryPriors* priors = calib ? calib->priors.get() : nullptr;

    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    // 大型数据的配对筛选可能耗时数分钟，通过统一事件流报告进度，避免表现为卡住。
    std::function<void(size_t, size_t)> sp = [&](size_t done, size_t total) {
        events::progress(Stage::Select, (int64_t)done, (int64_t)total);
        if (!verbose) return;
        // 进度原地重绘，自行携带标签，不能调用总会结束当前行的 L::err()。
        fprintf(stderr, "\r%s%s", L::prefix(Tag::Match).c_str(),
                spirula::i18n::format(M::match_pairs_scored,
                             {(long long)done, (long long)total}).c_str());
    };
    // 复用中断前选出的图像对，既节省筛选时间，也确保列表与匹配日志对齐。
    const bool reused_pairs =
        res && resume::readPairs(res->dir / "pairs.bin", res->signature, pairs);
    if (reused_pairs) {
        L::out(Tag::Match, M::match_reusing_pairs, {(long long)pairs.size()});
    } else if (mode == PairMode::Prefilter) {
        stats.scored = n_images * (n_images - 1) / 2;
        double t0 = now();
        events::stage_begin(Stage::Select, (int64_t)(n_images * (n_images - 1)));
        pairs = prefilterPairs(feats, popt, sp);
        events::stage_end(Stage::Select);
        stats.select_seconds = now() - t0;
        if (verbose)
            fprintf(stderr, "\n");
            L::err(Tag::Match, M::match_prefilter_kept,
                   {(long long)pairs.size(), (long long)stats.scored,
                    popt.num_features, popt.num_neighbors,
                    format_duration(stats.select_seconds)});
        // 子采样评分的 top-k 可能漏掉略低于阈值的真实弱连接，文件时序可补充这些关系。
        if (cfg.prefilter_sequential) {
            const size_t sel = pairs.size();
            const std::vector<std::pair<uint32_t, uint32_t>> win = sequentialPairs(
                (uint32_t)n_images, cfg.overlap, cfg.quadratic_overlap, folderRuns(image_names));
            pairs.insert(pairs.end(), win.begin(), win.end());
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
            L::err(Tag::Match, M::match_sequential_added,
                   {(long long)(pairs.size() - sel), (long long)sel, (long long)win.size()});
        }
    } else {
        pairs = mode == PairMode::Exhaustive
                    ? generatePairs((uint32_t)n_images, mode)
                    : sequentialPairs((uint32_t)n_images, cfg.overlap, cfg.quadratic_overlap,
                                      folderRuns(image_names));
        // 闭环通过内容筛选补充时间窗口无法连接的首尾回访；否则链中一个弱连接即可拆散重建。穷举模式已包含所有图像对。
        if (mode == PairMode::Sequential && cfg.loop_closure && n_images > 2) {
            const size_t seq = pairs.size();
            stats.scored = n_images * (n_images - 1) / 2;
            double t0 = now();
            events::stage_begin(Stage::Select, (int64_t)(n_images * (n_images - 1)));
            std::vector<std::pair<uint32_t, uint32_t>> extra = prefilterPairs(feats, popt, sp);
            events::stage_end(Stage::Select);
            stats.select_seconds = now() - t0;
            pairs.insert(pairs.end(), extra.begin(), extra.end());
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
            if (verbose)
                fprintf(stderr, "\n");
                L::err(Tag::Match, M::match_loop_closure_added,
                       {(long long)(pairs.size() - seq), (long long)seq,
                        (long long)extra.size(),
                        format_duration(stats.select_seconds)});
        }
    }
    // 让建图器优先信任序列对会降低已有大量匹配的数据集的稳健性，因此此设置保持禁用。
#if 0
    // 无论配对模式如何，都加入序列时间窗口，保证建图器优先信任的时序图像对存在（D79）。
    if (!reused_pairs && !cfg.sequences.empty()) {
        const size_t before = pairs.size();
        try {
            const SequenceTable st = buildSequenceTable(image_names, cfg.sequences);
            const std::vector<std::pair<uint32_t, uint32_t>> win =
                sequenceWindowPairs(st, cfg.overlap, cfg.quadratic_overlap);
            pairs.insert(pairs.end(), win.begin(), win.end());
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
            if (verbose)
                L::err(Tag::Match, M::match_sequence_added,
                       {(long long)(pairs.size() - before), (long long)before,
                        (long long)win.size()});
        } catch (const std::exception&) {
            // 名称无法匹配定义时交给建图器报告。
        }
    }
#endif
    // 补充 GPS 判定相近的图像对，不受候选列表限制。
    if (priors && cfg.sensor_pairs && !reused_pairs) {
        size_t positioned = 0;
        const std::vector<std::pair<uint32_t, uint32_t>> nearby = gpsProximityPairs(
            *priors, (uint32_t)n_images, cfg.sensor_pair_radius, 20, &positioned);
        const size_t before = pairs.size();
        pairs.insert(pairs.end(), nearby.begin(), nearby.end());
        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
        if (positioned && verbose)
            L::err(Tag::Match, M::sensor_gps_pairs_added,
                   {(long long)(pairs.size() - before), L::num(cfg.sensor_pair_radius, 0),
                    (long long)positioned});
    }
    if (res && !reused_pairs) resume::writePairs(res->dir / "pairs.bin", res->signature, pairs);
    stats.pairs = pairs.size();
    if (verbose)
        L::err(Tag::Match, M::match_plan,
               {(long long)n_images, (long long)pairs.size(),
                mode == PairMode::Exhaustive ? "exhaustive"
                : mode == PairMode::Sequential
                    ? (cfg.loop_closure ? "sequential + loop closure" : "sequential")
                    : (cfg.prefilter_sequential ? "prefilter + sequential" : "prefilter")});
    if (verbose && mode == PairMode::Prefilter)
        L::err(Tag::Match, M::match_prefilter_params,
               {popt.num_features, popt.num_neighbors});

    // 复用中断前已完成的验证；工作线程乱序结束，因此日志按图像对而非列表位置索引。
    std::unordered_map<uint64_t, TwoViewMatches> done_kept;
    std::vector<uint64_t> done_keys;
    const fs::path journal_path = res ? res->dir / "matches.part" : fs::path();
    const bool resumed_verify =
        res && verify &&
        resume::readJournal(journal_path, res->signature, db.images, done_kept, done_keys,
                            stats.putative);
    std::set<uint64_t> done_set(done_keys.begin(), done_keys.end());
    std::vector<std::pair<uint32_t, uint32_t>> todo;
    if (resumed_verify && !done_set.empty()) {
        for (const auto& p : pairs)
            if (!done_set.count(resume::pairKey(p.first, p.second))) todo.push_back(p);
        L::out(Tag::Match, M::match_resuming,
               {(long long)(pairs.size() - todo.size()), (long long)pairs.size()});
    } else {
        todo = pairs;
    }

    std::unique_ptr<IFeatureMatcher> matcher =
        createFeatureMatcher(cfg.matcher, opt, cfg.lightglue, cfg.loma_match);
    if (verbose && cfg.matcher != "bruteforce")
        L::err(Tag::Match, M::match_matcher_name, {matcher->name()});
    auto matchFn = [&](size_t b, size_t e, std::vector<std::vector<FeatureMatch>>& mout) {
        matcher->matchBatch(feats, todo, b, e, mout);
    };
    // 仅 CLI 事件接收端限制打印频率；verifyPairs 始终发出实际完成比例。
    std::function<void(size_t, size_t)> progress;

    if (verify) {
        // 串行 GPU 匹配器向工作池提供验证任务，几何验证是主要开销，见 sfm/feature/Verification.h。
        VerificationOptions vopt;
        vopt.two_view = tvopt;
        vopt.num_threads = cfg.threads;
        vopt.match_batch_pairs = opt.batch_pairs;

        // 鱼眼图像在标定后的单位视线上验证（D45）；普通针孔图像沿用像素坐标验证。
        BearingCache bc;
        std::vector<Camera> percam;
        VerifyPriorStats pstats;
        if (calib) {
            CameraSetup& cs = calib->cameras;
            // 两种焦距搜索共用遍布列表的候选匹配样本，避免只采到一个区域；鱼眼必须先估焦距再验证，普通镜头也共用该路径。
            const bool want_rect = cs.anyGuessedRectilinear();
            std::vector<std::pair<uint32_t, uint32_t>> sample;
            std::vector<std::vector<FeatureMatch>> sm;
            if (cs.anyWide() || want_rect) {
                const size_t want = calib->sample_pairs * std::max<size_t>(1, cs.count());
                size_t stride = std::max<size_t>(1, pairs.size() / std::max<size_t>(1, want));
                for (size_t p = 0; p < pairs.size() && sample.size() < want; p += stride)
                    sample.push_back(pairs[p]);
                std::vector<std::vector<FeatureMatch>> chunk;
                for (size_t b = 0; b < sample.size(); b += 16) {
                    size_t e = std::min(b + 16, sample.size());
                    matcher->matchBatch(feats, sample, b, e, chunk);
                    for (size_t k = b; k < e; k++) sm.push_back(std::move(chunk[k - b]));
                }
            }
            if (want_rect) {
                double t_f = now();
                bootstrapRectilinearFocals(feats, cs.ids, sample, sm, cs.cameras, cs.focal_given,
                                           cs.focal_measured, tvopt, calib->sample_pairs,
                                           cfg.threads, verbose);
                if (verbose && !cs.focal_measured.empty())
                    L::err(Tag::Match, M::focal_epipolar_search,
                           {format_duration(now() - t_f)});
            }
            if (cs.anyWide()) {
                bootstrapGroupFocals(feats, cs.ids, sample, sm, cs.cameras, cs.focal_given,
                                     cs.focal_measured, tvopt, calib->sample_pairs, cfg.threads,
                                     verbose);
                std::vector<Camera> percam(feats.size());
                for (size_t i = 0; i < feats.size(); i++) percam[i] = cs.cameras.at(cs.ids[i]);
                double t_b = now();
                // 混合镜头数据的跨组匹配包含鱼眼，因此全部使用标定视线验证；纯直线投影数据沿用像素路径。
                bc = precomputeBearings(feats, percam, /*wide_only=*/!cs.mixed(), cfg.threads);
                vopt.bearings = &bc;
                calib->used_bearings = true;
                if (verbose) {
                    // 焦距列表仅含标识符与数值，作为单个消息参数传递。
                    std::string focals;
                    for (const auto& kv : cs.cameras) {
                        if (!focals.empty()) focals += ", ";
                        focals += "cam " + std::to_string(kv.first) + ": " +
                                  L::num(kv.second.focal(), 1);
                    }
                    L::err(Tag::Match, M::match_bearings,
                           {format_duration(now() - t_b),
                            L::num(bc.bytes() / 1048576.0, 0), focals});
                }
            }
        }
        // 先用时间相邻图像对样本标定 IMU 到镜头的旋转，再向所有覆盖图像对提供陀螺先验；后续候选匹配将被释放。
        if (priors && cfg.sensor_verify && calib) {
            percam = perImageCameras(calib->cameras, feats.size());
            std::vector<std::pair<uint32_t, uint32_t>> sample;
            for (const auto& p : pairs)
                if (priors->calibrationPair(p.first, p.second)) sample.push_back(p);
            if (sample.size() > 900) {
                std::vector<std::pair<uint32_t, uint32_t>> thin;
                const double step = (double)sample.size() / 900.0;
                for (size_t k = 0; k < 900; k++) thin.push_back(sample[(size_t)(k * step)]);
                sample.swap(thin);
            }
            std::vector<std::vector<FeatureMatch>> sm, chunk;
            for (size_t b = 0; b < sample.size(); b += 16) {
                const size_t e = std::min(b + 16, sample.size());
                matcher->matchBatch(feats, sample, b, e, chunk);
                for (size_t k = b; k < e; k++) sm.push_back(std::move(chunk[k - b]));
            }
            const double t_c = now();
            calibrateSensorPriors(*priors, feats, sample, sm, percam, tvopt, cfg.threads, verbose);
            if (verbose)
                L::diag(Tag::Match, "[prior] calibrated on %zu pair(s) in %s", sample.size(),
                        format_duration(now() - t_c).c_str());
            if (priors->anyRotation()) {
                vopt.priors = priors;
                vopt.cameras = &percam;
            }
        }
        // 组内图像对估得的焦距标记为 focal_known，避免单图少量内点将可靠值拉偏；双鱼眼上曾出现 552 px 被 80 个内点拉到 397 px，再被联合优化拉回 517 px。
        // 不标记 focal_given，仍允许试探重建与 BA 精化，效果优于直接采用原始投票。
        if (calib)
            for (uint32_t id : calib->cameras.focal_measured)
                calib->cameras.focal_known.insert(id);
        // 将相机分组与焦距写入匹配数据库，使 map 沿用测量结果，而非从已筛选内点重新估计（D47）。
        if (calib) storeCameraSetup(db, calib->cameras);
        if (verbose)
            L::err(Tag::Match, M::match_verifying, {(long long)verificationThreadCount(vopt)});
        sfm::progress::begin_matching((uint32_t)feats.size(), pairs);
        {
            std::vector<std::string> names;
            std::vector<uint32_t> nfeat;
            names.reserve(db.images.size());
            nfeat.reserve(db.images.size());
            for (const ImageEntry& im : db.images) {
                names.push_back(im.name);
                nfeat.push_back(im.num_features);
            }
            sfm::progress::live_matches_begin(names, nfeat);
        }
        // 缓存日志中的图像对不会重新验证，必须主动报告进度，否则匹配图会一直将其显示为尚未处理。
        for (uint64_t key : done_keys) {
            const auto it = done_kept.find(key);
            sfm::progress::pair((uint32_t)(key >> 32), (uint32_t)key,
                                it == done_kept.end() ? 0u
                                                      : (uint32_t)it->second.matches.size());
        }
        resume::MatchJournal journal;
        if (res) journal.open(journal_path, res->signature, resumed_verify);
        vopt.journal = &journal;
        vopt.progress_done_base = pairs.size() - todo.size();
        vopt.progress_total = pairs.size();
        events::stage_begin(Stage::Match, (int64_t)pairs.size());
        // 累加而非覆盖：日志已统计先前送入验证的匹配，verifyPairs 另写本次数量。
        uint64_t putative = 0;
        std::vector<TwoViewMatches> fresh =
            verifyPairs(feats, todo, matchFn, vopt, &putative, progress, &pstats);
        stats.putative += putative;
        if (vopt.priors && verbose && pstats.pairs)
            L::err(Tag::Match, M::sensor_verify_summary,
                   {(long long)pstats.pairs, (long long)pstats.kept, (long long)pstats.dropped,
                    (long long)pstats.rescued, (long long)pstats.disagreed,
                    (long long)pstats.contradicted});
        // 恢复图像对列表顺序，保证恢复运行与完整运行的种子评分平局处理一致。
        if (done_kept.empty()) {
            db.pairs = std::move(fresh);
        } else {
            std::unordered_map<uint64_t, size_t> at;
            at.reserve(fresh.size() * 2);
            for (size_t i = 0; i < fresh.size(); i++)
                at[resume::pairKey(fresh[i].image1, fresh[i].image2)] = i;
            db.pairs.reserve(done_kept.size() + fresh.size());
            for (const auto& p : pairs) {
                const uint64_t key = resume::pairKey(p.first, p.second);
                const auto old = done_kept.find(key);
                if (old != done_kept.end()) db.pairs.push_back(std::move(old->second));
                else if (const auto n = at.find(key); n != at.end())
                    db.pairs.push_back(std::move(fresh[n->second]));
            }
        }
        // 仅将已验证连接扩展到 rig 的其他镜头；PortalCam 步行数据中，未验证候选对的伙伴匹配有四分之三失败。
        if (cfg.rig_pairs && !cfg.rigs.empty()) {
            std::vector<std::pair<uint32_t, uint32_t>> seeds, mates;
            try {
                const RigTable rt = buildRigTable(image_names, cfg.rigs);
                for (const TwoViewMatches& t : db.pairs)
                    if ((int)t.matches.size() >= cfg.rig_pair_min_inliers)
                        seeds.emplace_back(t.image1, t.image2);
                mates = rigMatePairs(rt, seeds, cfg.rig_pair_angle);
            } catch (const std::exception&) {
                // 名称无法匹配定义时交给建图器报告。
            }
            mates.erase(std::remove_if(mates.begin(), mates.end(),
                                       [&](const std::pair<uint32_t, uint32_t>& q) {
                                           return std::binary_search(pairs.begin(),
                                                                     pairs.end(), q);
                                       }),
                        mates.end());
            std::vector<std::pair<uint32_t, uint32_t>> mates_todo;
            for (const auto& q : mates)
                if (!done_set.count(resume::pairKey(q.first, q.second))) mates_todo.push_back(q);
            std::vector<TwoViewMatches> more;
            if (!mates_todo.empty()) {
                auto mateFn = [&](size_t b, size_t e,
                                  std::vector<std::vector<FeatureMatch>>& mout) {
                    matcher->matchBatch(feats, mates_todo, b, e, mout);
                };
                vopt.progress_done_base = pairs.size();
                vopt.progress_total = pairs.size() + mates_todo.size();
                uint64_t put2 = 0;
                more = verifyPairs(feats, mates_todo, mateFn, vopt, &put2, progress, &pstats);
                stats.putative += put2;
            }
            // 按伙伴列表顺序恢复结果，不受实际验证所属运行影响。
            std::unordered_map<uint64_t, size_t> at;
            for (size_t i = 0; i < more.size(); i++)
                at[resume::pairKey(more[i].image1, more[i].image2)] = i;
            size_t kept = 0;
            for (const auto& q : mates) {
                const uint64_t key = resume::pairKey(q.first, q.second);
                const auto old = done_kept.find(key);
                if (old != done_kept.end()) db.pairs.push_back(std::move(old->second));
                else if (const auto n = at.find(key); n != at.end())
                    db.pairs.push_back(std::move(more[n->second]));
                else continue;
                kept++;
            }
            stats.pairs += mates.size();
            if (!mates.empty())
                L::err(Tag::Match, M::match_rig_pairs_added,
                       {(long long)kept, (long long)mates.size(), (long long)seeds.size()});
        }
        journal.close();
        sfm::progress::flush();
        events::stage_end(Stage::Match);
        for (const TwoViewMatches& tvm : db.pairs) stats.inliers += tvm.matches.size();
        // 样本未完成标定时，改用已验证图像对；建图器同样需要这些旋转。
        if (priors && calib && !priors->anyRotation()) {
            if (percam.empty()) percam = perImageCameras(calib->cameras, feats.size());
            calibrateSensorPriorsFromDatabase(*priors, db, feats, percam, tvopt, cfg.threads,
                                              verbose);
        }
    } else {
        const size_t batch = std::max(1, opt.batch_pairs);
        std::vector<std::vector<FeatureMatch>> mout;
        for (size_t b = 0; b < todo.size(); b += batch) {
            size_t e = std::min(b + batch, todo.size());
            matchFn(b, e, mout);
            for (size_t p = b; p < e; p++) {
                uint32_t i = todo[p].first, j = todo[p].second;
                std::vector<FeatureMatch>& m = mout[p - b];
                stats.putative += m.size();
                if (!m.empty()) {
                    stats.inliers += m.size();
                    db.pairs.push_back({i, j, 0, std::move(m)});
                }
                if (progress) progress(p + 1, todo.size());
            }
        }
    }
    stats.kept = db.pairs.size();
    return 0;
}
// ---------------- 自动流水线 ----------------

AutoResult run_auto(SfmConfig& cfg, const AutoInputs& in) {
    AutoResult r;
    std::string _imagedir = in.image_dir;
    const std::string& _workspace = in.workspace;
    const bool verbose = !cfg.quiet;

    // ---------------- 图像与掩码位置（D39/D40）----------------
    // 默认数据集包含 images/ 与 masks/；直接指向图像目录时查找相邻 masks，在不存在其他图像的条件下两种输入形式等价。
    if (_imagedir.empty()) _imagedir = "images";
    if (!fs::is_directory(_imagedir)) {
        L::fail(Tag::Run, M::run_not_a_directory, {_imagedir});
        { r.exit_code = 1; return r; }
    }
    {
        fs::path nested = fs::path(_imagedir) / "images";
        if (fs::is_directory(nested) && !holdsImagesOutside(_imagedir, nested)) {
            L::out(Tag::Run, M::run_nested_images, {_imagedir, nested.string()});
            _imagedir = nested.string();
        }
    }
    if (!in.mask_dir_explicit) {
        fs::path p = fs::path(_imagedir);
        if (!p.empty() && p.filename().empty()) p = p.parent_path();  // 移除末尾的 /
        fs::path sibling = p.parent_path() / "masks";
        if (fs::is_directory(sibling)) cfg.mask_dir = sibling.string();
    }
    adoptExrColorSpace(cfg, _imagedir, in.explicit_flags);

    fs::path ws(_workspace);
    fs::create_directories(ws);
    const fs::path featdir = ws / "features";
    const fs::path matchpath = ws / "matches.bin";
    // COLMAP 布局：每个重建位于 sparse/<i>，0 为最大模型。
    const fs::path sparsedir = ws / "sparse";

    L::out(Tag::Run, M::run_header, {_imagedir, _workspace});
    L::out(Tag::Run, M::run_quality,
           {cfg.quality, (long long)cfg.max_image_size,
            (long long)cfg.sift.max_num_features});
    L::out(Tag::Run, M::run_data_type, {cfg.data_type});
    L::out(Tag::Run, M::run_cameras, {cfg.camera_model, cfg.camera_mode});
    if (!cfg.mask_dir.empty()) L::out(Tag::Run, M::run_masks, {cfg.mask_dir});
    // 报告质量与数据类型预设修改的选项，便于仅凭日志解释运行行为。
    for (const PresetChange& p : in.preset_changes)
        L::out(Tag::Run, M::run_preset_moved, {"--" + p.flag, p.to, p.from});

    // ---------------- 中断缓存及有效性 ----------------
    // 阶段开始前写入签名，使执行中断后留下的文件仍可供下一次复用。
    const fs::path rdir = resume::dir(_workspace);
    const std::string extract_sig =
        stageSignature(cfg, CMD_EXTRACT) + "images=" + _imagedir + "\n";
    std::error_code rm_ec;
    bool reuse = cfg.reuse;
    if (!reuse || resume::recorded(rdir / "extract.sig") != extract_sig) {
        // 图像对列表与日志索引特征文件，必须与特征缓存同步失效。
        reuse = false;
        resume::clear(_workspace);
        fs::remove_all(featdir, rm_ec);
        fs::remove(matchpath, rm_ec);
    }
    if (cfg.reuse) resume::store(rdir / "extract.sig", extract_sig);

    // ---------------- 1. 特征提取 ----------------
    double t0 = now();
    ExtractStats est;
    if (int rc = extractDirectory(_imagedir, featdir, cfg, est, reuse)) {
        r.exit_code = rc;
        return r;
    }
    double t_extract = now() - t0;
    if (est.images < 2) {
        L::fail(Tag::Run, M::run_too_few_images, {(long long)est.images});
        { r.exit_code = 1; return r; }
    }
    warnIfMasksLookInverted(est);

    // auto 在提取后得知图像数，超过穷举阈值时改用 GPU 内容筛选；视频预设同样如此，显式 --pairs sequential 则保持时序。
    // 262 帧步行数据中，时序模式得到 144/74/19/12 图像的四个模型，内容筛选得到一个 254 图像模型；阈值以下时序更便宜，loop-closure 补足回访连接。
    PairMode mode = cfg.pairMode();
    if ((mode == PairMode::Exhaustive || mode == PairMode::Sequential) &&
        est.images >= 100 && !in.explicit_flags.count("pairs")) {
        const char* was = mode == PairMode::Exhaustive ? "exhaustive" : "sequential";
        mode = PairMode::Prefilter;
        L::out(Tag::Match, M::match_switch_to_selection, {(long long)est.images, was});
    } else if (mode == PairMode::Exhaustive && est.images >= 100) {
        L::warn(Tag::Match, M::match_exhaustive_quadratic,
                {(long long)est.images, (long long)(est.images * (est.images - 1) / 2)});
    }

    // ---------------- 2. 匹配与几何验证 ----------------
    t0 = now();
    std::vector<FeatureSet> feats;
    MatchesDatabase db;
    MatchStats mstats;
    VerifyCalibration calib;
    calib.setup = cfg.camera;
    const SensorCaptures sensors = loadSensorCaptures(cfg, verbose);
    calib.sensors = &sensors;
    // 缓存签名包含匹配配置、提取配置及特征文件集合，保证列表与日志索引仍有效。
    MatchResume mres;
    mres.dir = rdir;
    mres.signature = extract_sig + stageSignature(cfg, CMD_MATCH) +
                     "pairs-resolved=" + std::to_string((int)mode) + "\n" +
                     "features=" + featureDirDigest(featdir) + "\n";
    // 完整 matches.bin 可直接复用；建图只需关键点与颜色，无需读取描述子。
    bool reused_matches = false;
    if (cfg.reuse && fs::exists(matchpath, rm_ec) &&
        resume::recorded(rdir / "match.sig") == mres.signature) {
        try {
            MatchesDatabase disk = readMatches(matchpath.string());
            if (loadFeatureDir(featdir.string(), cfg, /*with_descriptors=*/false, feats, db) == 0) {
                db.pairs = std::move(disk.pairs);
                db.cameras = std::move(disk.cameras);
                db.camera_ids = std::move(disk.camera_ids);
                db.focal_prior = std::move(disk.focal_prior);
                db.focal_measured = std::move(disk.focal_measured);
                loadCameraSetup(db, calib.cameras);
                mstats.images = feats.size();
                mstats.kept = mstats.pairs = db.pairs.size();
                for (const TwoViewMatches& tvm : db.pairs) mstats.inliers += tvm.matches.size();
                reused_matches = true;
                calib.priors = makeSensorPriors(cfg, sensors, db, calib.cameras.ids);
                if (calib.priors)
                    calibrateSensorPriorsFromDatabase(
                        *calib.priors, db, feats, perImageCameras(calib.cameras, feats.size()),
                        cfg.twoview, cfg.threads, verbose);
            }
        } catch (const std::exception& e) {
            L::warn(Tag::Match, M::match_reuse_failed, {e.what()});
            feats.clear();
            db = MatchesDatabase();
        }
    }
    if (reused_matches) {
        L::out(Tag::Match, M::match_reusing_matches,
               {(long long)mstats.kept, matchpath.string()});
    } else if (int rc = matchFeatureDir(featdir.string(), cfg, mode, /*verify=*/true, feats, db,
                                        mstats, &calib, cfg.reuse ? &mres : nullptr)) {
        r.exit_code = rc;
        return r;
    }
    double t_match = now() - t0;
    if (!reused_matches) {
        writeMatches(matchpath.string(), db);
        // matches.bin 已包含日志与图像对列表的全部信息，移除同样庞大的中间日志。
        resume::forget(rdir / "matches.part");
        resume::forget(rdir / "pairs.bin");
        if (cfg.reuse) resume::store(rdir / "match.sig", mres.signature);
    }
    // 后续仅使用关键点、对应图与颜色，立即释放描述子；每图 8k 个 128 字节特征时，每千张图约占 1 GB。
    for (FeatureSet& fs : feats) {
        std::vector<uint8_t>().swap(fs.descriptors);
    }
    // 必须在 writeMatches 之后压缩，磁盘匹配索引仍指向保留全部特征行的文件。
    if (cfg.compact_unused_features) {
        FeatureCompactionPlan plan = buildFeatureCompactionPlan(db);
        for (size_t i = 0; i < feats.size(); i++)
            feats[i] = compactFeatureSet(std::move(feats[i]), plan.old_to_new[i],
                                         plan.compact_counts[i]);
        remapMatches(db, plan, feats);
        if (verbose) reportFeatureCompaction(plan.stats);
    }

    // ---------------- 3. 增量建图 ----------------
    // 沿用双视图阶段的相机分组与测量焦距，避免小连通分量从几何默认值出发拟合出相互不一致的内参（D45/D46）。
    MapperOptions& mapopt = cfg.mapper;
    const CameraSetup& cs = calib.cameras;
    mapopt.initial_cameras = cs.cameras;
    mapopt.known_focal_cameras = cs.focal_known;
    mapopt.given_focal_cameras = cs.focal_given;
    mapopt.measured_focal_cameras = cs.focal_measured;

    t0 = now();
    events::stage_begin(Stage::Map, (int64_t)db.images.size());
    events::map_begin(db.images.size());
    RigTable rigs;
    SequenceTable seqs;
    try {
        rigs = buildRigs(db, cfg, verbose);
        seqs = buildSequences(db, cfg, verbose);
    } catch (const std::runtime_error& e) {
        L::fail(Tag::Map, M::rig_bad, {e.what()});
        r.exit_code = 2;
        return r;
    }
    Mapper mapper(db, feats, mapopt, cs.ids, &rigs, &seqs,
                  cfg.sensor_map ? calib.priors.get() : nullptr);
    AssembleStats ast;
    std::vector<Reconstruction> models = runMapper(mapper, db, feats, cfg, ast);
    double t_map = now() - t0;

    {
        // 每个模型执行一次全局优化，最多再加两轮；大型数据中可耗时数分钟，需单独报告进度。
        events::stage_begin(Stage::Refine);
        double t_finish = 0;
        models = finishModels(mapper, std::move(models), cfg, verbose, t_finish);
        t_map += t_finish;
        events::stage_end(Stage::Refine);
    }
    events::stage_end(Stage::Map);

    resolveImageNames(models, _imagedir);
    std::vector<ModelGauge> gauge;
    const bool auto_metric = fixGauge(models, cfg, _imagedir, verbose, gauge, &sensors);
    recolorPoints(models, cfg);
    // 规范对齐后刷新快照，避免屏幕仍显示种子坐标系下倾斜的模型。
    if (!gauge.empty()) progress::gauge(gauge[0].oriented, gauge[0].metric);
    if (!models.empty()) progress::model(models.front(), /*force=*/true);
    // 写出尺寸拆分前统计相机数，摘要应描述估计的分组，而非每种图像尺寸对应的输出相机。
    const size_t n_cameras = models.empty() ? 0 : models.front().cameras.size();
    splitCamerasBySize(models, feats);
    writeModels(models, sparsedir, verbose, gauge, &rigs);

    // run() 已报告建图计时，后续装配继续累加同一组计数器。
    g_map_prof.report(t_map, "map");

    // ---------------- 结果报告 ----------------
    const Reconstruction& rec = models.front();
    double mean = 0, median = 0;
    size_t nobs = 0;
    reprojStats(rec, feats, mean, median, nobs);
    const uint32_t reg = rec.numRegistered();
    L::out(Tag::Run, M::sum_header);
    L::out(Tag::Run, M::sum_extract,
           {format_duration(t_extract), (long long)est.images,
            (long long)est.features});
    if (est.masked_images) {
        // 仅统计本次实际提取部分，复用特征文件无法得知掩码排除数。
        const uint64_t before = est.features_new + est.masked_out;
        L::out(Tag::Run, M::sum_masks,
               {(long long)est.masked_images, (long long)est.images,
                (long long)est.masked_out,
                L::num(before ? 100.0 * est.masked_out / before : 0.0, 1)});
    }
    if (reused_matches)
        L::out(Tag::Run, M::sum_match_reused,
               {(long long)mstats.kept, (long long)mstats.inliers});
    else
        L::out(Tag::Run, M::sum_match,
               {format_duration(t_match), (long long)mstats.kept, (long long)mstats.pairs,
                (long long)mstats.inliers, (long long)mstats.putative});
    L::out(Tag::Run, M::sum_map,
           {format_duration(t_map), (long long)reg, (long long)est.images,
            (long long)rec.points3D.size(), (long long)n_cameras});
    printAssembly(ast, models.size(), Tag::Run);
    printFolderCoverage(models, db);
    writeUnregisteredList(models, db, _imagedir);
    L::out(Tag::Run, M::sum_total,
           {format_duration(t_extract + t_match + t_map)});
    L::out(Tag::Run, M::sum_model_error,
           {L::num(mean, 3), L::num(median, 3), (long long)nobs});
    if (models.size() > 1) {
        // 分散重建中 sparse/0 为最大模型，其余模型之间没有已确定的坐标变换（D41）。
        L::out(Tag::Run, M::sum_components,
               {(long long)models.size(), (long long)distinctRegistered(models),
                (long long)est.images});
        printExtraModels(models, feats);
    }
    L::out(Tag::Run, M::sum_written,
           {models.size() > 1
                ? sparsedir.string() + "/{0.." + std::to_string(models.size() - 1) + "}"
                : (sparsedir / "0").string()});

    // 结果阈值较宽松，用于快速识别明显失败，不表示已达到最优重建质量。
    const double frac = est.images ? (double)reg / est.images : 0.0;
    // 通过结构化数据分别报告重建质量与公制状态，避免单个退出码无法同时表示 3 和 4。
    auto emitResult = [&](bool part) {
        r.registered = reg;
        r.images = est.images;
        r.points = (int64_t)rec.points3D.size();
        r.models = (int64_t)models.size();
        r.mean_reproj = mean;
        r.median_reproj = median;
        r.partial = part;
        r.metric = auto_metric;
        r.sparse_dir = sparsedir;
        Event ev;
        ev.kind = Event::Kind::Result;
        ev.stage = Stage::Finish;
        ev.registered = reg;
        ev.images = est.images;
        ev.points = (int64_t)rec.points3D.size();
        ev.models = (int64_t)models.size();
        ev.mean_reproj = mean;
        ev.partial = part;
        ev.metric = auto_metric;
        events::emit(ev);
    };
    if (reg < 2 || rec.points3D.empty()) {
        L::out(Tag::Run, M::result_failed);
        emitResult(true);
        { r.exit_code = 2; return r; }
    }
    const bool partial = frac < 0.5 || mean > 2.0;
    emitResult(partial);
    if (partial) L::out(Tag::Run, M::result_partial, {L::num(100 * frac, 0), L::num(mean, 2)});
    // 重建质量优先决定退出状态；质量合格时才以公制规范结果决定退出码，两类结果始终输出供 GUI 读取。
    if (!auto_metric) L::out(Tag::Run, M::result_not_metric);
    if (!partial && auto_metric)
        L::out(Tag::Run, M::result_ok, {L::num(100 * frac, 0), L::num(mean, 2)});
    r.exit_code = partial ? 3 : (auto_metric ? 0 : 4);
    return r;
}

// ---------------- 运行上下文 ----------------

// 析构时恢复接收端，使前端可继续运行下一任务，并避免 Cancelled 异常留下悬空对象引用。
RunContext::~RunContext() {
    slog::set_sink({});
    events::set_sink({});
    cancel::set_token(nullptr);
    progress::set_dir("");
}

void RunContext::set_log(slog::Sink s) { slog::set_sink(std::move(s)); }
void RunContext::set_events(events::Sink s) { events::set_sink(std::move(s)); }
void RunContext::set_cancel(const std::atomic<bool>* flag) { cancel::set_token(flag); }
void RunContext::set_progress_dir(const std::string& dir) { progress::set_dir(dir); }

// ---------------- 读取设置列表 ----------------

namespace {

// 相机模型、焦距与畸变的裸值设置全局默认，PREFIX=VALUE 设置单个相机分组。
bool cameraOverrideArg(SfmConfig& cfg, OverrideKind kind, const std::string& v,
                       std::set<std::string>& seen, std::string& err) {
    const char* flag = kind == OverrideKind::Focal        ? "focal"
                       : kind == OverrideKind::Distortion ? "distortion"
                                                          : "camera-model";
    const char* form = kind == OverrideKind::Focal        ? "F or PREFIX=F"
                       : kind == OverrideKind::Distortion ? "k1,k2,... or PREFIX=k1,k2,..."
                                                          : "MODEL or PREFIX=MODEL";
    if (!parseCameraOverride(v, kind, cfg.camera.overrides)) {
        err = std::string("bad --") + flag + " '" + v + "' (" + form + ")";
        return false;
    }
    if (v.find('=') != std::string::npos) return true;  // 仅针对相机分组
    switch (kind) {
        case OverrideKind::Focal: cfg.focal = std::atof(v.c_str()); break;
        case OverrideKind::Distortion: cfg.distortion = v; break;
        case OverrideKind::Model: {
            CamModel m;
            if (!parseCamModelName(v, m)) {
                err = "unknown --camera-model '" + v + "'";
                return false;
            }
            cfg.camera_model = v;
            break;
        }
    }
    seen.insert(flag);
    return true;
}

bool cameraOverrideName(const std::string& a, OverrideKind& kind) {
    if (a == "--camera-model") kind = OverrideKind::Model;
    else if (a == "--focal") kind = OverrideKind::Focal;
    else if (a == "--distortion") kind = OverrideKind::Distortion;
    else return false;
    return true;
}

}  // 匿名命名空间

std::string parse_auto_args(const std::vector<std::string>& args, AutoRequest& out) {
    // 将设置转换成 setConfigField 所需的 argv 视图。
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    const int argc = (int)argv.size();

    SfmConfig& cfg = out.cfg;
    std::set<std::string> seen;
    std::string imagedir, workspace, manifest_path;
    bool maskdir_explicit = false;

    for (int i = 0; i < argc; i++) {
        const std::string a = args[(size_t)i];
        if (a == "--help" || a == "-h") { out.wants_help = true; return {}; }
        if (a == "--output" || a == "-o") {
            if (i + 1 >= argc) return "--output: missing value";
            workspace = args[(size_t)++i];
            continue;
        }
        if (a == "--manifest") {
            if (i + 1 >= argc) return "--manifest: missing value";
            manifest_path = args[(size_t)++i];
            continue;
        }
        if (a == "--rig") {
            if (i + 1 >= argc) return "--rig: missing value";
            RigDef d;
            if (std::string err = parseRigArg(args[(size_t)++i], d); !err.empty()) return err;
            cfg.rigs.push_back(std::move(d));
            continue;
        }
        if (a == "--sequence") {
            if (i + 1 >= argc) return "--sequence: missing value";
            SequenceDef d;
            if (std::string err = parseSequenceArg(args[(size_t)++i], d); !err.empty())
                return err;
            cfg.sequences.push_back(std::move(d));
            continue;
        }
        if (a == "--progress-dir") {
            if (i + 1 >= argc) return "--progress-dir: missing value";
            out.progress_dir = args[(size_t)++i];
            continue;
        }
        // 禁用掩码时还需标记该选项已显式设置，防止 manifest_apply 再用清单值补回掩码。
        if (a == "--no-masks") {
            cfg.mask_dir.clear();
            maskdir_explicit = true;
            seen.insert("masks");
            continue;
        }
        if (a == "--no-manage") {
            cfg.manager.do_merge = cfg.manager.do_grow = cfg.manager.do_reseed = false;
            cfg.manager.do_audit = cfg.manager.do_split = cfg.manager.do_duplicate_split = false;
            continue;
        }
        if (OverrideKind kind; cameraOverrideName(a, kind)) {
            if (i + 1 >= argc) return a + ": missing value";
            std::string err;
            if (!cameraOverrideArg(cfg, kind, args[(size_t)++i], seen, err)) return err;
            continue;
        }
        std::string err;
        const FieldResult r =
            setConfigField(cfg, CMD_AUTO, a, argc, argv.data(), i, seen, err);
        if (r == FieldResult::Error) return err;
        if (r == FieldResult::Ok) continue;
        if (a[0] == '-') return "unknown option " + a;
        if (!imagedir.empty()) return "unexpected argument '" + a + "'";
        imagedir = a;
    }
    if (workspace.empty()) return "--output WORKSPACE is required";
    maskdir_explicit = maskdir_explicit || seen.count("masks") || seen.count("mask-dir");

    // 优先级为显式命令行、清单文件、预设；各阶段共享已声明字段集合，避免覆盖高优先级值。
    std::vector<PresetChange> moved;
    if (std::string err = applyPresets(cfg, seen, moved); !err.empty()) return err;
    if (!manifest_path.empty()) {
        Manifest man;
        try {
            man = manifest_read(manifest_path);
        } catch (const std::exception& e) {
            return e.what();
        }
        if (std::string err = manifest_apply(man, cfg, seen, imagedir); !err.empty())
            return manifest_path + ": " + err;
        if (!man.mask_dir.empty()) maskdir_explicit = true;
    }
    if (std::string err = cfg.finalize(CMD_AUTO); !err.empty()) return err;
    if (std::string err = cfg.resolveDevice(); !err.empty()) return err;

    out.in.image_dir = imagedir;
    out.in.workspace = workspace;
    out.in.mask_dir_explicit = maskdir_explicit;
    out.in.explicit_flags = std::move(seen);
    out.in.preset_changes = std::move(moved);
    return {};
}

}  // 命名空间 sfm
