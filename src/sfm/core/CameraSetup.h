// 分别确定图像内参共享方式、相机模型与初始焦距；共享方式考虑文件夹、图像、EXIF 身份及尺寸。
// 模型和焦距可按 PREFIX=VALUE 覆盖各组，允许同一装置混用普通镜头与鱼眼。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "i18n/catalog/Sfm.h"
#include "sfm/core/Camera.h"
#include "sfm/core/Features.h"
#include "sfm/core/Matches.h"

namespace sfm {

enum class CameraMode { Single, Folder, Image };

// 选项值保留标识符，只翻译其含义说明。
inline const char* cameraModeName(CameraMode m) {
    namespace msg = spirula::i18n::msg::sfm;
    switch (m) {
        case CameraMode::Folder: return msg::camera_mode_folder.get();
        case CameraMode::Image:  return msg::camera_mode_image.get();
        default:                 return msg::camera_mode_single.get();
    }
}

inline bool parseCameraMode(const std::string& v, CameraMode& out) {
    if (v == "single" || v == "auto") out = CameraMode::Single;  // auto 为兼容名称
    else if (v == "folder") out = CameraMode::Folder;
    else if (v == "image") out = CameraMode::Image;
    else return false;
    return true;
}

// 按相对路径前缀匹配分组；cam0 匹配 cam0/00017 或 cam0/sub/x，空前缀表示全局默认。
struct CameraOverride {
    std::string prefix;
    bool has_model = false;
    CamModel model = CamModel::OpenCV;
    bool has_focal = false;
    double focal = 0;
    // 初始畸变按 packIntrinsics 的 BA 顺序，空值从零开始。
    bool has_extra = false;
    std::vector<double> extra;
};

// PREFIX=VALUE 指定的分组设置类型。
enum class OverrideKind { Model, Focal, Distortion };

// 将 k1,k2,... 解析为 BA 顺序系数；空列表或不能完整解析为数值的项均失败。
inline bool parseDistortion(const std::string& v, std::vector<double>& out) {
    out.clear();
    if (v.empty()) return false;
    for (size_t i = 0;;) {
        size_t c = v.find(',', i);
        if (c == std::string::npos) c = v.size();
        const std::string tok = v.substr(i, c - i);
        try {
            size_t used = 0;
            out.push_back(std::stod(tok, &used));
            if (used != tok.size()) return false;
        } catch (const std::exception&) {
            return false;
        }
        if (c == v.size()) break;
        i = c + 1;
    }
    return true;
}

// 解析 VALUE 或 PREFIX=VALUE，同前缀合并为一项，使模型、焦距、畸变可分别设置同一组。
inline bool parseCameraOverride(const std::string& arg, OverrideKind kind,
                                std::vector<CameraOverride>& out) {
    std::string prefix, value = arg;
    size_t eq = arg.find('=');
    if (eq != std::string::npos) {
        prefix = arg.substr(0, eq);
        value = arg.substr(eq + 1);
        while (!prefix.empty() && (prefix.back() == '/' || prefix.back() == '\\')) prefix.pop_back();
    }
    CamModel m = CamModel::OpenCV;
    double f = 0;
    std::vector<double> extra;
    if (kind == OverrideKind::Focal) {
        try {
            f = std::stod(value);
        } catch (const std::exception&) {
            return false;
        }
        if (!(f > 0)) return false;
    } else if (kind == OverrideKind::Distortion) {
        if (!parseDistortion(value, extra)) return false;
    } else if (!parseCamModelName(value, m)) {
        return false;
    }
    CameraOverride* e = nullptr;
    for (CameraOverride& o : out)
        if (o.prefix == prefix) { e = &o; break; }
    if (!e) {
        out.push_back(CameraOverride{});
        e = &out.back();
        e->prefix = prefix;
    }
    switch (kind) {
        case OverrideKind::Focal: e->has_focal = true; e->focal = f; break;
        case OverrideKind::Distortion: e->has_extra = true; e->extra = std::move(extra); break;
        case OverrideKind::Model: e->has_model = true; e->model = m; break;
    }
    return true;
}

struct CameraSetupOptions {
    CameraMode mode = CameraMode::Folder;
    // 未被覆盖项匹配的图像使用默认；focal <= 0 表示无先验，从几何估计开始。
    CamModel model = CamModel::OpenCV;
    double focal = 0;
    // 全局初始畸变使用模型 BA 顺序，组覆盖优先，空值为零。
    std::vector<double> extra;
    std::vector<CameraOverride> overrides;
    // 无显式组焦距时默认采用 EXIF 测量；24 mm 全画幅上几何猜测曾偏长 85%，真实镜头信息更可靠（D46）。
    bool exif_focal = true;
    // 在相机模式之外按 EXIF 机身、尺寸及焦距簇分组；固定镜头不受影响，变焦集合可避免错误共享。
    // 83 张网络照片焦距跨 5.4k–60k px，强制共享仅配准 55 张、AUC 8.5（D48）。
    bool exif_groups = true;
    // 显式 camera-mode 固定模式；否则明显的照片集合可自动从 Folder 改为 Image（D48）。
    bool mode_explicit = false;
    // 焦距单链接聚类的相对容差须超过 EXIF 1 mm 量化误差（24 mm 时约 4%），同时明显小于真实变焦跨度。
    double exif_focal_tol = 0.10;
};

struct CameraSetup {
    std::vector<uint32_t> ids;                  // 逐图像相机 ID，从 1 开始
    std::map<uint32_t, Camera> cameras;         // 逐相机 ID 初始内参
    // focal_given 来自 EXIF 或手动输入，禁止双视图搜索覆盖；focal_known 还要求属于当前组，禁止逐相机扫描覆盖，全局焦距仅 given。
    std::set<uint32_t> focal_given;
    std::set<uint32_t> focal_known;             // 当前组已知的焦距
    // focal_measured 为组内双视图测量，归入 known 而非 given，仍允许试探重建和 BA 精化，但不能由单图少量内点覆盖。
    std::set<uint32_t> focal_measured;
    std::map<uint32_t, std::string> labels;     // ID 到分组键的映射
    size_t exif_focal_images = 0;               // 含 EXIF 焦距的图像数
    size_t exif_camera_images = 0;              // 含 EXIF 相机身份的图像数
    size_t dim_buckets = 0;                     // 按 2% 容差统计的不同尺寸数
    // 下项统计因包含多种尺寸而进一步拆分的相机组。
    size_t size_split_groups = 0;
    CameraMode mode_used = CameraMode::Folder;  // 自动切换后的实际模式
    bool mode_switched = false;                 // 是否发生自动切换

    uint32_t count() const { return (uint32_t)cameras.size(); }
    // 混合宽角与普通镜头时，跨组对不能使用针孔像素几何，整个数据集必须采用标定视线验证（D46）。
    bool anyWide() const {
        for (const auto& kv : cameras)
            if (kv.second.wideFov()) return true;
        return false;
    }
    // 仍使用几何猜测焦距的普通镜头组，需要极线焦距搜索。
    bool anyGuessedRectilinear() const {
        for (const auto& kv : cameras)
            if (!kv.second.wideFov() && !focal_given.count(kv.first)) return true;
        return false;
    }
    bool mixed() const {
        bool wide = false, rect = false;
        for (const auto& kv : cameras) (kv.second.wideFov() ? wide : rect) = true;
        return wide && rect;
    }
};

namespace detail {

// 按路径分量检查 prefix 是否匹配 name，空前缀匹配全部。
inline bool cameraPrefixMatches(const std::string& name, const std::string& prefix) {
    if (prefix.empty()) return true;
    if (name.size() < prefix.size()) return false;
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    return name.size() == prefix.size() || name[prefix.size()] == '/';
}

// 最长匹配前缀优先，cam0 覆盖默认值，rig/cam0 覆盖 rig。
inline const CameraOverride* cameraOverrideFor(const std::string& name,
                                               const std::vector<CameraOverride>& ovr) {
    const CameraOverride* best = nullptr;
    for (const CameraOverride& o : ovr) {
        if (!cameraPrefixMatches(name, o.prefix)) continue;
        if (!best || o.prefix.size() > best->prefix.size()) best = &o;
    }
    return best;
}

// 以 / 分隔的相对路径的父目录，顶层文件返回空。
inline std::string parentPath(const std::string& name) {
    size_t s = name.find_last_of('/');
    return s == std::string::npos ? std::string() : name.substr(0, s);
}

// 按相邻排序焦距的相对差做单链接聚类，避免固定分桶边界拆开近似焦距；非正值统一归入 -1，与有效测量分开。
inline std::vector<int> exifFocalClusters(const std::vector<double>& focals, double tol) {
    std::vector<size_t> order;
    for (size_t i = 0; i < focals.size(); i++)
        if (focals[i] > 0) order.push_back(i);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return focals[a] < focals[b]; });
    std::vector<int> label(focals.size(), -1);
    int next = 0;
    for (size_t k = 0; k < order.size(); k++) {
        if (k && focals[order[k]] > focals[order[k - 1]] * (1.0 + tol)) next++;
        label[order[k]] = next;
    }
    return label;
}

}  // 命名空间 detail

// 按 2% 容差分桶后，用不同尺寸数与图像数之比识别照片集合，避免错误共享内参；真实采集比值约 0.005–0.022，三个网络集合为 0.181/0.184/0.322。
// 550 张 botanical_garden 预处理后约 24 种近似尺寸经容差归并为 4 组；最小样本限制避免少量图像造成误判。
inline bool looksLikePhotoCollection(const std::vector<FeatureSet>& feats,
                                     size_t* buckets_out = nullptr,
                                     double ratio = 0.08, size_t min_images = 20,
                                     size_t min_buckets = 4) {
    std::vector<std::pair<int, int>> reps;
    for (const FeatureSet& f : feats) {
        bool found = false;
        for (const auto& r : reps)
            if (std::abs(f.width - r.first) <= 0.02 * r.first &&
                std::abs(f.height - r.second) <= 0.02 * r.second) { found = true; break; }
        if (!found) reps.push_back({f.width, f.height});
    }
    if (buckets_out) *buckets_out = reps.size();
    if (feats.size() < min_images || reps.size() < min_buckets) return false;
    return (double)reps.size() > ratio * (double)feats.size();
}

// 所有模式先按 2% 容差分辨率分组，兼容裁剪、去畸变产生的小幅尺寸波动，主点偏差再由 BA 优化。
// 550 图数据若把约 24 个近似尺寸各自分组，会削弱焦距约束；不同相机模型或显式焦距始终不能跨组共享。
inline CameraSetup buildCameras(const std::vector<ImageEntry>& images,
                                const std::vector<FeatureSet>& feats,
                                const CameraSetupOptions& opt) {
    CameraSetup out;
    out.ids.assign(images.size(), 1);
    // 识别出照片集合时自动使用逐图像内参，相当于由数据选择网络照片预设。
    CameraMode mode = opt.mode;
    if (!opt.mode_explicit && mode == CameraMode::Folder &&
        looksLikePhotoCollection(feats, &out.dim_buckets)) {
        mode = CameraMode::Image;
        out.mode_switched = true;
    } else {
        looksLikePhotoCollection(feats, &out.dim_buckets);
    }
    out.mode_used = mode;
    std::map<std::string, uint32_t> key2id;
    std::vector<std::pair<int, int>> reps;  // 各分辨率桶的代表尺寸
    auto dimBucket = [&](int w, int h) {
        for (size_t b = 0; b < reps.size(); b++)
            if (std::abs(w - reps[b].first) <= 0.02 * reps[b].first &&
                std::abs(h - reps[b].second) <= 0.02 * reps[b].second)
                return b;
        reps.push_back({w, h});
        return reps.size() - 1;
    };

    std::map<uint32_t, std::vector<double>> exif_focals;  // 按 ID 收集，供取中位数
    std::map<uint32_t, std::vector<double>> px_scales;    // 同样按 ID 收集
    std::map<uint32_t, size_t> first_image;               // 按 ID 收集帧尺寸

    // 首遍按模式、尺寸与覆盖项确定基础分组；汇总组内焦距后再做 EXIF 聚类，第二遍生成相机。
    std::vector<std::string> base_key(images.size());
    std::map<std::string, std::set<size_t>> group_sizes;  // 模式键到分桶的映射
    for (size_t i = 0; i < images.size(); i++) {
        const std::string& name = images[i].name;
        const CameraOverride* ovr = detail::cameraOverrideFor(name, opt.overrides);
        const size_t bucket = dimBucket(feats[i].width, feats[i].height);
        char dims[64];
        snprintf(dims, sizeof dims, "r%zu", bucket);
        std::string key, group;
        switch (mode) {
            case CameraMode::Single: key = dims; break;
            case CameraMode::Image:  key = name; break;
            // 采用完整相对父路径，确保 rig/cam0、rig/cam1 与 rig 本身形成不同相机组。
            case CameraMode::Folder: group = detail::parentPath(name);
                                     key = group + "|" + dims; break;
        }
        if (mode != CameraMode::Image) group_sizes[group].insert(bucket);
        // 显式覆盖项将其匹配图像独立成组，即使基础模式原本会将其合并。
        if (ovr && !ovr->prefix.empty()) key += "|@" + ovr->prefix;
        base_key[i] = key;
    }

    for (const auto& kv : group_sizes)
        if (kv.second.size() > 1) out.size_split_groups++;

    // 仅在各基础组内部聚类 EXIF 焦距；不同基础组本来就不共享相机。
    std::vector<int> focal_cluster(images.size(), -1);
    if (opt.exif_groups) {
        std::map<std::string, std::vector<size_t>> by_base;
        for (size_t i = 0; i < images.size(); i++) by_base[base_key[i]].push_back(i);
        for (const auto& kv : by_base) {
            std::vector<double> f;
            f.reserve(kv.second.size());
            for (size_t i : kv.second) f.push_back(feats[i].exif_focal);
            std::vector<int> lab = detail::exifFocalClusters(f, opt.exif_focal_tol);
            for (size_t k = 0; k < kv.second.size(); k++) focal_cluster[kv.second[k]] = lab[k];
        }
    }

    for (size_t i = 0; i < images.size(); i++) {
        const std::string& name = images[i].name;
        const CameraOverride* ovr = detail::cameraOverrideFor(name, opt.overrides);
        std::string key = base_key[i];
        if (opt.exif_groups) {
            if (!feats[i].exif_camera.empty()) key += "|" + feats[i].exif_camera;
            if (focal_cluster[i] >= 0) key += "|f" + std::to_string(focal_cluster[i]);
        }

        auto it = key2id.find(key);
        if (it == key2id.end()) {
            it = key2id.emplace(key, (uint32_t)key2id.size() + 1).first;
            out.labels[it->second] = key;
            first_image[it->second] = i;
        }
        const uint32_t id = it->second;
        out.ids[i] = id;
        if (feats[i].exif_focal > 0) {
            exif_focals[id].push_back(feats[i].exif_focal);
            out.exif_focal_images++;
        }
        if (!feats[i].exif_camera.empty()) out.exif_camera_images++;
        px_scales[id].push_back(feats[i].pixelScale());
    }

    for (const auto& kv : first_image) {
        const uint32_t id = kv.first;
        const size_t i = kv.second;
        const CameraOverride* ovr = detail::cameraOverrideFor(images[i].name, opt.overrides);
        const CamModel model = ovr && ovr->has_model ? ovr->model : opt.model;
        double focal = 0;
        bool known = false, given = false;
        if (ovr && ovr->has_focal) {
            focal = ovr->focal;
            given = true;
            known = !ovr->prefix.empty();  // 参见 CameraSetup::focal_given
        } else if (opt.exif_focal && exif_focals.count(id)) {
            // 采用组内焦距中位数，避免固定变焦设置的量化抖动或单张异常改变初值。
            std::vector<double>& v = exif_focals[id];
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            focal = v[v.size() / 2];
            given = known = true;
        } else if (opt.focal > 0) {
            focal = opt.focal;   // 全局焦距：given，但不是 known
            given = true;
        }
        out.cameras[id] = Camera::defaultFor(id, feats[i].width, feats[i].height, focal, model);
        if (ovr && ovr->has_extra) setExtraParams(out.cameras[id], ovr->extra);
        else if (!opt.extra.empty()) setExtraParams(out.cameras[id], opt.extra);
        // 球面相机由图像尺寸精确标定，无论参数来源都视为焦距已知，跳过双视图与试探重建的焦距搜索（D49）。
        if (out.cameras[id].isSpherical()) given = known = true;
        // pixel_scale 同样取组内中位数，避免单帧异常影响整个组。
        std::vector<double>& ps = px_scales[id];
        if (!ps.empty()) {
            std::nth_element(ps.begin(), ps.begin() + ps.size() / 2, ps.end());
            out.cameras[id].pixel_scale = std::max(1.0, ps[ps.size() / 2]);
        }
        if (given) out.focal_given.insert(id);
        if (known) out.focal_known.insert(id);
    }
    return out;
}

// ---------------- 通过 matches.bin 传递相机设置（D47）----------------
// 双视图阶段基于原始候选匹配测量焦距；传给建图器可避免从已受验证焦距偏置的内点重新搜索。

inline void storeCameraSetup(MatchesDatabase& db, const CameraSetup& cs) {
    db.cameras.clear();
    db.focal_prior.clear();
    db.focal_measured.clear();
    for (const auto& kv : cs.cameras) {
        db.cameras.push_back(kv.second);
        db.focal_prior.push_back(cs.focal_known.count(kv.first) ? 1 : 0);
        db.focal_measured.push_back(cs.focal_measured.count(kv.first) ? 1 : 0);
    }
    db.camera_ids = cs.ids;
}

// 反向读取设置，文件无记录时返回 false 且不修改 cs。
inline bool loadCameraSetup(const MatchesDatabase& db, CameraSetup& cs) {
    if (!db.hasCameras()) return false;
    cs = CameraSetup();
    cs.ids = db.camera_ids;
    for (size_t i = 0; i < db.cameras.size(); i++) {
        const Camera& cam = db.cameras[i];
        cs.cameras[cam.id] = cam;
        const bool prior = i < db.focal_prior.size() && db.focal_prior[i];
        // 双视图测量焦距不升级为 given，仍允许建图试探与精化，保证读回重建与原运行一致。
        if (i < db.focal_measured.size() && db.focal_measured[i]) {
            cs.focal_measured.insert(cam.id);
            cs.focal_known.insert(cam.id);
            continue;
        }
        // 焦距偏离几何默认值则视为来自测量，避免重复搜索；等于默认值时仍视为猜测并如实报告。
        const Camera def = Camera::defaultFor(cam.id, cam.width, cam.height, 0, cam.model);
        if (prior || std::fabs(cam.focal() - def.focal()) > 1e-6) cs.focal_given.insert(cam.id);
        if (prior) cs.focal_known.insert(cam.id);
    }
    return true;
}

}  // 命名空间 sfm
