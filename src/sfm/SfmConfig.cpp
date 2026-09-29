// 参数解析、预设与帮助均通过 SfmConfig.h 的 SFM_CONFIG_FIELDS 展开，新增选项只需一条定义。
#include "sfm/SfmConfig.h"

#include "sfm/core/Log.h"
#include "sfm/vk/VkContext.h"
#include "i18n/catalog/SfmFields.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace sfm {
namespace {

// ---------------- 各字段类型的值与字符串转换 ----------------

std::string valueString(bool v) { return v ? "on" : "off"; }
std::string valueString(const std::string& v) { return v.empty() ? "none" : v; }

template <class T>
std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value, std::string>
valueString(T v) {
    char buf[32];
    if (std::is_signed<T>::value) std::snprintf(buf, sizeof buf, "%lld", (long long)v);
    else std::snprintf(buf, sizeof buf, "%llu", (unsigned long long)v);
    return buf;
}

template <class T>
std::enable_if_t<std::is_floating_point<T>::value, std::string> valueString(T v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", (double)v);
    return buf;
}

// ---------------- 单参数到字段的解析 ----------------

std::string rangeText(double lo, double hi) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "expected a value in [%g, %g]", lo, hi);
    return buf;
}

template <class T>
std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value, bool>
parseValue(T& out, const std::string& s, double lo, double hi, const char*, std::string& err) {
    long long v = 0;
    try {
        size_t p = 0;
        v = std::stoll(s, &p);
        if (p != s.size()) throw std::invalid_argument("trailing characters");
    } catch (...) {
        err = "expected an integer";
        return false;
    }
    if (std::is_unsigned<T>::value && v < 0) {
        err = "must not be negative";
        return false;
    }
    if (lo < hi && ((double)v < lo || (double)v > hi)) {
        err = rangeText(lo, hi);
        return false;
    }
    out = (T)v;
    return true;
}

template <class T>
std::enable_if_t<std::is_floating_point<T>::value, bool> parseValue(T& out, const std::string& s,
                                                                    double lo, double hi,
                                                                    const char*, std::string& err) {
    double v = 0;
    try {
        size_t p = 0;
        v = std::stod(s, &p);
        if (p != s.size()) throw std::invalid_argument("trailing characters");
    } catch (...) {
        err = "expected a number";
        return false;
    }
    if (lo < hi && (v < lo || v > hi)) {
        err = rangeText(lo, hi);
        return false;
    }
    out = (T)v;
    return true;
}

bool parseValue(std::string& out, const std::string& s, double, double, const char* choices,
                std::string& err) {
    std::string ch = choices ? choices : "";
    if (!ch.empty()) {
        for (size_t pos = 0; pos <= ch.size();) {
            size_t bar = ch.find('|', pos);
            std::string tok = ch.substr(pos, bar == std::string::npos ? bar : bar - pos);
            if (tok == s) { out = s; return true; }
            if (bar == std::string::npos) break;
            pos = bar + 1;
        }
        err = "expected one of: " + ch;
        for (char& c : err) if (c == '|') c = ' ';
        return false;
    }
    out = s;
    return true;
}

// 布尔字段只使用 --name / --no-name，不消耗后续值，保证开关后的路径仍是位置参数。
FieldResult trySetField(bool& out, const std::string& key, const char* name, int, char**, int&,
                        double, double, const char*, std::set<std::string>& seen, std::string&) {
    if (key == name) out = true;
    else if (key.size() > 3 && key.compare(0, 3, "no-") == 0 && key.compare(3, std::string::npos, name) == 0)
        out = false;
    else return FieldResult::Unknown;
    seen.insert(name);
    return FieldResult::Ok;
}

template <class T>
FieldResult trySetField(T& out, const std::string& key, const char* name, int argc, char** argv,
                        int& i, double lo, double hi, const char* choices,
                        std::set<std::string>& seen, std::string& error) {
    if (key != name) return FieldResult::Unknown;
    if (i + 1 >= argc) {
        error = std::string("--") + name + ": missing value";
        return FieldResult::Error;
    }
    std::string why;
    if (!parseValue(out, argv[++i], lo, hi, choices, why)) {
        error = std::string("--") + name + " " + argv[i] + ": " + why;
        return FieldResult::Error;
    }
    seen.insert(name);
    return FieldResult::Ok;
}

// --Max_Error 与 --max-error 等价，兼容 GUI 和配置字段中的下划线形式。
std::string normalizeKey(const std::string& arg) {
    std::string s = arg;
    while (!s.empty() && s[0] == '-') s.erase(s.begin());
    for (char& c : s) if (c == '_') c = '-';
    return s;
}

// ---------------- 帮助格式 ----------------

// 过长的候选值列表移到说明文本，不挤占选项列。
constexpr size_t kInlineChoices = 26;

std::string metavarFor(bool, const char*, const char*) { return ""; }
std::string metavarFor(const std::string&, const char* name, const char* choices) {
    if (choices && *choices) {
        // 有候选值的字段不接受路径；例如 features 在 map 中是目录，在 extract 中是前端名称，只有后者有候选列表。
        return std::string(choices).size() <= kInlineChoices
                   ? "{" + std::string(choices) + "}"
                   : "VALUE";
    }
    // 根据字段名生成参数占位说明，路径采用 DIR，无需另设配置表列。
    std::string n = name;
    if (n.find("dir") != std::string::npos || n == "masks" || n == "images" ||
        n == "features" || n == "resume")
        return "DIR";
    if (n.find("path") != std::string::npos) return "FILE";
    return "VALUE";
}
template <class T>
std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value, std::string>
metavarFor(T, const char*, const char*) {
    return "N";
}
template <class T>
std::enable_if_t<std::is_floating_point<T>::value, std::string> metavarFor(T, const char*,
                                                                          const char*) {
    return "X";
}

// 布尔选项显示能改变默认值的开关，如 --no-verify，而非 --verify 0。
std::string flagFor(bool v, const char* name) {
    return v ? "--no-" + std::string(name) : "--" + std::string(name);
}
template <class T>
std::string flagFor(const T&, const char* name) {
    return "--" + std::string(name);
}

constexpr int kFlagCol = 42;  // 选项列宽，容纳最长选项及参数占位符
constexpr int kWidth = 96;

// 使用 i18n::wrap 按终端列宽换行，支持无空格的中日文按字符断行，不能仅按字节和空格处理。
void printWrapped(FILE* out, const std::string& text, int indent) {
    for (const std::string& line : spirula::i18n::wrap(text, kWidth - indent))
        std::fprintf(out, "%*s%s\n", indent, "", line.c_str());
}

// 分组标题按配置表字符串查找；缺少译名时沿用原始标识符。
const char* groupLabel(const char* group) {
    namespace F = spirula::i18n::msg::sfmfield;
    struct Row { const char* key; const spirula::i18n::Msg* msg; };
    static const Row kRows[] = {
        {"pipeline", &F::group_pipeline}, {"colour", &F::group_colour},
        {"camera", &F::group_camera},
        {"features", &F::group_features}, {"matching", &F::group_matching},
        {"mapper", &F::group_mapper},     {"rig", &F::group_rig},
        {"manage", &F::group_manage},
        {"merge", &F::group_merge},       {"input", &F::group_input},
        {"runtime", &F::group_runtime},
    };
    for (const Row& r : kRows)
        if (std::strcmp(r.key, group) == 0) return r.msg->get();
    return group;
}

void printOption(FILE* out, const std::string& flag, const std::string& metavar,
                 const std::string& value, const std::string& help, const char* choices) {
    std::string left = "  " + flag + (metavar.empty() ? "" : " " + metavar);
    if (value.empty()) {
        std::fprintf(out, "%s\n", left.c_str());
    } else {
        if ((int)left.size() >= kFlagCol) std::fprintf(out, "%s\n%*s", left.c_str(), kFlagCol, "");
        else std::fprintf(out, "%-*s", kFlagCol, left.c_str());
        std::fprintf(out, "[%s]\n", value.c_str());
    }
    std::string h = help;
    std::string ch = choices ? choices : "";
    if (!ch.empty() && ch.size() > kInlineChoices) {
        for (char& c : ch) if (c == '|') c = ' ';
        h += " (one of: " + ch + ")";
    }
    printWrapped(out, h, 6);
}

}  // 匿名命名空间

// ---------------- 配置字段赋值 ----------------

FieldResult setConfigField(SfmConfig& cfg, uint32_t cmd, const std::string& arg, int argc,
                           char** argv, int& i, std::set<std::string>& seen, std::string& error) {
    const std::string key = normalizeKey(arg);
    if (key.empty()) return FieldResult::Unknown;
#define SFM_TRY_SET(member, name, cmds, tier, group, lo, hi, choices, help)                        \
    if ((uint32_t)(cmds) & cmd) {                                                                  \
        FieldResult r = trySetField(cfg.member, key, name, argc, argv, i, (double)(lo),            \
                                    (double)(hi), choices, seen, error);                           \
        if (r != FieldResult::Unknown) {                                                           \
            if (r == FieldResult::Ok && key == "device")                                           \
                cfg.device_request_set = true;                                                     \
            return r;                                                                               \
        }                                                                                           \
    }
    SFM_CONFIG_FIELDS(SFM_TRY_SET)
#undef SFM_TRY_SET
    return FieldResult::Unknown;
}

// ---------------- 预设 ----------------

namespace {
// 仅修改未被命令行声明的字段，并记录修改项供日志与 GUI 展示。
template <class T, class V>
void presetSet(std::set<std::string> const& seen, std::vector<PresetChange>& moved,
               const char* flag, T& member, V value) {
    if (seen.count(flag)) return;
    T next = (T)value;
    if (next == member) return;
    moved.push_back({flag, valueString(member), valueString(next)});
    member = next;
}
}  // 匿名命名空间

std::string applyPresets(SfmConfig& cfg, const std::set<std::string>& seen,
                         std::vector<PresetChange>& moved) {
    // 质量预设参考 COLMAP，按 SIFT 默认 3200 px 分级，同时调整 prefilter-neighbors 以权衡匹配耗时与图连通性。
    // 学习前端使用更低分辨率阶梯：全分辨率多通道特征图占用大量内存，检测器也天然输出更少但定位更准的点。
    const bool learned = isAlikedType(cfg.features) || isLomaType(cfg.features);
    if (cfg.quality == "low") {
        presetSet(seen, moved, "max-image-size", cfg.max_image_size, learned ? 800 : 1000);
        presetSet(seen, moved, "max-features", cfg.sift.max_num_features, 2048);
        presetSet(seen, moved, "aliked-max-features", cfg.aliked.max_num_features, 1024);
        presetSet(seen, moved, "loma-max-features", cfg.loma.max_num_features, 1024);
        presetSet(seen, moved, "prefilter-neighbors", cfg.prefilter.num_neighbors, 16);
    } else if (cfg.quality == "medium") {
        presetSet(seen, moved, "max-image-size", cfg.max_image_size, learned ? 1200 : 1600);
        presetSet(seen, moved, "max-features", cfg.sift.max_num_features, 4096);
        presetSet(seen, moved, "aliked-max-features", cfg.aliked.max_num_features, 2048);
        presetSet(seen, moved, "loma-max-features", cfg.loma.max_num_features, 2048);
        presetSet(seen, moved, "prefilter-neighbors", cfg.prefilter.num_neighbors, 24);
    } else if (cfg.quality == "high") {
        presetSet(seen, moved, "max-image-size", cfg.max_image_size, learned ? 1600 : 2400);
        presetSet(seen, moved, "max-features", cfg.sift.max_num_features, 8192);
        presetSet(seen, moved, "aliked-max-features", cfg.aliked.max_num_features, 4096);
        presetSet(seen, moved, "loma-max-features", cfg.loma.max_num_features, 4096);
    } else if (cfg.quality == "extreme") {
        presetSet(seen, moved, "max-image-size", cfg.max_image_size, learned ? 2400 : 3200);
        presetSet(seen, moved, "max-features", cfg.sift.max_num_features, 16384);
        presetSet(seen, moved, "aliked-max-features", cfg.aliked.max_num_features, 8192);
        presetSet(seen, moved, "loma-max-features", cfg.loma.max_num_features, 8192);
        presetSet(seen, moved, "prefilter-neighbors", cfg.prefilter.num_neighbors, 48);
    } else {
        return "unknown --quality '" + cfg.quality + "' (low, medium, high or extreme)";
    }

    // 学习描述子的最近邻距离更接近：20 图、190 对数据的互为最近邻距离比中位数为 0.826，需放宽 SIFT 的 0.8 阈值。
    // 比值 0.80/0.85/0.90/0.92/0.95 对应有效图像对 65/79/111/172/190，内点 6331/8426/11487/13633/16249，占比 90/79/54/44/29%；选择 0.92，继续放宽会引入大量错误。
    // 同数据 SIFT 以四倍特征预算仅得 68 对、7703 内点；余弦 0.85 阈值会拒绝约 70% 互近邻并仅保留 24 对，故不采用，仍提供 --min-similarity。
    if (learned && !isLearnedMatcher(cfg.matcher))
        presetSet(seen, moved, "ratio", cfg.match.max_ratio, 0.92f);

    // 学习匹配器自行决定对应关系，仅使用自身置信度阈值；距离比与交叉检查不适用于其输出。
    if (isLearnedMatcher(cfg.matcher)) {
        presetSet(seen, moved, "ratio", cfg.match.max_ratio, 1.0f);
        presetSet(seen, moved, "min-similarity", cfg.match.min_similarity, 0.0f);
        // 学习匹配器应处理筛选后的候选，穷举可能耗时数小时，而配对筛选只需数分钟。
        presetSet(seen, moved, "pairs", cfg.pairs, std::string("prefilter"));
    }

    if (cfg.data_type == "individual") {
        // 默认配置已适用于独立照片集合，无需调整。
    } else if (cfg.data_type == "video") {
        // 仅在 auto 的 100 图阈值以下采用视频时序预设；超过阈值改用内容筛选以捕获回访连接。
        presetSet(seen, moved, "pairs", cfg.pairs, std::string("sequential"));
        // 参考 COLMAP 的 ModifyForVideoData，视频相邻帧视差小，初始对不能沿用照片集的三角化角度阈值。
        presetSet(seen, moved, "init-min-tri-angle", cfg.mapper.init_min_tri_angle_deg,
                  cfg.mapper.init_min_tri_angle_deg / 2);
    } else if (cfg.data_type == "internet") {
        // 网络图片分辨率相同不代表相机相同，可能是不同设备缩放后的结果；每张图独立内参（D20）。
        presetSet(seen, moved, "camera-mode", cfg.camera_mode, std::string("image"));
        cfg.camera_mode_pinned = true;
        // 下载图片的文件顺序没有拍摄时序意义。
        presetSet(seen, moved, "prefilter-sequential", cfg.prefilter_sequential, false);
    } else {
        return "unknown --data-type '" + cfg.data_type + "' (individual, video or internet)";
    }
    return "";
}

// ---------------- 配置定稿 ----------------

std::string SfmConfig::finalize(uint32_t cmd) {
    if (!parseCamModelName(camera_model, camera.model))
        return "unknown --camera-model '" + camera_model + "'";
    if (!parseCameraMode(camera_mode, camera.mode))
        return "unknown --camera-mode '" + camera_mode + "'";
    camera.mode_explicit = camera_mode_pinned;
    camera.focal = focal;
    if (!distortion.empty() && !parseDistortion(distortion, camera.extra))
        return "bad --distortion '" + distortion + "' (k1,k2,... or PREFIX=k1,k2,...)";
    // 可优化内参必须为（焦距、畸变、主点）的前缀，固定中间项也固定后续项；显式冲突报错，收尾默认设置则让步（D50/D72）。
    if (mapper.refine_principal_point && !mapper.refine_extra_params)
        return "--refine-principal-point cannot be combined with "
               "--no-refine-extra-params: bundle adjustment frees a prefix of "
               "(focal, distortion, principal point)";
    if (!final_extra_params) final_principal_point = false;
    // 没有独立覆盖项的相机组使用此默认模型。
    mapper.camera_model = camera.model;

    // 同一容差同步到两个字段（D47）。
    twoview.ransac.max_error = max_error;
    mapper.max_reproj_error = max_error;
    // mapper.sequence_window = overlap;

    if (features != "sift" && !isAlikedType(features) && !isLomaType(features))
        return "unknown --features '" + features +
               "' (sift, aliked-n16rot, aliked-n32, loma-b128 or loma-b)";
    if (matcher != "bruteforce" && !isLearnedMatcher(matcher))
        return "unknown --matcher '" + matcher +
               "' (bruteforce, lightglue or loma-b128)";
    // auto 提前拒绝 SIFT 与学习匹配器组合；独立 match 从磁盘读取后由 LearnedMatcher 检查实际描述子。
    if ((cmd & (CMD_AUTO | CMD_EXTRACT)) && isLearnedMatcher(matcher) &&
        !isAlikedType(features) && !isLomaType(features))
        return "--matcher " + matcher + " needs learned descriptors; add "
               "--features aliked-n16rot";
    // LightGlue 使用 ALIKED 描述子，LoMa 使用 DeDoDe；两类不能混用，否则可能静默产生错误结果。
    if ((cmd & (CMD_AUTO | CMD_EXTRACT)) && isLomaType(matcher) != isLomaType(features) &&
        isLearnedMatcher(matcher))
        return "--matcher " + matcher + " and --features " + features +
               " are different frontends; a learned matcher only reads the "
               "descriptors it was trained on";
    // 提前拒绝描述子宽度不同的 LoMa 版本组合，避免完成提取后才失败。
    if ((cmd & (CMD_AUTO | CMD_EXTRACT)) && isLomaType(matcher) && isLomaType(features) &&
        lomaDescriptorDim(matcher) != lomaDescriptorDim(features))
        return "--matcher " + matcher + " wants " +
               std::to_string(lomaDescriptorDim(matcher)) + "-D descriptors and "
               "--features " + features + " makes " +
               std::to_string(lomaDescriptorDim(features)) + "-D ones";
    // 两个公制参考会争夺坐标规范，必须拒绝同时指定，避免结果依赖代码尝试顺序。
    const bool gps = metric_gps != "none";
    if (!metric_positions.empty() && gps)
        return "--metric-positions and --metric-gps are two references for one "
               "gauge; pass one";
    if (gps && image_dir.empty() && !(cmd & CMD_AUTO))
        return "--metric-gps reads each image's EXIF, so it needs --images";
    // GPS 通常为米级，位置文件通常为厘米级精度；阈值 0 表示使用对应来源的默认值。
    if (metric_max_error == 0)
        metric_max_error = gps ? 5.0 : 0.5;
    if (!telemetry.empty()) {
        bool listed = false;
        for (const TelemetryInput& t : telemetry_inputs)
            if (t.prefix.empty() && t.path == telemetry) listed = true;
        if (!listed) telemetry_inputs.push_back({"", telemetry, 0, 0});
    }

    lightglue.device = device;
    loma.device = loma_match.device = device;
    if (max_image_size <= 0) max_image_size = defaultMaxImageSize(features);

    sift.device = match.device = prefilter.device = mapper.device = device;
    aliked.device = device;
    // 跨阶段与工作线程传递 UUID；整数仅兼容旧输入形式。
    mapper.threads = threads;
    const bool v = !quiet;
    sift.verbose = mapper.verbose = manager.verbose = merge.verbose = aliked.verbose = v;
    lightglue.verbose = loma.verbose = loma_match.verbose = v;

    // 筛选评分问题约为完整匹配的 1/32，因此每次提交至少处理与匹配阶段一样多的图像对。
    prefilter.batch_pairs = std::max(prefilter.batch_pairs, match.batch_pairs);

    // 合并后轨迹尚未经过跨接缝 BA，过滤标准应与建图器一致；独立 merge 没有建图器，直接使用 filter-error/min-tri-angle。
    if (cmd != CMD_MERGE) {
        merge.filter_reproj_error = mapper.max_reproj_error;
        merge.min_tri_angle_deg = mapper.min_tri_angle_deg;
        merge.min_image_points = mapper.min_image_points;
    }
    manager.merge = merge;
    manager.merge.verbose = v;
    return "";
}

PairMode SfmConfig::pairMode() const {
    if (pairs == "sequential") return PairMode::Sequential;
    if (pairs == "prefilter") return PairMode::Prefilter;
    return PairMode::Exhaustive;  // exhaustive，以及尚未统计图像数的 auto
}

// ---------------- 设备解析 ----------------

std::string SfmConfig::selectorForDevice(const std::string& request, bool request_set,
                                          std::string& error) {
    error.clear();
    // 显式请求优先，其次环境变量，再其次 Auto。
    const bool supplied = request_set || !request.empty();
    const spirula::vkselect::Request req =
        spirula::vkselect::requestFrom(request, supplied);
    const spirula::vkselect::Resolution& res = VkContext::cachedSelector(req);
    if (res.ok()) return res.selector;
    // 仅有效默认选择为 Auto 时，缺少设备才允许退回 CPU。
    if (!supplied &&
        res.status == spirula::vkselect::ResolveStatus::NoDevice &&
        req.kind == spirula::vkselect::Request::Kind::Auto)
        return std::string();
    error = res.error;
    return std::string();
}

std::string SfmConfig::resolveDevice() {
    // 各阶段使用前统一解析设备请求。
    const bool request_set = device_request_set || !device_request.empty() ||
                             !device_selector.empty() || device >= 0;
    const bool request_text = device_request_set || !device_request.empty();
    const std::string request =
        request_text ? device_request
        : !device_selector.empty() ? device_selector
        : device >= 0 ? std::to_string(device) : std::string();
    std::string error;
    device_selector = selectorForDevice(request, request_set, error);
    if (!error.empty()) return error;
    // 序号请求继续填写兼容整数，但跨模块传递规范 UUID。
    if (request_text) {
        const spirula::vkselect::Request req = spirula::vkselect::parseRequest(request);
        if (req.kind == spirula::vkselect::Request::Kind::Ordinal) device = req.ordinal;
    }
    sift.device_selector = match.device_selector = prefilter.device_selector =
        mapper.device_selector = aliked.device_selector = lightglue.device_selector =
            loma.device_selector = loma_match.device_selector = device_selector;
    if (!device_selector.empty())
        sfm::slog::diag(sfm::slog::Tag::Device, "[gpu] running on %s",
                        device_selector.c_str());
    return "";
}

// ---------------- 帮助 ----------------

void printConfigOptions(FILE* out, uint32_t cmd, const SfmConfig& defaults) {
    const char* cur_group = "";
#define SFM_PRINT_FIELD(member, name, cmds, tier, group, lo, hi, choices, help)                    \
    if (((uint32_t)(cmds) & cmd) && (tier) != Tier::Alias) {                                       \
        if (std::string(cur_group) != group) {                                                     \
            cur_group = group;                                                                     \
            std::fprintf(out, "\n %s:\n", groupLabel(group));                                      \
        }                                                                                          \
        printOption(out, flagFor(defaults.member, name), metavarFor(defaults.member, name, choices),\
                    valueString(defaults.member),                                                  \
                    spirula::i18n::msg::sfmfield::help##_help.get(), choices);                     \
    }
    SFM_CONFIG_FIELDS(SFM_PRINT_FIELD)
#undef SFM_PRINT_FIELD
}

void printOptionLine(FILE* out, const std::string& flag, const std::string& value,
                     const std::string& help) {
    printOption(out, flag, "", value, help, "");
}

// ---------------- 阶段签名 ----------------

namespace {

// 设备、速度和日志详细度等不会改变阶段写出内容的选项不参与签名。
bool signatureRelevant(const char* name) {
    for (const char* n : {"threads", "decode-threads", "decode-budget", "device",
                          "quiet", "profile", "spv-path"})
        if (std::strcmp(name, n) == 0) return false;
    return true;
}

}  // 匿名命名空间

std::string stageSignature(const SfmConfig& cfg, uint32_t cmd) {
    std::string out;
#define SFM_SIG_FIELD(member, name, cmds, tier, group, lo, hi, choices, help)   \
    if (((uint32_t)(cmds) & cmd) && (tier) != Tier::Alias && signatureRelevant(name)) \
        out += std::string(name) + "=" + valueString(cfg.member) + "\n";
    SFM_CONFIG_FIELDS(SFM_SIG_FIELD)
#undef SFM_SIG_FIELD
    // 相机分组覆盖项不属于配置表行，但会影响相机设置与验证；命令行和清单均在此纳入签名。
    if (cmd & (CMD_MATCH | CMD_MAP))
        for (const CameraOverride& o : cfg.camera.overrides) {
            out += "override " + o.prefix + "=";
            if (o.has_model) out += camInfo(o.model).cli_name;
            if (o.has_focal) out += "," + valueString(o.focal);
            for (double e : o.extra) out += "," + valueString(e);
            out += "\n";
        }
    // 序列会将时间窗口加入图像对列表。
    if (cmd & (CMD_MATCH | CMD_MAP))
        for (const SequenceDef& d : cfg.sequences) {
            out += "sequence ";
            for (size_t m = 0; m < d.members.size(); m++)
                out += (m ? "," : "") + d.members[m];
            out += "\n";
        }
    return out;
}

}  // 命名空间 sfm
