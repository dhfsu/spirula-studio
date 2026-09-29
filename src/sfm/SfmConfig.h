// SfM 配置的唯一入口，聚合阶段选项并通过描述表供 CLI、帮助和 GUI 共用。
// 非单字段选项由前端优先手动解析，如相机分组覆盖、no-manage、审查命令及输出路径。
// 跨阶段参数仅由 finalize 统一展开，避免前端分别维护同步规则。
#pragma once

#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "sfm/core/CameraSetup.h"
#include "sfm/core/Sequence.h"
#include "sfm/feature/Matcher.h"
#include "sfm/feature/Extractor.h"
#include "sfm/feature/LearnedMatcher.h"
#include "sfm/feature/PairSelection.h"
#include "sfm/feature/Sift.h"
#include "sfm/geometry/TwoView.h"
#include "sfm/map/Bottomup.h"
#include "sfm/map/Mapper.h"
#include "sfm/map/Merge.h"

namespace sfm {

// 命令掩码互斥时允许复用选项名；max-error 在 auto/match/map 中表示验证与重投影容差，在 merge 中表示对齐容差。
enum CmdMask : uint32_t {
    CMD_AUTO    = 1u << 0,
    CMD_EXTRACT = 1u << 1,
    CMD_MATCH   = 1u << 2,
    CMD_MAP     = 1u << 3,
    CMD_MERGE   = 1u << 4,
    CMD_ALL     = CMD_AUTO | CMD_EXTRACT | CMD_MATCH | CMD_MAP | CMD_MERGE,
};

// Basic 为默认展开的基本选项，Advanced 为进阶选项，Alias 为仅参与解析、不重复展示的别名。
enum class Tier { Basic, Advanced, Alias };

// 视频的 IMU 与 GPS 覆盖 prefix 下图像；空前缀表示全部。
struct TelemetryInput {
    std::string prefix;
    std::string path;
    double fps = 0;           // 文件主干帧号到秒的换算率；0 使用文件帧率
    double time_offset = 0;   // 加到所有帧时间上的秒数
};

// 聚合配置保持各阶段选项结构不变，使库调用方与测试的默认行为独立于此层。
struct SfmConfig {
    // ---------------- 流水线选项 ----------------
    // 仅 auto 使用预设，显式字段覆盖优先于预设；参见 applyPresets()。
    std::string quality = "high";
    std::string data_type = "individual";
    // auto 运行时按数量选择：至少 100 张使用 prefilter，更少的视频使用 sequential，其余 exhaustive；独立 match 将 auto 视为 exhaustive。
    std::string pairs = "auto";
    int overlap = 10;
    // 时序窗口仅形成链，回到起点时缺少闭环；262 帧广场步行数据曾因此分成四个模型。
    // 在时间窗口上补充基于内容的候选列表，作用类似 COLMAP 词汇树闭环；仅适用于时序模式。
    bool loop_closure = true;
    // 时序窗口同时连接 i 与 i + 2^k，k < overlap，对应 COLMAP 的 quadratic_overlap。
    bool quadratic_overlap = false;
    // 为内容筛选补充时序窗口，保留得分略低于 top-k 但文件顺序相邻的真实连接。
    bool prefilter_sequential = false;
    // 依据成员已知旋转选择朝向重叠的 rig 伙伴图像对，见 RigPairs.h。
    bool rig_pairs = true;
    double rig_pair_angle = 30.0;
    int rig_pair_min_inliers = 30;

    // 以提取像素为单位的统一几何容差，同时用于双视图内点半径与建图重投影上限，由 finalize 同步（D47）。
    double max_error = 3.0;
    // 0 使用前端默认最长边：SIFT 为 3200，学习前端为 1600；由 finalize 解析，避免学习前端误用 SIFT 分辨率而占用四倍显存。
    int max_image_size = 0;
    std::string mask_dir;
    // 反转掩码保留与忽略区域，兼容以白色标记删除区域的导出器。
    bool flip_mask = false;

    // 输入色彩空间；解码后转换为检测器和学习模型所使用的 sRGB。
    std::string image_gamut = "Rec.709";
    bool image_is_linear = false;
    // srgb 保持点颜色为 sRGB；image 转回照片原色彩空间，对应训练器默认约定。
    std::string point_color_space = "srgb";

    // 相机配置的字符串供配置表与 GUI 使用，finalize 后解析到 camera；PREFIX=VALUE 由前端直接写入 camera.overrides。
    std::string camera_mode = "folder";
    std::string camera_model = "opencv";
    double focal = 0;
    // 初始畸变系数 k1,k2,...，按相机 BA 参数顺序；空值从零开始，与 COLMAP 一致。
    std::string distortion;
    // 显式 camera-mode 或网络照片预设固定分组模式；未固定时 buildCameras 可将明显的照片集合从 Folder 切换为 Image（D20/D48）。
    bool camera_mode_pinned = false;

    // 不属于单阶段选项的流程开关。
    bool verify = true;                 // 匹配阶段的几何验证
    bool compact_unused_features = true;   // 压缩已存储匹配引用的特征行
    bool final_principal_point = true;  // 最后执行放开主点的全局 BA（D51）
    bool final_extra_params = true;     // 收尾同时放开畸变（D72）
    // 还可追加逐图像独立内参优化（D73）。
    bool final_per_image_intrinsics = false;
    // 将最终模型归一化为立正、居中、单位尺度，避免保留任意种子对坐标系。
    bool orient = true;
    // 无测量时用 ground 的地面、墙面与水平投影，或 cameras 的平均向上轴确定朝向。
    std::string level = "ground";
    // EXIF Orientation 策略：none 忽略，orient 仅修正向上方向，apply 旋转像素。
    std::string exif_orientation = "orient";
    // 由外部米制测量确定公制坐标规范，见 MetricGauge.h（D74）。
    std::string metric_positions;       // 每行 image_name X Y Z
    // EXIF GPS 可为 none、仅经纬度的 horizontal，或含高度的 full。
    std::string metric_gps = "none";
    double metric_max_error = 0;        // 单位米，0 按来源选择默认值
    // 图像姿态 auto 使用向上与北向，up 仅倾斜，none 忽略。
    std::string exif_attitude = "auto";
    // 视频 IMU 与 GPS：telemetry 指定覆盖全部图像的文件，清单可指定多个；auto 接受有效的朝向、尺度与位置，up 仅朝向。
    std::string telemetry;
    std::string sensor_gauge = "auto";
    std::vector<TelemetryInput> telemetry_inputs;   // 清单项与 --telemetry 的合并结果
    // 传感器还用于固定验证旋转、约束配准与优化，并连接 GPS 距离小于 sensor_pair_radius 米的图像对。
    bool sensor_verify = true;
    bool sensor_map = true;
    bool sensor_pairs = true;
    double sensor_pair_radius = 20.0;
    double sensor_max_dt = 3.0;   // 陀螺旋转先验的最大跨度，秒
    // 公制拟合还按参考位置 RMS 半径的比例拒绝离群点，适应步行与长距离飞行的不同尺度。
    double metric_max_error_frac = 0.03;
    // rig 来自 --rig 与清单，待图像名确定后解析；final_free_rig 在末次 BA 中允许各图像独立位姿。
    std::vector<RigDef> rigs;
    bool final_free_rig = false;
    // 序列来自 --sequence 与清单；匹配加入时间窗口，建图优先相信相邻帧，空值保持普通流程（D79）。
    std::vector<SequenceDef> sequences;
    bool merge_ba = true;               // 合并后跨接缝执行 BA
    bool in_place = false;              // 将合并结果原地写回输入

    // map 通过选项指定的输入，auto 使用位置参数。
    std::string image_dir;
    std::string feature_dir;
    std::string resume;
    bool check = false;

    // auto 复用相同设置中断后留下的特征、图像对及已验证日志；禁用则各阶段从头开始。
    bool reuse = true;

    // 运行时选项。
    int threads = 0;           // 主机工作池大小，0 使用 hardware_concurrency
    int decode_threads = 0;    // 图像解码池大小，0 使用 hardware_concurrency
    int decode_budget_mb = 0;  // 0 使用 ImageLoadOptions 默认值
    // 设备整数保留输入写法，下游使用规范 UUID。
    int device = -1;
    // 入口解析后的规范 UUID，传递给各阶段；空值表示没有可用原生设备。
    std::string device_selector;
    bool quiet = false;
    // 原始 --device 写法；request_set 保留显式空值的含义。
    std::string device_request;
    bool device_request_set = false;

    // sift 使用 GPU SIFT，aliked-* 与 loma-* 需要推理层；提取与匹配分别配置，使学习描述子也能使用暴力匹配。
    std::string features = "sift";
    std::string matcher = "bruteforce";

    // ---------------- 各阶段选项结构 ----------------
    SiftOptions sift;
    AlikedOptions aliked;
    LomaOptions loma;
    LightGlueOptions lightglue;
    LomaMatchOptions loma_match;
    MatchOptions match;
    PairSelectionOptions prefilter;
    TwoViewOptions twoview;
    CameraSetupOptions camera;
    MapperOptions mapper;
    ManagerOptions manager;
    MergeOptions merge;
    BottomUpOptions bup;
    // 两种建图器共用逐层合并、扩展、联合求解及收尾流程；--bup-* 选项同样适用于 flat。
    AssembleOptions assemble;
    // flat 对整个采集执行增量重建，bottom-up 先重建小原子分组再向上合并，兼容 hierarchical 别名（D57）。
    // 默认始终为 flat，不按数据规模自动切换，避免未充分验证的调度变化干扰运行比较。
    std::string mapper_mode = "flat";

    // 解析与预设应用后调用一次，将公共参数同步到各阶段并执行额外验证；成功返回空字符串，否则返回错误说明。
    std::string finalize(uint32_t cmd);

    // 创建硬件前将请求解析为规范 UUID 并同步各阶段；选择错误立即失败，空值表示无可用 Vulkan 设备。
    std::string resolveDevice();

    // 为独立 BA 解析设备值，不修改此配置；request_set 区分显式空值 Auto。
    static std::string selectorForDevice(const std::string& request,
                                         bool request_set, std::string& error);

    // 独立解析 pairs 时 auto 等同 exhaustive；自动流水线在提取后根据图像数超过 100 的条件切换内容筛选。
    PairMode pairMode() const;
};

// ---------------- 配置描述表 ----------------
// F(member, name, cmds, tier, group, lo, hi, choices, help)：字段路径及类型、无 -- 的选项名、命令掩码、展示级别、分组、范围、字符串候选与帮助消息标识；lo >= hi 表示无界。
// 编辑器字段须包含 CMD_AUTO；help 引用 SfmFields.h，缺失译文编译失败。布尔名称保持正向，--name/--no-name 分别启用/禁用，按改变默认值的方向展示。
#define SFM_CONFIG_FIELDS(F)                                                                       \
    /* ---- 流水线 ---- */                                                                       \
    F(quality, "quality", CMD_AUTO, Tier::Basic, "pipeline", 0, 0, "low|medium|high|extreme",      \
      quality)                                                                                     \
    F(data_type, "data-type", CMD_AUTO, Tier::Basic, "pipeline", 0, 0,                             \
      "individual|video|internet", data_type)                                                      \
    F(pairs, "pairs", CMD_AUTO | CMD_MATCH, Tier::Basic, "pipeline", 0, 0,                         \
      "auto|exhaustive|sequential|prefilter", pairs)                                               \
    F(overlap, "overlap", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Advanced, "pipeline", 1, 1000000, \
      "",                                                                                          \
      overlap)                                                                                     \
    F(loop_closure, "loop-closure", CMD_AUTO | CMD_MATCH, Tier::Advanced, "pipeline", 0, 0, "",    \
      loop_closure)                                                                                \
    F(quadratic_overlap, "quadratic-overlap", CMD_AUTO | CMD_MATCH, Tier::Advanced, "pipeline", 0, \
      0, "", quadratic_overlap)                                                                    \
    F(prefilter_sequential, "prefilter-sequential", CMD_AUTO | CMD_MATCH, Tier::Advanced,          \
      "pipeline", 0, 0, "", prefilter_sequential)                                                  \
    F(rig_pairs, "rig-pairs", CMD_AUTO | CMD_MATCH, Tier::Advanced, "pipeline", 0, 0, "",          \
      rig_pairs)                                                                                   \
    F(rig_pair_angle, "rig-pair-angle", CMD_AUTO | CMD_MATCH, Tier::Advanced, "pipeline", 0, 180,  \
      "", rig_pair_angle)                                                                          \
    F(rig_pair_min_inliers, "rig-pair-min-inliers", CMD_AUTO | CMD_MATCH, Tier::Advanced,          \
      "pipeline", 0, 100000, "", rig_pair_min_inliers)                                             \
    F(max_error, "max-error", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Advanced, "pipeline", 0.1,     \
      100, "", max_error)                                                                          \
    F(max_image_size, "max-image-size", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "pipeline", 0,     \
      20000, "", max_image_size)                                                                   \
    F(mask_dir, "masks", CMD_AUTO | CMD_EXTRACT, Tier::Basic, "pipeline", 0, 0, "", masks)         \
    F(mask_dir, "mask-dir", CMD_AUTO | CMD_EXTRACT, Tier::Alias, "pipeline", 0, 0, "", mask_dir)   \
    F(flip_mask, "flip-mask", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "pipeline", 0, 0, "",        \
      flip_mask)                                                                                   \
    /* ---- 色彩 ---- */                                                                         \
    F(image_gamut, "image-gamut", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "colour", 0, 0,          \
      "Rec.709|ACES2065-1|ACEScg|Rec.2020|AdobeRGB|DCI-P3", image_gamut)                           \
    F(image_is_linear, "image-linear", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "colour", 0, 0, "", \
      image_linear)                                                                                \
    F(point_color_space, "point-color", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "colour", 0, 0,    \
      "srgb|image", point_color)                                                                   \
    /* ---- 相机 ---- */                                                                         \
    F(camera_mode, "camera-mode", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Basic, "camera", 0, 0,     \
      "single|folder|image", camera_mode)                                                          \
    F(camera_model, "camera-model", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Basic, "camera", 0, 0,   \
      "simple-pinhole|pinhole|radial|opencv|full-opencv|opencv-fisheye|thin-prism-fisheye|equirectangular",\
      camera_model)                                                                                \
    F(focal, "focal", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Basic, "camera", 0, 10000000, "",      \
      focal)                                                                                       \
    F(distortion, "distortion", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Advanced, "camera", 0, 0,    \
      "", distortion)                                                                              \
    F(camera.exif_focal, "exif-focal", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Advanced, "camera",   \
      0, 0, "", exif_focal)                                                                        \
    F(camera.exif_groups, "exif-groups", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Advanced, "camera", \
      0, 0, "", exif_groups)                                                                       \
    F(camera.exif_focal_tol, "exif-focal-tol", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Advanced,     \
      "camera", 0.001, 1.0, "", exif_focal_tol)                                                    \
    F(exif_orientation, "exif-orientation", CMD_AUTO | CMD_EXTRACT | CMD_MAP | CMD_MERGE,          \
      Tier::Advanced, "camera", 0, 0, "none|orient|apply", exif_orientation)                       \
    /* ---- 特征 ---- */                                                                       \
    F(features, "features", CMD_AUTO | CMD_EXTRACT, Tier::Basic, "features", 0, 0,                 \
      "sift|aliked-n16rot|aliked-n32|loma-b128|loma-b", features)                                  \
    F(sift.max_num_features, "max-features", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "features",   \
      128, 1000000, "", max_features)                                                              \
    F(aliked.max_num_features, "aliked-max-features", CMD_AUTO | CMD_EXTRACT, Tier::Advanced,      \
      "features", 128, 1000000, "", aliked_max_features)                                           \
    F(aliked.min_score, "aliked-min-score", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "features", 0, \
      1, "", aliked_min_score)                                                                     \
    F(aliked.model, "aliked-model", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "features", 0, 0, "",  \
      aliked_model)                                                                                \
    F(loma.max_num_features, "loma-max-features", CMD_AUTO | CMD_EXTRACT, Tier::Advanced,          \
      "features", 128, 1000000, "", aliked_max_features)                                           \
    F(loma.min_score, "loma-min-score", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "features", 0,     \
      1, "", aliked_min_score)                                                                     \
    F(loma.detector_model, "loma-detector-model", CMD_AUTO | CMD_EXTRACT, Tier::Advanced,          \
      "features", 0, 0, "", loma_model)                                                            \
    F(loma.descriptor_model, "loma-descriptor-model", CMD_AUTO | CMD_EXTRACT, Tier::Advanced,      \
      "features", 0, 0, "", loma_model)                                                            \
    F(sift.num_octaves, "octaves", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "features", 1, 8, "",   \
      octaves)                                                                                     \
    F(sift.peak_threshold, "peak-threshold", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "features",   \
      0, 1, "", peak_threshold)                                                                    \
    F(sift.edge_threshold, "edge-threshold", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "features",   \
      1, 1000, "", edge_threshold)                                                                 \
    F(sift.max_num_orientations, "max-orientations", CMD_AUTO | CMD_EXTRACT, Tier::Advanced,       \
      "features", 1, 8, "", max_orientations)                                                      \
    F(sift.profile, "profile", CMD_EXTRACT, Tier::Advanced, "features", 0, 0, "", profile)         \
    F(sift.spv_path, "spv-path", CMD_EXTRACT, Tier::Advanced, "features", 0, 0, "", spv_path)      \
    /* ---- 匹配 ---- */                                                                       \
    F(matcher, "matcher", CMD_AUTO | CMD_MATCH, Tier::Basic, "matching", 0, 0,                     \
      "bruteforce|lightglue|loma-b128|loma-b|loma-r|loma-l|loma-g", matcher)                       \
    F(lightglue.min_score, "lightglue-min-score", CMD_AUTO | CMD_MATCH, Tier::Advanced,            \
      "matching", 0, 1, "", lightglue_min_score)                                                   \
    F(lightglue.model, "lightglue-model", CMD_AUTO | CMD_MATCH, Tier::Advanced, "matching", 0, 0,  \
      "", lightglue_model)                                                                         \
    F(loma_match.min_score, "loma-min-match-score", CMD_AUTO | CMD_MATCH, Tier::Advanced,          \
      "matching", 0, 1, "", loma_min_match_score)                                                  \
    F(loma_match.model, "loma-matcher-model", CMD_AUTO | CMD_MATCH, Tier::Advanced, "matching",    \
      0, 0, "", loma_model)                                                                        \
    F(match.max_ratio, "ratio", CMD_AUTO | CMD_MATCH, Tier::Advanced, "matching", 0, 1, "", ratio) \
    F(match.min_similarity, "min-similarity", CMD_AUTO | CMD_MATCH, Tier::Advanced, "matching", 0, \
      1, "", min_similarity)                                                                       \
    F(match.cross_check, "cross-check", CMD_AUTO | CMD_MATCH, Tier::Advanced, "matching", 0, 0,    \
      "", cross_check)                                                                             \
    F(match.max_num_matches, "max-matches", CMD_AUTO | CMD_MATCH, Tier::Advanced, "matching", 0,   \
      1000000000, "", max_matches)                                                                 \
    F(verify, "verify", CMD_MATCH, Tier::Advanced, "matching", 0, 0, "", verify)                   \
    F(twoview.min_num_inliers, "min-inliers", CMD_AUTO | CMD_MATCH, Tier::Advanced, "matching", 4, \
      1000000, "", min_inliers)                                                                    \
    F(prefilter.num_features, "prefilter-features", CMD_AUTO | CMD_MATCH, Tier::Advanced,          \
      "matching", 16, 100000, "", prefilter_features)                                              \
    F(prefilter.train_features, "prefilter-train", CMD_AUTO | CMD_MATCH, Tier::Advanced,           \
      "matching", 0, 1000000, "", prefilter_train)                                                 \
    F(prefilter.num_neighbors, "prefilter-neighbors", CMD_AUTO | CMD_MATCH, Tier::Advanced,        \
      "matching", 1, 100000, "", prefilter_neighbors)                                              \
    F(prefilter.min_score, "prefilter-min-score", CMD_AUTO | CMD_MATCH, Tier::Advanced,            \
      "matching", 0, 1000000, "", prefilter_min_score)                                             \
    F(prefilter.ratio, "prefilter-ratio", CMD_AUTO | CMD_MATCH, Tier::Advanced, "matching", 0, 1,  \
      "", prefilter_ratio)                                                                         \
    /* ---- 建图 ---- */                                                                         \
    F(compact_unused_features, "compact-unused-features", CMD_AUTO | CMD_MAP, Tier::Advanced,     \
      "mapper", 0, 0, "", compact_unused_features)                                                 \
    F(mapper.focal_trials, "focal-trials", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 1000,  \
      "", focal_trials)                                                                            \
    F(mapper.refine_principal_point, "refine-principal-point", CMD_AUTO | CMD_MAP, Tier::Advanced, \
      "mapper", 0, 0, "", refine_principal_point)                                                  \
    F(final_principal_point, "final-principal-point", CMD_AUTO | CMD_MAP, Tier::Advanced,          \
      "mapper", 0, 0, "", final_principal_point)                                                   \
    F(mapper.pp_min_images, "pp-min-images", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 2,      \
      1000000, "", pp_min_images)                                                                  \
    F(mapper.refine_extra_params, "refine-extra-params", CMD_AUTO | CMD_MAP, Tier::Advanced,       \
      "mapper", 0, 0, "", refine_extra_params)                                                     \
    F(final_extra_params, "final-extra-params", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0,   \
      0, "", final_extra_params)                                                                   \
    F(final_per_image_intrinsics, "final-per-image-intrinsics", CMD_AUTO | CMD_MAP,                \
      Tier::Advanced, "mapper", 0, 0, "", final_per_image_intrinsics)                              \
    F(orient, "orient", CMD_AUTO | CMD_MAP | CMD_MERGE, Tier::Advanced, "mapper", 0, 0, "",        \
      orient)                                                                                      \
    F(level, "level", CMD_AUTO | CMD_MAP | CMD_MERGE, Tier::Advanced, "mapper", 0, 0,              \
      "ground|cameras", level)                                                                     \
    F(metric_positions, "metric-positions", CMD_AUTO | CMD_MAP | CMD_MERGE, Tier::Advanced,        \
      "mapper", 0, 0, "", metric_positions)                                                        \
    F(metric_gps, "metric-gps", CMD_AUTO | CMD_MAP | CMD_MERGE, Tier::Advanced, "mapper", 0, 0,    \
      "none|horizontal|full", metric_gps)                                                          \
    F(exif_attitude, "exif-attitude", CMD_AUTO | CMD_MAP | CMD_MERGE, Tier::Advanced, "mapper", 0, \
      0, "auto|up|none", exif_attitude)                                                            \
    F(metric_max_error, "metric-max-error", CMD_AUTO | CMD_MAP | CMD_MERGE, Tier::Advanced,        \
      "mapper", 0, 1000000, "", metric_max_error)                                                  \
    F(metric_max_error_frac, "metric-max-error-frac", CMD_AUTO | CMD_MAP | CMD_MERGE,              \
      Tier::Advanced, "mapper", 0, 1, "", metric_max_error_frac)                                   \
    F(telemetry, "telemetry", CMD_AUTO | CMD_MATCH | CMD_MAP | CMD_MERGE, Tier::Advanced,           \
      "mapper", 0, 0, "", telemetry)                                                               \
    F(sensor_gauge, "sensor-gauge", CMD_AUTO | CMD_MAP | CMD_MERGE, Tier::Advanced, "mapper", 0,   \
      0, "auto|up|none", sensor_gauge)                                                             \
    F(sensor_verify, "sensor-verify", CMD_AUTO | CMD_MATCH, Tier::Advanced, "mapper", 0, 0, "",    \
      sensor_verify)                                                                               \
    F(sensor_map, "sensor-map", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 0, "",            \
      sensor_map)                                                                                  \
    F(sensor_pairs, "sensor-pairs", CMD_AUTO | CMD_MATCH, Tier::Advanced, "mapper", 0, 0, "",      \
      sensor_pairs)                                                                                \
    F(sensor_pair_radius, "sensor-pair-radius", CMD_AUTO | CMD_MATCH, Tier::Advanced, "mapper",    \
      0, 100000, "", sensor_pair_radius)                                                           \
    F(sensor_max_dt, "sensor-max-dt", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Advanced, "mapper",    \
      0.01, 1000, "", sensor_max_dt)                                                               \
    F(mapper.min_tri_angle_deg, "min-tri-angle", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0,  \
      90, "", min_tri_angle)                                                                       \
    F(mapper.init_min_tri_angle_deg, "init-min-tri-angle", CMD_AUTO | CMD_MAP, Tier::Advanced,     \
      "mapper", 0, 90, "", init_min_tri_angle)                                                     \
    F(mapper.min_num_pnp_inliers, "min-pnp-inliers", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", \
      4, 1000000, "", min_pnp_inliers)                                                             \
    F(mapper.min_pnp_inlier_ratio, "min-pnp-ratio", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper",  \
      0, 1, "", min_pnp_ratio)                                                                     \
    F(mapper.min_image_points, "min-image-points", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper",   \
      1, 1000000, "", min_image_points)                                                            \
    F(mapper.ba_loss, "ba-loss", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 0,               \
      "trivial|huber|cauchy", ba_loss)                                                             \
    F(mapper.ba_loss_param, "ba-loss-param", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0,      \
      1000, "", ba_loss_param)                                                                     \
    F(mapper.seed_homography, "seed-homography", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0,  \
      0, "", seed_homography)                                                                      \
    F(mapper.ba_real, "ba-real", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 0,               \
      "float|double|df|cpu", ba_real)                                                              \
    F(mapper.ba_real_coarse, "ba-real-coarse", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 0, \
      "float|double|df|cpu", ba_real_coarse)                                                       \
    F(mapper.ba_solver, "ba-solver", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 0,           \
      "auto|dense|cg", ba_solver)                                                                  \
    F(mapper.retri_scale, "retri-scale", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 10, "",  \
      retri_scale)                                                                                 \
    F(mapper.merge_tracks, "merge-tracks", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 0, "", \
      merge_tracks)                                                                                \
    F(mapper.rank_by_visibility, "rank-by-visibility", CMD_AUTO | CMD_MAP, Tier::Advanced,         \
      "mapper", 0, 0, "", rank_by_visibility)                                                      \
    F(mapper.seed_blocking, "seed-blocking", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 0,   \
      "", seed_blocking)                                                                           \
    F(mapper_mode, "mapper", CMD_AUTO | CMD_MAP, Tier::Basic, "mapper", 0, 0, "flat|bottom-up",    \
      mapper)                                                                                      \
    F(bup.partition.leaf_max_images, "bup-atom-size", CMD_AUTO | CMD_MAP, Tier::Advanced,          \
      "mapper", 8, 100000, "", bup_atom_size)                                                      \
    F(bup.partition.overlap, "bup-overlap", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0,       \
      100000, "", bup_overlap)                                                                     \
    F(assemble.max_rounds, "bup-rounds", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 1, 100, "", \
      bup_rounds)                                                                                  \
    F(bup.atom.threads, "bup-atom-threads", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 1024, \
      "", bup_atom_threads)                                                                        \
    F(bup.atom.ba_growth, "bup-atom-ba-growth", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 1,   \
      1000000, "", bup_atom_ba_growth)                                                             \
    F(assemble.joint_every, "bup-joint-every", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 1,    \
      100, "", bup_joint_every)                                                                    \
    F(bup.atom.tight_final_ba, "bup-atom-tight-final", CMD_AUTO | CMD_MAP, Tier::Advanced,         \
      "mapper", 0, 0, "", bup_atom_tight_final)                                                    \
    F(assemble.coarse_joint_ba, "bup-coarse-ba", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0,  \
      0, "", bup_coarse_ba)                                                                        \
    F(bup.atom.init_trials, "bup-atom-init-trials", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper",  \
      1, 1000, "", bup_atom_init_trials)                                                           \
    F(bup.atom.min_model_fraction, "bup-atom-min-fraction", CMD_AUTO | CMD_MAP, Tier::Advanced,    \
      "mapper", 0, 1, "", bup_atom_min_fraction)                                                   \
    F(assemble.joint_intrinsics, "bup-joint-intrinsics", CMD_AUTO | CMD_MAP, Tier::Advanced,       \
      "mapper", 0, 0, "", bup_joint_intrinsics)                                                    \
    F(assemble.grow_every, "bup-grow-every", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 100, \
      "", bup_grow_every)                                                                          \
    F(assemble.grow_budget_frac, "bup-grow-budget", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper",  \
      0, 1000, "", bup_grow_budget)                                                                \
    F(mapper.ba_growth_ratio, "ba-growth", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 1, 100,   \
      "", ba_growth)                                                                               \
    F(mapper.ba_growth_rtol, "ba-growth-rtol", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0, 1, \
      "", ba_growth_rtol)                                                                          \
    F(mapper.ba_growth_patience, "ba-growth-patience", CMD_AUTO | CMD_MAP, Tier::Advanced,         \
      "mapper", 1, 1000, "", ba_growth_patience)                                                   \
    F(mapper.max_num_models, "max-models", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 1,        \
      100000, "", max_models)                                                                      \
    F(mapper.max_model_overlap, "model-overlap", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 0,  \
      1000000, "", model_overlap)                                                                  \
    F(mapper.model_overlap_ratio, "model-overlap-ratio", CMD_AUTO | CMD_MAP, Tier::Advanced,       \
      "mapper", 0, 1000, "", model_overlap_ratio)                                                  \
    F(mapper.min_model_size, "min-model-size", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper", 2,    \
      1000000, "", min_model_size)                                                                 \
    F(mapper.pnp_ratio_visible_only, "pnp-ratio-visible", CMD_AUTO | CMD_MAP, Tier::Advanced,      \
      "mapper", 0, 0, "", pnp_ratio_visible)                                                       \
    F(mapper.strong_pnp_inliers, "strong-pnp-inliers", CMD_AUTO | CMD_MAP, Tier::Advanced,         \
      "mapper", 0, 1000000, "", strong_pnp_inliers)                                                \
    F(mapper.strong_pnp_max_rival, "strong-pnp-max-rival", CMD_AUTO | CMD_MAP, Tier::Advanced,     \
      "mapper", 0, 1, "", strong_pnp_max_rival)                                                    \
    F(mapper.audit_min_evidence, "audit-evidence", CMD_AUTO | CMD_MAP, Tier::Advanced, "mapper",   \
      0, 1000000, "", audit_evidence)                                                              \
    /* ---- 相机装置 ---- */                                                                           \
    F(mapper.use_rigs, "rigs", CMD_AUTO | CMD_MAP, Tier::Advanced, "rig", 0, 0, "", rigs)          \
    F(mapper.refine_rigs, "refine-rigs", CMD_AUTO | CMD_MAP, Tier::Advanced, "rig", 0, 0, "",      \
      refine_rigs)                                                                                 \
    F(mapper.rig_complete_blind, "rig-blind", CMD_AUTO | CMD_MAP, Tier::Advanced, "rig", 0, 0,     \
      "", rig_blind)                                                                               \
    F(mapper.rig_calib.min_frames, "rig-min-frames", CMD_AUTO | CMD_MAP, Tier::Advanced, "rig",    \
      2, 1000000, "", rig_min_frames)                                                              \
    F(mapper.rig_calib.max_spread_deg, "rig-max-spread", CMD_AUTO | CMD_MAP, Tier::Advanced,       \
      "rig", 0, 180, "", rig_max_spread)                                                           \
    F(final_free_rig, "final-free-rig", CMD_AUTO | CMD_MAP, Tier::Advanced, "rig", 0, 0, "",       \
      final_free_rig)                                                                              \
    /* ---- 模型装配 ---- */                                                          \
    F(assemble.max_rounds, "rounds", CMD_AUTO | CMD_MAP, Tier::Alias, "manage", 1, 1000, "",       \
      rounds)                                                                                      \
    F(assemble.max_bridges, "max-bridges", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 0, 1000,  \
      "", max_bridges)                                                                             \
    F(manager.do_merge, "merge", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 0, 0, "", merge)    \
    F(manager.do_grow, "grow", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 0, 0, "", grow)       \
    F(manager.do_reseed, "reseed", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 0, 0, "", reseed) \
    F(manager.do_audit, "audit", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 0, 0, "", audit)    \
    F(manager.do_split, "split", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 0, 0, "", split)    \
    F(manager.do_duplicate_split, "fold-split", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 0,   \
      0, "", fold_split)                                                                           \
    F(manager.duplicate.max_cut_fraction, "fold-max-cut", CMD_AUTO | CMD_MAP, Tier::Advanced,      \
      "manage", 0, 1, "", fold_max_cut)                                                            \
    F(manager.duplicate.min_fold_overlap, "fold-min-overlap", CMD_AUTO | CMD_MAP, Tier::Advanced,  \
      "manage", 0, 1, "", fold_min_overlap)                                                        \
    F(assemble.joint_intrinsics, "joint-ba", CMD_AUTO | CMD_MAP, Tier::Alias, "manage", 0, 0, "",  \
      joint_ba)                                                                                    \
    F(manager.seam_min_agreement, "seam-min-agreement", CMD_AUTO | CMD_MAP, Tier::Advanced,        \
      "manage", 0, 1, "", seam_min_agreement)                                                      \
    F(manager.seam_relative_bar, "seam-relative-bar", CMD_AUTO | CMD_MAP, Tier::Advanced,          \
      "manage", 0, 10, "", seam_relative_bar)                                                      \
    F(manager.seam_min_pairs, "seam-min-pairs", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 1,   \
      100000, "", seam_min_pairs)                                                                  \
    F(manager.seam_rescue_frac, "seam-rescue", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage", 0, 1, \
      "", seam_rescue)                                                                             \
    F(manager.seam_max_rescues, "seam-max-rescues", CMD_AUTO | CMD_MAP, Tier::Advanced, "manage",  \
      0, 100000, "", seam_max_rescues)                                                             \
    /* ---- 合并 ---- */                                                                        \
    F(merge.max_reproj_error, "max-error", CMD_MERGE, Tier::Advanced, "merge", 0.1, 1000, "",      \
      merge_align_max_error)                                                                       \
    F(merge.max_reproj_error, "merge-max-error", CMD_AUTO | CMD_MAP, Tier::Advanced, "merge", 0.1, \
      1000, "", merge_max_error)                                                                   \
    F(merge.min_common_images, "min-common", CMD_MERGE, Tier::Advanced, "merge", 2, 1000000, "",   \
      min_common)                                                                                  \
    F(merge.min_common_images, "merge-min-common", CMD_AUTO | CMD_MAP, Tier::Advanced, "merge", 2, \
      1000000, "", merge_min_common)                                                               \
    F(merge.splice_arbitrate_inliers, "merge-arbitrate", CMD_AUTO | CMD_MAP | CMD_MERGE,           \
      Tier::Advanced, "merge", 0, 1000000, "", merge_arbitrate)                                    \
    F(merge.min_inlier_ratio, "min-inlier-ratio", CMD_MERGE, Tier::Advanced, "merge", 0, 1, "",    \
      min_inlier_ratio)                                                                            \
    F(merge.filter_reproj_error, "filter-error", CMD_MERGE, Tier::Advanced, "merge", 0, 1000, "",  \
      filter_error)                                                                                \
    F(merge.min_tri_angle_deg, "min-tri-angle", CMD_MERGE, Tier::Advanced, "merge", 0, 90, "",     \
      merge_min_tri_angle)                                                                         \
    F(merge_ba, "ba", CMD_MERGE, Tier::Advanced, "merge", 0, 0, "", ba)                            \
    F(in_place, "in-place", CMD_MERGE, Tier::Advanced, "merge", 0, 0, "", in_place)                \
    /* ---- 输入 ---- */                                                                         \
    F(image_dir, "images", CMD_MAP | CMD_MERGE, Tier::Advanced, "input", 0, 0, "", images)         \
    F(feature_dir, "features", CMD_MAP, Tier::Advanced, "input", 0, 0, "", feature_dir)            \
    F(resume, "resume", CMD_MAP, Tier::Advanced, "input", 0, 0, "", resume)                        \
    F(reuse, "resume", CMD_AUTO, Tier::Basic, "input", 0, 0, "", auto_resume)                      \
    F(check, "check", CMD_MAP, Tier::Advanced, "input", 0, 0, "", check)                           \
    /* ---- 运行时 ---- */                                                                        \
    F(threads, "threads", CMD_AUTO | CMD_MATCH | CMD_MAP, Tier::Advanced, "runtime", 0, 4096, "",  \
      threads)                                                                                     \
    F(decode_threads, "decode-threads", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "runtime", 0,      \
      4096, "", decode_threads)                                                                    \
    F(decode_budget_mb, "decode-budget", CMD_AUTO | CMD_EXTRACT, Tier::Advanced, "runtime", 0,     \
      1048576, "", decode_budget)                                                                  \
    F(device_request, "device", CMD_ALL, Tier::Advanced, "runtime", 0, 0, "", device)              \
    F(quiet, "quiet", CMD_ALL, Tier::Advanced, "runtime", 0, 0, "", quiet)

// ---------------- 配置使用方 ----------------

// 配置表解析单个命令行参数的结果。
enum class FieldResult { Unknown, Ok, Error };

// 解析 argv[i] 及所需值，arg 包含前导短横线；seen 记录显式字段，使其优先于预设并支持判断是否覆盖相机配置。
// 当前命令没有对应字段时返回 Unknown。
FieldResult setConfigField(SfmConfig& cfg, uint32_t cmd, const std::string& arg, int argc,
                           char** argv, int& i, std::set<std::string>& seen, std::string& error);

// 预设修改的字段，用于变更报告。
struct PresetChange {
    std::string flag, from, to;
};

// 应用 quality/data-type 预设，不修改 seen 中的显式字段，并记录所有变更。
std::string applyPresets(SfmConfig& cfg, const std::set<std::string>& seen,
                         std::vector<PresetChange>& moved);

// 按分组打印当前命令的配置选项，以字段当前值作为默认值。
void printConfigOptions(FILE* out, uint32_t cmd, const SfmConfig& defaults);

// 为手动解析选项使用同样的帮助布局；flag 包含参数占位符，value 为方括号说明，开关可为空。
void printOptionLine(FILE* out, const std::string& flag, const std::string& value,
                     const std::string& help);

// 阶段签名包含影响输出的配置行及相机覆盖项，排除不改变输出的选项；签名一致时才可复用中断缓存。
std::string stageSignature(const SfmConfig& cfg, uint32_t cmd);

}  // 命名空间 sfm
