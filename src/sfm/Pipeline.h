#pragma once

// SfM 各阶段的公共接口及 auto 调度，CLI 与进程内前端共用实现并安装各自日志、事件接收端。
// 阶段保留独立磁盘输入输出，便于通过 extract、match、map 分步定位故障。

#include "sfm/SfmConfig.h"
#include "sfm/core/CameraSetup.h"
#include "sfm/core/Events.h"
#include "sfm/core/FeatureCompaction.h"
#include "sfm/core/Features.h"
#include "sfm/core/Image.h"
#include "sfm/core/Log.h"
#include "sfm/core/Mask.h"
#include "sfm/core/Matches.h"
#include "sfm/core/Model.h"
#include "sfm/feature/Pairing.h"
#include "sfm/map/Assemble.h"
#include "sfm/map/Mapper.h"
#include "sfm/map/SensorPriors.h"
#include "sfm/map/MetricGauge.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace sfm {

// ---------------- 阶段统计 ----------------

struct ExtractStats {
    size_t images = 0, failed = 0, unreadable = 0;
    size_t reused = 0;            // 先前运行已写出的特征数量
    uint64_t features = 0;
    uint64_t features_new = 0;    // 本次实际提取的特征数
    // 未指定掩码目录时，下列掩码统计均为零。
    size_t masked_images = 0;     // 找到掩码文件的图像数
    size_t unmasked_images = 0;   // 未找到掩码文件的图像数
    size_t mask_unreadable = 0;   // 找到掩码但无法解码的图像数
    uint64_t masked_out = 0;      // 被掩码排除的关键点数
    std::string first_unmasked;   // 供警告使用的示例名称
    bool warned_empty = false;    // 掩码清除全部特征的警告仅输出一次
    bool warned_exif_mirror = false;   // 方向标签要求镜像的警告仅输出一次
};

struct MatchStats {
    size_t images = 0, pairs = 0, kept = 0, scored = 0;
    uint64_t inliers = 0, putative = 0;
    double select_seconds = 0;
};

// 每次运行只读取一次的遥测文件，供传感器先验与规范拟合查询。
struct LoadedCapture {
    SensorCapture cap;
    SensorTimeline timeline;
};

struct SensorCaptures {
    std::vector<std::unique_ptr<LoadedCapture>> loaded;
    bool empty() const { return loaded.empty(); }
    std::vector<SensorCapture> caps() const {
        std::vector<SensorCapture> out;
        for (const auto& lc : loaded) out.push_back(lc->cap);
        return out;
    }
};

// 配置指定的全部遥测文件；sensor-gauge 为 none 时为空。
SensorCaptures loadSensorCaptures(const SfmConfig& cfg, bool verbose);

// 为数据库图像创建尚未标定的传感器先验源；无遥测或全部用途被禁用时返回空。
std::unique_ptr<TelemetryPriors> makeSensorPriors(const SfmConfig& cfg,
                                                  const SensorCaptures& sensors,
                                                  const MatchesDatabase& db,
                                                  const std::vector<uint32_t>& cam_ids);

// 利用候选样本或数据库已验证匹配的旋转标定传感器先验，并按相机组报告。
void calibrateSensorPriors(TelemetryPriors& priors, const std::vector<FeatureSet>& feats,
                           const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                           const std::vector<std::vector<FeatureMatch>>& matches,
                           const std::vector<Camera>& cams, const TwoViewOptions& tvopt,
                           int threads, bool verbose);
void calibrateSensorPriorsFromDatabase(TelemetryPriors& priors, const MatchesDatabase& db,
                                       const std::vector<FeatureSet>& feats,
                                       const std::vector<Camera>& cams,
                                       const TwoViewOptions& tvopt, int threads, bool verbose);

// 从相机配置生成逐图像相机参数。
std::vector<Camera> perImageCameras(const CameraSetup& cs, size_t num_images);

// 鱼眼匹配在单位视线上验证，需要焦距；未给定时先在图像对样本上搜索，见 Verification.h 的 bootstrapFocal（D45）。
struct VerifyCalibration {
    CameraSetupOptions setup;
    size_t sample_pairs = 150;  // 每组用于焦距搜索的图像对数
    // 运行遥测为空时跳过传感器先验。
    const SensorCaptures* sensors = nullptr;
    // 输出结果
    CameraSetup cameras;        // 相机分组结果，供调用方复用
    bool used_bearings = false;
    // 数据库上的传感器先验，在图像对允许时完成标定。
    std::unique_ptr<TelemetryPriors> priors;
};

// ---------------- 流水线阶段 ----------------

// 按图像目录树写出特征；reuse 保留完整且未过期的缓存，无图像对应的旧特征始终删除。
int extractDirectory(const std::string& imagedir, const std::filesystem::path& outdir,
                     const SfmConfig& cfg, ExtractStats& stats, bool reuse = false);

// 匹配中断缓存的位置与签名要求，见 Resume.h；空指针表示从头开始。
struct MatchResume {
    std::filesystem::path dir;
    std::string signature;
};

// 按排序读取特征目录并固定下游图像索引，同时填充 db.images；失败返回非零。
int loadFeatureDir(const std::string& featdir, const SfmConfig& cfg, bool with_descriptors,
                   std::vector<FeatureSet>& feats, MatchesDatabase& db);

// 对特征目录执行配对、匹配和双视图验证。
int matchFeatureDir(const std::string& featdir, const SfmConfig& cfg, PairMode mode,
                    bool verify, std::vector<FeatureSet>& feats, MatchesDatabase& db,
                    MatchStats& stats, VerifyCalibration* calib = nullptr,
                    const MatchResume* res = nullptr);

std::vector<Reconstruction> runMapper(Mapper& mapper, const MatchesDatabase& db,
                                      const std::vector<FeatureSet>& feats, SfmConfig& cfg,
                                      AssembleStats& ast);

// 按数据库图像名解析 rig，详细模式下报告；定义无法匹配时抛出 std::runtime_error。
RigTable buildRigs(const MatchesDatabase& db, const SfmConfig& cfg, bool verbose);

// 同样按图像名解析序列，见 Sequence.h。
SequenceTable buildSequences(const MatchesDatabase& db, const SfmConfig& cfg, bool verbose);

// 建图后的合并、审查、扩展、裁剪与重新播种。
std::vector<Reconstruction> finishModels(Mapper& mapper,
                                         std::vector<Reconstruction> models,
                                         const SfmConfig& cfg, bool verbose,
                                         double& seconds);

// 模型坐标规范写入 gauge.txt；up 与 scale 使用 sensors、gps、cameras、none 等机器标识符，描述性句子仅出现在日志中。
struct ModelGauge {
    bool oriented = false;   // 测量确定 +Z 为上方
    bool metric = false;     // 一个单位等于一米
    std::string up = "none";
    std::string scale = "none";
    double scale_sigma = 0;  // 相对不确定度，未估计时为 0
};

// 调平、居中与公制规范；未拟合公制坐标时返回 false。
bool fixGauge(std::vector<Reconstruction>& models, const SfmConfig& cfg,
              const std::string& imagedir, bool verbose,
              std::vector<ModelGauge>& gauge, const SensorCaptures* sensors = nullptr);

void resolveImageNames(std::vector<Reconstruction>& models, const std::string& imagedir);
void recolorPoints(std::vector<Reconstruction>& models, const SfmConfig& cfg);
void splitCamerasBySize(std::vector<Reconstruction>& models,
                        const std::vector<FeatureSet>& feats);
void writeModels(const std::vector<Reconstruction>& models,
                 const std::filesystem::path& dir, bool verbose,
                 const std::vector<ModelGauge>& gauge = {}, const RigTable* rigs = nullptr);
void writeRigs(const std::filesystem::path& dir, const Reconstruction& m, const RigTable* rigs);
// 读取模型旁的 rigs.txt，按图像名构建表并将标定存入 m.rigs；无文件时返回空。
RigTable readRigs(const std::filesystem::path& dir, Reconstruction& m);

// ---------------- 摘要报告辅助函数 ----------------

void reprojStats(const Reconstruction& rec, const std::vector<FeatureSet>& feats,
                 double& mean, double& median, size_t& nobs);
size_t distinctRegistered(const std::vector<Reconstruction>& models);
void printExtraModels(const std::vector<Reconstruction>& models,
                      const std::vector<FeatureSet>& feats);
void printFolderCoverage(const std::vector<Reconstruction>& models,
                         const MatchesDatabase& db);
void printAssembly(const AssembleStats& ast, size_t models,
                   slog::Tag tag = slog::Tag::Map);
void printCameraSetup(slog::Tag tag, const CameraSetup& cs,
                      const CameraSetupOptions& sopt, size_t nimages);
void reportFeatureCompaction(const FeatureCompactionStats& stats);
void warnIfMasksLookInverted(const ExtractStats& st);

// 图像仍在内存中时采样关键点颜色，再恢复原图坐标。
void sampleFeatureColors(FeatureSet& fs, const GrayImage& img);
void finishFeatures(FeatureSet& fs, const GrayImage& img);

// 从 EXR 补齐 seen 中未显式指定的 image-gamut 与 image-linear。
void adoptExrColorSpace(SfmConfig& cfg, const std::string& imagedir,
                        const std::set<std::string>& seen);

bool holdsImagesOutside(const std::filesystem::path& root,
                        const std::filesystem::path& nested);

// 使用稳定时钟的秒数，统一用于阶段计时。
double now();

// ---------------- 自动流水线 ----------------

// 在对象生命周期内安装运行接收端；未设置时沿用 CLI 的打印、无事件和无取消行为。进程内每次运行一个任务，因此接收端为全局状态。
class RunContext {
public:
    RunContext() = default;
    RunContext(const RunContext&) = delete;
    RunContext& operator=(const RunContext&) = delete;
    ~RunContext();

    void set_log(slog::Sink s);
    void set_events(events::Sink s);
    void set_cancel(const std::atomic<bool>* flag);
    void set_progress_dir(const std::string& dir);
};

struct AutoInputs {
    std::string image_dir;
    std::string workspace;
    // 前端显式指定了掩码位置，不再查找相邻目录。
    bool mask_dir_explicit = false;
    // 前端显式设置的字段，预设展开与 EXR 色彩空间推断不得覆盖。
    std::set<std::string> explicit_flags;
    // 预设修改项，在运行开头报告。
    std::vector<PresetChange> preset_changes;
};

struct AutoResult {
    // 退出状态：0 成功、2 失败、3 部分成功、4 未确定公制坐标；结构化字段使前端能分别读取各项结果。
    int exit_code = 0;
    int64_t registered = 0, images = 0, points = 0, models = 0;
    double mean_reproj = 0.0, median_reproj = 0.0;
    bool partial = false;
    bool metric = true;
    std::filesystem::path sparse_dir;
};

// 按已定配置顺序运行全部阶段；取消抛出 Cancelled，其他失败写入 exit_code。
AutoResult run_auto(SfmConfig& cfg, const AutoInputs& in);

// 完整运行请求：已定配置及输入。
struct AutoRequest {
    SfmConfig cfg;
    AutoInputs in;
    // 快照输出位置由调用方安装，进程内前端可采用不同路由。
    std::string progress_dir;
    bool wants_help = false;
};

// CLI 与前端共用设置列表解析，避免维护第二份参数映射；成功返回空字符串，失败返回用户可读说明。
// 显式设置优先于清单，清单优先于预设，最后由 finalize 统一展开。
std::string parse_auto_args(const std::vector<std::string>& args, AutoRequest& out);

}  // 命名空间 sfm
