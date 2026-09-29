#pragma once

// 前端以 YAML/JSON 清单描述相机分组、镜头配置和传感器先验，避免子进程重复推导；支持两种格式读写。

#include "sfm/SfmConfig.h"
#include "sfm/core/Rig.h"
#include "sfm/core/Sequence.h"

#include <set>
#include <string>
#include <vector>

namespace sfm {

// 相机组的图像覆盖范围及已知镜头参数。
struct ManifestCamera {
    // 图像目录内的路径前缀，cam0 匹配其子路径，空值表示全局默认。
    std::string prefix;
    std::string model;               // 空值沿用本次运行的 --camera-model
    double focal = 0;                // 像素焦距，0 表示无先验
    std::vector<double> distortion;  // 模型 BA 顺序，空值表示全零
};

// 遥测来源可为视频或支持的独立文件；prefix 下图像按文件主干帧号除以 fps 定时，0 使用视频自身帧率。
struct ManifestCapture {
    std::string prefix;
    std::string telemetry;
    double fps = 0;
    double time_offset = 0;   // 加到所有帧时间上的秒数
};

struct Manifest {
    // 保存清单中的原始路径写法，应用时再相对 base_dir 解析，避免读写往返擅自改写路径。
    std::string image_dir;
    std::string mask_dir;
    // manifest_read 设置的清单目录，不写回文件。
    std::string base_dir;
    bool mask_flipped = false;
    bool has_mask_flipped = false;

    std::string camera_mode;         // single|folder|image，空值使用默认
    std::vector<ManifestCamera> cameras;
    std::vector<ManifestCapture> captures;
    // rig 成员使用路径前缀，同一后缀路径的图像组成一帧；可提供 cam_from_rig 四元数 (w,x,y,z) 和平移。
    std::vector<RigDef> rigs;
    // 序列成员同样按路径前缀指定，图像按文件名顺序排列。
    std::vector<SequenceDef> sequences;

    std::string image_gamut;         // 空值沿用运行配置
    int image_linear = -1;           // -1 未设置，0 禁用，1 启用
};

// 格式错误时抛出包含文件和行号的 std::runtime_error。
Manifest manifest_read(const std::string& path);

// 默认写 YAML，json 为真时写 JSON，均可由 manifest_read 读回。
std::string manifest_write(const Manifest& m, bool json = false);

// 将清单并入配置，seen 中的显式命令行字段优先；成功返回空字符串，否则返回不能应用的原因。
std::string manifest_apply(const Manifest& m, SfmConfig& cfg,
                           const std::set<std::string>& seen,
                           std::string& image_dir);

}  // 命名空间 sfm
