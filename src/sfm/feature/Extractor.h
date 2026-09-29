// 特征提取接口及实现工厂；仅创建所选提取器的设备上下文，避免 SIFT 与推理层同时占用同一 GPU。
// 每种提取器声明所需颜色和工作分辨率；接口上层不假定描述子为 128 维 uint8，也不假定存在尺度或方向。
#pragma once

#include <memory>
#include <string>

#include "sfm/core/Features.h"
#include "sfm/core/Image.h"
#include "sfm/feature/Sift.h"

namespace sfm {

// SfM 必须能在没有推理层时构建，因此使用独立的 ALIKED 配置，不能包含其模型头文件。
struct AlikedOptions {
    // 首次使用时下载并缓存 aliked-n16rot 或 aliked-n32，也可指定 .onnx 路径；两版本仅描述子头的采样位置数不同。
    std::string model = "aliked-n16rot";
    int   max_num_features = 2048;   // COLMAP 的 AlikedExtractionOptions 默认值
    double min_score = 0.2;
    bool  verbose = true;
    int   device = -1;
    // 规范形式 uuid:<hex>；学习前端在加载模型前用它配置共享 NN，不能仅靠上述整数选择设备。
    std::string device_selector;
};

// 独立的 LoMa 配置，与 AlikedOptions 一样，保证 SfM 可脱离推理层构建。
struct LomaOptions {
    // loma-b128 使用 DeDoDe-B 的 128 维描述子；loma-b / loma-r / loma-l / loma-g 使用 DeDoDe-G 的 256 维描述子。五种版本共享 DaD 检测器。
    std::string variant = "loma-b128";
    // .onnx 路径，覆盖 variant 对应的模型；空值表示下载。
    std::string detector_model;
    std::string descriptor_model;
    int    max_num_features = 2048;   // COLMAP 的 LomaExtractionOptions 默认值
    double min_score = 0.0;           // DaD 的密度没有实用的下限
    bool   verbose = true;
    int    device = -1;
    // 规范形式 uuid:<hex>；参见 AlikedOptions::device_selector。
    std::string device_selector;
};

struct IFeatureExtractor {
    virtual ~IFeatureExtractor() = default;

    // 特征坐标属于 img；调用方负责缩放回原始文件的坐标系（D46）。
    virtual FeatureSet extract(const GrayImage& img) = 0;

    virtual const char* name() const = 0;

    // SIFT 使用亮度；学习检测器以 RGB 训练，必须让加载器解码颜色。
    virtual bool wantsColor() const { return false; }
};

// 未显式设置时的最长边，与 COLMAP 的 FeatureExtractionOptions::EffMaxImageSize() 一致：SIFT 为 3200，学习前端为 1600；后者受全分辨率特征图开销限制。
int defaultMaxImageSize(const std::string& type);

// 判断 type 是否指定学习前端。
bool isAlikedType(const std::string& type);
bool isLomaType(const std::string& type);

// LoMa 描述子宽度：loma-b128 为 128，其余四种为 256，其他类型为 0；auto 在提取前拒绝宽度不一致的 --features / --matcher 组合。
int lomaDescriptorDim(const std::string& variant);

// 类型未知，或当前构建缺少所需推理层时，抛出包含类型名称的 std::runtime_error。
std::unique_ptr<IFeatureExtractor> createFeatureExtractor(const std::string& type,
                                                          const SiftOptions& sift,
                                                          const AlikedOptions& aliked,
                                                          const LomaOptions& loma);

}  // 命名空间 sfm
