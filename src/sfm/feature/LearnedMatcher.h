#pragma once
// 学习匹配器接口与实现工厂。暴力匹配每对约 3.5 ms，LightGlue 在双图特征上运行九层 Transformer，耗时数十毫秒，必须先筛选图像对。
// 千张图像使用 exhaustive 配对约有五十万对，可能耗时数小时。
// 接口不依赖推理层头文件；缺少推理层时由工厂在运行期报错。

#include <memory>
#include <string>

#include "sfm/feature/Matcher.h"

namespace sfm {

struct LightGlueOptions {
    // 首次使用时下载并缓存 aliked-lightglue，也可指定 .onnx 文件路径。
    std::string model = "aliked-lightglue";
    // COLMAP 的 LightGlueONNXMatchingOptions 默认值。
    double min_score = 0.1;
    int    device = -1;
    // 规范形式 uuid:<hex>；加载 LightGlue 权重前提交给共享 NN；空值沿用 NN 自身的设备选择优先级。
    std::string device_selector;
    bool   verbose = true;
};

// LoMa 的五种已发布匹配器均为九层；嵌入宽度和训练所用描述子从权重文件读取。
struct LomaMatchOptions {
    // 下载并缓存 loma-b、loma-b128、loma-r、loma-l、loma-g，也可指定 .onnx 路径；空值沿用 --matcher 指定的版本。
    std::string model;
    // COLMAP 的 LomaMatchingOptions 默认值，也是 LoMa 自身的过滤阈值。
    double min_score = 0.1;
    int    device = -1;
    // 规范形式 uuid:<hex>；参见 LightGlueOptions::device_selector。
    std::string device_selector;
    bool   verbose = true;
};

bool isLearnedMatcher(const std::string& type);

// 类型未知，或当前构建缺少所需推理层时，抛出包含类型名称的 std::runtime_error。
std::unique_ptr<IFeatureMatcher> createFeatureMatcher(const std::string& type,
                                                      const MatchOptions& match,
                                                      const LightGlueOptions& lightglue,
                                                      const LomaMatchOptions& loma);

}  // 命名空间 sfm
