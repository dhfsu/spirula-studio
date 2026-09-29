#pragma once

// 供外部前端轮询的进度快照，仅设置 progress-dir 时启用；普通 CLI 不产生额外文件。
// 快照先完整写临时文件再重命名，并按时间限流；实时匹配另用追加流，读取方处理未完成尾部。

#include "sfm/core/Events.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace sfm {

struct Point3D;
struct Reconstruction;

namespace progress {

// 图像对矩阵边长与模型点数预算按屏幕预览需求设置：512² 个 u32 约 1 MB，5 万点已足够预览。
inline constexpr uint32_t kMatrixBins = 512;
inline constexpr uint32_t kMaxPoints = 50000;

// 快照目录，空值关闭全部输出。
void set_dir(const std::string& dir);
bool enabled();

// 将模型朝向与公制状态带入下一份 model.bin，供前端标明单位。
void gauge(bool oriented, bool metric);

// model.bin：VKPM，u32 版本 4、标志（1 已定向，2 公制）、图像数、已配准数，u64 点数。
// 每已配准图像含 u32 ID、f32 OpenGL c2w[12]、u32 宽高/模型 ID/参数数与 f64 参数；点区为 u32 数量及 f32 xyz、u8 rgb。

// 当前模型抽样至 kMaxPoints，未到间隔时立即返回，force 用于阶段最终状态。
// color 只为实际写出的点计算 RGB，避免重建中尚未统一着色的预览全灰。
using PointColor = std::function<void(const Point3D&, uint8_t rgb[3])>;
void model(const Reconstruction& rec, bool force = false,
           const PointColor& color = {});

// pairs.bin：VKPP，u32 版本 2、图像数、桶数，随后为三层 bins*bins 的 u32：内点总数、候选对数、已验证对数。
// 候选与已验证计数区分尚未处理和从未计划匹配的单元。
void begin_matching(uint32_t n_images,
                    const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
// 记录一个已验证图像对，内点为 0 表示未通过；支持验证工作线程并发调用。
void pair(uint32_t image1, uint32_t image2, uint32_t inliers);

// status.bin：VKPS，u32 版本 1、阶段、标志（1 完成，2 部分成功，4 公制），i64 done/total/registered/images/points/models，f64 平均重投影误差。
// 外部前端读取与进程内事件相同的状态，普通进度限流，阶段切换与结果强制写出。
void status(const Event& e);

// thumbs/<rel_stem>.jpg 保存已解码工作图的缩略图，最长边 kThumbLong，避免前端再次解码高分辨率原图而落后于提取进度。
void thumbnail(const std::string& rel_stem, const uint8_t* rgb, int w, int h);
inline constexpr int kThumbLong = 640;

// live_matches.bin 使用 kStreamingPairs 对数标记，验证时逐对追加，供匹配图即时预览。
void live_matches_begin(const std::vector<std::string>& names,
                        const std::vector<uint32_t>& num_features);
void live_pair(uint32_t a, uint32_t b, int32_t config,
               const uint32_t* idx1, const uint32_t* idx2, size_t stride,
               uint32_t count);

// 无视时间门限立即刷新缓冲，阶段结束时保证屏幕显示最终状态。
void flush();

}  // 命名空间 progress
}  // 命名空间 sfm
