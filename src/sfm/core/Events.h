#pragma once

// 以结构化数据报告阶段、进度与结果，前端无需解析日志；Progress 写磁盘快照，本接口直接通知同进程前端。
// 默认不安装接收端，关闭时开销仅为分支判断；描述性文本仍由 Log 负责。

#include <cstdint>
#include <functional>
#include <string>

namespace sfm {

// 阶段编号会经 status.bin 传给前端，新增项只能追加，不能插入改变已有编号。
enum class Stage { Extract, Match, Map, Merge, Orient, Finish, Load, Select, Seed, Refine };
inline constexpr int kNumStages = 10;

struct Event {
    enum class Kind {
        StageBegin,
        StageEnd,
        Progress,        // 阶段中已完成 done，总计 total
        ImageExtracted,
        PairVerified,
        ModelUpdated,
        Result,          // 每次运行结束时仅一次
    };

    Kind  kind = Kind::Progress;
    Stage stage = Stage::Extract;

    int64_t done = 0, total = 0;

    // ImageExtracted 携带相对图像路径及检测结果。
    std::string name;
    int     width = 0, height = 0;
    int64_t features = 0;
    int64_t masked = 0;      // 图像掩码排除的关键点数

    // PairVerified 中 inliers 为 0 表示未通过验证。
    uint32_t image_a = 0, image_b = 0, inliers = 0;

    // 供 ModelUpdated 与 Result 使用。
    int64_t registered = 0, images = 0, points = 0, models = 0;
    double  mean_reproj = 0.0;
    // 仅 Result 使用，分别表示 CLI 退出码 3/4 所描述的覆盖或公制缺失。
    bool partial = false;
    bool metric = true;
};

namespace events {

// 进程级接收端，与日志一致；未安装时 emit 仅执行可预测分支。
using Sink = std::function<void(const Event&)>;
void set_sink(Sink s);
bool armed();

void emit(const Event& e);

// 统一事件构造入口，避免各阶段手工组装结构。
void stage_begin(Stage s, int64_t total = 0);
void stage_end(Stage s);
void progress(Stage s, int64_t done, int64_t total);

// 进度按整段采集去重统计，避免种子重试或原子重建从零计数使进度回退；接口幂等且线程安全。
void map_begin(size_t n_images);
void map_placed(uint32_t image);

}  // 命名空间 events
}  // 命名空间 sfm
