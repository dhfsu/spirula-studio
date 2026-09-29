// 运行事件实现，参见 Events.h。

#include "sfm/core/Events.h"

#include <mutex>
#include <vector>

namespace sfm {
namespace events {

namespace {
std::mutex g_mu;
Sink g_sink;

// 建图进度状态使用独立锁；原子工作线程会调用 map_placed，接收端则在事件发出线程执行。
std::mutex g_map_mu;
std::vector<char> g_placed;
int64_t g_placed_n = 0;
}  // 匿名命名空间

void set_sink(Sink s) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_sink = std::move(s);
}

bool armed() {
    std::lock_guard<std::mutex> lk(g_mu);
    return (bool)g_sink;
}

// 验证工作池并发发事件，通过锁保证顺序；接收端不得回调 sfm::events。
void emit(const Event& e) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_sink) g_sink(e);
}

void stage_begin(Stage s, int64_t total) {
    Event e;
    e.kind = Event::Kind::StageBegin;
    e.stage = s;
    e.total = total;
    emit(e);
}

void stage_end(Stage s) {
    Event e;
    e.kind = Event::Kind::StageEnd;
    e.stage = s;
    emit(e);
}

void progress(Stage s, int64_t done, int64_t total) {
    Event e;
    e.kind = Event::Kind::Progress;
    e.stage = s;
    e.done = done;
    e.total = total;
    emit(e);
}

void map_begin(size_t n_images) {
    std::lock_guard<std::mutex> lk(g_map_mu);
    g_placed.assign(n_images, 0);
    g_placed_n = 0;
}

void map_placed(uint32_t image) {
    int64_t done = 0, total = 0;
    {
        std::lock_guard<std::mutex> lk(g_map_mu);
        if (image >= g_placed.size() || g_placed[image]) return;
        g_placed[image] = 1;
        done = ++g_placed_n;
        total = (int64_t)g_placed.size();
    }
    // 每阶段每图像只发一次，无需限流。
    progress(Stage::Map, done, total);
}

}  // 命名空间 events
}  // 命名空间 sfm
