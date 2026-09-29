// 进度快照实现，参见 Progress.h。

#include "sfm/core/Progress.h"

#include "sfm/core/Model.h"

#include "external/stb_image_write.h"

#include "sfm/core/Matches.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace sfm {
namespace progress {

namespace {

using Clock = std::chrono::steady_clock;

// 按时间限制快照频率，兼顾实时预览与序列化成本，避免快速配准时每秒改写数千次。
constexpr double kInterval = 1.5;

// 70 万验证对可能额外产生 1.4 GB 实时匹配文件；达到上限后停止增长，读取方将缺失内容视为无法预览。
constexpr uint64_t kLiveCap = 256ull << 20;

// 排队写入的缩略图已缩小，单项约 1 MB，避免持有完整工作图。
struct ThumbJob {
    fs::path dst;
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
};

// 在 GPU 消费线程编码 JPEG 曾占提取阶段六分之一；有界写入队列满时丢最旧预览而不阻塞，查看器可回退解码原图。
class ThumbWriter {
public:
    ~ThumbWriter() { stop(); }

    void push(ThumbJob j) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!worker_.joinable()) {
            quit_ = false;
            worker_ = std::thread([this] { run(); });
        }
        while (q_.size() >= kQueue) q_.pop_front();
        q_.push_back(std::move(j));
        lk.unlock();
        cv_.notify_one();
    }

    // 返回时此前提交的缩略图全部写入磁盘。
    void drain() {
        std::unique_lock<std::mutex> lk(mu_);
        if (!worker_.joinable()) return;
        idle_.wait(lk, [this] { return q_.empty() && !busy_; });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!worker_.joinable()) return;
            quit_ = true;
        }
        cv_.notify_all();
        worker_.join();
        std::lock_guard<std::mutex> lk(mu_);
        q_.clear();
    }

private:
    static constexpr size_t kQueue = 8;

    void run() {
        fs::path made;   // 最近创建的目录
        std::error_code ec;   // 同一文件夹内后续文件共用该目录
        for (;;) {
            ThumbJob j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return quit_ || !q_.empty(); });
                if (q_.empty()) return;   // 已请求退出且队列为空
                j = std::move(q_.front());
                q_.pop_front();
                busy_ = true;
            }
            const fs::path parent = j.dst.parent_path();
            if (parent != made) {
                fs::create_directories(parent, ec);
                made = parent;
            }
            const fs::path tmp = fs::path(j.dst) += ".tmp";
            if (stbi_write_jpg(tmp.string().c_str(), j.w, j.h, 3, j.rgb.data(), 88))
                fs::rename(tmp, j.dst, ec);
            if (ec) fs::remove(tmp, ec);
            {
                std::lock_guard<std::mutex> lk(mu_);
                busy_ = false;
            }
            idle_.notify_all();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_, idle_;
    std::deque<ThumbJob> q_;
    std::thread worker_;
    bool quit_ = false, busy_ = false;
};

struct State {
    std::mutex mu;
    std::string dir;
    Clock::time_point model_at{};
    Clock::time_point pairs_at{};
    bool model_started = false, pairs_started = false;

    // 图像对矩阵每边聚合为 kMatrixBins 个桶。
    uint32_t n_images = 0, bins = 0;
    std::vector<uint32_t> counts, planned, verified;
    bool pairs_dirty = false;

    // 验证线程并发追加，因此使用独立锁，整个阶段保持文件打开。
    std::mutex live_mu;
    std::ofstream live;
    Clock::time_point live_at{};
    bool live_started = false;
    uint64_t live_bytes = 0;

    bool gauge_oriented = false, gauge_metric = false;

    Clock::time_point status_at{};
    bool status_started = false;
    Event last;                  // 下一次非强制写入将记录的状态

    ThumbWriter thumbs;
};

State& state() {
    static State s;
    return s;
}

// 先完整写文件再重命名，轮询读取者只能看到旧快照或完整新快照。
void write_atomic(const std::string& name, const std::string& bytes) {
    State& s = state();
    const fs::path dst = fs::path(s.dir) / name;
    const fs::path tmp = fs::path(s.dir) / (name + ".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        f.write(bytes.data(), (std::streamsize)bytes.size());
        if (!f) return;
    }
    std::error_code ec;
    fs::rename(tmp, dst, ec);
    if (ec) fs::remove(tmp, ec);
}

void put(std::string& b, const void* p, size_t n) {
    b.append((const char*)p, n);
}
void put_u32(std::string& b, uint32_t v) { put(b, &v, 4); }
void put_u64(std::string& b, uint64_t v) { put(b, &v, 8); }
void put_f32(std::string& b, float v) { put(b, &v, 4); }
void put_i64(std::string& b, int64_t v) { put(b, &v, 8); }
void put_f64(std::string& b, double v) { put(b, &v, 8); }

// 距 last 达到间隔时返回 true 并更新时间。
bool due(Clock::time_point& last, bool& started) {
    const auto now = Clock::now();
    if (started &&
        std::chrono::duration<double>(now - last).count() < kInterval)
        return false;
    started = true;
    last = now;
    return true;
}

void write_pairs_locked() {
    State& s = state();
    if (!s.pairs_dirty || s.counts.empty()) return;
    std::string b;
    b.reserve(16 + s.counts.size() * 12);
    put(b, "VKPP", 4);
    put_u32(b, 2);
    put_u32(b, s.n_images);
    put_u32(b, s.bins);
    put(b, s.counts.data(), s.counts.size() * 4);
    put(b, s.planned.data(), s.planned.size() * 4);
    put(b, s.verified.data(), s.verified.size() * 4);
    write_atomic("pairs.bin", b);
    s.pairs_dirty = false;
}

// 图像对对应的矩阵单元。
size_t cell_of(const State& s, uint32_t image1, uint32_t image2, size_t& mirror) {
    const uint32_t a = (uint32_t)((uint64_t)image1 * s.bins / s.n_images);
    const uint32_t b = (uint32_t)((uint64_t)image2 * s.bins / s.n_images);
    mirror = (size_t)b * s.bins + a;
    return (size_t)a * s.bins + b;
}

}  // 匿名命名空间

void set_dir(const std::string& dir) {
    State& s = state();
    // 改变目录前清空队列，已有任务仍指向旧目录。
    s.thumbs.stop();
    {
        // 立即关闭旧文件，避免同进程下一任务继续追加到前一任务。
        std::lock_guard<std::mutex> live(s.live_mu);
        s.live.close();
        s.live.clear();
    }
    std::lock_guard<std::mutex> lk(s.mu);
    s.dir = dir;
    s.gauge_oriented = s.gauge_metric = false;
    if (dir.empty()) return;
    std::error_code ec;
    fs::create_directories(dir, ec);
}

bool enabled() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    return !s.dir.empty();
}

void model(const Reconstruction& rec, bool force, const PointColor& color) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.dir.empty()) return;
    if (!due(s.model_at, s.model_started) && !force) return;

    // 只输出已配准图像，未配准图像没有可绘制位姿。
    std::vector<const Image*> imgs;
    for (const auto& kv : rec.images)
        if (kv.second.registered) imgs.push_back(&kv.second);

    const uint64_t n_pts = rec.points3D.size();
    const uint64_t stride = n_pts > kMaxPoints ? (n_pts / kMaxPoints + 1) : 1;

    std::string b;
    b.reserve(64 + imgs.size() * 128 + (size_t)(n_pts / stride + 1) * 15);
    put(b, "VKPM", 4);
    put_u32(b, 4);
    put_u32(b, (s.gauge_oriented ? 1u : 0u) | (s.gauge_metric ? 2u : 0u));
    put_u32(b, (uint32_t)rec.images.size());
    put_u32(b, (uint32_t)imgs.size());
    put_u64(b, n_pts);
    for (const Image* im : imgs) {
        put_u32(b, im->id);
        // COLMAP 世界到相机转为 OpenGL 相机到世界：R^T 的第 1、2 列取负，平移为 -R^T t，供读取方直接绘制。
        const Mat3& R = im->pose.R;      // 行主序，世界 -> 相机
        const Vec3& t = im->pose.t;
        const double C[3] = {
            -(R[0] * t.x + R[3] * t.y + R[6] * t.z),
            -(R[1] * t.x + R[4] * t.y + R[7] * t.z),
            -(R[2] * t.x + R[5] * t.y + R[8] * t.z)};
        for (int r = 0; r < 3; r++) {
            put_f32(b, (float)R[r]);            // R^T 的第 r 行、第 0 列
            put_f32(b, (float)-R[3 + r]);       // 第 1 列取负
            put_f32(b, (float)-R[6 + r]);       // 第 2 列取负
            put_f32(b, (float)C[r]);
        }
        // 按 cameras.bin 形式携带 COLMAP 模型 ID 与参数，读取方复用相机解析映射，使鱼眼视锥保持正确形状。
        const auto cam = rec.cameras.find(im->camera_id);
        const Camera c = cam == rec.cameras.end() ? Camera{} : cam->second;
        put_u32(b, (uint32_t)c.width);
        put_u32(b, (uint32_t)c.height);
        put_u32(b, (uint32_t)camColmapId(c.model));
        double ps[12] = {};
        packColmap(c, ps);
        const uint32_t np = (uint32_t)camColmapParams(c.model);
        put_u32(b, np);
        for (uint32_t k = 0; k < np; k++) put(b, &ps[k], 8);
    }

    // 先写数量，便于读取方预分配缓冲。
    uint32_t written = 0;
    for (uint64_t i = 0; i < n_pts; i += stride) written++;
    put_u32(b, written);
    uint64_t k = 0, next = 0;
    for (const auto& kv : rec.points3D) {
        if (k++ != next) continue;
        next += stride;
        put_f32(b, (float)kv.second.xyz.x);
        put_f32(b, (float)kv.second.xyz.y);
        put_f32(b, (float)kv.second.xyz.z);
        uint8_t rgb[3] = {kv.second.rgb[0], kv.second.rgb[1], kv.second.rgb[2]};
        if (color) color(kv.second, rgb);
        put(b, rgb, 3);
    }
    write_atomic("model.bin", b);
}

void gauge(bool oriented, bool metric) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.gauge_oriented = oriented;
    s.gauge_metric = metric;
}

void begin_matching(uint32_t n_images,
                    const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.dir.empty() || n_images == 0) return;
    s.n_images = n_images;
    s.bins = n_images < kMatrixBins ? n_images : kMatrixBins;
    s.counts.assign((size_t)s.bins * s.bins, 0);
    s.planned.assign((size_t)s.bins * s.bins, 0);
    s.verified.assign((size_t)s.bins * s.bins, 0);
    for (const auto& p : pairs) {
        if (p.first >= n_images || p.second >= n_images) continue;
        size_t mirror = 0;
        const size_t c = cell_of(s, p.first, p.second, mirror);
        s.planned[c]++;
        if (mirror != c) s.planned[mirror]++;
    }
    s.pairs_started = false;
    s.pairs_dirty = true;
}

void pair(uint32_t image1, uint32_t image2, uint32_t inliers) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.dir.empty() || s.counts.empty()) return;
    if (image1 >= s.n_images || image2 >= s.n_images) return;
    size_t mirror = 0;
    const size_t c = cell_of(s, image1, image2, mirror);
    s.counts[c] += inliers;
    s.verified[c]++;
    if (mirror != c) {
        s.counts[mirror] += inliers;
        s.verified[mirror]++;
    }
    s.pairs_dirty = true;
    if (due(s.pairs_at, s.pairs_started)) write_pairs_locked();
}

void status(const Event& e) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.dir.empty()) return;
    using K = Event::Kind;
    // 此处状态不依赖图像对，忽略逐对事件以免验证线程争锁而串行化。
    if (e.kind == K::PairVerified) return;
    // 阶段切换与最终结果必须立即写出，普通进度可等待限流间隔。
    const bool force = e.kind == K::StageBegin || e.kind == K::StageEnd ||
                       e.kind == K::Result;
    if (e.kind == K::Progress || e.kind == K::ImageExtracted) {
        s.last.stage = e.stage;
        s.last.done = e.done;
        s.last.total = e.total;
    } else if (e.kind == K::ModelUpdated) {
        // 这里是当前模型大小，阶段进度由 map_placed 统计，避免种子重试时回退。
        s.last.stage = e.stage;
        s.last.registered = e.registered;
        s.last.images = e.images;
        s.last.points = e.points;
    } else if (force) {
        Event keep = s.last;
        s.last = e;
        if (e.kind == K::StageBegin) { s.last.done = 0; }
        if (e.kind == K::StageEnd) { s.last.done = keep.total; s.last.total = keep.total; }
    }
    if (!force && !due(s.status_at, s.status_started)) return;
    if (force) { s.status_at = Clock::now(); s.status_started = true; }

    const Event& v = s.last;
    uint32_t flags = 0;
    if (e.kind == K::Result) flags |= 1u;
    if (v.partial) flags |= 2u;
    if (v.metric) flags |= 4u;
    std::string b;
    b += "VKPS";
    put_u32(b, 1);
    put_u32(b, (uint32_t)v.stage);
    put_u32(b, flags);
    put_i64(b, v.done);
    put_i64(b, v.total);
    put_i64(b, v.registered);
    put_i64(b, v.images);
    put_i64(b, v.points);
    put_i64(b, v.models);
    put_f64(b, v.mean_reproj);
    write_atomic("status.bin", b);
}

void live_matches_begin(const std::vector<std::string>& names,
                        const std::vector<uint32_t>& num_features) {
    State& s = state();
    std::string dir;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        dir = s.dir;
    }
    if (dir.empty()) return;
    std::lock_guard<std::mutex> lk(s.live_mu);
    s.live.close();
    s.live.clear();
    s.live.open(fs::path(dir) / "live_matches.bin",
                std::ios::binary | std::ios::trunc);
    if (!s.live) return;
    s.live_started = false;
    s.live_bytes = 0;
    const uint32_t version = 3, nimg = (uint32_t)names.size();
    s.live.write("VKMT", 4);
    s.live.write((const char*)&version, 4);
    s.live.write((const char*)&nimg, 4);
    for (uint32_t i = 0; i < nimg; i++) {
        const uint32_t len = (uint32_t)names[i].size();
        const uint32_t nf = i < num_features.size() ? num_features[i] : 0;
        s.live.write((const char*)&len, 4);
        s.live.write(names[i].data(), len);
        s.live.write((const char*)&nf, 4);
    }
    const uint32_t npairs = kStreamingPairs;
    s.live.write((const char*)&npairs, 4);
    s.live.flush();
}

// 锁外打包并一次写入，避免验证线程逐索引流写入而排队；按时间刷新，读取方允许尚未写完的尾部。
void live_pair(uint32_t a, uint32_t b, int32_t config,
               const uint32_t* idx1, const uint32_t* idx2, size_t stride,
               uint32_t count) {
    std::string rec;
    rec.reserve(16 + (size_t)count * 8);
    put_u32(rec, a);
    put_u32(rec, b);
    put(rec, &config, 4);
    put_u32(rec, count);
    for (uint32_t i = 0; i < count; i++) {
        put(rec, (const char*)idx1 + i * stride, 4);
        put(rec, (const char*)idx2 + i * stride, 4);
    }
    State& s = state();
    std::lock_guard<std::mutex> lk(s.live_mu);
    if (!s.live || s.live_bytes > kLiveCap) return;
    s.live.write(rec.data(), (std::streamsize)rec.size());
    s.live_bytes += rec.size();
    if (due(s.live_at, s.live_started)) s.live.flush();
}

// 预览使用区域均值缩小；在调用方缓冲仍有效时完成缩放，随后数据由写入线程独占。
void thumbnail(const std::string& rel_stem, const uint8_t* rgb, int w, int h) {
    State& s = state();
    std::string dir;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        if (s.dir.empty()) return;
        dir = s.dir;
    }
    if (!rgb || w <= 0 || h <= 0) return;
    const int longest = w > h ? w : h;
    const int step = longest > kThumbLong ? (longest + kThumbLong - 1) / kThumbLong : 1;
    ThumbJob j;
    j.w = (w + step - 1) / step;
    j.h = (h + step - 1) / step;
    j.dst = fs::path(dir) / "thumbs" / (rel_stem + ".jpg");
    j.rgb.resize((size_t)j.w * j.h * 3);
    for (int y = 0; y < j.h; y++) {
        for (int x = 0; x < j.w; x++) {
            uint32_t acc[3] = {0, 0, 0};
            uint32_t n = 0;
            for (int dy = 0; dy < step; dy++) {
                const int sy = y * step + dy;
                if (sy >= h) break;
                for (int dx = 0; dx < step; dx++) {
                    const int sx = x * step + dx;
                    if (sx >= w) break;
                    const uint8_t* p = rgb + ((size_t)sy * w + sx) * 3;
                    acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2];
                    n++;
                }
            }
            uint8_t* d = j.rgb.data() + ((size_t)y * j.w + x) * 3;
            for (int c = 0; c < 3; c++) d[c] = n ? (uint8_t)(acc[c] / n) : 0;
        }
    }
    s.thumbs.push(std::move(j));
}

void flush() {
    State& s = state();
    s.thumbs.drain();
    {
        std::lock_guard<std::mutex> lk(s.live_mu);
        if (s.live) s.live.flush();
    }
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.dir.empty()) return;
    write_pairs_locked();
}

}  // 命名空间 progress
}  // 命名空间 sfm
