// 断点恢复实现，参见 Resume.h。

#include "sfm/core/Resume.h"

#include <chrono>
#include <cstring>

namespace fs = std::filesystem;

namespace sfm {
namespace resume {

namespace {

// 按大小或时间任一阈值刷新；快任务避免逐对写盘，慢学习匹配避免长时间成果只留在内存。
constexpr size_t kFlushBytes = 4u << 20;
constexpr double kFlushSeconds = 5.0;

double steadyNow() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void put_u32(std::string& b, uint32_t v) { b.append((const char*)&v, 4); }
void put_i32(std::string& b, int32_t v) { b.append((const char*)&v, 4); }

bool read_u32(std::istream& f, uint32_t& v) {
    f.read((char*)&v, 4);
    return f.gcount() == 4;
}

// 签名块统一为 u32 文本长度加签名文本。
void put_signature(std::string& b, const std::string& sig) {
    put_u32(b, (uint32_t)sig.size());
    b += sig;
}

bool check_signature(std::istream& f, const std::string& sig) {
    uint32_t len = 0;
    if (!read_u32(f, len) || len > (1u << 20)) return false;
    std::string seen(len, '\0');
    f.read(&seen[0], (std::streamsize)len);
    return f.gcount() == (std::streamsize)len && seen == sig;
}

bool magic_is(std::istream& f, const char* want) {
    char m[4];
    f.read(m, 4);
    return f.gcount() == 4 && std::memcmp(m, want, 4) == 0;
}

}  // 匿名命名空间

fs::path dir(const std::string& workspace) { return fs::path(workspace) / kDir; }

std::string recorded(const fs::path& file) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return "";
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

void store(const fs::path& file, const std::string& signature) {
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    if (f) f << signature;
}

void forget(const fs::path& file) {
    std::error_code ec;
    fs::remove(file, ec);
}

void clear(const std::string& workspace) {
    if (workspace.empty()) return;
    std::error_code ec;
    fs::remove_all(dir(workspace), ec);
}

// ---------------- 图像对列表 ----------------

bool readPairs(const fs::path& file, const std::string& signature,
               std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    std::ifstream f(file, std::ios::binary);
    if (!f || !magic_is(f, "VKRP") || !check_signature(f, signature)) return false;
    uint32_t n = 0;
    if (!read_u32(f, n)) return false;
    std::vector<std::pair<uint32_t, uint32_t>> out(n);
    f.read((char*)out.data(), (std::streamsize)n * 8);
    if (f.gcount() != (std::streamsize)n * 8) return false;
    pairs.swap(out);
    return true;
}

void writePairs(const fs::path& file, const std::string& signature,
                const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    std::string head;
    head += "VKRP";
    put_signature(head, signature);
    put_u32(head, (uint32_t)pairs.size());
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    if (!f) return;
    f.write(head.data(), (std::streamsize)head.size());
    f.write((const char*)pairs.data(), (std::streamsize)pairs.size() * 8);
}

// ---------------- 验证日志 ----------------

bool MatchJournal::open(const fs::path& file, const std::string& signature,
                        bool append) {
    close();
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    _f.open(file, std::ios::binary |
                      (append ? std::ios::app : std::ios::trunc));
    if (!_f) return false;
    if (!append) {
        std::string head;
        head += "VKRJ";
        put_signature(head, signature);
        _f.write(head.data(), (std::streamsize)head.size());
    }
    _flushed_at = steadyNow();
    _armed = (bool)_f;
    return _armed;
}

void MatchJournal::record(uint32_t a, uint32_t b, int32_t config, uint32_t putative,
                          const uint32_t* idx1, const uint32_t* idx2,
                          size_t stride, uint32_t count) {
    if (!_armed) return;
    // 验证线程在加锁前完成打包。
    std::string rec;
    rec.reserve(20 + (size_t)count * 8);
    put_u32(rec, a);
    put_u32(rec, b);
    put_i32(rec, config);
    put_u32(rec, putative);
    put_u32(rec, count);
    for (uint32_t i = 0; i < count; i++) {
        rec.append((const char*)idx1 + i * stride, 4);
        rec.append((const char*)idx2 + i * stride, 4);
    }
    std::lock_guard<std::mutex> lk(_mu);
    _buf += rec;
    write_locked(false);
}

void MatchJournal::write_locked(bool force) {
    const double t = steadyNow();
    if (_buf.empty() ||
        (!force && _buf.size() < kFlushBytes && t - _flushed_at < kFlushSeconds))
        return;
    _f.write(_buf.data(), (std::streamsize)_buf.size());
    _f.flush();
    _buf.clear();
    _flushed_at = t;
}

void MatchJournal::flush() {
    std::lock_guard<std::mutex> lk(_mu);
    if (_armed) write_locked(true);
}

void MatchJournal::close() {
    std::lock_guard<std::mutex> lk(_mu);
    if (_armed) write_locked(true);
    _f.close();
    _f.clear();
    _buf.clear();
    _armed = false;
}

bool readJournal(const fs::path& file, const std::string& signature,
                 const std::vector<ImageEntry>& images,
                 std::unordered_map<uint64_t, TwoViewMatches>& kept,
                 std::vector<uint64_t>& done, uint64_t& putative) {
    std::ifstream f(file, std::ios::binary);
    if (!f || !magic_is(f, "VKRJ") || !check_signature(f, signature)) return false;
    std::unordered_map<uint64_t, TwoViewMatches> k;
    std::vector<uint64_t> d;
    uint64_t put = 0;
    const uint32_t nimg = (uint32_t)images.size();
    for (;;) {
        uint32_t a = 0, b = 0, count = 0, offered = 0;
        int32_t config = 0;
        if (!read_u32(f, a) || !read_u32(f, b)) break;
        f.read((char*)&config, 4);
        if (f.gcount() != 4 || !read_u32(f, offered) || !read_u32(f, count)) break;
        // 文件装不下所声明记录时视为尚未刷新的尾部，停止读取并保留此前完整记录。
        if (a >= nimg || b >= nimg || count > images[a].num_features) break;
        TwoViewMatches tvm;
        tvm.image1 = a;
        tvm.image2 = b;
        tvm.config = config;
        tvm.matches.resize(count);
        bool torn = false;
        for (uint32_t i = 0; i < count; i++) {
            f.read((char*)&tvm.matches[i].idx1, 4);
            f.read((char*)&tvm.matches[i].idx2, 4);
            if (!f) { torn = true; break; }
        }
        if (torn) break;
        const uint64_t key = pairKey(a, b);
        d.push_back(key);
        put += offered;
        if (count) k.emplace(key, std::move(tvm));
    }
    kept.swap(k);
    done.swap(d);
    putative = put;
    return true;
}

}  // 命名空间 resume
}  // 命名空间 sfm
