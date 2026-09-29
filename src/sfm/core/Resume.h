#pragma once

// 通过阶段签名判断中断缓存是否可复用，缓存位于工作区 .resume/，可删除后重新计算。
// 验证阶段按独立图像对追加记录，适合中途恢复；完整 matches.bin 写出后移除日志。

#include "sfm/core/Matches.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sfm {
namespace resume {

// 位于工作区、与 features 和 matches.bin 相邻，使用点前缀降低与用户目录命名冲突的可能。
inline constexpr const char* kDir = ".resume";

std::filesystem::path dir(const std::string& workspace);

// 读取已保存签名，无记录时为空；store 写入，forget 删除并使对应输出失效。
std::string recorded(const std::filesystem::path& file);
void store(const std::filesystem::path& file, const std::string& signature);
void forget(const std::filesystem::path& file);

// 工作区 .resume/ 下的全部内容。
void clear(const std::string& workspace);

// 复用已选择的图像对以节省筛选时间；文件不存在或设置不同则返回 false 且不修改 pairs。
bool readPairs(const std::filesystem::path& file, const std::string& signature,
               std::vector<std::pair<uint32_t, uint32_t>>& pairs);
void writePairs(const std::filesystem::path& file, const std::string& signature,
                const std::vector<std::pair<uint32_t, uint32_t>>& pairs);

// 每个无序图像对对应一个键。
inline uint64_t pairKey(uint32_t a, uint32_t b) {
    return a < b ? ((uint64_t)a << 32) | b : ((uint64_t)b << 32) | a;
}

// 记录所有已完成验证的图像对，无论保留与否，避免重复计算；工作线程并发追加，通过锁保护。
class MatchJournal {
public:
    ~MatchJournal() { close(); }

    // 恢复后继续追加或新建日志；打开失败则关闭日志功能，后续调用为空操作。
    bool open(const std::filesystem::path& file, const std::string& signature,
              bool append);
    // putative 保存验证前候选匹配数量，供最终摘要统计。
    void record(uint32_t a, uint32_t b, int32_t config, uint32_t putative,
                const uint32_t* idx1, const uint32_t* idx2, size_t stride,
                uint32_t count);
    void flush();
    void close();
    bool armed() const { return _armed; }

private:
    void write_locked(bool force);

    std::mutex _mu;
    std::ofstream _f;
    std::string _buf;
    double _flushed_at = 0;
    bool _armed = false;
};

// 读取已完成图像对及保留匹配；文件缺失、设置或图像不同则返回 false，不改输出。
bool readJournal(const std::filesystem::path& file, const std::string& signature,
                 const std::vector<ImageEntry>& images,
                 std::unordered_map<uint64_t, TwoViewMatches>& kept,
                 std::vector<uint64_t>& done, uint64_t& putative);

}  // 命名空间 resume
}  // 命名空间 sfm
