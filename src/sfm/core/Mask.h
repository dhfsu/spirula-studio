// 关键点掩码非零保留、零忽略；提取时压缩关键点、描述子和颜色，后续阶段无需携带掩码状态。
// 按归一化 UV 采样使掩码分辨率独立于图像，例如 1600² 掩码可覆盖 1920² 图像，避免像素索引越界误删特征。
// 文件查找兼容多种扩展名及 mask 后缀，并以唯一的剥离名称索引回退。
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "sfm/core/Features.h"

namespace sfm {

// 行主序二值掩码，1 保留，0 忽略。
struct Mask {
    int width = 0, height = 0;
    std::vector<uint8_t> bits;  // width*height 项，严格为 0 或 1

    bool empty() const { return bits.empty(); }
    size_t pixels() const { return (size_t)width * height; }

    // 按整图范围的 u,v∈[0,1) 最近邻采样，左上角为 (0,0)；掩码为分类值不能插值，越界钳位以保留边缘内部关键点。
    bool atUV(float u, float v) const {
        if (empty()) return true;
        int mx = (int)std::floor(u * width);
        int my = (int)std::floor(v * height);
        mx = std::max(0, std::min(width - 1, mx));
        my = std::max(0, std::min(height - 1, my));
        return bits[(size_t)my * width + mx] != 0;
    }

    // 交换保留与忽略区域，兼容标记删除区的掩码。
    void invert() {
        for (uint8_t& b : bits) b = (uint8_t)!b;
    }

    // 返回保留面积比例，便于发现反向掩码等几乎删除全图的问题。
    double keepFraction() const {
        if (empty()) return 1.0;
        size_t n = 0;
        for (uint8_t b : bits) n += b;
        return (double)n / (double)pixels();
    }
};

// 将支持的图像格式解码为严格 0/1 掩码，失败返回空；实现位于 Image.cpp。
Mask loadMask(const std::string& path);

// 原地删除落在零掩码处的关键点及对应描述子、颜色，返回删除数。
// 关键点仍在工作图坐标，按像素中心约定 uv=(x+0.5)/width 采样，与 resizeGray 一致。
inline uint32_t applyMask(FeatureSet& fs, const Mask& m) {
    if (m.empty() || fs.width <= 0 || fs.height <= 0 || fs.keypoints.empty()) return 0;
    const uint32_t n = fs.count();
    // 合成测试可只有关键点而无描述子，此时每行字节数为 0，无需搬移描述子。
    const size_t dbytes = fs.descriptors.size() >= (size_t)n * fs.dim * dtypeSize(fs.dtype)
                              ? (size_t)fs.dim * dtypeSize(fs.dtype)
                              : 0;
    const bool colors = fs.hasColors();
    uint32_t out = 0;
    for (uint32_t i = 0; i < n; i++) {
        const Keypoint& k = fs.keypoints[i];
        if (!m.atUV((k.x + 0.5f) / fs.width, (k.y + 0.5f) / fs.height)) continue;
        if (out != i) {
            fs.keypoints[out] = fs.keypoints[i];
            std::copy_n(fs.descriptors.begin() + (size_t)i * dbytes, dbytes,
                        fs.descriptors.begin() + (size_t)out * dbytes);
            if (colors)
                std::copy_n(fs.colors.begin() + (size_t)i * 3, (size_t)3,
                            fs.colors.begin() + (size_t)out * 3);
        }
        out++;
    }
    fs.keypoints.resize(out);
    if (dbytes) fs.descriptors.resize((size_t)out * dbytes);
    if (colors) fs.colors.resize((size_t)out * 3);
    return n - out;
}

// ---------------- 查找图像对应的掩码 ----------------

namespace detail {

inline std::string lowerStr(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// 移除最后一个扩展名，如 a/b.jpg.png -> a/b.jpg。
inline std::string stripExt(const std::string& s) {
    size_t dot = s.rfind('.');
    size_t slash = s.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return s;
    return s.substr(0, dot);
}

// 移除文件主干末尾常见掩码标记，如 img_mask -> img；不存在时保留原文。
inline std::string stripMaskTag(const std::string& s) {
    static const char* kTags[] = {"_mask", "-mask", ".mask", "_masks", "_alpha", "_seg"};
    for (const char* t : kTags) {
        size_t n = std::strlen(t);
        if (s.size() > n && s.compare(s.size() - n, n, t) == 0) return s.substr(0, s.size() - n);
    }
    return s;
}

inline std::string baseName(const std::string& s) {
    size_t slash = s.find_last_of("/\\");
    return slash == std::string::npos ? s : s.substr(slash + 1);
}

}  // 命名空间 detail

// 优先按完整图像名加扩展名、主干名、主干加 _mask 查找，支持 png/jpg/jpeg 及大小写形式。
// 未命中时延迟构建剥离扩展名和掩码后缀的索引，兼容平铺掩码目录；仅接受唯一键，避免同名文件错误配对。
// find 虽为 const 但延迟建表不具线程安全性，调用方须在单线程中预先解析全部掩码路径。
class MaskIndex {
public:
    MaskIndex() = default;
    explicit MaskIndex(const std::string& mask_dir) : dir_(mask_dir) {
        std::error_code ec;
        valid_ = !mask_dir.empty() && std::filesystem::is_directory(dir_, ec);
    }

    bool valid() const { return valid_; }
    const std::filesystem::path& dir() const { return dir_; }

    // 返回 rel_name 的掩码路径，无匹配时为空；rel_name 使用 / 分隔符。
    std::string find(const std::string& rel_name) const {
        if (!valid_) return "";
        namespace fs = std::filesystem;
        const std::string stem = detail::stripExt(rel_name);
        static const char* kExts[] = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"};
        std::error_code ec;
        for (const char* e : kExts) {
            fs::path p = dir_ / (rel_name + e);
            if (fs::exists(p, ec)) return p.string();
            p = dir_ / (stem + e);
            if (fs::exists(p, ec)) return p.string();
        }
        for (const char* e : kExts) {
            fs::path p = dir_ / (stem + "_mask" + e);
            if (fs::exists(p, ec)) return p.string();
        }
        // 回退到剥离名称索引。
        if (!indexed_) buildIndex();
        for (const std::string& key : probeKeys(rel_name)) {
            auto it = index_.find(key);
            if (it != index_.end() && !it->second.ambiguous)
                return (dir_ / it->second.rel).string();
        }
        return "";
    }

private:
    // 键仅能对应一个文件，否则标为歧义；level 表示匹配具体程度，完整名称优先于剥离后的名称。
    struct Entry {
        std::string rel;
        int level = 0;
        bool ambiguous = false;
    };

    // 按具体程度降序生成图像名查询键。
    static std::vector<std::string> probeKeys(const std::string& rel_name) {
        using namespace detail;
        const std::string low = lowerStr(rel_name);
        const std::string base = baseName(low);
        std::vector<std::string> k;
        for (const std::string& c : {low, stripExt(low), base, stripExt(base)})
            if (std::find(k.begin(), k.end(), c) == k.end()) k.push_back(c);
        return k;
    }

    void claim(const std::string& key, const std::string& rel, int level) const {
        if (key.empty()) return;
        auto it = index_.find(key);
        if (it == index_.end()) {
            index_.emplace(key, Entry{rel, level, false});
            return;
        }
        if (it->second.rel == rel) return;             // 同一文件的另一别名
        if (level < it->second.level) it->second = Entry{rel, level, false};
        else if (level == it->second.level) it->second.ambiguous = true;
    }

    void buildIndex() const {
        using namespace detail;
        indexed_ = true;
        std::error_code ec;
        for (auto it = std::filesystem::recursive_directory_iterator(dir_, ec);
             it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file(ec)) continue;
            const std::string rel = it->path().lexically_relative(dir_).generic_string();
            const std::string low = lowerStr(rel);
            const std::string s1 = stripExt(low);       // "00023.jpg.png" -> "00023.jpg"
            const std::string s2 = stripExt(s1);        // 继续剥离得到 00023
            // 级别 0 为完整名，1 去一层扩展名，2 去两层或掩码标记；仅基本文件名形式加 4。
            claim(low, rel, 0);
            claim(s1, rel, 1);
            claim(s2, rel, 2);
            claim(stripMaskTag(s1), rel, 2);
            claim(stripMaskTag(s2), rel, 2);
            claim(baseName(low), rel, 4);
            claim(baseName(s1), rel, 5);
            claim(baseName(s2), rel, 6);
            claim(baseName(stripMaskTag(s1)), rel, 6);
            claim(baseName(stripMaskTag(s2)), rel, 6);
        }
    }

    std::filesystem::path dir_;
    bool valid_ = false;
    mutable bool indexed_ = false;
    mutable std::map<std::string, Entry> index_;
};

}  // 命名空间 sfm
