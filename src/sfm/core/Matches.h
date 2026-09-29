// 特征匹配及独立二进制格式；每条匹配记录两图像的特征索引，数据库汇集整个数据集的双视图对应。
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sfm/core/Camera.h"

namespace sfm {

// 候选对应：图像 image1 的 idx1 <-> 图像 image2 的 idx2。
struct FeatureMatch {
    uint32_t idx1 = 0, idx2 = 0;
    float distance = 0;   // 描述子 L2 距离，不持久化
};

// 有序图像对的匹配；验证后仅保存内点，config 为 TwoViewConfig，0 表示原始未验证数据。
struct TwoViewMatches {
    uint32_t image1 = 0, image2 = 0;      // MatchesDatabase::images 的索引
    int32_t config = 0;                   // 0 表示未验证
    std::vector<FeatureMatch> matches;
};

// 匹配数据库中的图像身份。
struct ImageEntry {
    std::string name;         // 特征文件主干路径，即内部图像名
    uint32_t num_features = 0;
};

struct MatchesDatabase {
    std::vector<ImageEntry> images;
    std::vector<TwoViewMatches> pairs;
    // 保存验证使用的相机分组、内参与焦距来源，避免建图从受验证选择偏置的内点重新估计；旧文件或未验证输出可为空（D47）。
    std::vector<Camera> cameras;         // 每个相机 ID 一项
    std::vector<uint32_t> camera_ids;    // 与 images 一一对应
    std::vector<uint8_t> focal_prior;    // 与 cameras 对应，1 表示非猜测
    // focal_measured 区分本阶段测量与外部先验，使建图继续优化测量值而保留给定值（D45）。
    std::vector<uint8_t> focal_measured;
    bool hasCameras() const {
        return !cameras.empty() && camera_ids.size() == images.size();
    }
};

// ---------------- 匹配文件格式 ----------------
// 魔数 VKMT，u32 版本 4；v2 在匹配后结束，无相机记录，v3 有相机但没有逐相机 focal_measured 字节。

inline void writeMatches(const std::string& path, const MatchesDatabase& db) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    uint32_t version = 4, nimg = (uint32_t)db.images.size();
    f.write("VKMT", 4);
    f.write((const char*)&version, 4);
    f.write((const char*)&nimg, 4);
    for (const ImageEntry& im : db.images) {
        uint32_t len = (uint32_t)im.name.size();
        f.write((const char*)&len, 4);
        f.write(im.name.data(), len);
        f.write((const char*)&im.num_features, 4);
    }
    uint32_t npairs = (uint32_t)db.pairs.size();
    f.write((const char*)&npairs, 4);
    for (const TwoViewMatches& p : db.pairs) {
        uint32_t nm = (uint32_t)p.matches.size();
        f.write((const char*)&p.image1, 4);
        f.write((const char*)&p.image2, 4);
        f.write((const char*)&p.config, 4);
        f.write((const char*)&nm, 4);
        for (const FeatureMatch& m : p.matches) {
            f.write((const char*)&m.idx1, 4);
            f.write((const char*)&m.idx2, 4);
        }
    }
    uint32_t ncam = db.hasCameras() ? (uint32_t)db.cameras.size() : 0;
    f.write((const char*)&ncam, 4);
    for (uint32_t c = 0; c < ncam; c++) {
        const Camera& cam = db.cameras[c];
        int32_t model_id = camColmapId(cam.model);
        uint32_t np = (uint32_t)camColmapParams(cam.model);
        uint8_t prior = c < db.focal_prior.size() ? db.focal_prior[c] : 0;
        double params[16] = {0};
        packColmap(cam, params);
        f.write((const char*)&cam.id, 4);
        f.write((const char*)&cam.width, 4);
        f.write((const char*)&cam.height, 4);
        f.write((const char*)&model_id, 4);
        f.write((const char*)&prior, 1);
        f.write((const char*)&cam.pixel_scale, 8);
        f.write((const char*)&np, 4);
        f.write((const char*)params, np * 8);
    }
    if (ncam) {
        uint32_t nid = (uint32_t)db.camera_ids.size();
        f.write((const char*)&nid, 4);
        f.write((const char*)db.camera_ids.data(), (std::streamsize)nid * 4);
        for (uint32_t c = 0; c < ncam; c++) {
            uint8_t m = c < db.focal_measured.size() ? db.focal_measured[c] : 0;
            f.write((const char*)&m, 1);
        }
    }
}

inline MatchesDatabase readMatches(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "VKMT", 4) != 0) throw std::runtime_error("bad magic in " + path);
    uint32_t version, nimg;
    f.read((char*)&version, 4);
    f.read((char*)&nimg, 4);
    MatchesDatabase db;
    db.images.resize(nimg);
    for (uint32_t i = 0; i < nimg; i++) {
        uint32_t len;
        f.read((char*)&len, 4);
        db.images[i].name.resize(len);
        f.read(&db.images[i].name[0], len);
        f.read((char*)&db.images[i].num_features, 4);
    }
    uint32_t npairs;
    f.read((char*)&npairs, 4);
    db.pairs.resize(npairs);
    for (uint32_t i = 0; i < npairs; i++) {
        TwoViewMatches& p = db.pairs[i];
        uint32_t nm;
        f.read((char*)&p.image1, 4);
        f.read((char*)&p.image2, 4);
        f.read((char*)&p.config, 4);
        f.read((char*)&nm, 4);
        p.matches.resize(nm);
        for (uint32_t j = 0; j < nm; j++) {
            f.read((char*)&p.matches[j].idx1, 4);
            f.read((char*)&p.matches[j].idx2, 4);
        }
    }
    if (version >= 3) {
        uint32_t ncam = 0;
        f.read((char*)&ncam, 4);
        if (f.gcount() != 4) ncam = 0;
        for (uint32_t c = 0; c < ncam; c++) {
            Camera cam;
            int32_t model_id = 0;
            uint32_t np = 0;
            uint8_t prior = 0;
            double params[16] = {0};
            f.read((char*)&cam.id, 4);
            f.read((char*)&cam.width, 4);
            f.read((char*)&cam.height, 4);
            f.read((char*)&model_id, 4);
            f.read((char*)&prior, 1);
            f.read((char*)&cam.pixel_scale, 8);
            f.read((char*)&np, 4);
            if (np > 16) throw std::runtime_error("bad camera in " + path);
            f.read((char*)params, (std::streamsize)np * 8);
            cam.model = camFromColmapId(model_id);
            unpackColmap(cam, params);
            db.cameras.push_back(cam);
            db.focal_prior.push_back(prior);
        }
        if (ncam) {
            uint32_t nid = 0;
            f.read((char*)&nid, 4);
            if (nid == db.images.size()) {
                db.camera_ids.resize(nid);
                f.read((char*)db.camera_ids.data(), (std::streamsize)nid * 4);
            }
        }
        if (ncam && version >= 4) {
            std::vector<uint8_t> measured(ncam);
            f.read((char*)measured.data(), (std::streamsize)ncam);
            if (f.gcount() == (std::streamsize)ncam) db.focal_measured = std::move(measured);
        }
        if (!db.hasCameras()) {   // 附加段截断时视为无相机，不发布不完整参数
            db.cameras.clear();
            db.camera_ids.clear();
            db.focal_prior.clear();
            db.focal_measured.clear();
        }
    }
    return db;
}

// ---------------- 逐图像对读取 ----------------
// 仅索引匹配表与数量，交互查看某一图像对时才读取其内容，避免整份大型匹配数组进入内存。

// 此对数标记表示持续读取到文件尾，用于仍在追加的 live_matches.bin。
inline constexpr uint32_t kStreamingPairs = 0xFFFFFFFFu;

struct MatchesIndex {
    struct Entry {
        uint32_t image1 = 0, image2 = 0, count = 0;
        uint64_t offset = 0;      // 本图像对首个 idx1 的文件位置
    };
    std::vector<ImageEntry> images;
    std::vector<Entry> pairs;
};

inline bool indexMatches(const std::string& path, MatchesIndex& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    // 运行中读取的计数可能尚未完整，每图像对至少 16 字节，以文件大小限制可接受数量。
    f.seekg(0, std::ios::end);
    const uint64_t bytes = (uint64_t)f.tellg();
    f.seekg(0);
    char magic[4];
    f.read(magic, 4);
    uint32_t version = 0, nimg = 0;
    f.read((char*)&version, 4);
    f.read((char*)&nimg, 4);
    if (!f || std::memcmp(magic, "VKMT", 4) != 0) return false;
    MatchesIndex idx;
    idx.images.resize(nimg);
    for (uint32_t i = 0; i < nimg; i++) {
        uint32_t len = 0;
        f.read((char*)&len, 4);
        if (!f || len > (1u << 20)) return false;
        idx.images[i].name.resize(len);
        f.read(&idx.images[i].name[0], len);
        f.read((char*)&idx.images[i].num_features, 4);
    }
    uint32_t npairs = 0;
    f.read((char*)&npairs, 4);
    const bool streaming = npairs == kStreamingPairs;
    if (!f || (!streaming && (uint64_t)npairs * 16 > bytes)) return false;
    if (!streaming) idx.pairs.reserve(npairs);
    for (uint32_t i = 0; streaming || i < npairs; i++) {
        MatchesIndex::Entry e;
        int32_t config = 0;
        f.read((char*)&e.image1, 4);
        f.read((char*)&e.image2, 4);
        f.read((char*)&config, 4);
        f.read((char*)&e.count, 4);
        // 流式文件遇到尚未写完的尾部则停止；固定计数文件缺数据仍报截断错误。
        if (!f) return streaming ? (out = std::move(idx), true) : false;
        e.offset = (uint64_t)f.tellg();
        // seekg 越过文件尾不会立即失败，因此显式检查边界；流式文件允许未完成尾部，固定文件视为损坏。
        if (e.offset + (uint64_t)e.count * 8 > bytes)
            return streaming ? (out = std::move(idx), true) : false;
        idx.pairs.push_back(e);
        f.seekg((std::streamoff)e.count * 8, std::ios::cur);
        if (!f) return streaming ? (out = std::move(idx), true) : false;
    }
    out = std::move(idx);
    return true;
}

inline bool readPairMatches(const std::string& path,
                            const MatchesIndex::Entry& e,
                            std::vector<FeatureMatch>& out) {
    out.clear();
    if (!e.count) return true;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg((std::streamoff)e.offset);
    out.resize(e.count);
    for (uint32_t i = 0; i < e.count && f; i++) {
        f.read((char*)&out[i].idx1, 4);
        f.read((char*)&out[i].idx2, 4);
    }
    if (!f) { out.clear(); return false; }
    return true;
}

}  // 命名空间 sfm
