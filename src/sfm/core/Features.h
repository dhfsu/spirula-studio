// 关键点、描述子及独立二进制格式；布局不假定数据类型或维度，兼容 SIFT 的 128 维 uint8 与学习前端浮点描述子。
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sfm {

// 关键点使用原图像素坐标，左上原点、x 向右、y 向下；提取器将缩小工作图的结果映回原尺寸，使相机参数对应磁盘图像（D46）。
struct Keypoint {
    float x = 0, y = 0;         // 亚像素位置
    float scale = 0;            // 原图像素单位的 sigma
    float orientation = 0;      // 弧度，从 +x 逆时针
    // 响应值为 SIFT 的 DoG 极值或学习检测分数，v5 起保存，无尺度检测器依赖它排序。
    float response = 0;
};

enum class DType : uint32_t { U8 = 0, F32 = 1 };

inline uint32_t dtypeSize(DType t) { return t == DType::U8 ? 1u : 4u; }

// 单图特征，描述子按行排列，第 i 个占 data[i*dim .. i*dim+dim)，元素类型为 dtype。
struct FeatureSet {
    int width = 0, height = 0;          // 特征坐标所属的源图尺寸
    // extract 尺寸为实际检测分辨率，0 表示相同；坐标属于源图，但定位噪声与像素阈值属于提取分辨率（D47）。
    int extract_width = 0, extract_height = 0;
    // EXIF 像素焦距与相机身份，无有效值时为 0/空；提取阶段是最后读取原图的阶段，因此须在此记录。
    double exif_focal = 0;
    std::string exif_camera;
    // 建图所见像素的 Orientation，范围 1..8；apply 旋转后为 1，供规范对齐修正竖拍向上方向。
    uint8_t exif_orientation = 1;
    uint32_t dim = 128;
    DType dtype = DType::U8;
    std::vector<Keypoint> keypoints;
    std::vector<uint8_t> descriptors;   // count*dim*dtypeSize 字节
    // 可选 RGB 为 count*3 字节，三维点沿轨迹平均颜色；无图像来源或旧 v1 特征时可为空。
    std::vector<uint8_t> colors;

    uint32_t count() const { return (uint32_t)keypoints.size(); }
    bool hasColors() const { return colors.size() == (size_t)count() * 3; }
    // 是否存在需要保存的检测分数；SIFT 的 response 保持 0，学习检测器填充分数。
    bool hasScores() const {
        for (const Keypoint& k : keypoints)
            if (k.response != 0) return true;
        return false;
    }
    // SIFT 按更具可重复性的尺度排序（D16），无尺度检测器按检测分数排序，不混合两者。
    float rank(uint32_t i) const {
        const Keypoint& k = keypoints[i];
        return k.scale > 0 ? k.scale : k.response;
    }
    // 每提取像素对应的源像素数，至少为 1；两轴仅有尺寸取整差异，取平均得到各向同性尺度。
    double pixelScale() const {
        if (extract_width <= 0 || extract_height <= 0 || width <= 0 || height <= 0) return 1.0;
        return 0.5 * ((double)width / extract_width + (double)height / extract_height);
    }
};

// 将关键点从工作尺寸映回原图并记录尺寸，使用双线性重采样的精确逆：源像素中心为 (x+0.5)*sx。
// sigma 按两轴比例的几何平均缩放，处理宽高取整差异，保证 cameras.bin 对应原图。
inline void scaleKeypoints(FeatureSet& fs, int w, int h) {
    if (w <= 0 || h <= 0 || fs.width <= 0 || fs.height <= 0) return;
    fs.extract_width = fs.width;    // 关键点实际测量所在的分辨率
    fs.extract_height = fs.height;
    if (w == fs.width && h == fs.height) return;
    const float sx = (float)w / fs.width, sy = (float)h / fs.height;
    const float ss = std::sqrt(sx * sy);
    for (Keypoint& k : fs.keypoints) {
        k.x = (k.x + 0.5f) * sx - 0.5f;
        k.y = (k.y + 0.5f) * sy - 0.5f;
        k.scale *= ss;
    }
    fs.width = w;
    fs.height = h;
}

// ---------------- 特征文件格式 ----------------
// 各版本附加段见 src/sfm/README.md；旧版本缺失字段按默认值读取，允许复用旧缓存。

// 先写相邻临时文件再重命名替换，保证中断后文件完整或不存在，无需重读描述子即可安全复用（D76）。
inline void writeFeatures(const std::string& path, const FeatureSet& fs) {
    const std::string tmp = path + ".part";
    {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    uint32_t version = 6, count = fs.count(), dtype = (uint32_t)fs.dtype;
    f.write("VKFT", 4);
    f.write((const char*)&version, 4);
    f.write((const char*)&fs.width, 4);
    f.write((const char*)&fs.height, 4);
    f.write((const char*)&count, 4);
    f.write((const char*)&fs.dim, 4);
    f.write((const char*)&dtype, 4);
    for (const Keypoint& k : fs.keypoints) {
        float v[4] = {k.x, k.y, k.scale, k.orientation};
        f.write((const char*)v, sizeof v);
    }
    f.write((const char*)fs.descriptors.data(), (std::streamsize)fs.descriptors.size());
    uint8_t has_colors = fs.hasColors() ? 1 : 0;
    f.write((const char*)&has_colors, 1);
    if (has_colors) f.write((const char*)fs.colors.data(), (std::streamsize)fs.colors.size());
    uint32_t cam_len = (uint32_t)fs.exif_camera.size();
    f.write((const char*)&fs.exif_focal, 8);
    f.write((const char*)&cam_len, 4);
    f.write(fs.exif_camera.data(), (std::streamsize)cam_len);
    f.write((const char*)&fs.extract_width, 4);
    f.write((const char*)&fs.extract_height, 4);
    uint8_t has_scores = fs.hasScores() ? 1 : 0;
    f.write((const char*)&has_scores, 1);
    if (has_scores)
        for (const Keypoint& k : fs.keypoints) f.write((const char*)&k.response, 4);
    f.write((const char*)&fs.exif_orientation, 1);
    if (!f) throw std::runtime_error("cannot write " + path);
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("cannot write " + path);
    }
}

// 通过文件头与文件大小检查完整性和关键点数，避免为发现截断而读取整个描述子块。
inline bool peekFeatures(const std::string& path, uint32_t& count) {
    std::error_code ec;
    const uint64_t bytes = (uint64_t)std::filesystem::file_size(path, ec);
    if (ec) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    uint32_t version = 0, n = 0, dim = 0, dtype = 0;
    int w = 0, h = 0;
    f.read(magic, 4);
    f.read((char*)&version, 4);
    f.read((char*)&w, 4);
    f.read((char*)&h, 4);
    f.read((char*)&n, 4);
    f.read((char*)&dim, 4);
    f.read((char*)&dtype, 4);
    if (!f || std::memcmp(magic, "VKFT", 4) != 0 || dtype > 1) return false;
    const uint64_t need = 28 + (uint64_t)n * 16 + (uint64_t)n * dim * dtypeSize((DType)dtype);
    if (bytes < need) return false;
    count = n;
    return true;
}

// with_descriptors=false 时直接跳过描述子；建图只需关键点与颜色，每千张图可避免约 1 GB 无用读取。
inline FeatureSet readFeatures(const std::string& path, bool with_descriptors = true) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "VKFT", 4) != 0) throw std::runtime_error("bad magic in " + path);
    uint32_t version, count, dim, dtype;
    f.read((char*)&version, 4);
    FeatureSet fs;
    f.read((char*)&fs.width, 4);
    f.read((char*)&fs.height, 4);
    f.read((char*)&count, 4);
    f.read((char*)&dim, 4);
    f.read((char*)&dtype, 4);
    fs.dim = dim;
    fs.dtype = (DType)dtype;
    fs.keypoints.resize(count);
    {   // 一次批量读取关键点，避免千万点级数据的逐点流调用开销。
        std::vector<float> raw((size_t)count * 4);
        f.read((char*)raw.data(), (std::streamsize)(raw.size() * sizeof(float)));
        for (uint32_t i = 0; i < count; i++) {
            const float* v = &raw[(size_t)i * 4];
            fs.keypoints[i] = {v[0], v[1], v[2], v[3], 0.0f};
        }
    }
    const std::streamsize desc_bytes = (std::streamsize)((size_t)count * dim *
                                                         dtypeSize(fs.dtype));
    if (with_descriptors) {
        fs.descriptors.resize((size_t)desc_bytes);
        f.read((char*)fs.descriptors.data(), desc_bytes);
    } else {
        f.seekg(desc_bytes, std::ios::cur);
    }
    if (version >= 2) {
        uint8_t has_colors = 0;
        f.read((char*)&has_colors, 1);
        if (has_colors) {
            fs.colors.resize((size_t)count * 3);
            f.read((char*)fs.colors.data(), (std::streamsize)fs.colors.size());
        }
    }
    if (version >= 3) {
        // 附加段先读到局部变量，完整后才发布；截断 EXIF 应表现为缺失，而非可信的错误焦距。
        double focal = 0;
        uint32_t cam_len = 0;
        f.read((char*)&focal, 8);
        f.read((char*)&cam_len, 4);
        if (f.gcount() == 4) {
            fs.exif_focal = focal;
            if (cam_len > 0 && cam_len < (1u << 16)) {
                std::string cam(cam_len, '\0');
                f.read(&cam[0], (std::streamsize)cam_len);
                if (f.gcount() == (std::streamsize)cam_len) fs.exif_camera = std::move(cam);
            }
        }
    }
    if (version >= 4) {
        int ew = 0, eh = 0;
        f.read((char*)&ew, 4);
        f.read((char*)&eh, 4);
        if (f.gcount() == 4 && ew > 0 && eh > 0) {
            fs.extract_width = ew;
            fs.extract_height = eh;
        }
    }
    if (version >= 5) {
        uint8_t has_scores = 0;
        f.read((char*)&has_scores, 1);
        if (f.gcount() == 1 && has_scores) {
            std::vector<float> raw(count);
            f.read((char*)raw.data(), (std::streamsize)(raw.size() * sizeof(float)));
            for (uint32_t i = 0; i < count; i++) fs.keypoints[i].response = raw[i];
        }
    }
    if (version >= 6) {
        uint8_t o = 0;
        f.read((char*)&o, 1);
        if (f.gcount() == 1 && o >= 1 && o <= 8) fs.exif_orientation = o;
    }
    return fs;
}

}  // 命名空间 sfm
