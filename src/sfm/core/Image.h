// 主机图像容器与解码，使用 stb_image 读取常见格式，以 Rec.601 权重转为 [0,1] 单通道 float，并可限制最长边。
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sfm/core/Exif.h"
#include "sfm/core/Mask.h"

namespace sfm {

// 灰度按行存储，范围 [0,1]；可选 RGB 为同工作尺寸的每像素三字节交错缓冲，仅用于关键点颜色采样，不供 SIFT 使用。
struct GrayImage {
    int width = 0;
    int height = 0;
    // 源文件尺寸可能不同于工作尺寸，特征写出前映回此坐标，使内参描述磁盘图像（D46）。
    int orig_width = 0;
    int orig_height = 0;
    std::vector<float> data;  // width*height
    std::vector<uint8_t> rgb; // 空值或 width*height*3 字节的交错 RGB
    // 可选掩码保留自身分辨率，按 UV 采样，无需匹配原图或工作图尺寸。
    Mask mask;
    // 在解码池中解析 EXIF，吸收读取开销；此阶段之后不再接触原图文件。
    ExifData exif;
    // 已应用旋转，但标签要求的镜像被忽略。
    bool exif_mirror_dropped = false;

    float at(int x, int y) const { return data[(size_t)y * width + x]; }
    size_t pixels() const { return (size_t)width * height; }
    bool hasColor() const { return rgb.size() == pixels() * 3; }
};

// 在灰度同一坐标系双线性采样颜色，越界钳位；无颜色缓冲时保留灰度。
inline void sampleColor(const GrayImage& img, float x, float y, uint8_t out[3]) {
    if (!img.hasColor()) { out[0] = out[1] = out[2] = 128; return; }
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    float wx = x - x0, wy = y - y0;
    int x0c = std::max(0, std::min(img.width - 1, x0));
    int x1c = std::max(0, std::min(img.width - 1, x0 + 1));
    int y0c = std::max(0, std::min(img.height - 1, y0));
    int y1c = std::max(0, std::min(img.height - 1, y0 + 1));
    const uint8_t* p = img.rgb.data();
    for (int c = 0; c < 3; c++) {
        float a = p[3 * ((size_t)y0c * img.width + x0c) + c];
        float b = p[3 * ((size_t)y0c * img.width + x1c) + c];
        float cc = p[3 * ((size_t)y1c * img.width + x0c) + c];
        float d = p[3 * ((size_t)y1c * img.width + x1c) + c];
        float top = a + (b - a) * wx;
        float bot = cc + (d - cc) * wx;
        out[c] = (uint8_t)std::lround(std::max(0.0f, std::min(255.0f, top + (bot - top) * wy)));
    }
}

// 掩码解码失败返回空，不因此丢弃图像；gamut/is_linear 描述原文件，解码后统一转为 sRGB。
GrayImage loadGrayImage(const std::string& path, int max_image_size = 3200,
                        bool want_color = false, const std::string& mask_path = "",
                        const std::string& gamut = "",
                        std::optional<bool> is_linear = std::nullopt,
                        bool flip_mask = false,
                        // 旋转像素和掩码并将 orientation 设为 1；忽略镜像部分。
                        bool apply_exif_orientation = false);

// 仅从图像头读取宽高，不完整解码；格式不可解码时返回 false。
bool imageSize(const std::string& path, int& width, int& height);

// 双线性重采样到精确目标尺寸，供加载器尺寸限制与合成测试使用。
inline GrayImage resizeGray(const GrayImage& src, int dw, int dh) {
    GrayImage out;
    out.width = dw;
    out.height = dh;
    out.data.resize((size_t)dw * dh);
    // 将目标像素中心反向映射到源图，边界钳位。
    const float sx = src.width / (float)dw;
    const float sy = src.height / (float)dh;
    for (int y = 0; y < dh; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float wy = fy - y0;
        int y0c = std::max(0, std::min(src.height - 1, y0));
        int y1c = std::max(0, std::min(src.height - 1, y0 + 1));
        for (int x = 0; x < dw; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            float wx = fx - x0;
            int x0c = std::max(0, std::min(src.width - 1, x0));
            int x1c = std::max(0, std::min(src.width - 1, x0 + 1));
            float a = src.at(x0c, y0c), b = src.at(x1c, y0c);
            float c = src.at(x0c, y1c), d = src.at(x1c, y1c);
            float top = a + (b - a) * wx;
            float bot = c + (d - c) * wx;
            out.data[(size_t)y * dw + x] = top + (bot - top) * wy;
        }
    }
    return out;
}

}  // 命名空间 sfm
