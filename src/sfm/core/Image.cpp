// Image.h 的解码实现；stb_image 在 src/external/stb_image_impl.cpp 中统一实例化，此处不能再次定义 STB_IMAGE_IMPLEMENTATION。
#include "sfm/core/Image.h"

#include "core/ColorSpace.h"
#include "core/ExrImage.h"
#include "core/ImageOrient.h"

#include <algorithm>

#include "external/stb_image.h"

namespace sfm {

// 用区域均值将交错 RGB 缩到目标尺寸，颜色仅用于点云采样，无需昂贵滤波；避免与灰度同时保留全分辨率彩色缓冲。
static std::vector<uint8_t> downscaleRgb(const unsigned char* src, int w, int h, int dw, int dh) {
    std::vector<uint8_t> out((size_t)dw * dh * 3);
    for (int y = 0; y < dh; y++) {
        int sy0 = (int)((int64_t)y * h / dh), sy1 = std::max(sy0 + 1, (int)((int64_t)(y + 1) * h / dh));
        for (int x = 0; x < dw; x++) {
            int sx0 = (int)((int64_t)x * w / dw), sx1 = std::max(sx0 + 1, (int)((int64_t)(x + 1) * w / dw));
            uint32_t acc[3] = {0, 0, 0}, n = 0;
            for (int sy = sy0; sy < sy1; sy++)
                for (int sx = sx0; sx < sx1; sx++) {
                    const unsigned char* p = src + 3 * ((size_t)sy * w + sx);
                    acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2]; n++;
                }
            uint8_t* o = &out[3 * ((size_t)y * dw + x)];
            o[0] = (uint8_t)(acc[0] / n); o[1] = (uint8_t)(acc[1] / n); o[2] = (uint8_t)(acc[2] / n);
        }
    }
    return out;
}

// Rec.601 亮度权重，与 COLMAP 的 FreeImage 灰度转换一致。
static constexpr float kLumaR = 0.299f / 255.0f;
static constexpr float kLumaG = 0.587f / 255.0f;
static constexpr float kLumaB = 0.114f / 255.0f;

static inline float lumaAt(const unsigned char* rgb, int w, int x, int y) {
    const unsigned char* p = rgb + 3 * ((size_t)y * w + x);
    return kLumaR * p[0] + kLumaG * p[1] + kLumaB * p[2];
}

// 直接从 RGB 四邻点计算亮度并双线性重采样为目标灰度，与先转全尺寸灰度再缩放的运算顺序和舍入一致。
// 省去每源像素 4 字节的全尺寸浮点缓冲，改善大图解码并发预算。
static void resizeGrayFromRgb(const unsigned char* rgb, int w, int h, int dw, int dh,
                              std::vector<float>& out) {
    out.resize((size_t)dw * dh);
    const float sx = w / (float)dw;
    const float sy = h / (float)dh;
    for (int y = 0; y < dh; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float wy = fy - y0;
        int y0c = std::max(0, std::min(h - 1, y0));
        int y1c = std::max(0, std::min(h - 1, y0 + 1));
        for (int x = 0; x < dw; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            float wx = fx - x0;
            int x0c = std::max(0, std::min(w - 1, x0));
            int x1c = std::max(0, std::min(w - 1, x0 + 1));
            float a = lumaAt(rgb, w, x0c, y0c), b = lumaAt(rgb, w, x1c, y0c);
            float c = lumaAt(rgb, w, x0c, y1c), d = lumaAt(rgb, w, x1c, y1c);
            float top = a + (b - a) * wx;
            float bot = c + (d - c) * wx;
            out[(size_t)y * dw + x] = top + (bot - top) * wy;
        }
    }
}

namespace {

// 在缩小后的灰度、颜色和掩码上应用 EXIF 旋转；四分之一圈旋转与重采样可交换，避免全尺寸 RGB 旋转使峰值内存翻倍。
void applyExifOrientation(GrayImage& img) {
    const ExifTransform xf = exifTransform(img.exif.orientation);
    if (xf.turns_cw != 0) {
        std::vector<float> gray((size_t)img.width * img.height);
        spirula::orient_pixels(img.data.data(), img.width, img.height, 1,
                               xf.turns_cw, false, gray.data());
        img.data.swap(gray);
        if (img.hasColor()) {
            std::vector<uint8_t> rgb(img.rgb.size());
            spirula::orient_pixels(img.rgb.data(), img.width, img.height, 3,
                                   xf.turns_cw, false, rgb.data());
            img.rgb.swap(rgb);
        }
        if (!img.mask.empty()) {
            std::vector<uint8_t> bits(img.mask.bits.size());
            spirula::orient_pixels(img.mask.bits.data(), img.mask.width,
                                   img.mask.height, 1, xf.turns_cw, false,
                                   bits.data());
            img.mask.bits.swap(bits);
            std::swap(img.mask.width, img.mask.height);
        }
        std::swap(img.width, img.height);
        std::swap(img.orig_width, img.orig_height);
        std::swap(img.exif.pixel_width, img.exif.pixel_height);
    }
    img.exif_mirror_dropped = xf.mirror;
    img.exif.orientation = 1;
}

}  // 匿名命名空间

GrayImage loadGrayImage(const std::string& path, int max_image_size, bool want_color,
                        const std::string& mask_path,
                        const std::string& gamut, std::optional<bool> is_linear,
                        bool flip_mask, bool apply_exif_orientation) {
    int w = 0, h = 0, chan = 0;
    // 强制解码三通道并统一计算亮度，消除解码器差异；EXR 使用当前线程，外层池已占用全部核心。
    std::vector<uint8_t> exr_rgb;
    unsigned char* rgb = nullptr;
    if (exr::is_exr(path)) {
        exr::Info info;
        exr::Options opt;
        opt.threads = 1;
        const std::string err =
            exr::decode_srgb8(path, opt, info, exr_rgb, gamut, is_linear);
        if (!err.empty())
            throw std::runtime_error("cannot decode image " + path + ": " + err);
        w = info.width;
        h = info.height;
        rgb = exr_rgb.data();
    } else {
        rgb = stbi_load(path.c_str(), &w, &h, &chan, 3);
        if (!rgb)
            throw std::runtime_error("cannot decode image " + path + ": " + stbi_failure_reason());
        colorspace::to_srgb_inplace(rgb, (size_t)w * h, gamut,
                                    is_linear.value_or(false));
    }

    GrayImage img;
    img.orig_width = w;
    img.orig_height = h;

    // 将最长边限制到 max_image_size，COLMAP 默认值为 3200。
    int dw = w, dh = h;
    int longEdge = std::max(w, h);
    if (max_image_size > 0 && longEdge > max_image_size) {
        double scale = (double)max_image_size / longEdge;
        dw = std::max(1, (int)std::lround(w * scale));
        dh = std::max(1, (int)std::lround(h * scale));
    }
    img.width = dw;
    img.height = dh;
    // 颜色缓冲与缩小后的灰度尺寸一致，使关键点可直接索引。
    if (want_color) {
        img.rgb = (dw == w && dh == h) ? std::vector<uint8_t>(rgb, rgb + (size_t)w * h * 3)
                                       : downscaleRgb(rgb, w, h, dw, dh);
    }
    if (dw == w && dh == h) {
        img.data.resize((size_t)w * h);
        for (size_t i = 0; i < img.data.size(); i++)
            img.data[i] = kLumaR * rgb[3 * i] + kLumaG * rgb[3 * i + 1] + kLumaB * rgb[3 * i + 2];
    } else {
        resizeGrayFromRgb(rgb, w, h, dw, dh, img.data);
    }
    if (exr_rgb.empty()) stbi_image_free(rgb);
    // 保留掩码自身分辨率并按 UV 采样，避免额外缩放损失边界细节（D39）。
    if (!mask_path.empty()) {
        img.mask = loadMask(mask_path);
        if (flip_mask) img.mask.invert();
    }
    img.exif = readExif(path);  // 仅读取文件头，见 Exif.h
    if (apply_exif_orientation) applyExifOrientation(img);
    return img;
}

Mask loadMask(const std::string& path) {
    int w = 0, h = 0, chan = 0;
    // 常见二值、灰度、RGB 掩码统一转为单通道；仅 alpha 携带形状的掩码须另行处理，灰度转换会丢失 alpha。
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &chan, 1);
    if (!px || w <= 0 || h <= 0) {
        if (px) stbi_image_free(px);
        return Mask();
    }
    Mask m;
    m.width = w;
    m.height = h;
    m.bits.resize((size_t)w * h);
    for (size_t i = 0; i < m.bits.size(); i++) m.bits[i] = px[i] != 0 ? 1 : 0;
    stbi_image_free(px);

    // RGB 全白的 RGBA 掩码可能仅在 alpha 中保存主体形状；仅当灰度读取全部饱和时重读 alpha，普通不透明掩码保持原样。
    if (chan == 4) {
        bool all_keep = true;
        for (uint8_t b : m.bits)
            if (!b) { all_keep = false; break; }
        if (all_keep) {
            int w2 = 0, h2 = 0, c2 = 0;
            unsigned char* rgba = stbi_load(path.c_str(), &w2, &h2, &c2, 4);
            if (rgba) {
                if (w2 == w && h2 == h)
                    for (size_t i = 0; i < m.bits.size(); i++)
                        m.bits[i] = rgba[4 * i + 3] != 0 ? 1 : 0;
                stbi_image_free(rgba);
            }
        }
    }
    return m;
}

bool imageSize(const std::string& path, int& width, int& height) {
    if (exr::is_exr(path)) {
        exr::Info info;
        if (!exr::probe(path, info).empty()) return false;
        width = info.width;
        height = info.height;
        return true;
    }
    int comp = 0;
    return stbi_info(path.c_str(), &width, &height, &comp) != 0;
}

}  // 命名空间 sfm
