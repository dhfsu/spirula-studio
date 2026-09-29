#pragma once

// OpenEXR 读取器，支持扫描线与瓦片部件、HALF/FLOAT/UINT 通道及 NONE/RLE/ZIP/ZIPS/PIZ/PXR24/B44/B44A 压缩。
// 不支持的格式通过返回字符串说明；无需 OpenEXR/Imath，inflate 使用随仓库提供的 miniz。

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace exr {

// gamut 由文件色度信息解析为 ColorSpace.h 的名称；声明的基色无匹配时，gamut_known 为 false 并回退到 Rec.709。
struct Info {
    int width = 0;                  // 显示窗口尺寸，而非数据窗口
    int height = 0;
    int channels = 0;               // 1（亮度）、3（RGB）或 4（RGBA）
    std::string compression;
    std::string gamut;              // 空字符串表示 Rec.709
    bool is_linear = true;          // EXR 规范要求场景线性像素
    bool chromaticities = false;    // 文件是否声明了基色
    bool gamut_known = true;
    bool tiled = false;
    int parts = 1;
};

// 检查前四字节的 EXR 魔数；非 EXR 文件交给 stb_image。
bool is_exr(const std::string& path);

struct Options {
    int channels = 3;      // 请求 1、3 或 4 通道，按相应顺序交错排列
    int threads = 0;       // 0 使用全部核心；1 在调用线程中解码
};

// 仅读取文件头；成功返回空字符串，失败返回问题说明。
std::string probe(const std::string& path, Info& info);

// 读取文件声明的色彩空间，用于补齐未设置选项；不是可读 EXR 时返回 false。
bool declared_color_space(const std::string& path, Info& info);

// 交错排列的 float，每像素 opt.channels 个值，保留文件的场景线性色彩空间；大于 1 的值合法且不截断。
std::string decode(const std::string& path, const Options& opt, Info& info,
                   std::vector<float>& out);

// 输出交错排列的 8 位 sRGB；gamut 为空或 is_linear 未设置时分别沿用文件信息，可用 Rec.709 与 false 覆盖错误文件头。
std::string decode_srgb8(const std::string& path, const Options& opt, Info& info,
                         std::vector<uint8_t>& out, const std::string& gamut = "",
                         std::optional<bool> is_linear = std::nullopt);

}  // 命名空间 exr
