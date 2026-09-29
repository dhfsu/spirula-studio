// 纯主机掩码测试，覆盖 UV 采样、解码及文件查找。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "sfm/core/Features.h"
#include "sfm/core/Mask.h"
#include "sfm/tests/TestMain.h"

namespace fs = std::filesystem;

// 按文件身份而非字符串比较路径，兼容 Windows 混合分隔符及扩展名大小写差异。
bool samePath(const std::string& a, const std::string& b) {
    if (a == b) return true;
    std::error_code ec;
    return !a.empty() && !b.empty() && fs::equivalent(a, b, ec);
}
using namespace sfm;

// ---------------- 掩码测试 ----------------
// 覆盖分辨率独立采样、并行特征数组同步压缩及正确文件配对，数据写入临时目录。

// 用无需额外库即可写出的二进制 PGM 代替 PNG；stb_image 按内容识别格式。
static void writePgm(const fs::path& p, int w, int h, const std::vector<uint8_t>& px) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << "P5\n" << w << " " << h << "\n255\n";
    f.write((const char*)px.data(), (std::streamsize)px.size());
}

// 左侧 keep_frac 宽度为 255，其余忽略。
static std::vector<uint8_t> leftHalfMask(int w, int h, double keep_frac) {
    std::vector<uint8_t> px((size_t)w * h, 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            px[(size_t)y * w + x] = (x < keep_frac * w) ? 255 : 0;
    return px;
}

int cmdMaskSelftest(int, char**) {
    int fails = 0;
    const fs::path tmp = fs::temp_directory_path() / "spirula_sfm_mask_selftest";
    std::error_code ec;
    fs::remove_all(tmp, ec);

    // ---------------- UV 采样与分辨率无关 ----------------
    // 同几何掩码的四种分辨率应给出一致关键点分类，不能按原图像素直接索引小掩码。
    {
        const int IW = 400, IH = 300;
        const int res[][2] = {{400, 300}, {40, 30}, {1200, 900}, {97, 73}};
        // 测试边界两侧与极端角落关键点，内部角点不能因索引钳位而被误删。
        std::vector<Keypoint> kps;
        for (int i = 0; i < 40; i++)
            kps.push_back({(float)(i * 10 + 0.5f), (float)(IH / 2), 2, 0, 0});
        kps.push_back({0.0f, 0.0f, 2, 0, 0});
        kps.push_back({(float)(IW - 0.01f), (float)(IH - 0.01f), 2, 0, 0});

        std::vector<uint32_t> kept_per_res;
        for (const auto& r : res) {
            Mask m;
            m.width = r[0];
            m.height = r[1];
            m.bits.resize((size_t)r[0] * r[1]);
            std::vector<uint8_t> px = leftHalfMask(r[0], r[1], 0.5);
            for (size_t i = 0; i < m.bits.size(); i++) m.bits[i] = px[i] != 0;

            FeatureSet fs;
            fs.width = IW;
            fs.height = IH;
            fs.dim = 4;
            fs.keypoints = kps;
            fs.descriptors.resize(kps.size() * 4);
            for (size_t i = 0; i < kps.size(); i++)
                for (int d = 0; d < 4; d++) fs.descriptors[i * 4 + d] = (uint8_t)(i + d);
            fs.colors.assign(kps.size() * 3, 0);
            for (size_t i = 0; i < kps.size(); i++) fs.colors[i * 3] = (uint8_t)i;

            applyMask(fs, m);
            kept_per_res.push_back(fs.count());
            // 保留点位于左半区，描述子与颜色仍须对应原关键点；边界容许一个掩码单元的量化误差。
            const float tol = (float)IW / r[0];
            for (uint32_t i = 0; i < fs.count(); i++) {
                if (fs.keypoints[i].x >= IW * 0.5f + tol) {
                    printf("  FAIL: mask %dx%d kept a right-half keypoint at x=%.1f\n", r[0], r[1],
                           fs.keypoints[i].x);
                    fails++;
                    break;
                }
                uint8_t tag = fs.colors[i * 3];
                if (fs.descriptors[i * 4] != tag || fs.descriptors[i * 4 + 3] != (uint8_t)(tag + 3)) {
                    printf("  FAIL: mask %dx%d desaligned descriptors/colors at %u\n", r[0], r[1], i);
                    fails++;
                    break;
                }
            }
        }
        printf("mask: uv sampling at %dx%d over masks 400x300/40x30/1200x900/97x73 kept "
               "%u/%u/%u/%u of %zu\n", IW, IH, kept_per_res[0], kept_per_res[1], kept_per_res[2],
               kept_per_res[3], kps.size());
        // 40×30 掩码将边界量化到十个图像像素，允许一个关键点差异，其余精确一致。
        for (size_t i = 1; i < kept_per_res.size(); i++) {
            uint32_t d = kept_per_res[i] > kept_per_res[0] ? kept_per_res[i] - kept_per_res[0]
                                                           : kept_per_res[0] - kept_per_res[i];
            if (d > 1) {
                printf("  FAIL: resolution %d changed the kept count by %u\n", res[i][0], d);
                fails++;
            }
        }
        // 图内角点不能因采样不可达而被删除。
        Mask allkeep;
        allkeep.width = allkeep.height = 3;
        allkeep.bits.assign(9, 1);
        FeatureSet fs2;
        fs2.width = IW; fs2.height = IH; fs2.dim = 1;
        fs2.keypoints = kps;
        fs2.descriptors.assign(kps.size(), 0);
        if (applyMask(fs2, allkeep) != 0) { printf("  FAIL: all-keep mask dropped keypoints\n"); fails++; }
    }

    // ---------------- 解码、二值化与失败 ----------------
    {
        std::vector<uint8_t> px = leftHalfMask(64, 48, 0.25);
        // 保留区采用灰度渐变，验证非零即保留，而非按 128 阈值。
        for (int y = 0; y < 48; y++)
            for (int x = 0; x < 16; x++) px[(size_t)y * 64 + x] = (uint8_t)(1 + (x % 3));
        writePgm(tmp / "decode" / "m.pgm", 64, 48, px);
        Mask m = loadMask((tmp / "decode" / "m.pgm").string());
        double keep = m.keepFraction();
        printf("mask: decoded %dx%d, keep fraction %.3f (expect 0.250)\n", m.width, m.height, keep);
        if (m.width != 64 || m.height != 48 || std::fabs(keep - 0.25) > 1e-6) {
            printf("  FAIL: mask decode / binarization\n");
            fails++;
        }
        for (uint8_t b : m.bits)
            if (b > 1) { printf("  FAIL: mask not binarized to 0/1\n"); fails++; break; }
        // 非图像文件返回空掩码，不抛异常、不丢整次运行。
        {
            fs::create_directories(tmp / "decode");
            std::ofstream(tmp / "decode" / "junk.png", std::ios::binary) << "not an image";
        }
        Mask bad = loadMask((tmp / "decode" / "junk.png").string());
        if (!bad.empty()) { printf("  FAIL: undecodable mask did not come back empty\n"); fails++; }
        FeatureSet fs;
        fs.width = 100; fs.height = 100; fs.dim = 1;
        fs.keypoints.assign(5, Keypoint{50, 50, 2, 0, 0});
        fs.descriptors.assign(5, 0);
        if (applyMask(fs, bad) != 0 || fs.count() != 5) {
            printf("  FAIL: empty mask must keep every keypoint\n");
            fails++;
        }
    }

    // ---------------- 常见掩码命名查找 ----------------
    {
        const fs::path md = tmp / "masks";
        std::vector<uint8_t> px = leftHalfMask(8, 8, 0.5);
        struct Case { const char* image; const char* mask; };
        const Case cases[] = {
            {"cam0/a.jpg", "cam0/a.jpg.png"},      // COLMAP、Spirula Studio 与 SAM 的命名形式
            {"cam0/b.jpg", "cam0/b.png"},          // COLMAP 备选及 Nerfstudio 命名形式
            {"cam0/c.jpg", "cam0/c_mask.png"},     // 带 mask 后缀
            {"cam0/d.jpg", "cam0/d.jpg"},          // 文件名与扩展名相同
            {"cam1/e.jpg", "cam1/e.jpg.JPEG"},     // 大写扩展名且容器不同
            {"cam1/f.jpg", "f.png"},               // 平铺掩码目录与嵌套图像目录
            {"cam1/g.png", "cam1/g.png.png"},      // PNG 图像与 PNG 掩码
        };
        for (const Case& c : cases) writePgm(md / c.mask, 8, 8, px);
        MaskIndex idx(md.string());
        for (const Case& c : cases) {
            std::string got = idx.find(c.image);
            std::string want = (md / c.mask).string();
            if (!samePath(got, want)) {
                printf("  FAIL: %s -> \"%s\", expected \"%s\"\n", c.image, got.c_str(), want.c_str());
                fails++;
            }
        }
        printf("mask: resolved %zu naming conventions\n", sizeof(cases) / sizeof(cases[0]));

        // 没有掩码时应返回空，不能误配其他图像掩码。
        if (!idx.find("cam0/nosuch.jpg").empty()) {
            printf("  FAIL: invented a mask for an unmasked image\n");
            fails++;
        }
        // 不同子目录的同名掩码不能匹配歧义平铺名称，但各嵌套图像仍匹配自身文件。
        writePgm(md / "x" / "dup.png", 8, 8, px);
        writePgm(md / "y" / "dup.png", 8, 8, px);
        MaskIndex idx2(md.string());
        if (!idx2.find("dup.jpg").empty()) {
            printf("  FAIL: ambiguous basename resolved anyway\n");
            fails++;
        }
        if (!samePath(idx2.find("x/dup.jpg"), (md / "x" / "dup.png").string())) {
            printf("  FAIL: unambiguous nested name did not resolve\n");
            fails++;
        }
        // 掩码目录缺失时不生效，不作为致命错误。
        if (MaskIndex((tmp / "nope").string()).valid()) {
            printf("  FAIL: missing mask directory reported valid\n");
            fails++;
        }
    }

    fs::remove_all(tmp, ec);
    printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

int main() { return sfmTestMain(0, nullptr, cmdMaskSelftest); }
