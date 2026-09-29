// 需要设备的 SIFT、暴力匹配、并行解码与相机模型测试，输出 PASS/FAIL。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "sfm/core/CameraSetup.h"
#include "sfm/core/Exif.h"
#include "sfm/core/Features.h"
#include "sfm/core/Image.h"
#include "sfm/core/ImageLoader.h"
#include "sfm/core/Matches.h"
#include "sfm/core/Model.h"
#include "sfm/feature/Matcher.h"
#include "sfm/feature/PairSelection.h"
#include "sfm/feature/Pairing.h"
#include "sfm/feature/Sift.h"
#include "sfm/feature/Verification.h"
#include "sfm/geometry/TwoView.h"
#include "sfm/tests/TestMain.h"

namespace fs = std::filesystem;
using namespace sfm;

static GrayImage syntheticScene(int w, int h) {
    GrayImage img;
    img.width = w;
    img.height = h;
    img.data.assign((size_t)w * h, 0.9f);
    struct Blob { float cx, cy, r; };
    std::vector<Blob> blobs = {
        {64, 64, 8}, {180, 70, 14}, {90, 170, 6}, {200, 190, 20}, {128, 128, 11}};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float v = 0.9f;
            for (const Blob& b : blobs) {
                float d2 = (x - b.cx) * (x - b.cx) + (y - b.cy) * (y - b.cy);
                v -= 0.8f * std::exp(-d2 / (2 * b.r * b.r));
            }
            // 加入少量高频纹理，使描述子包含结构
            if (x > w * 3 / 4) v += 0.05f * std::sin(x * 0.7f) * std::cos(y * 0.6f);
            img.data[(size_t)y * w + x] = std::min(1.0f, std::max(0.0f, v));
        }
    return img;
}

int cmdSelftest(int argc, char** argv) {
    SiftOptions opt;
    opt.max_num_features = 0;  // 保留全部特征以确定完整结果集
    opt.verbose = false;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) opt.device = std::stoi(argv[++i]);
    }

    GrayImage img = syntheticScene(256, 256);
    SiftExtractor ext(opt);
    FeatureSet a = ext.extract(img);

    int fails = 0;

    // 1）检测数量合理性
    printf("selftest: extracted %u features from 256x256 synthetic scene\n", a.count());
    if (a.count() < 10) { printf("  FAIL: too few features\n"); fails++; }

    // 2）每个斑点中心附近应有关键点
    struct C { float x, y; };
    std::vector<C> centers = {{64, 64}, {180, 70}, {200, 190}, {128, 128}};
    int hit = 0;
    for (const C& c : centers) {
        float best = 1e9f;
        for (const Keypoint& k : a.keypoints)
            best = std::min(best, std::hypot(k.x - c.x, k.y - c.y));
        if (best < 4.0f) hit++;
    }
    printf("  blob localization: %d/%zu centers within 4px\n", hit, centers.size());
    if (hit < (int)centers.size() - 1) { printf("  FAIL: blob localization\n"); fails++; }

    // 3）描述子须合理归一化、量化且非退化
    bool descOk = a.count() > 0;
    for (uint32_t i = 0; i < a.count() && descOk; i++) {
        int nz = 0;
        double ss = 0;
        for (int j = 0; j < 128; j++) {
            uint8_t q = a.descriptors[(size_t)i * 128 + j];
            if (q) nz++;
            ss += (double)q * q;
        }
        if (nz == 0) descOk = false;  // 全零描述子
    }
    printf("  descriptor sanity: %s\n", descOk ? "ok" : "BAD");
    if (!descOk) fails++;

    // 4）确定性：不额外排序，两次关键点与描述子须逐项一致，下游依赖稳定特征索引（D16）。
    FeatureSet b = ext.extract(img);
    bool det = a.count() == b.count() && a.descriptors == b.descriptors;
    for (uint32_t i = 0; det && i < a.count(); i++) {
        const Keypoint& p = a.keypoints[i];
        const Keypoint& q = b.keypoints[i];
        if (std::fabs(p.x - q.x) > 1e-4f || std::fabs(p.y - q.y) > 1e-4f ||
            std::fabs(p.scale - q.scale) > 1e-4f ||
            std::fabs(p.orientation - q.orientation) > 1e-4f)
            det = false;
    }
    printf("  determinism (canonical order, unsorted compare): %s (%u vs %u)\n",
           det ? "ok" : "BAD", a.count(), b.count());
    if (!det) fails++;

    // 5）GPU 尺度 top-K 应保留大尺度特征
    {
        SiftOptions kopt = opt;
        kopt.max_num_features = 200;
        SiftExtractor extK(kopt);
        FeatureSet c = extK.extract(img);
        // 截断结果的尺度须不低于完整集合第 K 大尺度。
        std::vector<float> scales;
        for (const Keypoint& k : a.keypoints) scales.push_back(k.scale);
        std::sort(scales.begin(), scales.end(), std::greater<float>());
        float kth = scales.size() > 200 ? scales[199] : 0.0f;
        bool topk = c.count() <= 200 && c.count() > 0;
        float minKept = 1e9f;
        for (const Keypoint& k : c.keypoints) minKept = std::min(minKept, k.scale);
        // 允许直方图桶边界的极小余量
        if (a.count() > 200 && minKept < kth * 0.98f) topk = false;
        printf("  top-K (cap 200): kept %u, min scale %.3f vs 200th-largest %.3f -> %s\n",
               c.count(), minKept, kth, topk ? "ok" : "BAD");
        if (!topk) fails++;
    }

    // 6）特征文件往返，含 v2 颜色；为合成特征添加颜色渐变以覆盖颜色段。
    FeatureSet ac = a;
    ac.colors.resize((size_t)ac.count() * 3);
    for (uint32_t i = 0; i < ac.count(); i++) {
        ac.colors[3 * i + 0] = (uint8_t)(i & 0xff);
        ac.colors[3 * i + 1] = (uint8_t)((i * 3) & 0xff);
        ac.colors[3 * i + 2] = (uint8_t)((i * 7) & 0xff);
    }
    ac.exif_focal = 3637.05;                    // v3 EXIF 附加段（D46）
    ac.exif_camera = "Canon-EOS 5D-24.000000-5616x3744";
    ac.extract_width = ac.width / 2;            // v4 提取尺度附加段（D47）
    ac.extract_height = ac.height / 2;
    std::string tmp = "/tmp/spirula_sfm_selftest_features.bin";
    writeFeatures(tmp, ac);
    FeatureSet r = readFeatures(tmp);
    bool rt = r.count() == ac.count() && r.dim == ac.dim && r.width == ac.width &&
              r.descriptors == ac.descriptors && r.colors == ac.colors &&
              r.exif_focal == ac.exif_focal && r.exif_camera == ac.exif_camera &&
              r.extract_width == ac.extract_width && r.extract_height == ac.extract_height &&
              std::fabs(r.pixelScale() - 2.0) < 1e-12;
    for (uint32_t i = 0; i < ac.count() && rt; i++)
        if (std::fabs(r.keypoints[i].x - ac.keypoints[i].x) > 1e-6f) rt = false;
    printf("  features.bin round-trip (+colors): %s\n", rt ? "ok" : "BAD");
    if (!rt) fails++;

    // 6b）已知纯色区域的关键点采样应返回该颜色。
    {
        GrayImage ci;
        ci.width = 8; ci.height = 8;
        ci.data.assign(64, 0.5f);
        ci.rgb.resize(64 * 3);
        for (int i = 0; i < 64; i++) { ci.rgb[3 * i] = 10; ci.rgb[3 * i + 1] = 200; ci.rgb[3 * i + 2] = 90; }
        uint8_t c[3];
        sampleColor(ci, 3.5f, 4.5f, c);
        bool cok = c[0] == 10 && c[1] == 200 && c[2] == 90;
        // 越界钳位到边缘，仍为同一纯色
        sampleColor(ci, -5.0f, 100.0f, c);
        cok = cok && c[0] == 10 && c[1] == 200 && c[2] == 90;
        printf("  color sampling: %s\n", cok ? "ok" : "BAD");
        if (!cok) fails++;
    }

    // 6d）缩小图提取后恢复源坐标，关键点应对应与全分辨率相同的真值斑点（D46）。
    {
        // 源图为 256 px 场景的两倍，加载器使用原 256 px 工作图，恢复坐标应落到双倍斑点中心。
        GrayImage half = syntheticScene(256, 256);
        FeatureSet hf = ext.extract(half);
        scaleKeypoints(hf, 512, 512);
        double worst = 0;
        for (const C& c : centers) {
            double best = 1e9;
            for (const Keypoint& k : hf.keypoints)
                best = std::min(best, (double)std::hypot(k.x - 2 * c.x, k.y - 2 * c.y));
            worst = std::max(worst, best);
        }
        // 工作图允许 4 px 定位误差，源图相应加倍；错误坐标约定会造成倍率或半图级误差。
        bool sok = hf.width == 512 && hf.height == 512 && worst < 8.0;
        printf("  keypoints scaled to source resolution: %dx%d, blob error %.2f px -> %s\n",
               hf.width, hf.height, worst, sok ? "ok" : "BAD");
        if (!sok) fails++;
    }

    // 6e）手工 TIFF 覆盖两种 EXIF 焦距规则及相机身份分组（D46）。
    {
        std::vector<uint8_t> t;
        auto pu16 = [&](uint16_t v) { t.push_back(v & 0xff); t.push_back((uint8_t)(v >> 8)); };
        auto pu32 = [&](uint32_t v) {
            for (int i = 0; i < 4; i++) t.push_back((uint8_t)((v >> (8 * i)) & 0xff));
        };
        auto entry = [&](uint16_t tag, uint16_t type, uint32_t count, uint32_t val) {
            pu16(tag); pu16(type); pu32(count); pu32(val);
        };
        const uint32_t kExif = 50, kMake = 104, kModel = 112, kFocal = 120, kRes = 128;
        t.push_back('I'); t.push_back('I');
        pu16(42);
        pu32(8);
        pu16(3);                              // IFD0
        entry(0x010F, 2, 6, kMake);           // 制造商标签
        entry(0x0110, 2, 8, kModel);          // 相机型号标签
        entry(0x8769, 4, 1, kExif);           // EXIF 子 IFD 指针
        pu32(0);
        pu16(4);                              // EXIF IFD
        entry(0x920A, 5, 1, kFocal);          // FocalLength=24 mm
        entry(0xA20E, 5, 1, kRes);            // 焦平面横向分辨率标签
        entry(0xA210, 3, 1, 2);               // 分辨率单位为英寸
        entry(0xA002, 4, 1, 5616);            // 像素宽度标签
        pu32(0);
        const char* mk = "Canon\0";
        const char* md = "EOS 5D\0";
        t.insert(t.end(), mk, mk + 6);
        t.resize(kModel, 0);
        t.insert(t.end(), md, md + 7);
        t.resize(kFocal, 0);
        pu32(24); pu32(1);                    // 24/1 mm
        pu32(38492117u); pu32(10000u);        // 3849.2118 像素/英寸
        ExifData e = parseExifTiff(t.data(), t.size());
        // 24 mm × 3849.2118 像素/英寸 = 24 × 151.5437 像素/毫米，焦距约 3637.0 px。
        double f = exifFocalPx(e, 5616, 3744);
        bool eok = e.valid && e.make == "Canon" && e.model == "EOS 5D" &&
                   std::fabs(e.focal_mm - 24.0) < 1e-9 && std::fabs(f - 3637.05) < 1.0;
        // 缩放后的文件保留 EXIF 时，按实际尺寸比例修正焦平面分辨率，使像素焦距同步缩放。
        eok = eok && std::fabs(exifFocalPx(e, 2808, 1872) - 0.5 * f) < 1.0;
        // 存在等效 35 mm 焦距时优先采用规则 1。
        ExifData e35 = e;
        e35.focal_35mm = 24;
        double f35 = exifFocalPx(e35, 5616, 3744);
        eok = eok && std::fabs(f35 - 24.0 / 43.27 * std::hypot(5616.0, 3744.0)) < 1e-6;
        // 身份只含机身和尺寸，焦距按容差而非字符串精确相等判断（D48）。
        std::string key = exifCameraKey(e, 5616, 3744);
        eok = eok && key == "Canon-EOS 5D-5616x3744";
        ExifData e25 = e;
        e25.focal_mm = 25;
        eok = eok && exifCameraKey(e25, 5616, 3744) == key;
        // 24/25 mm 的约 4% 量化差应视为同一设置，真实变焦另分组，无焦距图像不能进入测量簇。
        {
            std::vector<double> f = {3637, 3800, 0, 5391, 5500, 59903, 0, 3700};
            std::vector<int> lab = detail::exifFocalClusters(f, 0.10);
            eok = eok && lab[0] == lab[1] && lab[1] == lab[7] &&   // 3637/3700/3800
                  lab[3] == lab[4] &&                              // 5391/5500
                  lab[0] != lab[3] && lab[3] != lab[5] &&
                  lab[2] == -1 && lab[6] == -1;
            // 容差低于量化步长时会错误拆分固定镜头。
            std::vector<int> tight = detail::exifFocalClusters(f, 0.01);
            eok = eok && tight[0] != tight[1];
        }
        // 截断输入不能越界或凭空生成焦距。
        for (size_t cut = 1; cut < t.size(); cut += 7) {
            ExifData tr = parseExifTiff(t.data(), cut);
            double ft = exifFocalPx(tr, 5616, 3744);
            if (!(ft == 0 || std::fabs(ft - 3637.05) < 1.0)) eok = false;
        }
        printf("  EXIF: focal %.1f px (plane res), %.1f px (35mm), key \"%s\" -> %s\n", f, f35,
               key.c_str(), eok ? "ok" : "BAD");
        if (!eok) fails++;
    }

    // 6f）camera-mode 基础分组后，PREFIX=VALUE 应进一步拆分并标记组焦距先验（D46）。
    {
        std::vector<ImageEntry> imgs = {{"cam/0", 0},  {"cam/1", 0},  {"cam0/0", 0},
                                        {"cam0/1", 0}, {"cam1/0", 0}, {"cam1/1", 0}};
        std::vector<FeatureSet> fsv(6);
        for (int i = 0; i < 6; i++) {
            fsv[i].width = i < 2 ? 720 : 960;
            fsv[i].height = i < 2 ? 540 : 960;
        }
        fsv[0].exif_focal = fsv[1].exif_focal = 700;  // 仅针孔图像包含 EXIF
        CameraSetupOptions so;
        so.mode = CameraMode::Folder;
        so.model = CamModel::OpenCV;
        parseCameraOverride("cam0=thin-prism-fisheye", OverrideKind::Model, so.overrides);
        parseCameraOverride("cam1=thin-prism-fisheye", OverrideKind::Model, so.overrides);
        parseCameraOverride("cam0=520", OverrideKind::Focal, so.overrides);
        CameraSetup cs = buildCameras(imgs, fsv, so);
        bool gok = cs.count() == 3 && cs.ids[0] == cs.ids[1] && cs.ids[2] == cs.ids[3] &&
                   cs.ids[4] == cs.ids[5] && cs.ids[0] != cs.ids[2] && cs.ids[2] != cs.ids[4];
        const Camera& pin = cs.cameras.at(cs.ids[0]);
        const Camera& f0 = cs.cameras.at(cs.ids[2]);
        const Camera& f1 = cs.cameras.at(cs.ids[4]);
        gok = gok && pin.model == CamModel::OpenCV && f0.isFisheye() && f1.isFisheye();
        gok = gok && std::fabs(pin.focal() - 700) < 1e-9 &&      // EXIF 来源
                     std::fabs(f0.focal() - 520) < 1e-9;         // 显式焦距
        // cam1 仅指定模型，焦距仍采用鱼眼几何猜测。
        gok = gok && std::fabs(f1.focal() - std::hypot(960.0, 960.0) / M_PI) < 1e-6;
        gok = gok && cs.focal_known.count(cs.ids[0]) && cs.focal_known.count(cs.ids[2]) &&
              !cs.focal_known.count(cs.ids[4]);
        gok = gok && cs.mixed() && cs.anyWide();
        // 全局和组畸变列表按各模型 BA 顺序解释，opencv 为 k1,k2,p1,p2，薄棱镜还含 k3,k4,sx1,sy1（D72）。
        CameraSetupOptions sod = so;
        parseDistortion("-0.11,0.02,0.001,-0.002", sod.extra);
        parseCameraOverride("cam1=-0.4,0.09,0,0,0.01", OverrideKind::Distortion, sod.overrides);
        CameraSetup csd = buildCameras(imgs, fsv, sod);
        const Camera& dpin = csd.cameras.at(csd.ids[0]);
        const Camera& dfar = csd.cameras.at(csd.ids[4]);
        gok = gok && dpin.k1 == -0.11 && dpin.k2 == 0.02 && dpin.p1 == 0.001 &&
              dpin.p2 == -0.002 && dpin.k3 == 0;
        gok = gok && dfar.k1 == -0.4 && dfar.k2 == 0.09 && dfar.p1 == 0 && dfar.p2 == 0 &&
              dfar.k3 == 0.01 && dfar.k4 == 0 && dfar.sx1 == 0 && dfar.sy1 == 0;
        // 尾逗号、杂字符和空列表均须拒绝。
        std::vector<double> junk;
        gok = gok && !parseDistortion("0.1,", junk) && !parseDistortion("0.1,x", junk) &&
              !parseDistortion("", junk);
        // 禁用 EXIF 焦距后，针孔使用 1.2*max(w,h) 猜测。
        CameraSetupOptions so2 = so;
        so2.exif_focal = false;
        CameraSetup cs2 = buildCameras(imgs, fsv, so2);
        gok = gok && std::fabs(cs2.cameras.at(cs2.ids[0]).focal() - 1.2 * 720) < 1e-9 &&
              !cs2.focal_known.count(cs2.ids[0]);
        // 默认 EXIF 分组应区分真实变焦，容忍整毫米舍入差（D48）。
        {
            std::vector<ImageEntry> zi = {{"a", 0}, {"b", 0}, {"c", 0}};
            std::vector<FeatureSet> zf(3);
            for (FeatureSet& f : zf) {
                f.width = 3888;
                f.height = 5184;
                f.exif_camera = "Panasonic-DC-G9-3888x5184";
            }
            zf[0].exif_focal = 14976;   // 50 mm 焦距
            zf[1].exif_focal = 15300;   // 相同镜头设置的不同舍入记录
            zf[2].exif_focal = 59903;   // 200 mm 焦距
            CameraSetupOptions zo;
            zo.mode = CameraMode::Folder;
            CameraSetup zs = buildCameras(zi, zf, zo);
            gok = gok && zs.count() == 2 && zs.ids[0] == zs.ids[1] && zs.ids[0] != zs.ids[2];
            gok = gok && std::fabs(zs.cameras.at(zs.ids[2]).focal() - 59903) < 1.0;
            zo.exif_groups = false;
            gok = gok && buildCameras(zi, zf, zo).count() == 1;
        }
        // 不同尺寸占比高时识别为照片集合，显式 camera-mode 则固定模式；2% 桶须吸收预处理抖动，少量单相机集不能误触发。
        {
            auto make = [](const std::vector<std::pair<int, int>>& dims) {
                std::vector<FeatureSet> v(dims.size());
                for (size_t i = 0; i < dims.size(); i++) {
                    v[i].width = dims[i].first;
                    v[i].height = dims[i].second;
                }
                return v;
            };
            std::vector<std::pair<int, int>> capture, jittered, collection, tiny;
            for (int i = 0; i < 60; i++) {
                capture.push_back({4032, 3024});
                // 每图 ±1% 尺寸变化，模拟同一相机的预处理裁剪。
                jittered.push_back({4032 + (i % 5) * 8, 3024 + (i % 5) * 6});
                // 按相对 5% 递增尺寸，使 2% 桶在各尺度都能分开六十组。
                collection.push_back({(int)(700 * std::pow(1.05, i)),
                                      (int)(500 * std::pow(1.05, i))});
            }
            for (int i = 0; i < 8; i++) tiny.push_back({1000 + i * 300, 800 + i * 200});
            size_t nb = 0;
            gok = gok && !looksLikePhotoCollection(make(capture), &nb) && nb == 1;
            gok = gok && !looksLikePhotoCollection(make(jittered), &nb) && nb == 1;
            gok = gok && looksLikePhotoCollection(make(collection), &nb) && nb == 60;
            gok = gok && !looksLikePhotoCollection(make(tiny), &nb);  // 图像数量不足
            std::vector<ImageEntry> ci(60, {"x", 0});
            for (int i = 0; i < 60; i++) ci[i].name = "img" + std::to_string(i);
            CameraSetupOptions co;
            CameraSetup ccs = buildCameras(ci, make(collection), co);
            gok = gok && ccs.mode_switched && ccs.mode_used == CameraMode::Image &&
                  ccs.count() == 60;
            co.mode_explicit = true;   // 显式指定 camera-mode=folder
            CameraSetup pcs = buildCameras(ci, make(collection), co);
            gok = gok && !pcs.mode_switched && pcs.count() == 60;  // 仍有六十种尺寸
            CameraSetup kcs = buildCameras(std::vector<ImageEntry>(60, {"y", 0}),
                                           make(jittered), CameraSetupOptions{});
            gok = gok && !kcs.mode_switched && kcs.count() == 1;
        }
        printf("  camera grouping: %u groups, focals %.0f/%.0f/%.0f -> %s\n", cs.count(),
               pin.focal(), f0.focal(), f1.focal(), gok ? "ok" : "BAD");
        if (!gok) fails++;
    }

    // 6c）投影与反投影互逆，RADIAL/OPENCV 的 COLMAP 读写保留模型 ID 和参数（D29）。
    {
        Camera cam;
        cam.model = CamModel::OpenCV;
        cam.width = 1600; cam.height = 1200;
        cam.fx = 1300; cam.fy = 1280; cam.cx = 802; cam.cy = 598;
        cam.k1 = -0.12; cam.k2 = 0.03; cam.p1 = 0.001; cam.p2 = -0.0007;
        double maxErr = 0;
        for (int gy = 1; gy < 12; gy++)
            for (int gx = 1; gx < 12; gx++) {
                Vec2 px = {cam.cx + (gx - 6) * 100.0, cam.cy + (gy - 6) * 80.0};
                Vec2 xn = cam.unproject(px);
                Vec2 back = cam.project({xn.x, xn.y, 1.0});
                maxErr = std::max(maxErr, std::hypot(back.x - px.x, back.y - px.y));
            }
        bool projok = maxErr < 1e-3;
        printf("  camera project/unproject inverse (opencv): max %.2e px -> %s\n", maxErr,
               projok ? "ok" : "BAD");
        if (!projok) fails++;

        // FULL_OPENCV 的有理分母也须被固定点去畸变正确反解。
        {
            Camera fc = cam;
            fc.model = CamModel::FullOpenCV;
            fc.k3 = 0.004; fc.k4 = 0.02; fc.k5 = -0.003; fc.k6 = 0.0005;
            double err = 0;
            for (int gy = 1; gy < 12; gy++)
                for (int gx = 1; gx < 12; gx++) {
                    Vec2 px = {fc.cx + (gx - 6) * 100.0, fc.cy + (gy - 6) * 80.0};
                    Vec2 xn = fc.unproject(px);
                    Vec2 back = fc.project({xn.x, xn.y, 1.0});
                    err = std::max(err, std::hypot(back.x - px.x, back.y - px.y));
                }
            bool rok = err < 1e-3;
            printf("  camera project/unproject inverse (full-opencv rational): max %.2e px -> %s\n",
                   err, rok ? "ok" : "BAD");
            if (!rok) fails++;
        }

        // 两类鱼眼的 project->bearing 须恢复包括 z<0 的超过 90 度视线。
        {
            Camera feKB;
            feKB.model = CamModel::OpenCVFisheye;
            feKB.width = feKB.height = 1920; feKB.fx = feKB.fy = 560; feKB.cx = feKB.cy = 960;
            // 正径向系数保持 theta_d(theta) 严格单调，使全视场可逆；任意符号系数可能产生折返。
            feKB.k1 = 0.05; feKB.k2 = 0.01; feKB.k3 = 0.002; feKB.k4 = 0.0005;
            Camera feTP = feKB;
            feTP.model = CamModel::ThinPrismFisheye;   // 加切向与薄棱镜项
            feTP.p1 = 0.001; feTP.p2 = -0.0008; feTP.sx1 = 0.002; feTP.sy1 = -0.0015;
            for (auto* pcam : {&feKB, &feTP}) {
                double maxAng = 0;
                int wide = 0;
                for (double th = 5; th <= 130; th += 5)    // 离轴至 130 度，覆盖大于 180 度视场
                    for (double phi = 0; phi < 360; phi += 45) {
                        double t = th * M_PI / 180, ph2 = phi * M_PI / 180;
                        Vec3 ray = {std::sin(t)*std::cos(ph2), std::sin(t)*std::sin(ph2), std::cos(t)};
                        if (ray.z < 0) wide++;
                        Vec3 b = pcam->bearing(pcam->project(ray));
                        double d = std::max(-1.0, std::min(1.0, b.dot(ray)));
                        maxAng = std::max(maxAng, std::acos(d) * 180.0 / M_PI);
                    }
                bool feok = maxAng < 1e-3 && wide > 0;
                printf("  %-14s project/bearing round-trip (to 130 deg, %d past 90): "
                       "max %.2e deg -> %s\n",
                       pcam->model == CamModel::OpenCVFisheye ? "opencv-fisheye" : "thin-prism",
                       wide, maxAng, feok ? "ok" : "BAD");
                if (!feok) fails++;
            }
        }

        // 等距柱状投影覆盖整球及正后方，像素公式为 x=(theta/2pi+1/2)w，y=(1/2-phi/pi)h，角度约定与 COLMAP 一致（D49）。
        {
            const int W = 5760, H = 2880;
            Camera eq = Camera::defaultFor(8, W, H, 0, CamModel::Equirect);
            double maxAng = 0, maxPx = 0;
            int back = 0;
            for (double th = 2; th <= 178; th += 4)
                for (double phi = 0; phi < 360; phi += 15) {
                    double t = th * M_PI / 180, ph2 = phi * M_PI / 180;
                    Vec3 ray = {std::sin(t) * std::cos(ph2), std::sin(t) * std::sin(ph2),
                                std::cos(t)};
                    if (ray.z < 0) back++;
                    Vec2 px = eq.project(ray);
                    Vec3 b = eq.bearing(px);
                    maxAng = std::max(maxAng,
                                      std::acos(std::max(-1.0, std::min(1.0, b.dot(ray)))) *
                                          180.0 / M_PI);
                    // 独立展开 COLMAP 投影公式作为参考
                    double az = std::atan2(ray.x, ray.z);
                    double el = std::atan2(-ray.y, std::hypot(ray.x, ray.z));
                    Vec2 ref = {(az / (2 * M_PI) + 0.5) * W, (0.5 - el / M_PI) * H};
                    maxPx = std::max(maxPx, std::hypot(px.x - ref.x, px.y - ref.y));
                }
            // 横向周期回绕后必须为同一射线，保证全景接缝连续。
            Vec3 l = eq.bearing({-3.0, H * 0.5}), r = eq.bearing({W - 3.0, H * 0.5});
            double seam = std::acos(std::max(-1.0, std::min(1.0, l.dot(r)))) * 180.0 / M_PI;
            // acos(1-eps)≈sqrt(2eps) 使双精度角误差底约 1e-6 度，像素公式比较无此损失，要求 1e-9。
            bool eqok = maxAng < 1e-4 && maxPx < 1e-9 && back > 0 && std::fabs(seam) < 1e-9;
            printf("  equirect project/bearing round-trip (full sphere, %d behind): "
                   "%.2e deg, vs COLMAP %.2e px, seam %.2e deg -> %s\n",
                   back, maxAng, maxPx, seam, eqok ? "ok" : "BAD");
            if (!eqok) fails++;
        }

        // 全部模型的 COLMAP 读写保留 ID 和各参数，同时验证统一打包实现（D30）。
        Reconstruction rc;
        Camera sp = Camera::defaultFor(1, 800, 600, 700.0, CamModel::SimplePinhole);
        Camera ph = Camera::defaultFor(2, 1024, 768, 900.0, CamModel::Pinhole);
        ph.fx = 905; ph.fy = 898;
        Camera rad = Camera::defaultFor(3, 1920, 1080, 2000.0, CamModel::Radial);
        rad.k1 = -0.05; rad.k2 = 0.01;
        cam.id = 4;
        Camera fe = Camera::defaultFor(5, 1920, 1920, 560.0, CamModel::OpenCVFisheye);
        fe.k1 = 0.02; fe.k2 = -0.01; fe.k3 = 0.003; fe.k4 = -0.001;
        Camera fo = Camera::defaultFor(6, 1600, 1200, 1300.0, CamModel::FullOpenCV);
        fo.k1 = -0.11; fo.k2 = 0.02; fo.p1 = 0.001; fo.p2 = -0.0005;
        fo.k3 = 0.004; fo.k4 = 0.021; fo.k5 = -0.003; fo.k6 = 0.0006;
        Camera tp = Camera::defaultFor(7, 1920, 1920, 560.0, CamModel::ThinPrismFisheye);
        tp.k1 = 0.02; tp.k2 = -0.005; tp.p1 = 0.001; tp.p2 = -0.0008;
        tp.k3 = 0.001; tp.k4 = -0.0003; tp.sx1 = 0.002; tp.sy1 = -0.0015;
        Camera eqc = Camera::defaultFor(8, 5760, 2880, 0, CamModel::Equirect);
        rc.cameras[1] = sp; rc.cameras[2] = ph; rc.cameras[3] = rad; rc.cameras[4] = cam;
        rc.cameras[5] = fe; rc.cameras[6] = fo; rc.cameras[7] = tp; rc.cameras[8] = eqc;
        std::string cdir = "/tmp/spirula_sfm_selftest_model";
        fs::create_directories(cdir);
        rc.writeBinary(cdir);
        Reconstruction rr = Reconstruction::readBinary(cdir);
        auto eq = [](double a, double b) { return std::fabs(a - b) < 1e-9; };
        bool iook = rr.cameras.size() == 8 &&
                    rr.cameras[8].model == CamModel::Equirect &&
                    rr.cameras[8].width == 5760 && rr.cameras[8].height == 2880 &&
                    eq(rr.cameras[8].fx, 5760 / (2 * M_PI)) &&
                    eq(rr.cameras[8].fy, 2880 / M_PI) &&
                    rr.cameras[1].model == CamModel::SimplePinhole && eq(rr.cameras[1].focal(), 700) &&
                    rr.cameras[2].model == CamModel::Pinhole &&
                    eq(rr.cameras[2].fx, 905) && eq(rr.cameras[2].fy, 898) &&
                    rr.cameras[3].model == CamModel::Radial && eq(rr.cameras[3].focal(), 2000) &&
                    eq(rr.cameras[3].k1, -0.05) &&
                    rr.cameras[4].model == CamModel::OpenCV &&
                    eq(rr.cameras[4].fx, 1300) && eq(rr.cameras[4].fy, 1280) &&
                    eq(rr.cameras[4].p1, 0.001) && eq(rr.cameras[4].p2, -0.0007) &&
                    rr.cameras[5].model == CamModel::OpenCVFisheye &&
                    eq(rr.cameras[5].k3, 0.003) && eq(rr.cameras[5].k4, -0.001) &&
                    rr.cameras[6].model == CamModel::FullOpenCV &&
                    eq(rr.cameras[6].k1, -0.11) && eq(rr.cameras[6].p1, 0.001) &&
                    eq(rr.cameras[6].k3, 0.004) && eq(rr.cameras[6].k4, 0.021) &&
                    eq(rr.cameras[6].k5, -0.003) && eq(rr.cameras[6].k6, 0.0006) &&
                    rr.cameras[7].model == CamModel::ThinPrismFisheye &&
                    eq(rr.cameras[7].k4, -0.0003) && eq(rr.cameras[7].sx1, 0.002) &&
                    eq(rr.cameras[7].sy1, -0.0015);
        printf("  COLMAP camera IO round-trip (all 8 models): %s\n", iook ? "ok" : "BAD");
        if (!iook) fails++;

        // BA 布局将主点移到尾部，各模型单独往返，并检查固定参数确实为末尾两项（D50）。
        {
            bool ba_ok = true;
            for (const auto& kv : rc.cameras) {
                const Camera& c = kv.second;
                double d[12] = {0};
                packIntrinsics(c, d);
                Camera back = c;
                back.fx = back.fy = back.cx = back.cy = 0;
                back.k1 = back.k2 = back.p1 = back.p2 = 0;
                unpackIntrinsics(back, d);
                if (!(eq(back.fx, c.fx) && eq(back.fy, c.fy) && eq(back.cx, c.cx) &&
                      eq(back.cy, c.cy) && eq(back.k1, c.k1) && eq(back.k2, c.k2) &&
                      eq(back.p1, c.p1) && eq(back.p2, c.p2) && eq(back.k3, c.k3) &&
                      eq(back.k4, c.k4) && eq(back.k5, c.k5) && eq(back.k6, c.k6) &&
                      eq(back.sx1, c.sx1) && eq(back.sy1, c.sy1)))
                    ba_ok = false;
                const int n = camNumParams(c.model);
                const int nf = camNumFreeParams(c.model);
                if (c.model == CamModel::Equirect) {
                    if (nf != 0) ba_ok = false;          // 没有任何可优化参数
                } else {
                    if (nf != n - 2) ba_ok = false;      // 除 cx/cy 外均可优化
                    if (!(eq(d[n - 2], c.cx) && eq(d[n - 1], c.cy))) ba_ok = false;
                }
            }
            printf("  BA intrinsics layout (round-trip, principal point last): %s\n",
                   ba_ok ? "ok" : "BAD");
            if (!ba_ok) fails++;
        }

        // FULL_OPENCV 必须写为模型 6，十二参数顺序为 fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6。
        {
            std::ifstream cf(cdir + "/cameras.bin", std::ios::binary);
            bool full_ok = false, eq_ok = false;
            uint64_t ncam = 0; cf.read((char*)&ncam, 8);
            for (uint64_t i = 0; i < ncam; i++) {
                uint32_t cid; int32_t mdl; uint64_t cw, ch;
                cf.read((char*)&cid, 4); cf.read((char*)&mdl, 4);
                cf.read((char*)&cw, 8); cf.read((char*)&ch, 8);
                int np = mdl == 0 ? 3 : mdl == 1 ? 4 : mdl == 3 ? 5 : mdl == 4 || mdl == 5 ? 8
                       : mdl == 17 ? 2 : 12;
                double ps[12] = {0};
                for (int k = 0; k < np; k++) cf.read((char*)&ps[k], 8);
                if (mdl == 6)  // FULL_OPENCV
                    full_ok = np == 12 && ps[4] == -0.11 && ps[5] == 0.02 && ps[6] == 0.001 &&
                              ps[7] == -0.0005 && ps[8] == 0.004 && ps[9] == 0.021 &&
                              ps[10] == -0.003 && ps[11] == 0.0006;
                if (mdl == 17)  // EQUIRECTANGULAR 参数严格为 (w,h)
                    eq_ok = np == 2 && ps[0] == (double)cw && ps[1] == (double)ch &&
                            cw == 5760 && ch == 2880;
            }
            printf("  FULL_OPENCV emit (12 params, COLMAP order): %s\n", full_ok ? "ok" : "BAD");
            if (!full_ok) fails++;
            printf("  EQUIRECTANGULAR emit (model 17, params = w,h): %s\n", eq_ok ? "ok" : "BAD");
            if (!eq_ok) fails++;
        }
    }

    // 7b）256 维浮点描述子匹配，单独覆盖该宽度 SPIR-V，主机用相同距离作为参考。
    {
        auto make = [](uint32_t n, uint32_t seed) {
            FeatureSet f;
            f.width = f.height = 1000;
            f.dim = 256;
            f.dtype = DType::F32;
            f.keypoints.resize(n);
            f.descriptors.resize((size_t)n * 256 * sizeof(float));
            float* d = reinterpret_cast<float*>(f.descriptors.data());
            uint32_t st = seed;
            auto rnd = [&] {
                st = st * 1664525u + 1013904223u;
                return (float)((st >> 8) & 0xffff) / 32768.0f - 1.0f;
            };
            for (uint32_t i = 0; i < n; i++) {
                f.keypoints[i] = {(float)(i % 100) * 7.0f, (float)(i / 100) * 7.0f, 0, 0, 1};
                double sq = 0;
                for (int c = 0; c < 256; c++) {
                    const float v = rnd();
                    d[(size_t)i * 256 + c] = v;
                    sq += (double)v * v;
                }
                // 余弦阈值要求单位范数。
                const float inv = (float)(1.0 / std::sqrt(sq));
                for (int c = 0; c < 256; c++) d[(size_t)i * 256 + c] *= inv;
            }
            return f;
        };
        FeatureSet fa = make(300, 12345u);
        // B 为 A 的扰动而非独立噪声，避免高维随机向量近正交导致几乎无匹配，无法充分测试归约。
        FeatureSet fb = fa;
        fb.keypoints.resize(280);
        fb.descriptors.resize((size_t)280 * 256 * sizeof(float));
        {
            float* d = reinterpret_cast<float*>(fb.descriptors.data());
            uint32_t st = 4242u;
            for (uint32_t i = 0; i < 280; i++) {
                double sq = 0;
                for (int c = 0; c < 256; c++) {
                    st = st * 1664525u + 1013904223u;
                    const float n = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.05f;
                    const float v = d[(size_t)i * 256 + c] + n;
                    d[(size_t)i * 256 + c] = v;
                    sq += (double)v * v;
                }
                const float inv = (float)(1.0 / std::sqrt(sq));
                for (int c = 0; c < 256; c++) d[(size_t)i * 256 + c] *= inv;
            }
        }

        MatchOptions mo;
        mo.device = opt.device;
        mo.max_ratio = 0.95f;
        mo.cross_check = true;
        BruteForceMatcher m256(mo);
        const std::vector<FeatureMatch> gm = m256.match(fa, fb);

        // 主机参考使用上传器相同量化，隔离内核逻辑与舍入差异。
        auto quant = [](float v) {
            const float q = v * (127.0f / 0.4f) + 128.0f;
            return (int)std::lround(std::min(255.0f, std::max(0.0f, q)));
        };
        std::vector<int> qa((size_t)fa.count() * 256), qb((size_t)fb.count() * 256);
        const float* pa = reinterpret_cast<const float*>(fa.descriptors.data());
        const float* pb = reinterpret_cast<const float*>(fb.descriptors.data());
        for (size_t i = 0; i < qa.size(); i++) qa[i] = quant(pa[i]);
        for (size_t i = 0; i < qb.size(); i++) qb[i] = quant(pb[i]);
        auto d2 = [&](uint32_t i, uint32_t j) {
            long long s = 0;
            for (int c = 0; c < 256; c++) {
                const long long e = qa[(size_t)i * 256 + c] - qb[(size_t)j * 256 + c];
                s += e * e;
            }
            return s;
        };
        std::vector<uint32_t> bestB(fb.count(), 0u);
        for (uint32_t j = 0; j < fb.count(); j++) {
            long long bd = -1;
            for (uint32_t i = 0; i < fa.count(); i++) {
                const long long v = d2(i, j);
                if (bd < 0 || v < bd) { bd = v; bestB[j] = i; }
            }
        }
        std::vector<FeatureMatch> ref;
        for (uint32_t i = 0; i < fa.count(); i++) {
            long long b1 = -1, b2 = -1;
            uint32_t bj = 0;
            for (uint32_t j = 0; j < fb.count(); j++) {
                const long long v = d2(i, j);
                if (b1 < 0 || v < b1) { b2 = b1; b1 = v; bj = j; }
                else if (b2 < 0 || v < b2) { b2 = v; }
            }
            if (b2 >= 0 && (double)b1 >= (double)mo.max_ratio * mo.max_ratio * (double)b2)
                continue;
            if (bestB[bj] != i) continue;
            ref.push_back({i, bj, (float)std::sqrt((double)b1)});
        }
        bool ok = gm.size() == ref.size() && ref.size() > 200;
        for (size_t k = 0; k < ref.size() && ok; k++)
            if (gm[k].idx1 != ref[k].idx1 || gm[k].idx2 != ref[k].idx2) ok = false;
        printf("  matcher 256-D vs host reference: %zu / %zu matches -> %s\n", gm.size(),
               ref.size(), ok ? "ok" : "BAD");
        if (!ok) fails++;
    }

    // 7）自匹配应近似恒等且距离为零
    {
        MatchOptions mo;
        mo.device = opt.device;
        BruteForceMatcher matcher(mo);
        std::vector<FeatureMatch> ms = matcher.match(a, a);
        size_t ident = 0;
        bool distOk = true;
        for (const FeatureMatch& m : ms) {
            if (m.idx1 == m.idx2) {
                ident++;
                if (m.distance != 0.0f) distOk = false;
            }
        }
        float frac = ms.empty() ? 0.0f : (float)ident / ms.size();
        bool mok = ms.size() > 10 && frac > 0.9f && distOk;
        printf("  matcher self-match: %zu matches, %.1f%% identity (dist 0) -> %s\n", ms.size(),
               100.0f * frac, mok ? "ok" : "BAD");
        if (!mok) fails++;

        // 匹配文件读写往返
        MatchesDatabase db;
        db.images = {{"a", a.count()}, {"a", a.count()}};
        db.pairs = {{0, 1, 0, ms}};
        // 匹配数据库须保存验证使用的相机配置及 cameras.bin 不保存的测量尺度（D47）。
        Camera vc = Camera::defaultFor(1, 4000, 3000, 2600, CamModel::OpenCVFisheye);
        vc.pixel_scale = 1.6;
        vc.k1 = -0.03;
        db.cameras = {vc};
        db.camera_ids = {1, 1};
        db.focal_prior = {1};
        std::string mtmp = "/tmp/spirula_sfm_selftest_matches.bin";
        writeMatches(mtmp, db);
        MatchesDatabase rd = readMatches(mtmp);
        bool mrt = rd.images.size() == 2 && rd.pairs.size() == 1 &&
                   rd.pairs[0].matches.size() == ms.size() &&
                   rd.images[0].num_features == a.count();
        for (size_t k = 0; k < ms.size() && mrt; k++)
            if (rd.pairs[0].matches[k].idx1 != ms[k].idx1 ||
                rd.pairs[0].matches[k].idx2 != ms[k].idx2)
                mrt = false;
        mrt = mrt && rd.hasCameras() && rd.cameras.size() == 1 &&
              rd.cameras[0].model == CamModel::OpenCVFisheye && rd.cameras[0].width == 4000 &&
              std::fabs(rd.cameras[0].focal() - 2600) < 1e-9 &&
              std::fabs(rd.cameras[0].k1 + 0.03) < 1e-12 &&
              std::fabs(rd.cameras[0].pixel_scale - 1.6) < 1e-12 &&
              rd.camera_ids == db.camera_ids && rd.focal_prior == db.focal_prior;
        CameraSetup rcs;
        mrt = mrt && loadCameraSetup(rd, rcs) && rcs.count() == 1 &&
              rcs.focal_known.count(1) && rcs.focal_given.count(1);
        printf("  matches.bin round-trip (+verification cameras): %s\n", mrt ? "ok" : "BAD");
        if (!mrt) fails++;

        // 验证提取像素阈值正确换算到相机源像素（D47）。
        {
            FeatureSet fs;
            fs.width = 4000; fs.height = 3000;
            fs.extract_width = 2500; fs.extract_height = 1875;
            bool tok = std::fabs(fs.pixelScale() - 1.6) < 1e-12;
            Camera c = Camera::defaultFor(1, 4000, 3000, 2600, CamModel::OpenCV);
            c.pixel_scale = fs.pixelScale();
            tok = tok && std::fabs(c.errPx(4.0) - 6.4) < 1e-12 &&
                  std::fabs(c.errRad(4.0) - 6.4 / 2600) < 1e-15;
            // 未缩小的图像不受换算影响。
            Camera c1 = c;
            c1.pixel_scale = 1.0;
            tok = tok && std::fabs(c1.errPx(4.0) - 4.0) < 1e-12;
            // buildCameras 须从特征恢复该尺度。
            std::vector<ImageEntry> ie = {{"a/0", 0}, {"a/1", 0}};
            std::vector<FeatureSet> fv = {fs, fs};
            CameraSetup bcs = buildCameras(ie, fv, CameraSetupOptions{});
            tok = tok && bcs.count() == 1 &&
                  std::fabs(bcs.cameras.at(bcs.ids[0]).pixel_scale - 1.6) < 1e-12;
            printf("  pixel thresholds: scale %.2f, 4 extraction px = %.2f source px -> %s\n",
                   fs.pixelScale(), c.errPx(4.0), tok ? "ok" : "BAD");
            if (!tok) fails++;
        }

        // 8）六份同特征产生十五对真实验证任务，串行与并行结果须逐对相同。
        std::vector<FeatureSet> vf(6, a);
        auto vpairs = generatePairs((uint32_t)vf.size(), PairMode::Exhaustive);
        auto vmatch = [&](size_t b, size_t e, std::vector<std::vector<FeatureMatch>>& mo) {
            matcher.matchBatch(vf, vpairs, b, e, mo);
        };
        VerificationOptions vo;
        vo.num_threads = 1;
        std::vector<TwoViewMatches> serial = verifyPairs(vf, vpairs, vmatch, vo);
        vo.num_threads = 8;
        std::vector<TwoViewMatches> par = verifyPairs(vf, vpairs, vmatch, vo);
        bool vok = !serial.empty() && serial.size() == par.size();
        for (size_t k = 0; k < serial.size() && vok; k++) {
            if (serial[k].image1 != par[k].image1 || serial[k].image2 != par[k].image2 ||
                serial[k].config != par[k].config ||
                serial[k].matches.size() != par[k].matches.size())
                vok = false;
            for (size_t q = 0; q < serial[k].matches.size() && vok; q++)
                if (serial[k].matches[q].idx1 != par[k].matches[q].idx1 ||
                    serial[k].matches[q].idx2 != par[k].matches[q].idx2)
                    vok = false;
        }
        printf("  verification 1 vs 8 threads: %zu/%zu pairs kept, identical -> %s\n",
               serial.size(), vpairs.size(), vok ? "ok" : "BAD");
        if (!vok) fails++;
    }

    // 8b）GPU 图像对筛选对两组互不相关的重复图像仅保留组内边，结果须确定。
    {
        std::mt19937 rng(1234);
        auto randomSet = [&](uint32_t count) {
            FeatureSet f;
            f.width = f.height = 256;
            f.keypoints.resize(count);
            f.descriptors.resize((size_t)count * 128);
            for (uint32_t i = 0; i < count; i++) {
                f.keypoints[i] = {(float)(rng() % 256), (float)(rng() % 256),
                                  1.0f + (float)(rng() % 1024) / 256.0f, 0.0f, 0.0f};
                for (uint32_t d = 0; d < 128; d++)
                    f.descriptors[(size_t)i * 128 + d] = (uint8_t)(rng() % 128);
            }
            return f;
        };
        FeatureSet r1 = randomSet(512), r2 = randomSet(512);
        std::vector<FeatureSet> pf = {r1, r1, r2, r2, r1};
        PairSelectionOptions po;
        po.device = opt.device;
        po.num_features = 256;  // 覆盖按尺度从 512 收集到 256 特征
        po.num_neighbors = 4;
        auto sel = prefilterPairs(pf, po);
        auto sel2 = prefilterPairs(pf, po);
        std::vector<std::pair<uint32_t, uint32_t>> want = {{0, 1}, {0, 4}, {1, 4}, {2, 3}};
        bool pok = sel == want && sel2 == sel;
        printf("  pair selection: kept %zu/10 pairs (want the 4 within-group), "
               "deterministic -> %s\n",
               sel.size(), pok ? "ok" : "BAD");
        if (!pok) fails++;
    }

    // 9）并行解码须按序交付、内容正确且窗口有界。
    {
        // 用单行编码图像索引的微型 PGM，直接验证交付顺序和内容，无需 PNG 写入器。
        fs::path dir = fs::temp_directory_path() / "spirula_sfm_selftest_imgs";
        fs::remove_all(dir);
        fs::create_directories(dir);
        const int N = 24;
        std::vector<std::string> paths;
        std::vector<std::pair<int, int>> dims;
        for (int i = 0; i < N; i++) {
            // 宽度递减，使索引顺序同时为最大图优先。
            int w = 64 - i, h = 4;
            char buf[64];
            snprintf(buf, sizeof buf, "img%03d.pgm", i);
            fs::path p = dir / buf;
            std::vector<unsigned char> px((size_t)w * h, (unsigned char)(i * 7 + 3));
            std::ofstream f(p.string(), std::ios::binary);
            f << "P5\n" << w << " " << h << "\n255\n";
            f.write((const char*)px.data(), (std::streamsize)px.size());
            f.close();
            paths.push_back(p.string());
            dims.emplace_back(w, h);
        }
        ImageLoadOptions lo;
        lo.max_image_size = 0;  // 不缩放，仅测试顺序
        lo.num_threads = 8;
        lo.memory_budget_bytes = 1 << 20;  // 刻意使用很小预算
        ImageLoadPlan pl = planImageLoad(dims, lo);
        std::vector<size_t> seen;
        bool content_ok = true;
        loadImagesInOrder(paths, pl, lo, [&](size_t k, GrayImage& img) {
            seen.push_back(k);
            float want = (float)(int(k) * 7 + 3) / 255.0f;
            if (img.width != 64 - (int)k || img.height != 4 ||
                std::fabs(img.data[0] - want) > 1.0f / 255.0f)
                content_ok = false;
        });
        bool order_ok = seen.size() == (size_t)N;
        for (size_t k = 0; k < seen.size() && order_ok; k++)
            if (seen[k] != k) order_ok = false;
        bool plan_ok = pl.num_threads >= 1 && pl.num_threads <= 8 && pl.window >= pl.num_threads;
        printf("  parallel decode: %zu/%d in order, content %s, plan %d thr / window %d -> %s\n",
               seen.size(), N, content_ok ? "ok" : "BAD", pl.num_threads, pl.window,
               (order_ok && content_ok && plan_ok) ? "ok" : "BAD");
        if (!order_ok || !content_ok || !plan_ok) fails++;

        // 缺文件应报告并跳过，不能终止整批。
        std::vector<std::string> withBad = paths;
        withBad.insert(withBad.begin() + 5, (dir / "does_not_exist.pgm").string());
        size_t errs = 0, got = 0;
        ImageLoadPlan pl2 = planImageLoad(dims, lo);
        loadImagesInOrder(withBad, pl2, lo, [&](size_t, GrayImage&) { got++; },
                          [&](size_t, const std::string&) { errs++; });
        bool skip_ok = errs == 1 && got == (size_t)N;
        printf("  decode error handling: %zu ok / %zu failed -> %s\n", got, errs,
               skip_ok ? "ok" : "BAD");
        if (!skip_ok) fails++;

        // 消费者异常须在线程全部汇合后传播，避免析构可 join 线程导致 MSVC 直接以 0xC0000409 退出而丢失错误。
        bool threw = false;
        ImageLoadPlan pl3 = planImageLoad(dims, lo);
        try {
            loadImagesInOrder(paths, pl3, lo, [&](size_t k, GrayImage&) {
                if (k == 3) throw std::runtime_error("consumer");
            });
        } catch (const std::exception&) {
            threw = true;
        }
        printf("  throwing consumer propagates -> %s\n", threw ? "ok" : "BAD");
        if (!threw) fails++;
        fs::remove_all(dir);
    }

    printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    return sfmTestMain(argc - 1, argv + 1, cmdSelftest);
}
