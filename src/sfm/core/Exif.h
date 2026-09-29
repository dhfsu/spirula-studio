// 从文件头读取 EXIF 焦距和相机身份；24 mm 全画幅的几何猜测可达 6739 px，而真实约 3637 px，过长焦距容易被畸变与浅基线吸收。
// 独立解析少量 TIFF 标签，不修改第三方像素解码器，仅需读取数 KB。缺少可换算传感器尺度的标签时回退几何猜测。
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sfm {

struct ExifData {
    bool valid = false;         // 已找到并解析 EXIF 块
    std::string make, model;
    double focal_mm = 0;        // Exif:FocalLength
    double focal_35mm = 0;      // Exif:FocalLengthIn35mmFilm
    double focal_plane_x_res = 0;   // Exif:FocalPlaneXResolution
    int focal_plane_unit = 0;       // Exif:FocalPlaneResolutionUnit，2=英寸、3=厘米、4=毫米、5=微米
    int pixel_width = 0, pixel_height = 0;  // Exif:PixelXDimension/PixelYDimension
    int orientation = 1;        // Exif:Orientation，范围 1..8，1 表示存储方向即显示方向

    double exposure_time = 0;   // Exif:ExposureTime，单位秒
    double f_number = 0;        // Exif:FNumber
    double iso = 0;             // Exif:PhotographicSensitivity，兼容 SOS/REI/ISOSpeed
    // APEX 缺值为 NaN，因为 Tv=0 是合法的 1 s 曝光：t=2^-Tv，N=2^(Av/2)。
    double shutter_apex  = std::numeric_limits<double>::quiet_NaN();  // Exif:ShutterSpeedValue
    double aperture_apex = std::numeric_limits<double>::quiet_NaN();  // Exif:ApertureValue

    // GPS 位于 0x8825 指向的独立 IFD；经纬度以度表示并按半球决定符号，海平面以下高度取负。
    bool has_gps = false;
    bool has_alt = false;
    double lat_deg = 0, lon_deg = 0, alt_m = 0;

    bool hasFocal() const { return focal_35mm > 0 || focal_mm > 0; }
};

namespace detail {

// TIFF 大小端读取均检查边界，越界返回 0，使截断或恶意文件只产生空字段，不越界访问。
struct TiffReader {
    const uint8_t* p = nullptr;
    size_t n = 0;
    bool le = true;

    uint16_t u16(size_t o) const {
        if (o + 2 > n) return 0;
        return le ? (uint16_t)(p[o] | (p[o + 1] << 8)) : (uint16_t)((p[o] << 8) | p[o + 1]);
    }
    uint32_t u32(size_t o) const {
        if (o + 4 > n) return 0;
        return le ? (uint32_t)(p[o] | (p[o + 1] << 8) | (p[o + 2] << 16) | ((uint32_t)p[o + 3] << 24))
                  : (uint32_t)(((uint32_t)p[o] << 24) | (p[o + 1] << 16) | (p[o + 2] << 8) | p[o + 3]);
    }
};

inline size_t tiffTypeSize(uint16_t t) {
    switch (t) {
        case 1: case 2: case 6: case 7: return 1;   // BYTE, ASCII, SBYTE, UNDEFINED
        case 3: case 8: return 2;                   // SHORT, SSHORT
        case 4: case 9: case 11: return 4;          // LONG, SLONG, FLOAT
        case 5: case 10: case 12: return 8;         // RATIONAL, SRATIONAL, DOUBLE
        default: return 0;
    }
}

// 将 IFD 条目的首分量转为 double，不支持的类型返回 false。
inline bool tiffValue(const TiffReader& r, size_t entry, double& out) {
    uint16_t type = r.u16(entry + 2);
    uint32_t count = r.u32(entry + 4);
    if (count == 0) return false;
    size_t esz = tiffTypeSize(type);
    if (esz == 0) return false;
    size_t vo = (esz * count <= 4) ? entry + 8 : r.u32(entry + 8);
    switch (type) {
        case 1: case 7: out = vo < r.n ? r.p[vo] : 0; return vo < r.n;
        case 3: out = r.u16(vo); return true;
        case 4: out = r.u32(vo); return true;
        case 9: out = (int32_t)r.u32(vo); return true;
        case 5: {
            uint32_t num = r.u32(vo), den = r.u32(vo + 4);
            if (den == 0) return false;
            out = (double)num / den;
            return true;
        }
        case 10: {
            int32_t num = (int32_t)r.u32(vo), den = (int32_t)r.u32(vo + 4);
            if (den == 0) return false;
            out = (double)num / den;
            return true;
        }
        default: return false;
    }
}

// 读取 RATIONAL 的前 n 个分量；镜头标签只需一个，经纬度的度分秒需要三个。
inline bool tiffRationals(const TiffReader& r, size_t entry, int n, double* out) {
    if (r.u16(entry + 2) != 5 || (int)r.u32(entry + 4) < n) return false;
    const size_t vo = (8u * (unsigned)n <= 4) ? entry + 8 : r.u32(entry + 8);
    for (int i = 0; i < n; i++) {
        const uint32_t num = r.u32(vo + 8 * (size_t)i), den = r.u32(vo + 8 * (size_t)i + 4);
        if (den == 0) return false;
        out[i] = (double)num / den;
    }
    return true;
}

inline std::string tiffString(const TiffReader& r, size_t entry) {
    uint32_t count = r.u32(entry + 4);
    if (count == 0) return {};
    size_t vo = (count <= 4) ? entry + 8 : r.u32(entry + 8);
    if (vo >= r.n) return {};
    size_t len = std::min((size_t)count, r.n - vo);
    while (len > 0 && r.p[vo + len - 1] == '\0') len--;
    std::string s((const char*)r.p + vo, len);
    // 相机制造商名称常带末尾空格，需要裁剪。
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
}

// GPS IFD 使用独立标签编号，与 IFD0 编号重叠，因此必须单独分派。
inline void parseGpsIfd(const TiffReader& r, size_t off, ExifData& out) {
    if (off + 2 > r.n) return;
    uint16_t count = r.u16(off);
    if ((size_t)count * 12 + off + 2 > r.n) count = (uint16_t)((r.n - off - 2) / 12);
    double lat[3] = {0, 0, 0}, lon[3] = {0, 0, 0}, alt = 0, v = 0;
    bool has_lat = false, has_lon = false, has_alt = false;
    char lat_ref = 0, lon_ref = 0;
    int alt_ref = 0;
    for (uint16_t i = 0; i < count; i++) {
        const size_t e = off + 2 + (size_t)i * 12;
        switch (r.u16(e)) {
            case 0x0001: { std::string s = tiffString(r, e); if (!s.empty()) lat_ref = s[0]; } break;
            case 0x0002: has_lat = tiffRationals(r, e, 3, lat); break;
            case 0x0003: { std::string s = tiffString(r, e); if (!s.empty()) lon_ref = s[0]; } break;
            case 0x0004: has_lon = tiffRationals(r, e, 3, lon); break;
            case 0x0005: if (tiffValue(r, e, v)) alt_ref = (int)v; break;
            case 0x0006: has_alt = tiffRationals(r, e, 1, &alt); break;
            default: break;
        }
    }
    if (!has_lat || !has_lon || !lat_ref || !lon_ref) return;
    out.lat_deg = lat[0] + lat[1] / 60.0 + lat[2] / 3600.0;
    out.lon_deg = lon[0] + lon[1] / 60.0 + lon[2] / 3600.0;
    if (lat_ref == 'S' || lat_ref == 's') out.lat_deg = -out.lat_deg;
    if (lon_ref == 'W' || lon_ref == 'w') out.lon_deg = -out.lon_deg;
    out.has_alt = has_alt;
    out.alt_m = alt_ref == 1 ? -alt : alt;
    out.has_gps = true;
}

// 遍历 IFD 填充 out，depth 限制防止恶意子 IFD 指针形成循环。
inline void parseIfd(const TiffReader& r, size_t off, ExifData& out, int depth) {
    if (depth > 3 || off + 2 > r.n) return;
    uint16_t count = r.u16(off);
    if ((size_t)count * 12 + off + 2 > r.n) count = (uint16_t)((r.n - off - 2) / 12);
    for (uint16_t i = 0; i < count; i++) {
        size_t e = off + 2 + (size_t)i * 12;
        uint16_t tag = r.u16(e);
        double v = 0;
        switch (tag) {
            case 0x0112:
                if (tiffValue(r, e, v) && v >= 1 && v <= 8) out.orientation = (int)v;
                break;
            case 0x010F: out.make = tiffString(r, e); break;
            case 0x0110: out.model = tiffString(r, e); break;
            case 0x8769:  // 镜头标签所在的 EXIF 子 IFD
                if (tiffValue(r, e, v) && v > 0) parseIfd(r, (size_t)v, out, depth + 1);
                break;
            case 0x8825:  // GPS IFD 使用独立遍历
                if (tiffValue(r, e, v) && v > 0) parseGpsIfd(r, (size_t)v, out);
                break;
            case 0x829A: if (tiffValue(r, e, v) && v > 0) out.exposure_time = v; break;
            case 0x829D: if (tiffValue(r, e, v) && v > 0) out.f_number = v; break;
            case 0x8827: if (tiffValue(r, e, v) && v > 0) out.iso = v; break;
            case 0x8831: case 0x8832: case 0x8833:  // SOS / REI / ISOSpeed
                if (out.iso == 0 && tiffValue(r, e, v) && v > 0) out.iso = v;
                break;
            case 0x9201: if (tiffValue(r, e, v)) out.shutter_apex = v; break;
            case 0x9202: if (tiffValue(r, e, v)) out.aperture_apex = v; break;
            case 0x920A: if (tiffValue(r, e, v)) out.focal_mm = v; break;
            case 0xA405: if (tiffValue(r, e, v)) out.focal_35mm = v; break;
            case 0xA20E: if (tiffValue(r, e, v)) out.focal_plane_x_res = v; break;
            case 0xA210: if (tiffValue(r, e, v)) out.focal_plane_unit = (int)v; break;
            case 0xA002: if (tiffValue(r, e, v)) out.pixel_width = (int)v; break;
            case 0xA003: if (tiffValue(r, e, v)) out.pixel_height = (int)v; break;
            default: break;
        }
    }
}

}  // 命名空间 detail

// 从 II/MM 字节序标记开始解析 TIFF 块。
inline ExifData parseExifTiff(const uint8_t* data, size_t size) {
    ExifData out;
    if (size < 8) return out;
    detail::TiffReader r;
    r.p = data;
    r.n = size;
    if (data[0] == 'I' && data[1] == 'I') r.le = true;
    else if (data[0] == 'M' && data[1] == 'M') r.le = false;
    else return out;
    if (r.u16(2) != 42) return out;
    uint32_t ifd0 = r.u32(4);
    if (ifd0 == 0 || ifd0 >= size) return out;
    detail::parseIfd(r, ifd0, out, 0);
    out.valid = true;
    return out;
}

namespace detail {

// 查找首个以 sig 开头的 APP1 载荷并保留 sig；通过 seek 跳过其他段，避免每张照片读取数 MB 前缀。
inline std::vector<uint8_t> readApp1Segment(const std::string& path, const char* sig,
                                            size_t sig_len) {
    std::vector<uint8_t> seg_buf;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return seg_buf;
    auto at = [f](long off, void* dst, size_t n) {
        return std::fseek(f, off, SEEK_SET) == 0 && std::fread(dst, 1, n, f) == n;
    };
    uint8_t hdr[2];
    long o = 2;
    if (!at(0, hdr, 2) || hdr[0] != 0xFF || hdr[1] != 0xD8) { fclose(f); return seg_buf; }
    while (at(o, hdr, 2)) {
        if (hdr[0] != 0xFF) break;
        const uint8_t marker = hdr[1];
        // 写入器可在标记前用 0xFF 填充。
        if (marker == 0xFF) { o += 1; continue; }
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            o += 2;
            continue;
        }
        if (marker == 0xDA || marker == 0xD9) break;  // 遇到扫描数据或结束标记后不再有元数据
        if (!at(o + 2, hdr, 2)) break;
        const size_t seg = (size_t)(hdr[0] << 8 | hdr[1]);
        if (seg < 2) break;
        // APP1 可装 EXIF 或 XMP，遇到另一类型时继续查找。
        if (marker == 0xE1 && seg >= 2 + sig_len) {
            seg_buf.resize(sig_len);
            if (!at(o + 4, seg_buf.data(), sig_len)) break;
            if (std::memcmp(seg_buf.data(), sig, sig_len) == 0) {
                seg_buf.resize(seg - 2);
                const bool whole = at(o + 4, seg_buf.data(), seg_buf.size());
                fclose(f);
                if (!whole) seg_buf.clear();
                return seg_buf;
            }
            seg_buf.clear();
        }
        o += 2 + (long)seg;
    }
    fclose(f);
    seg_buf.clear();
    return seg_buf;
}

}  // 命名空间 detail

// 返回 JPEG APP1 中的 Exif\0\0 与随后 TIFF 块，不存在时为空。
inline std::vector<uint8_t> readExifSegment(const std::string& path) {
    return detail::readApp1Segment(path, "Exif\0\0", 6);
}

// 返回 JPEG XMP 命名空间头之后的文本，不存在时为空。
inline std::string readXmpPacket(const std::string& path) {
    static const char kSig[] = "http://ns.adobe.com/xap/1.0/";   // 末尾 NUL 属于标记的一部分
    const std::vector<uint8_t> seg = detail::readApp1Segment(path, kSig, sizeof kSig);
    if (seg.size() <= sizeof kSig) return {};
    return std::string((const char*)seg.data() + sizeof kSig, seg.size() - sizeof kSig);
}

// 没有 EXIF 的图像返回无效结果，调用方将其视为没有先验。
inline ExifData readExif(const std::string& path) {
    const std::vector<uint8_t> seg = readExifSegment(path);
    if (seg.size() <= 6) return ExifData();
    return parseExifTiff(seg.data() + 6, seg.size() - 6);
}

// ---------------- 图像方向 ----------------

// Orientation 将存储像素变成显示图像：先顺时针旋转，再水平镜像。
struct ExifTransform {
    int  turns_cw = 0;      // 0..3 个四分之一圈
    bool mirror = false;    // 旋转后应用水平镜像
    bool identity() const { return turns_cw == 0 && !mirror; }
};

inline ExifTransform exifTransform(int orientation) {
    switch (orientation) {
        case 2:  return {0, true};
        case 3:  return {2, false};
        case 4:  return {2, true};
        case 5:  return {1, true};
        case 6:  return {1, false};
        case 7:  return {3, true};
        case 8:  return {3, false};
        default: return {0, false};
    }
}

// 存储图像相机坐标系的向上方向，轴为右、下、前；镜像不改变向上方向，因此无需修改像素也能用于规范对齐。
inline void exifUpInCamera(int orientation, double up[3]) {
    static const double kX[4] = {0, -1, 0, 1};
    static const double kY[4] = {-1, 0, 1, 0};
    const int t = exifTransform(orientation).turns_cw;
    up[0] = kX[t];
    up[1] = kY[t];
    up[2] = 0;
}

// 仅读取 Orientation 标签，无标签时返回 1。
inline int exifOrientation(const std::string& path) {
    return readExif(path).orientation;
}

// 像素已旋转后更新 EXIF，并断开仍保留旧方向的缩略图；仅改 IFD 条目内的 SHORT/LONG 值，不移动块内偏移。
inline void exifFlattenOrientation(uint8_t* tiff, size_t size, int w, int h) {
    if (size < 8) return;
    detail::TiffReader r;
    r.p = tiff;
    r.n = size;
    if (tiff[0] == 'I' && tiff[1] == 'I') r.le = true;
    else if (tiff[0] == 'M' && tiff[1] == 'M') r.le = false;
    else return;
    if (r.u16(2) != 42) return;
    const uint32_t ifd0 = r.u32(4);
    if (ifd0 == 0 || (size_t)ifd0 + 2 > size) return;

    auto put = [&](size_t entry, uint32_t value) {
        const uint16_t type = r.u16(entry + 2);
        if (r.u32(entry + 4) != 1 || entry + 12 > size) return;
        uint8_t* v = tiff + entry + 8;
        if (type == 3) {              // SHORT 使用值字段的低半部分
            for (int i = 0; i < 2; i++) v[r.le ? i : 1 - i] = (uint8_t)(value >> (8 * i));
        } else if (type == 4) {       // LONG
            for (int i = 0; i < 4; i++) v[r.le ? i : 3 - i] = (uint8_t)(value >> (8 * i));
        }
    };
    auto entries = [&](size_t off) {
        uint16_t n = r.u16(off);
        if ((size_t)n * 12 + off + 2 > size) n = (uint16_t)((size - off - 2) / 12);
        return n;
    };

    const uint16_t n0 = entries(ifd0);
    size_t exif_ifd = 0;
    for (uint16_t i = 0; i < n0; i++) {
        const size_t e = ifd0 + 2 + (size_t)i * 12;
        const uint16_t tag = r.u16(e);
        if (tag == 0x0112) put(e, 1);
        else if (tag == 0x8769) exif_ifd = r.u32(e + 8);
    }
    if (exif_ifd > 0 && exif_ifd + 2 <= size) {
        const uint16_t n1 = entries(exif_ifd);
        for (uint16_t i = 0; i < n1; i++) {
            const size_t e = exif_ifd + 2 + (size_t)i * 12;
            const uint16_t tag = r.u16(e);
            if (tag == 0xA002) put(e, (uint32_t)w);
            else if (tag == 0xA003) put(e, (uint32_t)h);
        }
    }
    const size_t next = ifd0 + 2 + (size_t)n0 * 12;
    if (next + 4 <= size) for (int i = 0; i < 4; i++) tiff[next + i] = 0;
}

// 对已不适用原焦距的视图清零 FocalLength 与等效 35 mm 焦距；360 全帧针孔等效焦距不能误作为鱼眼已知焦距。
inline void exifClearFocal(uint8_t* tiff, size_t size) {
    if (size < 8) return;
    detail::TiffReader r;
    r.p = tiff;
    r.n = size;
    if (tiff[0] == 'I' && tiff[1] == 'I') r.le = true;
    else if (tiff[0] == 'M' && tiff[1] == 'M') r.le = false;
    else return;
    if (r.u16(2) != 42) return;
    auto entries = [&](size_t off) {
        uint16_t n = r.u16(off);
        if ((size_t)n * 12 + off + 2 > size) n = (uint16_t)((size - off - 2) / 12);
        return n;
    };
    const uint32_t ifd0 = r.u32(4);
    if (ifd0 == 0 || (size_t)ifd0 + 2 > size) return;
    size_t exif_ifd = 0;
    for (uint16_t i = 0, n = entries(ifd0); i < n; i++) {
        const size_t e = ifd0 + 2 + (size_t)i * 12;
        if (r.u16(e) == 0x8769) exif_ifd = r.u32(e + 8);
    }
    if (exif_ifd == 0 || exif_ifd + 2 > size) return;
    for (uint16_t i = 0, n = entries(exif_ifd); i < n; i++) {
        const size_t e = exif_ifd + 2 + (size_t)i * 12;
        const uint16_t tag = r.u16(e), type = r.u16(e + 2);
        size_t at = 0, len = 0;
        if (tag == 0xA405 && (type == 3 || type == 4)) {
            at = e + 8;
            len = type == 3 ? 2 : 4;
        } else if (tag == 0x920A && type == 5) {
            at = r.u32(e + 8);   // 仅清零分子，分母保持非零
            len = 4;
        }
        if (len > 0 && at + len <= size) std::memset(tiff + at, 0, len);
    }
}

// 计算存储图像的像素焦距；优先 f_px = f35 / 43.27 * image_diagonal，其次 f_px = f_mm * px_per_mm，无法确定时为 0。
// 焦平面分辨率属于原始传感器读出；若实际图像已缩放且 EXIF 保留原尺寸，则按尺寸比修正 px_per_mm。
inline double exifFocalPx(const ExifData& e, int width, int height) {
    if (!e.valid || width <= 0 || height <= 0) return 0;
    if (e.focal_35mm > 0) {
        const double diag = std::sqrt((double)width * width + (double)height * height);
        return e.focal_35mm / 43.27 * diag;
    }
    if (e.focal_mm > 0 && e.focal_plane_x_res > 0 && e.focal_plane_unit >= 2 &&
        e.focal_plane_unit <= 5) {
        double px_per_mm = 0;
        switch (e.focal_plane_unit) {
            case 2: px_per_mm = e.focal_plane_x_res / 25.4; break;   // 英寸
            case 3: px_per_mm = e.focal_plane_x_res / 10.0; break;   // 厘米
            case 4: px_per_mm = e.focal_plane_x_res; break;          // 毫米
            case 5: px_per_mm = e.focal_plane_x_res * 1000.0; break; // 微米
            default: return 0;
        }
        if (e.pixel_width > 0 && e.pixel_width != width)
            px_per_mm *= (double)width / e.pixel_width;
        double f = e.focal_mm * px_per_mm;
        // 焦距超出图像长边的 [0.1,100] 倍通常意味着标签误读，应放弃先验而非传给建图器。
        if (f > 0.1 * std::max(width, height) && f < 100.0 * std::max(width, height)) return f;
    }
    return 0;
}

// 相对曝光 EV 为 log2(t / N^2 * ISO)，直接标签优先于 APEX；缺项按 1 处理，三项全缺时返回 false。
inline bool exifExposureEv(const ExifData& e, double& ev) {
    if (!e.valid) return false;
    double t = e.exposure_time > 0 ? e.exposure_time
             : std::isfinite(e.shutter_apex) ? std::exp2(-e.shutter_apex) : 0;
    double N = e.f_number > 0 ? e.f_number
             : std::isfinite(e.aperture_apex) ? std::exp2(e.aperture_apex / 2) : 0;
    double s = e.iso;
    if (t <= 0 && N <= 0 && s <= 0) return false;
    double rel = (t > 0 ? t : 1) / ((N > 0 ? N : 1) * (N > 0 ? N : 1)) * (s > 0 ? s : 1);
    if (!(rel > 0) || !std::isfinite(rel)) return false;
    ev = std::log2(rel);
    return true;
}

// 相机身份由制造商、型号和尺寸组成，未知时为空且不据此分组。
// 焦距单独按容差聚类，避免 EXIF 整毫米量化将固定镜头的 24/25 mm 记录误拆为两个相机（D48）。
inline std::string exifCameraKey(const ExifData& e, int width, int height) {
    if (!e.valid || e.make.empty() || e.model.empty() || !e.hasFocal()) return {};
    char buf[32];
    snprintf(buf, sizeof buf, "-%dx%d", width, height);
    return e.make + "-" + e.model + buf;
}

}  // 命名空间 sfm
