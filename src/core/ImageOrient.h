#pragma once

// 主机交错图像的四分之一圈旋转与镜像，统一实现 ExifTransform 所描述的像素方向变换。

#include <cstddef>
#include <utility>

namespace spirula {

// w × h 图像顺时针旋转 turns_cw 个四分之一圈后的尺寸。
inline void oriented_size(int turns_cw, int& w, int& h) {
    if (turns_cw & 1) std::swap(w, h);
}

// 先顺时针旋转，再水平镜像，与 ExifTransform 一致；dst 容纳 w*h*channels 个元素且不能与 src 重叠。
template <typename T>
void orient_pixels(const T* src, int w, int h, int channels,
                   int turns_cw, bool mirror, T* dst) {
    const int t = turns_cw & 3;
    int dw = w, dh = h;
    oriented_size(t, dw, dh);
    const size_t c = (size_t)channels;
    for (int dy = 0; dy < dh; dy++) {
        for (int dx = 0; dx < dw; dx++) {
            const int mx = mirror ? dw - 1 - dx : dx;
            int sx = 0, sy = 0;
            switch (t) {
                case 1:  sx = dy;         sy = h - 1 - mx; break;
                case 2:  sx = w - 1 - mx; sy = h - 1 - dy; break;
                case 3:  sx = w - 1 - dy; sy = mx;         break;
                default: sx = mx;         sy = dy;         break;
            }
            const T* s = src + ((size_t)sy * w + sx) * c;
            T* d = dst + ((size_t)dy * dw + dx) * c;
            for (size_t k = 0; k < c; k++) d[k] = s[k];
        }
    }
}

}  // 命名空间 spirula
