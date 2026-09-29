// GPU DoG/SIFT 参考 VLFeat 与 COLMAP 默认：首 octave=-1，S=3，sigma0=1.6，峰阈值 0.02/3，边缘阈值 10，每点至多两方向，4×4×8 描述子并 RootSIFT 量化。
// 主机提供各层偏移和尺寸，设备使用平铺 gauss/dog 缓冲；仅用 float，无需 BA 标量变体。
// 编译期常量必须与 Sift.h 一致。

static const int   S            = 3;        // 每 octave 的尺度分辨率
static const int   GAUSS_PER_OCT = S + 3;   // 每 octave 六个高斯层
static const int   DOG_PER_OCT   = S + 2;   // 每 octave 五个 DoG 层
static const float SIGMA0       = 1.6;      // octave 第零层基础尺度
static const int   FIRST_OCTAVE = -1;       // -1 表示输入放大两倍
static const float PI2          = 6.2831853071795864;

static const int NBP  = 4;   // 描述子每轴空间桶数，4×4
static const int NBO  = 8;   // 描述子方向桶数
static const int NORI = 36;  // 方向分配的直方图桶数
static const float MAGNIF = 3.0;  // 描述子放大系数，桶宽=MAGNIF*sigma

// 原始关键点浮点布局：[x_oct,y_oct,octave,s_cont,sigma_oct,resp]
static const int KP_STRIDE  = 6;
// 定向关键点布局：[x_orig,y_orig,scale_orig,orientation,x_oct,y_oct,sigma_oct,octave]
static const int OKP_STRIDE = 8;

[[vk::push_constant]] cbuffer PC {
    uint p0; uint p1; uint p2; uint p3; uint p4; uint p5; uint p6; uint p7;
};

[[vk::binding(0)]]  StructuredBuffer<float>   img_in;   // 原始灰度，范围 [0,1]
[[vk::binding(1)]]  RWStructuredBuffer<float> gauss;    // 全部高斯层
[[vk::binding(2)]]  RWStructuredBuffer<float> dog;      // 全部 DoG 层
[[vk::binding(3)]]  RWStructuredBuffer<float> tmpH;     // 可分离模糊临时区
[[vk::binding(4)]]  StructuredBuffer<float>   weights;  // 拼接的模糊核
[[vk::binding(5)]]  StructuredBuffer<uint4>   glev;     // 每高斯层的（偏移，宽，高，保留）
[[vk::binding(6)]]  StructuredBuffer<uint4>   dlev;     // 每 DoG 层的（偏移，宽，高，octave）
[[vk::binding(7)]]  RWStructuredBuffer<float> kp;       // 原始关键点
[[vk::binding(8)]]  RWStructuredBuffer<uint>  kpcount;  // 第 0 项为原始关键点数
[[vk::binding(9)]]  RWStructuredBuffer<float> okp;      // 已分配方向的关键点
[[vk::binding(10)]] RWStructuredBuffer<uint>  okpcount; // 第 0 项为定向关键点数
[[vk::binding(11)]] RWStructuredBuffer<uint>  desc;     // 每保留关键点 32 个 uint，即 128 字节
[[vk::binding(12)]] RWStructuredBuffer<uint>  scalehist;// 用于 top-K 的 log2(scale) 直方图
[[vk::binding(13)]] RWStructuredBuffer<float> fokp;     // 压缩后的选中定向关键点
[[vk::binding(14)]] RWStructuredBuffer<uint>  selcount; // 第 0 项为选中数量

int clampi(int v, int lo, int hi) { return max(lo, min(hi, v)); }

// ================ 金字塔构建 ================

// 将原图双线性放大两倍到高斯层槽；p0=输出偏移，p1/p2=输入宽高，p3/p4=输出宽高。
[shader("compute")] [numthreads(16, 16, 1)]
void upsample(uint3 t : SV_DispatchThreadID) {
    uint ox = t.x, oy = t.y;
    if (ox >= p3 || oy >= p4) return;
    int inW = int(p1), inH = int(p2);
    // 目标像素中心反向映射到 (x-0.5)/2，按标准两倍上采样约定对齐。
    float fx = 0.5 * float(ox) - 0.25;
    float fy = 0.5 * float(oy) - 0.25;
    int x0 = int(floor(fx)), y0 = int(floor(fy));
    float wx = fx - float(x0), wy = fy - float(y0);
    int x0c = clampi(x0, 0, inW - 1), x1c = clampi(x0 + 1, 0, inW - 1);
    int y0c = clampi(y0, 0, inH - 1), y1c = clampi(y0 + 1, 0, inH - 1);
    float a = img_in[y0c * inW + x0c], b = img_in[y0c * inW + x1c];
    float c = img_in[y1c * inW + x0c], d = img_in[y1c * inW + x1c];
    float top = a + (b - a) * wx, bot = c + (d - c) * wx;
    gauss[p0 + oy * p3 + ox] = top + (bot - top) * wy;
}

// 水平高斯：gauss[srcOff]->tmpH，边界复制；p0=源偏移，p1/p2=宽高，p3=半径，p4=权重偏移。
[shader("compute")] [numthreads(16, 16, 1)]
void blur_h(uint3 t : SV_DispatchThreadID) {
    uint x = t.x, y = t.y;
    if (x >= p1 || y >= p2) return;
    int W = int(p1), r = int(p3);
    float acc = 0.0;
    for (int k = -r; k <= r; k++) {
        int xc = clampi(int(x) + k, 0, W - 1);
        acc += weights[p4 + uint(k + r)] * gauss[p0 + y * p1 + uint(xc)];
    }
    tmpH[y * p1 + x] = acc;
}

// 垂直高斯：tmpH->gauss[dstOff]；p0=目标偏移，p1/p2=宽高，p3=半径，p4=权重偏移。
[shader("compute")] [numthreads(16, 16, 1)]
void blur_v(uint3 t : SV_DispatchThreadID) {
    uint x = t.x, y = t.y;
    if (x >= p1 || y >= p2) return;
    int H = int(p2), r = int(p3);
    float acc = 0.0;
    for (int k = -r; k <= r; k++) {
        int yc = clampi(int(y) + k, 0, H - 1);
        acc += weights[p4 + uint(k + r)] * tmpH[uint(yc) * p1 + x];
    }
    gauss[p0 + y * p1 + x] = acc;
}

// 最近邻二倍抽取；p0/p1=源/目标偏移，p2=源宽，p3/p4=目标宽高。
[shader("compute")] [numthreads(16, 16, 1)]
void downsample(uint3 t : SV_DispatchThreadID) {
    uint x = t.x, y = t.y;
    if (x >= p3 || y >= p4) return;
    gauss[p1 + y * p3 + x] = gauss[p0 + (2 * y) * p2 + (2 * x)];
}

// DoG=gauss(o,d+1)-gauss(o,d)；p0/p1=低/高层偏移，p2=DoG 偏移，p3/p4=宽高。
[shader("compute")] [numthreads(16, 16, 1)]
void dog_diff(uint3 t : SV_DispatchThreadID) {
    uint x = t.x, y = t.y;
    if (x >= p3 || y >= p4) return;
    uint i = y * p3 + x;
    dog[p2 + i] = gauss[p1 + i] - gauss[p0 + i];
}

// ================ 极值检测与亚像素精化 ================

float dogAt(uint off, uint W, int x, int y) { return dog[off + uint(y) * W + uint(x)]; }

// octave 中 d∈[1,S] 的 DoG 每像素一线程；p0=oct，p1=d，p2=首层偏移，p3/p4=宽高，p5=容量，p6/p7=峰/边缘阈值位表示。
[shader("compute")] [numthreads(16, 16, 1)]
void extrema(uint3 t : SV_DispatchThreadID) {
    int W = int(p3), H = int(p4);
    int x = int(t.x), y = int(t.y);
    if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) return;
    float peakThresh = asfloat(p6);
    float edgeThresh = asfloat(p7);
    uint layer = p3 * p4;                 // 该 octave 每 DoG 层像素数
    uint offC = p2 + p1 * layer;          // dog(oct,d)
    uint offL = offC - layer;             // dog(oct,d-1)
    uint offU = offC + layer;             // dog(oct,d+1)

    float v = dogAt(offC, p3, x, y);
    if (abs(v) < 0.8 * peakThresh) return;

    // 在相邻三层的 26 个邻居中检测严格极值。
    bool isMax = true, isMin = true;
    [unroll] for (int dy = -1; dy <= 1; dy++)
        [unroll] for (int dx = -1; dx <= 1; dx++) {
            float n0 = dogAt(offL, p3, x + dx, y + dy);
            float n1 = dogAt(offC, p3, x + dx, y + dy);
            float n2 = dogAt(offU, p3, x + dx, y + dy);
            if (!(dx == 0 && dy == 0)) { if (n1 >= v) isMax = false; if (n1 <= v) isMin = false; }
            if (n0 >= v) isMax = false; if (n0 <= v) isMin = false;
            if (n2 >= v) isMax = false; if (n2 <= v) isMin = false;
        }
    if (!isMax && !isMin) return;

    // 三维二次拟合最多重定位五次，得到亚像素位置。
    int xi = x, yi = y, si = int(p1);
    float ox = 0, oy = 0, os = 0;
    float Dval = v;
    bool ok = false;
    for (int it = 0; it < 5; it++) {
        uint oC = p2 + uint(si) * layer, oL = oC - layer, oU = oC + layer;
        float c   = dogAt(oC, p3, xi, yi);
        float dx  = 0.5 * (dogAt(oC, p3, xi + 1, yi) - dogAt(oC, p3, xi - 1, yi));
        float dy  = 0.5 * (dogAt(oC, p3, xi, yi + 1) - dogAt(oC, p3, xi, yi - 1));
        float ds  = 0.5 * (dogAt(oU, p3, xi, yi)     - dogAt(oL, p3, xi, yi));
        float dxx = dogAt(oC, p3, xi + 1, yi) + dogAt(oC, p3, xi - 1, yi) - 2 * c;
        float dyy = dogAt(oC, p3, xi, yi + 1) + dogAt(oC, p3, xi, yi - 1) - 2 * c;
        float dss = dogAt(oU, p3, xi, yi)     + dogAt(oL, p3, xi, yi)     - 2 * c;
        float dxy = 0.25 * (dogAt(oC, p3, xi + 1, yi + 1) - dogAt(oC, p3, xi - 1, yi + 1)
                          - dogAt(oC, p3, xi + 1, yi - 1) + dogAt(oC, p3, xi - 1, yi - 1));
        float dxs = 0.25 * (dogAt(oU, p3, xi + 1, yi) - dogAt(oU, p3, xi - 1, yi)
                          - dogAt(oL, p3, xi + 1, yi) + dogAt(oL, p3, xi - 1, yi));
        float dys = 0.25 * (dogAt(oU, p3, xi, yi + 1) - dogAt(oU, p3, xi, yi - 1)
                          - dogAt(oL, p3, xi, yi + 1) + dogAt(oL, p3, xi, yi - 1));
        // 用伴随矩阵逆求 H*offset=-grad，尺寸 3×3。
        float A00 = dxx, A01 = dxy, A02 = dxs;
        float A11 = dyy, A12 = dys, A22 = dss;
        float c00 = A11 * A22 - A12 * A12;
        float c01 = A02 * A12 - A01 * A22;
        float c02 = A01 * A12 - A02 * A11;
        float det = A00 * c00 + A01 * c01 + A02 * c02;
        if (abs(det) < 1e-18) return;
        float idet = 1.0 / det;
        float c11 = A00 * A22 - A02 * A02;
        float c12 = A02 * A01 - A00 * A12;
        float c22 = A00 * A11 - A01 * A01;
        // offset = -Hinv * grad
        ox = -idet * (c00 * dx + c01 * dy + c02 * ds);
        oy = -idet * (c01 * dx + c11 * dy + c12 * ds);
        os = -idet * (c02 * dx + c12 * dy + c22 * ds);
        Dval = c + 0.5 * (dx * ox + dy * oy + ds * os);
        if (abs(ox) < 0.5 && abs(oy) < 0.5 && abs(os) < 0.5) { ok = true; break; }
        xi += int(round(clamp(ox, -1.0, 1.0)));
        yi += int(round(clamp(oy, -1.0, 1.0)));
        si += int(round(clamp(os, -1.0, 1.0)));
        if (xi < 1 || yi < 1 || xi >= W - 1 || yi >= H - 1 || si < 1 || si > S) return;
    }
    if (!ok) return;
    if (abs(Dval) < peakThresh) return;

    // 用精化位置的 2×2 空间 Hessian 排除边缘响应。
    {
        uint oC = p2 + uint(si) * layer;
        float c   = dogAt(oC, p3, xi, yi);
        float dxx = dogAt(oC, p3, xi + 1, yi) + dogAt(oC, p3, xi - 1, yi) - 2 * c;
        float dyy = dogAt(oC, p3, xi, yi + 1) + dogAt(oC, p3, xi, yi - 1) - 2 * c;
        float dxy = 0.25 * (dogAt(oC, p3, xi + 1, yi + 1) - dogAt(oC, p3, xi - 1, yi + 1)
                          - dogAt(oC, p3, xi + 1, yi - 1) + dogAt(oC, p3, xi - 1, yi - 1));
        float tr = dxx + dyy, det = dxx * dyy - dxy * dxy;
        if (det <= 0) return;
        float r = edgeThresh;
        if (tr * tr * r >= (r + 1.0) * (r + 1.0) * det) return;
    }

    float s_cont = float(si) + os;
    uint slot;
    InterlockedAdd(kpcount[0], 1u, slot);
    if (slot >= p5) return;
    uint b = slot * uint(KP_STRIDE);
    kp[b + 0] = float(xi) + ox;
    kp[b + 1] = float(yi) + oy;
    kp[b + 2] = float(p0);
    kp[b + 3] = s_cont;
    kp[b + 4] = SIGMA0 * exp2(s_cont / float(S));
    kp[b + 5] = abs(Dval);
}

// ================ 方向与描述子共用梯度采样 ================

// 高斯层中心差分梯度，返回幅值和 [0,2pi) 方向。
void gradAt(uint off, int W, int H, int x, int y, out float mag, out float ang) {
    int xa = clampi(x + 1, 0, W - 1), xb = clampi(x - 1, 0, W - 1);
    int ya = clampi(y + 1, 0, H - 1), yb = clampi(y - 1, 0, H - 1);
    float gx = 0.5 * (gauss[off + uint(y) * W + xa] - gauss[off + uint(y) * W + xb]);
    float gy = 0.5 * (gauss[off + uint(ya) * W + x] - gauss[off + uint(yb) * W + x]);
    mag = sqrt(gx * gx + gy * gy);
    float a = atan2(gy, gx);
    if (a < 0) a += PI2;
    ang = a;
}

// 为连续尺度 s_cont 选择当前 octave 内最近高斯层。
uint gaussLevelIndex(uint oct, float s_cont) {
    int s = clampi(int(round(s_cont)), 0, GAUSS_PER_OCT - 1);
    return oct * uint(GAUSS_PER_OCT) + uint(s);
}

// ================ 方向分配 ================

// [p3,p0) 每原始关键点一线程，至多输出 p1 个方向；p0=结束，p1=方向上限，p2=容量，p3=开始。
[shader("compute")] [numthreads(64, 1, 1)]
void orient(uint3 t : SV_DispatchThreadID) {
    uint ki = t.x + p3;
    if (ki >= p0) return;
    uint b = ki * uint(KP_STRIDE);
    float x_oct = kp[b + 0], y_oct = kp[b + 1];
    uint oct = uint(kp[b + 2]);
    float s_cont = kp[b + 3], sigma_oct = kp[b + 4];

    uint gi = gaussLevelIndex(oct, s_cont);
    uint4 L = glev[gi];
    uint off = L.x; int W = int(L.y), H = int(L.z);

    float sigmaw = 1.5 * sigma_oct;
    int R = int(floor(3.0 * sigmaw));
    int xi = int(round(x_oct)), yi = int(round(y_oct));

    float hist[NORI];
    [unroll] for (int i = 0; i < NORI; i++) hist[i] = 0.0;
    float inv2s2 = 1.0 / (2.0 * sigmaw * sigmaw);
    for (int dy = -R; dy <= R; dy++)
        for (int dx = -R; dx <= R; dx++) {
            int px = xi + dx, py = yi + dy;
            if (px < 0 || py < 0 || px >= W || py >= H) continue;
            float r2 = float(dx * dx + dy * dy);
            if (r2 > float(R * R) + 0.5) continue;
            float mag, ang;
            gradAt(off, W, H, px, py, mag, ang);
            float w = exp(-r2 * inv2s2) * mag;
            int bin = int(floor(ang / PI2 * float(NORI))) % NORI;
            hist[bin] += w;
        }

    // 循环三点均值平滑直方图六次，与 VLFeat 一致。
    for (int pass = 0; pass < 6; pass++) {
        float prev = hist[NORI - 1];
        float first = hist[0];
        [unroll] for (int i = 0; i < NORI; i++) {
            float next = (i + 1 < NORI) ? hist[i + 1] : first;
            float cur = hist[i];
            hist[i] = (prev + cur + next) / 3.0;
            prev = cur;
        }
    }

    float maxh = 0.0;
    [unroll] for (int i = 0; i < NORI; i++) maxh = max(maxh, hist[i]);
    if (maxh <= 0.0) return;
    float thresh = 0.8 * maxh;

    // 峰值做抛物线插值并保留最强 p1 个；仅前 nbest 项被读，仍清零以避免编译器未初始化警告。
    float bestVal[4] = {0.0, 0.0, 0.0, 0.0};
    float bestAng[4] = {0.0, 0.0, 0.0, 0.0};
    int nbest = 0;
    int cap = int(min(p1, 4u));
    for (int i = 0; i < NORI; i++) {
        float h0 = hist[(i + NORI - 1) % NORI], h1 = hist[i], h2 = hist[(i + 1) % NORI];
        if (h1 <= h0 || h1 <= h2 || h1 < thresh) continue;
        float di = 0.5 * (h0 - h2) / (h0 - 2.0 * h1 + h2);
        float ang = PI2 * (float(i) + di) / float(NORI);
        if (ang < 0) ang += PI2; if (ang >= PI2) ang -= PI2;
        // 按峰高插入小型 top-k 列表
        int pos = nbest;
        while (pos > 0 && bestVal[pos - 1] < h1) {
            if (pos < cap) { bestVal[pos] = bestVal[pos - 1]; bestAng[pos] = bestAng[pos - 1]; }
            pos--;
        }
        if (pos < cap) {
            bestVal[pos] = h1; bestAng[pos] = ang;
            if (nbest < cap) nbest++;
        }
    }

    float octSize = exp2(float(int(oct) + FIRST_OCTAVE));
    for (int i = 0; i < nbest; i++) {
        uint slot;
        InterlockedAdd(okpcount[0], 1u, slot);
        if (slot >= p2) return;
        uint ob = slot * uint(OKP_STRIDE);
        okp[ob + 0] = x_oct * octSize;
        okp[ob + 1] = y_oct * octSize;
        okp[ob + 2] = sigma_oct * octSize;
        okp[ob + 3] = bestAng[i];
        okp[ob + 4] = x_oct;
        okp[ob + 5] = y_oct;
        okp[ob + 6] = sigma_oct;
        okp[ob + 7] = float(oct);
    }
}

// ================ 按尺度选择 top-K ================
// log2(scale) 分桶后由主机选近似阈值，压缩到 fokp，仅为保留点运行昂贵描述子；主机最后精确截到 K。

// 每定向点一线程累加尺度直方图；p0=点数，p1=桶数，p2/p3=最小对数尺度/逆范围的位表示。
[shader("compute")] [numthreads(64, 1, 1)]
void scale_hist(uint3 t : SV_DispatchThreadID) {
    uint i = t.x;
    if (i >= p0) return;
    float scale = okp[i * uint(OKP_STRIDE) + 2];
    int b = int(floor((log2(max(scale, 1e-8)) - asfloat(p2)) * asfloat(p3) * float(p1)));
    b = clampi(b, 0, int(p1) - 1);
    uint old;
    InterlockedAdd(scalehist[uint(b)], 1u, old);
}

// 将 scale>=threshold 的点压缩到 fokp；p0=点数，p1=阈值位表示，p2=容量。
[shader("compute")] [numthreads(64, 1, 1)]
void select_topk(uint3 t : SV_DispatchThreadID) {
    uint i = t.x;
    if (i >= p0) return;
    uint ib = i * uint(OKP_STRIDE);
    if (okp[ib + 2] < asfloat(p1)) return;
    uint slot;
    InterlockedAdd(selcount[0], 1u, slot);
    if (slot >= p2) return;
    uint ob = slot * uint(OKP_STRIDE);
    [unroll] for (int k = 0; k < OKP_STRIDE; k++) fokp[ob + k] = okp[ib + k];
}

// ================ 128 维描述子 ================

// [p1,p0) 每个选中关键点一线程。
[shader("compute")] [numthreads(64, 1, 1)]
void descriptor(uint3 t : SV_DispatchThreadID) {
    uint ki = t.x + p1;
    if (ki >= p0) return;
    uint ob = ki * uint(OKP_STRIDE);
    float theta = fokp[ob + 3];
    float x_oct = fokp[ob + 4], y_oct = fokp[ob + 5], sigma_oct = fokp[ob + 6];
    uint oct = uint(fokp[ob + 7]);

    float s_cont = SIGMA0;  // 由 sigma_oct=SIGMA0*2^(s_cont/S) 恢复 s_cont=S*log2(sigma_oct/SIGMA0)，用于选层。
    s_cont = float(S) * log2(sigma_oct / SIGMA0);
    uint gi = gaussLevelIndex(oct, s_cont);
    uint4 L = glev[gi];
    uint off = L.x; int W = int(L.y), H = int(L.z);

    float binSize = MAGNIF * sigma_oct;
    // 支持半径覆盖 NBP+1 桶，旋转引入 sqrt(2) 系数，再加 0.5 取整。
    int R = int(floor(1.41421356 * binSize * float(NBP + 1) * 0.5 + 0.5));
    float ct = cos(theta), st = sin(theta);
    int xi = int(round(x_oct)), yi = int(round(y_oct));

    float d[128];
    [unroll] for (int i = 0; i < 128; i++) d[i] = 0.0;
    float sigScale = float(NBP) * 0.5;
    float inv2s2 = 1.0 / (2.0 * sigScale * sigScale);

    for (int dy = -R; dy <= R; dy++)
        for (int dx = -R; dx <= R; dx++) {
            int px = xi + dx, py = yi + dy;
            if (px < 0 || py < 0 || px >= W || py >= H) continue;
            // 将采样偏移旋到描述子坐标系，以桶宽为单位。
            float rx = (ct * float(dx) + st * float(dy)) / binSize;
            float ry = (-st * float(dx) + ct * float(dy)) / binSize;
            // 连续桶坐标居中，使桶中心编号为 0..NBP-1
            float bx = rx + float(NBP) * 0.5 - 0.5;
            float by = ry + float(NBP) * 0.5 - 0.5;
            if (bx <= -1.0 || bx >= float(NBP) || by <= -1.0 || by >= float(NBP)) continue;

            float mag, ang;
            gradAt(off, W, H, px, py, mag, ang);
            float w = exp(-(rx * rx + ry * ry) * inv2s2) * mag;
            float ro = ang - theta;
            while (ro < 0) ro += PI2;
            while (ro >= PI2) ro -= PI2;
            float ob_c = ro / PI2 * float(NBO);

            int bx0 = int(floor(bx)), by0 = int(floor(by)), ob0 = int(floor(ob_c));
            float fx = bx - float(bx0), fy = by - float(by0), fo = ob_c - float(ob0);
            // 向 2×2×2 邻域做三线性散射
            [unroll] for (int iy = 0; iy <= 1; iy++) {
                int yb = by0 + iy;
                if (yb < 0 || yb >= NBP) continue;
                float wy = (iy == 0) ? (1.0 - fy) : fy;
                [unroll] for (int ix = 0; ix <= 1; ix++) {
                    int xb = bx0 + ix;
                    if (xb < 0 || xb >= NBP) continue;
                    float wx = (ix == 0) ? (1.0 - fx) : fx;
                    [unroll] for (int io = 0; io <= 1; io++) {
                        int obn = (ob0 + io) % NBO;
                        float wo = (io == 0) ? (1.0 - fo) : fo;
                        d[(yb * NBP + xb) * NBO + obn] += w * wx * wy * wo;
                    }
                }
            }
        }

    // 先 L2 归一化、截断到 0.2*||d||、再 L2，随后 RootSIFT 的 L1+sqrt，最后 round(512*x) 并钳位到 255。
    float norm = 0.0;
    [unroll] for (int i = 0; i < 128; i++) norm += d[i] * d[i];
    norm = sqrt(norm) + 1e-12;
    float clip = 0.2 * norm;
    float norm2 = 0.0;
    [unroll] for (int i = 0; i < 128; i++) { d[i] = min(d[i], clip); norm2 += d[i] * d[i]; }
    norm2 = sqrt(norm2) + 1e-12;
    float l1 = 0.0;
    [unroll] for (int i = 0; i < 128; i++) { d[i] /= norm2; l1 += d[i]; }
    l1 += 1e-12;
    uint base = ki * 32u;
    [unroll] for (int j = 0; j < 32; j++) {
        uint packed = 0;
        [unroll] for (int k = 0; k < 4; k++) {
            float v = sqrt(d[j * 4 + k] / l1);
            int q = int(round(512.0 * v));
            q = clampi(q, 0, 255);
            packed |= uint(q) << (8 * k);
        }
        desc[base + j] = packed;
    }
}
