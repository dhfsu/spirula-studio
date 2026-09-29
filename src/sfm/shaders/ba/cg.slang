// 隐式 Schur PCG 不生成 S，通过逐观测 Jc/Jp 与逐点 W 计算 Sx=Bx-sum_p Acp_p W_p(Acp_p^T x)。
// 共享列采用原子累加，u1 区分独占/共享；预条件器为块 Jacobi。标量状态存于 u_cg_scal，收敛后各内核空操作，解累积到 u_g。

[[vk::binding(29)]] StructuredBuffer<uint> u_cam_ranges;  // num_images+1 个按图像分组的 CSR 偏移
[[vk::binding(30)]] StructuredBuffer<uint> u_cam_obs;     // 逐图像观测索引
[[vk::binding(31)]] RWStructuredBuffer<Real> u_cg_r;      // n_dim 维残差
[[vk::binding(32)]] RWStructuredBuffer<Real> u_cg_z;      // n_dim 维 M^-1 r
[[vk::binding(33)]] RWStructuredBuffer<Real> u_cg_p;      // n_dim 维搜索方向
[[vk::binding(34)]] RWStructuredBuffer<Real> u_cg_sp;     // n_dim 维 S p
[[vk::binding(35)]] RWStructuredBuffer<Real> u_cg_v;      // 每点 3 项 W(Acp^T p)
[[vk::binding(36)]] RWStructuredBuffer<Real> u_cg_B;      // 逐相机压缩存储的带阻尼 Jc^T Jc
[[vk::binding(37)]] RWStructuredBuffer<Real> u_cg_M;      // 逐相机压缩 S_cc，原地转为 L 因子
[[vk::binding(38)]] RWStructuredBuffer<Real> u_cg_scal;   // 由 kCg* 索引的标量状态
[[vk::binding(39)]] RWStructuredBuffer<Real> u_cg_part;   // 点积局部和，共两个槽
[[vk::binding(40)]] RWStructuredBuffer<AWord> u_cg_sp_a;  // 分块累加使用的原子视图
[[vk::binding(41)]] RWStructuredBuffer<AWord> u_cg_B_a;
[[vk::binding(42)]] RWStructuredBuffer<AWord> u_cg_M_a;
[[vk::binding(43)]] StructuredBuffer<uint> u_cam_chunks;  // 每块 3 项：图像、起点、数量
[[vk::binding(44)]] StructuredBuffer<uint4> u_prec_blocks;  // 每块 (col0,len0,col1,len1)

static const uint kCgRho = 0;    // r^T z
static const uint kCgPAp = 1;    // p^T S p
static const uint kCgAlpha = 2;
static const uint kCgBeta = 3;
static const uint kCgRR = 4;     // ||r||^2
static const uint kCgTol2 = 5;   // eta^2 ||r0||^2
static const uint kCgConv = 6;   // 大于 0.5 后循环内核不再执行
static const uint kCgIters = 7;  // 已完成 CG 迭代数
static const uint kCgQ = 8;      // x 处二次模型 1/2 x^T S x-g^T x
static const uint kCgDQ = 9;     // 上次迭代的二次模型变化量

// 下三角块步长，预条件器宽度至多 kPrecDof，逐图像 B 块宽度按档位决定。
static const uint kCamBlk = kPrecDof * (kPrecDof + 1) / 2;
uint bblk<let MD : int>() { return uint(MD * (MD + 1) / 2); }

bool cgDone() { return u_cg_scal[kCgConv] > Real(0.5); }

// 预条件块 b 的全局列，由两个连续区间表示
void precCols(uint b, out uint dof, out uint cols[kPrecDof]) {
    uint4 blk = u_prec_blocks[b];
    dof = blk.y + blk.w;
    for (uint i = 0; i < blk.y; i++) cols[i] = blk.x + i;
    for (uint i = 0; i < blk.w; i++) cols[blk.y + i] = blk.z + i;
}

// ---------------- 每阻尼值装配 B_c、M_c=S_cc 与约化右端 ----------------

// 每 warp 处理相机观测的一块，累加 B_c、所属预条件块 M_c 与 g_c，忽略跨预条件块项。
// u0=块数，u1=独占标记，u2/u3=成员/组块起点。
static const uint kShcJ = 0;                    // Jc    2 x dof
static const uint kShcZ = 2 * kDofRig;          // Z = Q^T Jc   2 x dof
static const uint kShcQ = 4 * kDofRig;          // Q = Jp W Jp^T   2 x 2
static const uint kShcY = 4 * kDofRig + 4;      // Y = (Jp W)^T    2 x 3
static const uint kShcR = 4 * kDofRig + 10;     // 残差，2 项
static const uint kShcB = 4 * kDofRig + 12;     // Bp，3 项
static const uint kShcP = 4 * kDofRig + 15;     // Jp，2×3

void cgCamDiagImpl<let MD : int>(uint wgx, uint gi) {
    const uint kEl = (MD * MD + kPairWg - 1) / kPairWg;
    if (wgx >= u0) return;
    uint img = u_cam_chunks[3 * wgx];
    uint t0 = u_cam_chunks[3 * wgx + 1];
    uint t1 = t0 + u_cam_chunks[3 * wgx + 2];
    uint dof;
    uint cols[MD];
    camCols<MD>(img, dof, cols);
    uint nel = dof * dof;
    uint4 ii = u_image_info[img];
    uint mfree = ii.y != kNoMember ? u_member_info[ii.y].z : 0;

    Real accB[kEl], accM[kEl];
    uint elr[kEl], elc[kEl];
    [ForceUnroll]
    for (uint k = 0; k < kEl; k++) {
        accB[k] = Real(0.0);
        accM[k] = Real(0.0);
        uint el = gi + kPairWg * k;
        elr[k] = el / dof;
        elc[k] = el - elr[k] * dof;
    }
    Real gacc = Real(0.0);

    for (uint t = t0; t < t1; t++) {
        uint o = u_cam_obs[t];
        uint pt = u_obs_point[o];
        uint jb = u_jc_off[o];
        GroupMemoryBarrierWithGroupSync();  // 上一观测已读取完毕
        for (uint i = gi; i < 2 * dof; i += kPairWg) g_sh[kShcJ + i] = u_Jc[jb + i];
        if (gi < 6) g_sh[kShcP + gi] = u_Jp[6 * o + gi];
        if (gi < 2) g_sh[kShcR + gi] = u_res[2 * o + gi];
        if (gi < 3) g_sh[kShcB + gi] = u_Bp[3 * pt + gi];
        // 先算 Y=(Jp W)^T，再算 Q=Jp W Jp^T，用 2×2 中间矩阵替代每观测 dof×3 的 Acp W。
        if (gi < 6) {
            uint j = gi >> 1, row = gi & 1;
            g_sh[kShcY + gi] = u_Jp[6 * o + row] * u_W[9 * pt + j]
                             + u_Jp[6 * o + 2 + row] * u_W[9 * pt + 3 + j]
                             + u_Jp[6 * o + 4 + row] * u_W[9 * pt + 6 + j];
        }
        GroupMemoryBarrierWithGroupSync();
        if (gi < 4) {
            uint qr = gi >> 1, qs = gi & 1;
            g_sh[kShcQ + gi] = g_sh[kShcY + qr] * g_sh[kShcP + qs]
                             + g_sh[kShcY + 2 + qr] * g_sh[kShcP + 2 + qs]
                             + g_sh[kShcY + 4 + qr] * g_sh[kShcP + 4 + qs];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint i = gi; i < 2 * dof; i += kPairWg) {
            uint a = i >> 1, s = i & 1;
            g_sh[kShcZ + i] = g_sh[kShcJ + 2 * a] * g_sh[kShcQ + s]
                            + g_sh[kShcJ + 2 * a + 1] * g_sh[kShcQ + 2 + s];
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint k = 0; k < kEl; k++) {
            uint el = gi + kPairWg * k;
            if (el >= nel) continue;
            uint r = elr[k], c = elc[k];
            if (r < c) continue;  // 对称块仅计算下三角
            accM[k] = accM[k] + g_sh[kShcZ + 2 * r] * g_sh[kShcJ + 2 * c]
                              + g_sh[kShcZ + 2 * r + 1] * g_sh[kShcJ + 2 * c + 1];
            accB[k] = accB[k] + g_sh[kShcJ + 2 * r] * g_sh[kShcJ + 2 * c]
                              + g_sh[kShcJ + 2 * r + 1] * g_sh[kShcJ + 2 * c + 1];
        }
        if (gi < dof) {
            Real yb0 = g_sh[kShcY] * g_sh[kShcB] + g_sh[kShcY + 2] * g_sh[kShcB + 1]
                     + g_sh[kShcY + 4] * g_sh[kShcB + 2];
            Real yb1 = g_sh[kShcY + 1] * g_sh[kShcB] + g_sh[kShcY + 3] * g_sh[kShcB + 1]
                     + g_sh[kShcY + 5] * g_sh[kShcB + 2];
            gacc = gacc + g_sh[kShcJ + 2 * gi] * (yb0 - g_sh[kShcR])
                        + g_sh[kShcJ + 2 * gi + 1] * (yb1 - g_sh[kShcR + 1]);
        }
    }

    uint bb = bblk<MD>() * img;
    // 共享布局按行确定所属块和局部行：帧 [0,6)，成员 [6,6+mfree)，其余为内参。
    uint fb = kCamBlk * ii.x;
    uint mb = kCamBlk * (u2 + ii.y);
    uint gb = kCamBlk * (u3 + ii.z);
    [ForceUnroll]
    for (uint k = 0; k < kEl; k++) {
        uint el = gi + kPairWg * k;
        if (el >= nel) continue;
        uint r = elr[k], c = elc[k];
        if (r < c) continue;
        Real jj = accB[k];
        if (r == c) jj = jj * (Real(1.0) + Real(f0));  // 乘法阻尼
        Real m = jj - accM[k];
        if (isfinite(jj)) atomicAccum(u_cg_B_a, bb + pidx(r, c), jj);
        if (!isfinite(m)) continue;
        if (u1 != 0) {
            atomicAccum(u_cg_M_a, kCamBlk * img + pidx(r, c), m);
        } else if (r < 6) {
            atomicAccum(u_cg_M_a, fb + pidx(r, c), m);
        } else if (r < 6 + mfree) {
            if (c >= 6) atomicAccum(u_cg_M_a, mb + pidx(r - 6, c - 6), m);
        } else if (c >= 6 + mfree) {
            atomicAccum(u_cg_M_a, gb + pidx(r - 6 - mfree, c - 6 - mfree), m);
        }
    }
    if (gi < dof && isfinite(gacc))
        atomicAccum(u_g_a, cols[gi], -gacc);
}

// 逐块原地 Cholesky；若主元低于最大对角的 1e-10 则设下限并解除耦合，防止巨大 Gram 项消减噪声破坏分解。u0=块数。
[shader("compute")] [numthreads(256, 1, 1)]
void cg_prec_fact(uint3 t : SV_DispatchThreadID) {
    uint b = t.x;
    if (b >= u0) return;
    uint4 blk = u_prec_blocks[b];
    uint dof = blk.y + blk.w;
    if (dof == 0) return;
    uint cb = kCamBlk * b;
    Real L[kCamBlk];
    for (uint i = 0; i < dof * (dof + 1) / 2; i++) L[i] = u_cg_M[cb + i];
    Real f = Real(1e-30);
    for (uint j = 0; j < dof; j++)
        if (L[pidx(j, j)] > f) f = L[pidx(j, j)];
    f = max(Real(1e-10) * f, Real(1e-30));
    for (uint j = 0; j < dof; j++) {
        bool ok = L[pidx(j, j)] > f;
        Real d = sqrt(ok ? L[pidx(j, j)] : f);
        L[pidx(j, j)] = d;
        for (uint i = j + 1; i < dof; i++) L[pidx(i, j)] = ok ? L[pidx(i, j)] / d : Real(0.0);
        for (uint c = j + 1; c < dof; c++)
            for (uint i = c; i < dof; i++)
                L[pidx(i, c)] = L[pidx(i, c)] - L[pidx(i, j)] * L[pidx(c, j)];
    }
    for (uint i = 0; i < dof * (dof + 1) / 2; i++) u_cg_M[cb + i] = L[i];
}

// z=M^-1 r，每块一线程执行小矩阵前后代；u0=块数，u1=1 时遵守收敛标志。
[shader("compute")] [numthreads(256, 1, 1)]
void cg_prec_apply(uint3 t : SV_DispatchThreadID) {
    if (u1 != 0 && cgDone()) return;
    uint b = t.x;
    if (b >= u0) return;
    uint dof;
    uint cols[kPrecDof];
    precCols(b, dof, cols);
    if (dof == 0) return;
    uint cb = kCamBlk * b;
    // i 以下元素均先写后读，仍显式清零使 slangc 能识别初始化。
    Real y[kPrecDof];
    for (uint i = 0; i < kPrecDof; i++) y[i] = Real(0);
    for (uint i = 0; i < dof; i++) {
        Real v = u_cg_r[cols[i]];
        for (uint j = 0; j < i; j++) v = v - u_cg_M[cb + pidx(i, j)] * y[j];
        y[i] = v / u_cg_M[cb + pidx(i, i)];
    }
    for (int i = int(dof) - 1; i >= 0; i--) {
        uint ui = uint(i);
        Real v = y[ui];
        for (uint j = ui + 1; j < dof; j++) v = v - u_cg_M[cb + pidx(j, ui)] * y[j];
        y[ui] = v / u_cg_M[cb + pidx(ui, ui)];
    }
    for (uint i = 0; i < dof; i++) u_cg_z[cols[i]] = y[i];
}

// ---------------- CG 循环内核 ----------------

// r=g，u_g 中的 x 清零并重置标量；u0=n_dim。
[shader("compute")] [numthreads(256, 1, 1)]
void cg_init(uint3 t : SV_DispatchThreadID) {
    uint i = t.x;
    if (i < 8) u_cg_scal[i] = Real(0.0);
    if (i >= u0) return;
    u_cg_r[i] = u_g[i];
    u_g[i] = Real(0.0);
}

// p=z，u0=n_dim。
[shader("compute")] [numthreads(256, 1, 1)]
void cg_copy(uint3 t : SV_DispatchThreadID) {
    uint i = t.x;
    if (i >= u0) return;
    u_cg_p[i] = u_cg_z[i];
}

// 每点一线程沿轨迹计算 v_p=W_p(Acp_p^T p)，成本随轨迹线性；u0=点数。
void cgGatherImpl<let MD : int>(uint p) {
    if (cgDone()) return;
    if (p >= u0) return;
    Real u0v = Real(0.0), u1v = Real(0.0), u2v = Real(0.0);
    uint t0 = u_obs_ranges[p], t1 = u_obs_ranges[p + 1];
    for (uint oi = t0; oi < t1; oi++) {
        uint dof, base;
        uint cols[MD];
        obsCamCols<MD>(oi, dof, base, cols);
        // Acp^T p=Jp^T(Jc p)，先收缩二维残差，仅两个累加器跨自由度循环。
        Real d0 = Real(0.0), d1 = Real(0.0);
        for (uint a = 0; a < dof; a++) {
            Real xa = u_cg_p[cols[a]];
            d0 = d0 + u_Jc[base + 2 * a] * xa;
            d1 = d1 + u_Jc[base + 2 * a + 1] * xa;
        }
        u0v = u0v + u_Jp[6 * oi] * d0 + u_Jp[6 * oi + 1] * d1;
        u1v = u1v + u_Jp[6 * oi + 2] * d0 + u_Jp[6 * oi + 3] * d1;
        u2v = u2v + u_Jp[6 * oi + 4] * d0 + u_Jp[6 * oi + 5] * d1;
    }
    for (int i = 0; i < 3; i++)
        u_cg_v[3 * p + i] = u_W[9 * p + 3 * i] * u0v + u_W[9 * p + 3 * i + 1] * u1v
                          + u_W[9 * p + 3 * i + 2] * u2v;
}

// 每相机 warp 计算 Sp_c=B_c p_c，后续再减交叉项；独占帧行直接写，共享行累加到预先清零的切片。u0=图像数，u1=独占标记。
void cgBmulImpl<let MD : int>(uint img, uint gi) {
    if (cgDone()) return;
    if (img >= u0) return;
    uint dof;
    uint cols[MD];
    camCols<MD>(img, dof, cols);
    if (gi >= dof) return;
    uint cb = bblk<MD>() * img;
    Real acc = Real(0.0);
    for (uint b = 0; b < dof; b++) {
        uint idx = gi >= b ? pidx(gi, b) : pidx(b, gi);
        acc = acc + u_cg_B[cb + idx] * u_cg_p[cols[b]];
    }
    bool own = u1 != 0 || (gi < 6 && (u_image_info[img].w & 1u) == 0);
    if (own) storeAccum(u_cg_sp_a, cols[gi], acc);
    else if (isfinite(acc)) atomicAccum(u_cg_sp_a, cols[gi], acc);
}

// 每观测块 warp 计算 Sp_c-=sum Jc^T(Jp v)，各 lane 固定读取雅可比元素，成对归约后每自由度一次原子加。
// 二维 Jp v 由各 lane 重算，省去屏障；u0=块数。
void cgScatterImpl<let MD : int>(uint wgx, uint gi) {
    if (cgDone()) return;
    if (wgx >= u0) return;
    uint img = u_cam_chunks[3 * wgx];
    uint t0 = u_cam_chunks[3 * wgx + 1];
    uint t1 = t0 + u_cam_chunks[3 * wgx + 2];
    uint dof;
    uint cols[MD];
    camCols<MD>(img, dof, cols);

    // 2*dof 可达 48，超过 warp 宽度，因此每 lane 最多负责两个元素
    uint gi2 = gi + kPairWg;
    uint row = gi & 1, row2 = gi2 & 1;
    Real pacc = Real(0.0), pacc2 = Real(0.0);
    bool live = gi < 2 * dof, live2 = gi2 < 2 * dof;
    for (uint t = t0; t < t1; t++) {
        uint o = u_cam_obs[t];
        uint pt = u_obs_point[o];
        uint base = u_jc_off[o];
        Real d0 = u_Jp[6 * o] * u_cg_v[3 * pt] + u_Jp[6 * o + 2] * u_cg_v[3 * pt + 1]
                + u_Jp[6 * o + 4] * u_cg_v[3 * pt + 2];
        Real d1 = u_Jp[6 * o + 1] * u_cg_v[3 * pt] + u_Jp[6 * o + 3] * u_cg_v[3 * pt + 1]
                + u_Jp[6 * o + 5] * u_cg_v[3 * pt + 2];
        if (live)
            pacc = pacc + u_Jc[base + gi] * (row == 0 ? d0 : d1);
        if (live2)
            pacc2 = pacc2 + u_Jc[base + gi2] * (row2 == 0 ? d0 : d1);
    }
    g_sh[gi] = pacc;
    g_sh[gi2] = pacc2;
    GroupMemoryBarrierWithGroupSync();
    if (gi < dof) {
        Real v = g_sh[2 * gi] + g_sh[2 * gi + 1];
        if (isfinite(v))
            atomicAccum(u_cg_sp_a, cols[gi], -v);
    }
}

// 无自由成员外参时用 wide 档，其余用 rig 档。
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void cg_cam_diag_w(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { cgCamDiagImpl<kDofWide>(wg.x + u4, gi); }
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void cg_cam_diag_x(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { cgCamDiagImpl<kDofRig>(wg.x + u4, gi); }
[shader("compute")] [numthreads(256, 1, 1)]
void cg_gather_w(uint3 t : SV_DispatchThreadID) { cgGatherImpl<kDofWide>(t.x + u4); }
[shader("compute")] [numthreads(256, 1, 1)]
void cg_gather_x(uint3 t : SV_DispatchThreadID) { cgGatherImpl<kDofRig>(t.x + u4); }
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void cg_bmul_w(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { cgBmulImpl<kDofWide>(wg.x, gi); }
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void cg_bmul_x(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { cgBmulImpl<kDofRig>(wg.x, gi); }
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void cg_scatter_w(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { cgScatterImpl<kDofWide>(wg.x + u4, gi); }
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void cg_scatter_x(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { cgScatterImpl<kDofRig>(wg.x + u4, gi); }

// 逐工作组产生点积局部和；u1=0 计算 p·Sp，u1=1 计算 r·z 与 r·r。u0=维度，u2=局部和步长，u3 控制收敛跳过。
[shader("compute")] [numthreads(256, 1, 1)]
void cg_red2(uint3 t : SV_DispatchThreadID, uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) {
    if (u3 != 0 && cgDone()) return;
    uint i = t.x;
    Real a = Real(0.0), b = Real(0.0);
    if (i < u0) {
        if (u1 == 0) {
            a = u_cg_p[i] * u_cg_sp[i];
        } else {
            a = u_cg_r[i] * u_cg_z[i];
            b = u_cg_r[i] * u_cg_r[i];
        }
    }
    g_sh[gi] = a;
    g_sh[256 + gi] = b;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) {
            g_sh[gi] = g_sh[gi] + g_sh[gi + s];
            g_sh[256 + gi] = g_sh[256 + gi] + g_sh[256 + gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) {
        u_cg_part[wg.x] = g_sh[0];
        if (u1 != 0) u_cg_part[u2 + wg.x] = g_sh[256];
    }
}

// 单工作组归约并更新标量；u1 模式 0 初始化、1 更新 alpha、2 更新 beta/收敛，f0 为残差容差，f1 为 Nash-Sofer 容差，0 禁用。
[shader("compute")] [numthreads(256, 1, 1)]
void cg_fin(uint gi : SV_GroupIndex) {
    if (u1 != 0 && cgDone()) return;
    Real a = Real(0.0), b = Real(0.0);
    for (uint i = gi; i < u0; i += 256) {
        a = a + u_cg_part[i];
        if (u1 != 1) b = b + u_cg_part[u2 + i];
    }
    g_sh[gi] = a;
    g_sh[256 + gi] = b;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) {
            g_sh[gi] = g_sh[gi] + g_sh[gi + s];
            g_sh[256 + gi] = g_sh[256 + gi] + g_sh[256 + gi + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi != 0) return;
    Real A = g_sh[0], B = g_sh[256];
    if (u1 == 0) {  // 初始残差的 A=r·z，B=||r||²
        u_cg_scal[kCgRho] = A;
        u_cg_scal[kCgRR] = B;
        u_cg_scal[kCgTol2] = Real(f0) * Real(f0) * B;
        u_cg_scal[kCgIters] = Real(0.0);
        u_cg_scal[kCgQ] = Real(0.0);
        // 右端为零或预条件器失效时保持零步长
        u_cg_scal[kCgConv] = (isfinite(A) && A > Real(0.0)) ? Real(0.0) : Real(1.0);
    } else if (u1 == 1) {  // A = p.Sp
        Real rho = u_cg_scal[kCgRho];
        bool ok = isfinite(A) && A > Real(0.0);
        u_cg_scal[kCgPAp] = A;
        u_cg_scal[kCgAlpha] = ok ? rho / A : Real(0.0);
        // 沿 p 的步骤使模型代价下降 alpha*rho/2
        Real dq = ok ? Real(-0.5) * rho * rho / A : Real(0.0);
        u_cg_scal[kCgDQ] = dq;
        u_cg_scal[kCgQ] = u_cg_scal[kCgQ] + dq;
        if (!ok) u_cg_scal[kCgConv] = Real(1.0);  // 失去正定性
    } else {  // A 为新 r·z，B 为新 ||r||²
        Real rho = u_cg_scal[kCgRho];
        u_cg_scal[kCgBeta] = A / rho;
        u_cg_scal[kCgRho] = A;
        u_cg_scal[kCgRR] = B;
        u_cg_scal[kCgIters] = u_cg_scal[kCgIters] + Real(1.0);
        if (!(B > u_cg_scal[kCgTol2]) || !isfinite(A) || !(A > Real(0.0)))
            u_cg_scal[kCgConv] = Real(1.0);
        // Nash-Sofer 判据 i(Q_i-Q_i-1)/Q_i < f1，与 Ceres 一致；4102 图测量同最终代价下少约 40% 迭代，标记为容差满足。
        Real q = u_cg_scal[kCgQ];
        if (f1 > 0.0f && q < Real(0.0) &&
            u_cg_scal[kCgIters] * u_cg_scal[kCgDQ] / q < Real(f1)) {
            u_cg_scal[kCgConv] = Real(1.0);
            u_cg_scal[kCgTol2] = B;
        }
    }
}

// x+=alpha p，r-=alpha S p，x 累积于 u_g，u0=n_dim。
[shader("compute")] [numthreads(256, 1, 1)]
void cg_axpy(uint3 t : SV_DispatchThreadID) {
    if (cgDone()) return;
    uint i = t.x;
    if (i >= u0) return;
    Real alpha = u_cg_scal[kCgAlpha];
    u_g[i] = u_g[i] + alpha * u_cg_p[i];
    u_cg_r[i] = u_cg_r[i] - alpha * u_cg_sp[i];
}

// p=z+beta p，u0=n_dim。
[shader("compute")] [numthreads(256, 1, 1)]
void cg_updp(uint3 t : SV_DispatchThreadID) {
    if (cgDone()) return;
    uint i = t.x;
    if (i >= u0) return;
    u_cg_p[i] = u_cg_z[i] + u_cg_scal[kCgBeta] * u_cg_p[i];
}

// ---------------- 粗层校正：M^-1=块 Jacobi+P A_c^-1 P^T ----------------

// 每 u2 帧簇提供七自由度相似运动基 P；CG 复用 u_S 存 A_c，u3 后存逐帧 6×7 基，相关表位于 pair-Schur 表之后。

static const uint kTcDof = 7;

uint tcCluster(uint o) { return u_image_info[u_obs_image[o]].x / u2; }

// 世界相似运动 (w,tau,s) 产生 d(angle-axis)=-Jr^-1 w，dt=s t-R tau；基仅需覆盖慢模态，可用 fp32。u0=帧数。
[shader("compute")] [numthreads(64, 1, 1)]
void tc_basis(uint3 t : SV_DispatchThreadID) {
    uint f = t.x;
    if (f >= u0) return;
    float a[3], tr[3];
    for (int i = 0; i < 3; i++) {
        a[i] = rToFloat(u_poses[6 * f + i]);
        tr[i] = rToFloat(u_poses[6 * f + 3 + i]);
    }
    float th2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2], th = sqrt(th2);
    float K[9] = {0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0};
    float K2[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            K2[3 * i + j] = K[3 * i] * K[j] + K[3 * i + 1] * K[3 + j] + K[3 * i + 2] * K[6 + j];
    bool nearZero = th < 1e-3f;
    float s1 = nearZero ? 1.0f : sin(th) / th;
    float c1 = nearZero ? 0.5f : (1.0f - cos(th)) / th2;
    float c2 = nearZero ? 1.0f / 12.0f : 1.0f / th2 - (1.0f + cos(th)) / (2.0f * th * sin(th));
    uint b = u3 + 42 * f;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 7; j++) {
            float rot = j < 3 ? -(float(i == j) + 0.5f * K[3 * i + j] + c2 * K2[3 * i + j]) : 0.0f;
            float R = j >= 3 && j < 6 ? float(i == j - 3) + s1 * K[3 * i + j - 3] + c1 * K2[3 * i + j - 3]
                                      : 0.0f;
            u_S[b + 7 * i + j] = Real(rot);
            u_S[b + 7 * (3 + i) + j] = Real(j == 6 ? tr[i] : -R);
        }
}

// 逐图像位姿行累加 A_c+=P_f^T B_img P_f；u0=图像数，u2=每簇帧数，u3=基偏移。
void tcBpartImpl<let MD : int>(uint img) {
    if (img >= u0) return;
    uint f = u_image_info[img].x, c = f / u2;
    uint cb = bblk<MD>() * img, pb = u3 + 42 * f;
    Real BP[42];
    for (uint r = 0; r < 6; r++)
        for (uint j = 0; j < 7; j++) {
            Real s = Real(0.0);
            for (uint l = 0; l < 6; l++)
                s = s + u_cg_B[cb + (r >= l ? pidx(r, l) : pidx(l, r))] * u_S[pb + 7 * l + j];
            BP[7 * r + j] = s;
        }
    for (uint i = 0; i < 7; i++)
        for (uint j = 0; j <= i; j++) {
            Real s = Real(0.0);
            for (uint l = 0; l < 6; l++) s = s + u_S[pb + 7 * l + i] * BP[7 * l + j];
            if (isfinite(s)) atomicAccum(u_S_a, pidx(kTcDof * c + i, kTcDof * c + j), s);
        }
}

// 从观测 o 开始的一段内求 G=sum P_f^T Jc_pose^T Jp，尺寸 7×3。
void tcRunG(uint o, uint t1, out Real G[21]) {
    for (int i = 0; i < 21; i++) G[i] = Real(0.0);
    uint c = tcCluster(o);
    for (; o < t1 && tcCluster(o) == c; o++) {
        uint pb = u3 + 42 * u_image_info[u_obs_image[o]].x, jb = u_jc_off[o];
        for (uint i = 0; i < 7; i++) {
            Real q0 = Real(0.0), q1 = Real(0.0);
            for (uint a = 0; a < 6; a++) {
                q0 = q0 + u_S[pb + 7 * a + i] * u_Jc[jb + 2 * a];
                q1 = q1 + u_S[pb + 7 * a + i] * u_Jc[jb + 2 * a + 1];
            }
            for (uint j = 0; j < 3; j++)
                G[3 * i + j] = G[3 * i + j] + q0 * u_Jp[6 * o + 2 * j] + q1 * u_Jp[6 * o + 2 * j + 1];
        }
    }
}

// 每簇对块 warp 计算 A_c[cu,cv]-=sum G_u W G_v^T，共享暂存 7×7 乘积并归约 49 项后一次写出。
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void tc_schur(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) {
    uint wgx = wg.x + u4;
    if (wgx >= u0) return;
    uint2 ck = u_pair_chunks[u1 + wgx];
    uint2 e0 = u_pair_entries[ck.x];
    uint cu = tcCluster(e0.x), cv = tcCluster(e0.y);
    Real acc0 = Real(0.0), acc1 = Real(0.0);
    for (uint b0 = 0; b0 < ck.y; b0 += kPairWg) {
        GroupMemoryBarrierWithGroupSync();
        if (b0 + gi < ck.y) {
            uint2 e = u_pair_entries[ck.x + b0 + gi];
            uint p = u_obs_point[e.x], t1 = u_obs_ranges[p + 1];
            Real G[21], GW[21];
            tcRunG(e.x, t1, G);
            for (uint j = 0; j < 3; j++) {
                Real w0 = u_W[9 * p + j], w1 = u_W[9 * p + 3 + j], w2 = u_W[9 * p + 6 + j];
                for (uint i = 0; i < 7; i++)
                    GW[3 * i + j] = G[3 * i] * w0 + G[3 * i + 1] * w1 + G[3 * i + 2] * w2;
            }
            if (e.y != e.x) tcRunG(e.y, t1, G);
            for (uint i = 0; i < 7; i++)
                for (uint j = 0; j < 7; j++) {
                    Real v = GW[3 * i] * G[3 * j] + GW[3 * i + 1] * G[3 * j + 1]
                           + GW[3 * i + 2] * G[3 * j + 2];
                    g_chol[49 * gi + 7 * i + j] = isfinite(v) ? v : Real(0.0);
                }
        } else {
            for (uint i = 0; i < 49; i++) g_chol[49 * gi + i] = Real(0.0);
        }
        GroupMemoryBarrierWithGroupSync();
        uint nl = min(kPairWg, ck.y - b0);
        for (uint l = 0; l < nl; l++) {
            acc0 = acc0 + g_chol[49 * l + gi];
            if (gi + kPairWg < 49) acc1 = acc1 + g_chol[49 * l + gi + kPairWg];
        }
    }
    for (uint k = 0; k < 2; k++) {
        uint el = gi + kPairWg * k;
        if (el >= 49) continue;
        uint R = kTcDof * cu + el / 7, C = kTcDof * cv + el % 7;
        if (R >= C) atomicAccum(u_S_a, pidx(R, C), -(k == 0 ? acc0 : acc1));
    }
}

// A_c 对角乘 1+f0，并保存到 u_y 供主元下限使用；全局相似规范使无阻尼矩阵奇异。u0=粗维度。
[shader("compute")] [numthreads(256, 1, 1)]
void tc_reg(uint3 t : SV_DispatchThreadID) {
    uint i = t.x;
    if (i >= u0) return;
    Real d = u_S[pidx(i, i)];
    u_y[i] = d;
    u_S[pidx(i, i)] = d * (Real(1.0) + Real(f0));
}

// ---------------- 原地 L -> L^-1，以两次矩阵向量乘替代依赖的三角分派 ----------------

// 逐 32×32 对角块求逆，每列一线程，u0=n。
[shader("compute")] [numthreads(kBs, 1, 1)]
void tc_dinv(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) {
    uint base = wg.x * kBs, m = min(kBs, u0 - base), c = gi;
    for (uint r = c; r < m; r++)
        g_chol[r * kBs + c] = u_S[pidx(base + r, base + c)];
    GroupMemoryBarrierWithGroupSync();
    // 逆矩阵第 c 列写入共享池上半部分
    if (c < m) {
        g_chol[kSh2 + c * kPitch + c] = Real(1.0) / g_chol[c * kBs + c];
        for (uint r = c + 1; r < m; r++) {
            Real s = Real(0.0);
            for (uint k = c; k < r; k++) s = s + g_chol[r * kBs + k] * g_chol[kSh2 + k * kPitch + c];
            g_chol[kSh2 + r * kPitch + c] = -s / g_chol[r * kBs + r];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint r = c; r < m; r++)
        u_S[pidx(base + r, base + c)] = g_chol[kSh2 + r * kPitch + c];
}

// 逆矩阵块行 X_ij=-X_ii sum_{j<=k<i} L_ik X_kj 先写临时行，保证原 L 尚可读取；每块一个 32×32 工作组。
[shader("compute")] [numthreads(kBs, kBs, 1)]
void tc_inv_row(uint3 wg : SV_GroupID, uint3 lt : SV_GroupThreadID) {
    uint n = u0, i = u1, j = wg.x;
    uint r = lt.y, c = lt.x;
    uint bi = i * kBs, bj = j * kBs;
    Real acc = Real(0.0);
    for (uint k = j; k < i; k++) {
        uint bk = k * kBs;
        GroupMemoryBarrierWithGroupSync();
        g_chol[r * kBs + c] = bi + r < n ? u_S[pidx(bi + r, bk + c)] : Real(0.0);
        g_chol[kSh2 + r * kPitch + c] = bk + r >= bj + c ? u_S[pidx(bk + r, bj + c)] : Real(0.0);
        GroupMemoryBarrierWithGroupSync();
        for (uint m = 0; m < kBs; m++)
            acc = acc + g_chol[r * kBs + m] * g_chol[kSh2 + m * kPitch + c];
    }
    GroupMemoryBarrierWithGroupSync();
    g_chol[kSh2 + r * kPitch + c] = acc;
    g_chol[r * kBs + c] = bi + r < n && c <= r ? u_S[pidx(bi + r, bi + c)] : Real(0.0);
    GroupMemoryBarrierWithGroupSync();
    Real x = Real(0.0);
    for (uint m = 0; m <= r; m++) x = x + g_chol[r * kBs + m] * g_chol[kSh2 + m * kPitch + c];
    if (bi + r < n) u_y[u2 + r * n + bj + c] = -x;
}

// 将临时行复制回第 u1 块行，每元素一线程，u0=n。
[shader("compute")] [numthreads(256, 1, 1)]
void tc_inv_copy(uint3 t : SV_DispatchThreadID) {
    uint n = u0, bi = u1 * kBs;
    uint r = t.x / bi, col = t.x - r * bi;
    if (r >= kBs || bi + r >= n) return;
    u_S[pidx(bi + r, col)] = u_y[u2 + r * n + col];
}

// ---------------- 应用粗层校正 ----------------

// v=P^T r 写入 u_y[0,u0)，每粗自由度一线程，u1=帧数。
[shader("compute")] [numthreads(256, 1, 1)]
void tc_restrict(uint3 t : SV_DispatchThreadID) {
    if (cgDone()) return;
    uint q = t.x;
    if (q >= u0) return;
    uint c = q / kTcDof, j = q - kTcDof * c;
    uint fend = min(u1, (c + 1) * u2);
    Real s = Real(0.0);
    for (uint f = c * u2; f < fend; f++)
        for (uint a = 0; a < 6; a++) s = s + u_S[u3 + 42 * f + 7 * a + j] * u_cg_r[6 * f + a];
    u_y[q] = s;
}

// w=Xv 写入 u_y[u0,2u0)，每行一 warp，u0=粗维度。
[shader("compute")] [numthreads(256, 1, 1)]
void tc_lmul(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (cgDone()) return;
    uint row = t.x / 32, lane = gi % 32, w = gi / 32;
    Real s = Real(0.0);
    if (row < u0)
        for (uint j = lane; j <= row; j += 32) s = s + u_S[pidx(row, j)] * u_y[j];
    g_sh[gi] = s;
    GroupMemoryBarrierWithGroupSync();
    for (uint h = 16; h > 0; h >>= 1) {
        if (lane < h) g_sh[gi] = g_sh[gi] + g_sh[gi + h];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0 && row < u0) u_y[u0 + row] = g_sh[32 * w];
}

// z=X^T w 写入 u_y[0,u0)，每工作组 32 列合并读取压缩行段，8 条行 lane 在共享内存归约。
[shader("compute")] [numthreads(256, 1, 1)]
void tc_ltmul(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) {
    if (cgDone()) return;
    uint cc = gi % 32, rl = gi / 32;
    uint col = wg.x * 32 + cc;
    Real s = Real(0.0);
    if (col < u0)
        for (uint r = wg.x * 32 + rl; r < u0; r += 8)
            if (r >= col) s = s + u_S[pidx(r, col)] * u_y[u0 + r];
    g_sh[gi] = s;
    GroupMemoryBarrierWithGroupSync();
    if (rl == 0 && col < u0) {
        Real z = Real(0.0);
        for (uint k = 0; k < 8; k++) z = z + g_sh[32 * k + cc];
        u_y[col] = z;
    }
}

// 在位姿列上 z+=P y，u0=pose_dim。
[shader("compute")] [numthreads(256, 1, 1)]
void tc_prolong(uint3 t : SV_DispatchThreadID) {
    if (cgDone()) return;
    uint i = t.x;
    if (i >= u0) return;
    uint f = i / 6, a = i - 6 * f, c = f / u2;
    Real s = Real(0.0);
    for (uint j = 0; j < 7; j++) s = s + u_S[u3 + 42 * f + 7 * a + j] * u_y[kTcDof * c + j];
    u_cg_z[i] = u_cg_z[i] + s;
}

[shader("compute")] [numthreads(64, 1, 1)]
void tc_bpart_w(uint3 t : SV_DispatchThreadID) { tcBpartImpl<kDofWide>(t.x); }
[shader("compute")] [numthreads(64, 1, 1)]
void tc_bpart_x(uint3 t : SV_DispatchThreadID) { tcBpartImpl<kDofRig>(t.x); }
