// 原地稠密 SPD 求解，u_S 按行压缩下三角，第 r 行从 r(r+1)/2 开始，右端 u_g，不额外分配矩阵。
// 分块右看式 Cholesky 后执行前后代；u0=维度，u1=块索引，u2=分派项数，u4=分段首块，共用 ba.slang 绑定。

static const uint kBs = 32;

// 主元至少 1e-30，包含 NaN 保护；启用 u3 时，低于原对角 f0 比例的主元替换为原值，限制粗矩阵舍入造成的不定性。
Real pivot(Real x, uint row) {
    if (u3 == 0) return sqrt(x > Real(1e-30) ? x : Real(1e-30));
    Real d = u_y[row];
    return sqrt(x > Real(f0) * d ? x : max(d, Real(1e-30)));
}
static const uint kPitch = 33;           // 填充共享行距以避免存储体冲突
static const uint kShV = kSh2 + 32 * 33; // 共享池中的右端向量槽

// 一个 kBs 线程工作组原地分解第 k 对角块。
[shader("compute")] [numthreads(kBs, 1, 1)]
void chol_diag(uint3 t : SV_DispatchThreadID) {
    uint r = t.x;
    uint n = u0;
    uint base = u1 * kBs;
    uint m = min(kBs, n - base);

    if (r < m)
        for (uint c = 0; c <= r; c++)
            g_chol[r * kPitch + c] = u_S[pidx(base + r, base + c)];

    for (uint j = 0; j < m; j++) {
        GroupMemoryBarrierWithGroupSync();
        if (r == j)
            g_chol[j * kPitch + j] = pivot(g_chol[j * kPitch + j], base + j);
        GroupMemoryBarrierWithGroupSync();
        if (r > j && r < m)
            g_chol[r * kPitch + j] = g_chol[r * kPitch + j] / g_chol[j * kPitch + j];
        GroupMemoryBarrierWithGroupSync();
        if (r > j && r < m)
            for (uint c = j + 1; c <= r; c++)
                g_chol[r * kPitch + c] = g_chol[r * kPitch + c]
                    - g_chol[r * kPitch + j] * g_chol[c * kPitch + j];
    }

    GroupMemoryBarrierWithGroupSync();
    if (r < m)
        for (uint c = 0; c <= r; c++)
            u_S[pidx(base + r, base + c)] = g_chol[r * kPitch + c];
}

// 对下方块求 L_ik=A_ik L_kk^-T，每行块一个 kBs 线程组，grid.x 为下方行块数。
[shader("compute")] [numthreads(kBs, 1, 1)]
void chol_panel(uint3 t : SV_DispatchThreadID, uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) {
    uint n = u0;
    uint k = u1;
    uint r = gi;
    uint base = k * kBs;
    uint rbase = (k + 1 + wg.x + u4) * kBs;
    uint gr = rbase + r;
    bool valid = gr < n;

    for (uint c = 0; c <= r; c++)
        g_chol[r * kBs + c] = u_S[pidx(base + r, base + c)];
    for (uint c = 0; c < kBs; c++)
        g_chol[kSh2 + r * kPitch + c] = valid ? u_S[pidx(gr, base + c)] : Real(0.0);
    GroupMemoryBarrierWithGroupSync();

    for (uint j = 0; j < kBs; j++) {
        Real v = g_chol[kSh2 + r * kPitch + j];
        for (uint mm = 0; mm < j; mm++)
            v = v - g_chol[kSh2 + r * kPitch + mm] * g_chol[j * kBs + mm];
        g_chol[kSh2 + r * kPitch + j] = v / g_chol[j * kBs + j];
    }

    if (valid)
        for (uint c = 0; c < kBs; c++)
            u_S[pidx(gr, base + c)] = g_chol[kSh2 + r * kPitch + c];
}

// 对 k<j<=i 执行 A_ij-=L_ik L_jk^T，每块对一个 kBs×kBs 工作组。
// 拥有下一对角块的组同时原地分解它，省去后续独立 chol_diag，仅首块单独分解。
[shader("compute")] [numthreads(kBs, kBs, 1)]
void chol_update(uint3 wg : SV_GroupID, uint3 lt : SV_GroupThreadID) {
    uint tid = wg.x + u4;
    if (tid >= u2) return;
    uint n = u0;
    uint k = u1;

    uint ii = uint((sqrt(8.0f * float(tid) + 1.0f) - 1.0f) * 0.5f);
    while ((ii + 1) * (ii + 2) / 2 <= tid) ii++;
    while (ii * (ii + 1) / 2 > tid) ii--;
    uint jj = tid - ii * (ii + 1) / 2;
    uint bi = (k + 1 + ii) * kBs; // 行块起点
    uint bj = (k + 1 + jj) * kBs; // 列块起点
    uint bk = k * kBs;

    uint r = lt.y, c = lt.x;
    g_chol[r * kBs + c] = (bi + r < n) ? u_S[pidx(bi + r, bk + c)] : Real(0.0);
    g_chol[kSh2 + r * kPitch + c] = (bj + r < n) ? u_S[pidx(bj + r, bk + c)] : Real(0.0);
    GroupMemoryBarrierWithGroupSync();

    uint gr = bi + r, gc = bj + c;
    Real acc = Real(0.0);
    bool live = gr < n && gc <= gr;
    if (live) {
        for (uint m = 0; m < kBs; m++)
            acc = acc + g_chol[r * kBs + m] * g_chol[kSh2 + c * kPitch + m];
    }

    if (tid != 0) {
        if (live) {
            uint idx = pidx(gr, gc);
            u_S[idx] = u_S[idx] - acc;
        }
        return;
    }

    // 在共享内存更新并分解下一对角块
    uint m2 = min(kBs, n - bi);
    GroupMemoryBarrierWithGroupSync();
    if (live)
        g_chol[r * kPitch + c] = u_S[pidx(gr, gc)] - acc;
    for (uint j = 0; j < m2; j++) {
        GroupMemoryBarrierWithGroupSync();
        if (r == j && c == j)
            g_chol[j * kPitch + j] = pivot(g_chol[j * kPitch + j], bi + j);
        GroupMemoryBarrierWithGroupSync();
        if (c == j && r > j && r < m2)
            g_chol[r * kPitch + j] = g_chol[r * kPitch + j] / g_chol[j * kPitch + j];
        GroupMemoryBarrierWithGroupSync();
        if (c > j && c <= r && r < m2)
            g_chol[r * kPitch + c] = g_chol[r * kPitch + c]
                - g_chol[r * kPitch + j] * g_chol[c * kPitch + j];
    }
    GroupMemoryBarrierWithGroupSync();
    if (live)
        u_S[pidx(gr, gc)] = g_chol[r * kPitch + c];
}

// ---------------- 每块一次分派的三角求解 ----------------
// 前代各组重复解 32×32 对角系统，组 0 写临时 y_k，各线程更新下方右端；后代从 u_y 求 x_k 写入 u_g，再更新上方 u_y，避免跨组同步依赖。
// u0=维度，u1=块索引，u2=更新行数。

static const uint kTriWg = 256;

[shader("compute")] [numthreads(kTriWg, 1, 1)]
void tri_fwd(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) {
    uint n = u0;
    uint base = u1 * kBs;
    uint m = min(kBs, n - base);
    uint r = gi;

    if (r < m) {
        for (uint c = 0; c <= r; c++)
            g_chol[r * kPitch + c] = u_S[pidx(base + r, base + c)];
        g_chol[kShV + r] = u_g[base + r];
    }
    for (uint j = 0; j < m; j++) {
        GroupMemoryBarrierWithGroupSync();
        if (r == j)
            g_chol[kShV + j] = g_chol[kShV + j] / g_chol[j * kPitch + j];
        GroupMemoryBarrierWithGroupSync();
        if (r > j && r < m)
            g_chol[kShV + r] = g_chol[kShV + r] - g_chol[r * kPitch + j] * g_chol[kShV + j];
    }
    GroupMemoryBarrierWithGroupSync();
    if (wg.x == 0 && r < m)
        u_y[base + r] = g_chol[kShV + r];

    uint row = wg.x * kTriWg + gi;
    if (row < u2) {
        uint gr = base + kBs + row;
        Real acc = Real(0.0);
        for (uint j = 0; j < kBs; j++)
            acc = acc + u_S[pidx(gr, base + j)] * g_chol[kShV + j];
        u_g[gr] = u_g[gr] - acc;
    }
}

[shader("compute")] [numthreads(kTriWg, 1, 1)]
void tri_bwd(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) {
    uint n = u0;
    uint base = u1 * kBs;
    uint m = min(kBs, n - base);
    uint r = gi;

    if (r < m) {
        for (uint c = 0; c <= r; c++)
            g_chol[r * kPitch + c] = u_S[pidx(base + r, base + c)];
        g_chol[kShV + r] = u_y[base + r];
    }
    for (int j = int(m) - 1; j >= 0; j--) {
        GroupMemoryBarrierWithGroupSync();
        if (int(r) == j)
            g_chol[kShV + j] = g_chol[kShV + j] / g_chol[j * kPitch + j];
        GroupMemoryBarrierWithGroupSync();
        if (int(r) < j && r < m)
            g_chol[kShV + r] = g_chol[kShV + r] - g_chol[uint(j) * kPitch + r] * g_chol[kShV + j];
    }
    GroupMemoryBarrierWithGroupSync();
    if (wg.x == 0 && r < m)
        u_g[base + r] = g_chol[kShV + r];

    uint gr = wg.x * kTriWg + gi;
    if (gr < u2) {
        Real acc = Real(0.0);
        for (uint j = 0; j < m; j++)
            acc = acc + u_S[pidx(base + j, gr)] * g_chol[kShV + j];
        u_y[gr] = u_y[gr] - acc;
    }
}
