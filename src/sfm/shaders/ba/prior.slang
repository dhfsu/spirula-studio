// 主机装配的稀疏 6×6 帧先验块，加入稠密 S/g 或 CG 预条件器与隐式乘积；由 ba.slang 包含。

[[vk::binding(47)]] StructuredBuffer<uint> u_prior_rows;   // num_frames+1 个 CSR 偏移
[[vk::binding(48)]] StructuredBuffer<uint> u_prior_cols;   // 每条目的列帧索引
[[vk::binding(49)]] StructuredBuffer<uint> u_prior_erow;   // 每条目的行帧索引
[[vk::binding(50)]] StructuredBuffer<Real> u_prior_blk;    // 每条目 36 项，[6a+b] 对应 y[6 row+a]+=blk*x[6 col+b]
[[vk::binding(51)]] StructuredBuffer<Real> u_prior_g;      // pose_dim

// S 加入先验块，双向条目只写行索引不小于列索引的下三角；u0=条目数。
[shader("compute")] [numthreads(256, 1, 1)]
void prior_add_s(uint3 t : SV_DispatchThreadID) {
    uint e = t.x;
    if (e >= u0) return;
    uint r = u_prior_erow[e], c = u_prior_cols[e];
    if (r < c) return;
    for (uint a = 0; a < 6; a++)
        for (uint b = 0; b < 6; b++) {
            if (r == c && b > a) continue;
            uint idx = pidx(6 * r + a, 6 * c + b);
            u_S[idx] = u_S[idx] + u_prior_blk[36 * e + 6 * a + b];
        }
}

// g 加入先验梯度，u0=pose_dim。
[shader("compute")] [numthreads(256, 1, 1)]
void prior_add_g(uint3 t : SV_DispatchThreadID) {
    uint i = t.x;
    if (i >= u0) return;
    u_g[i] = u_g[i] + u_prior_g[i];
}

// 对角先验加入块 Jacobi；两种划分均以帧 f 对应块 f，位姿行在前，u0=条目数。
[shader("compute")] [numthreads(256, 1, 1)]
void prior_add_m(uint3 t : SV_DispatchThreadID) {
    uint e = t.x;
    if (e >= u0) return;
    uint r = u_prior_erow[e];
    if (u_prior_cols[e] != r) return;
    uint cb = kCamBlk * r;
    for (uint a = 0; a < 6; a++)
        for (uint b = 0; b <= a; b++)
            u_cg_M[cb + pidx(a, b)] = u_cg_M[cb + pidx(a, b)] + u_prior_blk[36 * e + 6 * a + b];
}

// cg_scatter 完成后，在位姿行执行 Sp+=先验块*p，每帧行一线程，u0=帧数。
[shader("compute")] [numthreads(256, 1, 1)]
void prior_matvec(uint3 t : SV_DispatchThreadID) {
    if (cgDone()) return;
    uint f = t.x;
    if (f >= u0) return;
    uint e0 = u_prior_rows[f], e1 = u_prior_rows[f + 1];
    if (e0 == e1) return;
    Real acc[6];
    for (uint a = 0; a < 6; a++) acc[a] = Real(0.0);
    for (uint e = e0; e < e1; e++) {
        uint c = 6 * u_prior_cols[e];
        for (uint a = 0; a < 6; a++)
            for (uint b = 0; b < 6; b++)
                acc[a] = acc[a] + u_prior_blk[36 * e + 6 * a + b] * u_cg_p[c + b];
    }
    for (uint a = 0; a < 6; a++)
        u_cg_sp[6 * f + a] = u_cg_sp[6 * f + a] + acc[a];
}
