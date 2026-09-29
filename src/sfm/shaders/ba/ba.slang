// BA 内核，未知量为 [帧位姿 | 自由成员外参 | 自由组内参]；点经 Schur 消元，S=J_u^T J_u-A_up(App+lambda)^-1 A_up^T，可显式压缩存储或隐式应用。
// 逐观测保存 Jc(2×dof)、Jp(2×3) 与加权残差，不显式生成 A_cp=Jc^T Jp。
// 推送常量 u0..u3 为数量/偏移，f0..f3 为阻尼/损失，u4 为主机分段后的首项。

#include "common/camera.slang"
#include "common/loss.slang"

// 主机选择能容纳问题的最窄自由度档；rig 档为 6+6+12，须与 Problem.h 的 kMaxCamDof 一致。
static const int kDofCompact = 12;
static const int kDofMid = 14;
static const int kDofWide = 18;
static const int kDofRig = 24;
// 预条件块不超过宽档；有 rig 时按帧、成员和组分别成块。
static const int kPrecDof = kDofWide;
static const uint kNoMember = 0xFFFFFFFFu;

[[vk::push_constant]] cbuffer PC {
    uint u0; uint u1; uint u2; uint u3;
    float f0; float f1; float f2; float f3;
    uint u4;
};

// ---------------- 问题数据 ----------------
[[vk::binding(0)]]  StructuredBuffer<Real> u_obs;          // 每观测 2 项
[[vk::binding(1)]]  StructuredBuffer<uint> u_obs_image;    // 逐观测
[[vk::binding(2)]]  StructuredBuffer<uint> u_obs_point;    // 逐观测
[[vk::binding(3)]]  StructuredBuffer<uint4> u_image_info;  // 逐图像：x=帧，y=成员（kNoMember 表示无），z=组，w=标志，第 0 位表示共享帧
[[vk::binding(4)]]  StructuredBuffer<uint4> u_group_info;  // 逐组：x=内参偏移，y=内参列起点，z=自由内参数，w=模型 ID
[[vk::binding(5)]]  RWStructuredBuffer<Real> u_poses;      // 每帧 6 项
[[vk::binding(6)]]  RWStructuredBuffer<Real> u_intr;       // 平铺内参
[[vk::binding(7)]]  RWStructuredBuffer<Real> u_points;     // 每点 3 项
[[vk::binding(8)]]  StructuredBuffer<uint> u_obs_ranges;   // num_points+1 项，观测按点排序
[[vk::binding(9)]]  StructuredBuffer<uint> u_model_obs;    // 拼接的逐模型观测索引列表
[[vk::binding(10)]] StructuredBuffer<uint> u_jc_off;       // 逐观测在 u_Jc 中的元素偏移

// ---------------- 工作缓冲：Real 与原子字双视图 ----------------
[[vk::binding(11)]] RWStructuredBuffer<Real> u_Jp;         // 点雅可比，索引 [6*o+2*j+row]
[[vk::binding(12)]] RWStructuredBuffer<Real> u_S;          // 压缩下三角，共 n_dim*(n_dim+1)/2 项
[[vk::binding(13)]] RWStructuredBuffer<AWord> u_S_a;
[[vk::binding(14)]] RWStructuredBuffer<Real> u_g;          // 梯度缓冲原地改写为解 dU
[[vk::binding(15)]] RWStructuredBuffer<AWord> u_g_a;
[[vk::binding(16)]] RWStructuredBuffer<Real> u_App;        // 每点 9 项
[[vk::binding(17)]] RWStructuredBuffer<AWord> u_App_a;
[[vk::binding(18)]] RWStructuredBuffer<Real> u_Bp;         // 每点 3 项
[[vk::binding(19)]] RWStructuredBuffer<AWord> u_Bp_a;
[[vk::binding(20)]] RWStructuredBuffer<Real> u_cost;       // (1)
[[vk::binding(21)]] RWStructuredBuffer<AWord> u_cost_a;

// ---------------- 按图像对聚合的 Schur 表，见 buildPairTables ----------------
[[vk::binding(22)]] StructuredBuffer<uint2> u_pair_entries; // 每条目保存 (obs_i,obs_j)
[[vk::binding(23)]] StructuredBuffer<uint2> u_pair_chunks;  // 偏移与 count|atomic_flag
[[vk::binding(24)]] RWStructuredBuffer<Real> u_W;           // 每点 9 项，(App+l)^-1
[[vk::binding(25)]] RWStructuredBuffer<Real> u_Y;           // 逐观测 (Jp W)^T，索引 [6*o+2*j+row]
[[vk::binding(26)]] RWStructuredBuffer<Real> u_y;           // n_dim 项的三角求解临时区
[[vk::binding(27)]] RWStructuredBuffer<Real> u_Jc;          // 逐观测相机雅可比，块内 [2*a+row]，起点 u_jc_off
[[vk::binding(28)]] RWStructuredBuffer<Real> u_res;         // 加权残差，每观测 2 项

// ---------------- rig：29..44 为 CG 绑定，47..51 为先验绑定 ----------------
[[vk::binding(45)]] StructuredBuffer<uint4> u_member_info;  // 逐成员：x=外参偏移，y=外参列起点，z=自由参数数，w=自由参数掩码
[[vk::binding(46)]] RWStructuredBuffer<Real> u_exts;        // 每成员 6 项 cam_from_rig

// 压缩下三角索引，row >= col
uint pidx(uint r, uint c) { return r * (r + 1) / 2 + c; }

// ================ 代价 ================

static const uint kCostWg = 256;

// 共享内存分三池，模块总量 fp64 约 26 KB，低于 32 KB 基线；流水线仅分配入口实际访问的池。
// g_chol 存两个 32×32 块及求解向量，第二块行距 33；g_pair 存 Schur 分批中间量，g_sh 存小型归约与 CG 临时量。
// 若全部共用 16.9 KB Cholesky 池，32 线程组占用受限，RTX 4080 Super 的 schur_pair 实测慢 2.2 倍。
static const uint kSh2 = 32 * 32;
static const uint kShPool = 32 * 32 + 32 * 33 + 32;
groupshared Real g_chol[kShPool];
groupshared Real g_sh[512];

// u0 为模型观测数，u1 为 u_model_obs 偏移，f0 为损失参数
void cost_impl<M : ICameraModel, let RIG : int>(uint tid, uint lid) {
    Real c = Real(0.0);
    if (tid < u0) {
        uint o = u_model_obs[u1 + tid];
        uint img = u_obs_image[o];
        uint pt = u_obs_point[o];
        uint4 ii = u_image_info[img];
        uint4 gi = u_group_info[ii.z];

        Real pose[6];
        for (int i = 0; i < 6; i++) pose[i] = u_poses[6 * ii.x + i];
        Real intr[M.kNumIntr];
        for (int i = 0; i < M.kNumIntr; i++) intr[i] = u_intr[gi.x + i];
        RVec3 X = rvec3(u_points[3 * pt + 0], u_points[3 * pt + 1], u_points[3 * pt + 2]);
        RVec2 obs = rvec2(u_obs[2 * o + 0], u_obs[2 * o + 1]);

        RVec2 r;
        if (RIG != 0) {
            uint4 mi = u_member_info[ii.y];
            Real ext[6];
            for (int i = 0; i < 6; i++) ext[i] = u_exts[mi.x + i];
            r = reprojectRig<M>(pose, ext, intr, X, obs);
        } else {
            r = reproject<M>(pose, intr, X, obs);
        }
        Real s = r.x * r.x + r.y * r.y;
        c = Real(0.5) * Loss.cost(s, f0);
    }
    g_sh[lid] = c;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint stride = kCostWg / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            g_sh[lid] = g_sh[lid] + g_sh[lid + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0)
        atomicAccum(u_cost_a, 0, g_sh[0]);
}

// ================ 残差、雅可比、App 与 Bp ================

// S/g 统一由 Schur 阶段累加，可在寄存器合并整相机 J^T J，避免逐观测 dof² 次高争用原子操作。
// 雅可比阶段对参数幂等，拒绝步骤恢复参数后可直接重跑下游，无需备份 S；u0/u1 为观测数量/偏移，f0 为损失参数。
void jac_impl<M : ICameraModel, let RIG : int>(uint tid) {
    if (tid >= u0) return;
    uint o = u_model_obs[u1 + tid];
    uint img = u_obs_image[o];
    uint pt = u_obs_point[o];
    uint4 ii = u_image_info[img];
    uint4 gi = u_group_info[ii.z];

    Real pose[6];
    for (int i = 0; i < 6; i++) pose[i] = u_poses[6 * ii.x + i];
    Real intr[M.kNumIntr];
    for (int i = 0; i < M.kNumIntr; i++) intr[i] = u_intr[gi.x + i];
    RVec3 X = rvec3(u_points[3 * pt + 0], u_points[3 * pt + 1], u_points[3 * pt + 2]);
    RVec2 obs = rvec2(u_obs[2 * o + 0], u_obs[2 * o + 1]);

    // 求值布局为 [帧6 | rig 成员6 | 模型全部内参]，存储仅保留 [6 | mfree | gi.z] 自由列；固定参数不占列（D50）。
    const int NE = RIG != 0 ? 6 : 0;
    const int DOF = 6 + NE + M.kNumIntr;
    uint mfree = 0, mmask = 0;
    Real ext[6];
    for (int i = 0; i < 6; i++) ext[i] = Real(0.0);
    if (RIG != 0) {
        uint4 mi = u_member_info[ii.y];
        for (int i = 0; i < 6; i++) ext[i] = u_exts[mi.x + i];
        mfree = mi.z;
        mmask = mfree != 0 ? mi.w : 0;
    }

    Real Jc[2][DOF];
    Real Jp[2][3];
    RVec2 r;
    if (RIG != 0) {
        r = reprojectRig<M>(pose, ext, intr, X, obs);
    } else {
        r = reproject<M>(pose, intr, X, obs);
    }
    Real s = r.x * r.x + r.y * r.y;
    Real sw = sqrt(Loss.weight(s, f0));

    // 两次反向微分得到雅可比的两行
    {
        var dp_pose = diffPair(pose);
        var dp_ext = diffPair(ext);
        var dp_intr = diffPair(intr);
        var dp_X = diffPair(X);
        [ForceUnroll]
        for (int row = 0; row < 2; row++) {
            RVec2 seed = rvec2(row == 0 ? Real(1.0) : Real(0.0),
                               row == 0 ? Real(0.0) : Real(1.0));
            if (RIG != 0) {
                bwd_diff(reprojectRig<M>)(dp_pose, dp_ext, dp_intr, dp_X, obs, seed);
            } else {
                bwd_diff(reproject<M>)(dp_pose, dp_intr, dp_X, obs, seed);
            }
            for (int i = 0; i < 6; i++) Jc[row][i] = dp_pose.d[i] * sw;
            if (RIG != 0)
                for (int i = 0; i < 6; i++) Jc[row][6 + i] = dp_ext.d[i] * sw;
            for (int i = 0; i < M.kNumIntr; i++) Jc[row][6 + NE + i] = dp_intr.d[i] * sw;
            Jp[row][0] = dp_X.d.x * sw; Jp[row][1] = dp_X.d.y * sw; Jp[row][2] = dp_X.d.z * sw;
        }
    }
    r.x = r.x * sw;
    r.y = r.y * sw;

    uint jb = u_jc_off[o];
    [ForceUnroll]
    for (int a = 0; a < DOF; a++) {
        // 求值列 a 在压缩自由列中的位置。
        uint dst;
        bool write;
        if (a < 6) {
            dst = uint(a);
            write = true;
        } else if (a < 6 + NE) {
            uint i = uint(a - 6);
            dst = 6 + countbits(mmask & ((1u << i) - 1u));
            write = ((mmask >> i) & 1u) != 0;
        } else {
            uint i = uint(a - 6 - NE);
            dst = 6 + mfree + i;
            write = i < gi.z;
        }
        if (write) {
            u_Jc[jb + 2 * dst] = Jc[0][a];
            u_Jc[jb + 2 * dst + 1] = Jc[1][a];
        }
    }
    for (int j = 0; j < 3; j++) {
        u_Jp[6 * o + 2 * j] = Jp[0][j];
        u_Jp[6 * o + 2 * j + 1] = Jp[1][j];
    }
    u_res[2 * o] = r.x;
    u_res[2 * o + 1] = r.y;

    // App += Jp^T Jp（完整 3×3），Bp += Jp^T r
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Real v = Jp[0][i] * Jp[0][j] + Jp[1][i] * Jp[1][j];
            if (isfinite(v))
                atomicAccum(u_App_a, 9 * pt + 3 * i + j, v);
        }
        Real bv = Jp[0][i] * r.x + Jp[1][i] * r.y;
        if (isfinite(bv))
            atomicAccum(u_Bp_a, 3 * pt + i, bv);
    }
}

// ================ 与相机模型无关的 Schur 补 ================

// 图像全局列按 [帧 | 成员外参 | 内参] 排列，与压缩雅可比一致。
void camCols<let MD : int>(uint img, out uint dof, out uint cols[MD]) {
    uint4 ii = u_image_info[img];
    uint4 gi = u_group_info[ii.z];
    for (int i = 0; i < 6; i++) cols[i] = 6 * ii.x + i;
    uint k = 6;
    if (ii.y != kNoMember) {
        uint4 mi = u_member_info[ii.y];
        for (uint i = 0; i < mi.z; i++) cols[k + i] = mi.y + i;
        k += mi.z;
    }
    for (uint i = 0; i < gi.z; i++) cols[k + i] = gi.y + i;
    dof = k + gi.z;
}

void obsCamCols<let MD : int>(uint o, out uint dof, out uint base, out uint cols[MD]) {
    camCols<MD>(u_obs_image[o], dof, cols);
    base = u_jc_off[o];
}

// 每观测一线程，累加 Jc^T Jc 与梯度，再减 Schur 项；每线程成本随轨迹长度线性，避免逐点平方任务触发看门狗。
// 此为支持共享列的原子回退，图像对内核要求独占列；对角乘法阻尼同样在此合并。u0=观测数，f0=点阻尼。
void schurObsImpl<let MD : int>(uint ti) {
    if (ti >= u0) return;
    uint p = u_obs_point[ti];

    // W=(App+lambda)^-1，已由 point_prep 逐点预计算
    Real W[9];
    for (int i = 0; i < 9; i++) W[i] = u_W[9 * p + i];

    uint dof_i, jb_i;
    uint cols_i[MD];
    obsCamCols<MD>(ti, dof_i, jb_i, cols_i);
    Real Jci[2 * MD];
    for (uint a = 0; a < 2 * dof_i; a++) Jci[a] = u_Jc[jb_i + a];

    // Y=(Jp_i W)^T，2×3 按列存储，与 u_Jp 一致
    Real Y[6];
    for (int j = 0; j < 3; j++)
        for (int row = 0; row < 2; row++)
            Y[2 * j + row] = u_Jp[6 * ti + row] * W[j]
                           + u_Jp[6 * ti + 2 + row] * W[3 + j]
                           + u_Jp[6 * ti + 4 + row] * W[6 + j];

    // S+=Jc^T Jc，含对角阻尼；g+=Jc^T r-Jc^T(Y Bp)
    Real yb0 = Y[0] * u_Bp[3 * p] + Y[2] * u_Bp[3 * p + 1] + Y[4] * u_Bp[3 * p + 2];
    Real yb1 = Y[1] * u_Bp[3 * p] + Y[3] * u_Bp[3 * p + 1] + Y[5] * u_Bp[3 * p + 2];
    Real r0 = u_res[2 * ti], r1 = u_res[2 * ti + 1];
    for (uint a = 0; a < dof_i; a++) {
        for (uint b = 0; b <= a; b++) {
            Real v = Jci[2 * a] * Jci[2 * b] + Jci[2 * a + 1] * Jci[2 * b + 1];
            if (a == b) v = v * (Real(1.0) + Real(f0));
            if (isfinite(v))
                atomicAccum(u_S_a, pidx(cols_i[a], cols_i[b]), v);
        }
        Real gv = Jci[2 * a] * (r0 - yb0) + Jci[2 * a + 1] * (r1 - yb1);
        if (isfinite(gv))
            atomicAccum(u_g_a, cols_i[a], gv);
    }

    // 对同点全部观测 j 累加 S-=Jc_i^T(Jp_i W Jp_j^T)Jc_j，仅写 row>=col。
    uint t0 = u_obs_ranges[p];
    uint t1 = u_obs_ranges[p + 1];
    for (uint tj = t0; tj < t1; tj++) {
        uint dof_j, jb_j;
        uint cols_j[MD];
        obsCamCols<MD>(tj, dof_j, jb_j, cols_j);
        // Q=Y Jp_j^T，尺寸 2×2
        Real q00 = Real(0.0), q01 = Real(0.0), q10 = Real(0.0), q11 = Real(0.0);
        for (int j = 0; j < 3; j++) {
            Real b0 = u_Jp[6 * tj + 2 * j], b1 = u_Jp[6 * tj + 2 * j + 1];
            q00 = q00 + Y[2 * j] * b0;
            q01 = q01 + Y[2 * j] * b1;
            q10 = q10 + Y[2 * j + 1] * b0;
            q11 = q11 + Y[2 * j + 1] * b1;
        }
        for (uint a = 0; a < dof_i; a++) {
            uint row = cols_i[a];
            Real z0 = Jci[2 * a] * q00 + Jci[2 * a + 1] * q10;
            Real z1 = Jci[2 * a] * q01 + Jci[2 * a + 1] * q11;
            for (uint c = 0; c < dof_j; c++) {
                uint col = cols_j[c];
                if (row < col) continue;
                Real v = z0 * u_Jc[jb_j + 2 * c] + z1 * u_Jc[jb_j + 2 * c + 1];
                if (isfinite(v))
                    atomicAccum(u_S_a, pidx(row, col), -v);
            }
        }
    }
}

// 逐点预计算 W=(App+l)^-1，供 acp_prep 与点更新使用；u0=点数，f0=点阻尼。
[shader("compute")] [numthreads(256, 1, 1)]
void point_prep(uint3 t : SV_DispatchThreadID) {
    uint p = t.x;
    if (p >= u0) return;
    Real A[9];
    for (int i = 0; i < 9; i++) A[i] = u_App[9 * p + i];
    A[0] = A[0] * (Real(1.0) + Real(f0));
    A[4] = A[4] * (Real(1.0) + Real(f0));
    A[8] = A[8] * (Real(1.0) + Real(f0));
    Real Ai[9];
    inverse3(A, Ai);
    for (int i = 0; i < 9; i++) u_W[9 * p + i] = Ai[i];
}

// 逐观测预计算 Y=(Jp W)^T，使 pair-Schur 中间 2×2 矩阵仅需小矩阵乘；u0=观测数。
[shader("compute")] [numthreads(256, 1, 1)]
void y_prep(uint3 t : SV_DispatchThreadID) {
    uint o = t.x + u4;
    if (o >= u0) return;
    uint p = u_obs_point[o];
    Real W[9];
    for (int i = 0; i < 9; i++) W[i] = u_W[9 * p + i];
    Real Jp[6];
    for (int i = 0; i < 6; i++) Jp[i] = u_Jp[6 * o + i];
    for (int j = 0; j < 3; j++)
        for (int row = 0; row < 2; row++)
            u_Y[6 * o + 2 * j + row] =
                Jp[row] * W[j] + Jp[2 + row] * W[3 + j] + Jp[4 + row] * W[6 + j];
}

// 每 warp 处理一个无序图像对块，独占 S 的块时无原子写入，跨块共享时由计数第 31 位标记原子累加。
// 对角对额外累加 J^T J 与梯度，并合并全局对角 (1+lambda) 阻尼，因此雅可比与独立阻尼阶段不再写 S/g。u0=块数，f0=lambda。
static const uint kPairWg = 32;
// 宽档元素数，供未分档的 CG 图像对内核使用。
static const uint kPairEl = (kPrecDof * kPrecDof + kPairWg - 1) / kPairWg;

// 共享暂存 Q=Y_i Jp_j^T、Y_i Bp 与 Z=Q^T Jc_i，使每块元素仅需两项点积。
// 每屏障批量处理 kPrB 条目；逐条屏障曾比显式 A_cp 版本慢 2.3 倍，批量摊薄同步成本。
static const uint kPrB = 8;         // 每轮屏障暂存的条目数
static const uint kPrQ = 0;         // 每条目 6 项：Q 占 4，Y_i Bp 占 2
static const uint kPrZ = 6 * kPrB;  // 每条目 2*MD 项，再接同大小 Jc_j
groupshared Real g_pair[6 * kPrB + 4 * kPrB * kPrecDof];

void schurPairImpl<let MD : int>(uint wgx, uint gi) {
    const uint kEl = (MD * MD + kPairWg - 1) / kPairWg;
    if (wgx >= u0) return;
    uint2 ck = u_pair_chunks[wgx];
    uint off = ck.x;
    uint count = ck.y & 0x7FFFFFFFu;
    bool useAtomic = (ck.y & 0x80000000u) != 0;

    uint2 e0 = u_pair_entries[off];
    uint dof_i, base0_i, dof_j, base0_j;
    uint cols_i[MD], cols_j[MD];
    obsCamCols<MD>(e0.x, dof_i, base0_i, cols_i);
    obsCamCols<MD>(e0.y, dof_j, base0_j, cols_j);
    bool diag = e0.x == e0.y;

    // 元素 el=r*dof_j+c 由 lane el%32 固定负责
    uint nel = dof_i * dof_j;
    Real acc[kEl];
    Real acc2[kEl];  // 延后的 J^T J，仅对角图像对使用
    uint elr[kEl], elc[kEl];
    [ForceUnroll]
    for (uint k = 0; k < kEl; k++) {
        acc[k] = Real(0.0);
        acc2[k] = Real(0.0);
        uint el = gi + kPairWg * k;
        elr[k] = el / dof_j;
        elc[k] = el - elr[k] * dof_j;
    }
    Real gacc = Real(0.0);

    const uint kZs = 2 * MD;              // 暂存块的逐条目步长
    const uint kPrJ = kPrZ + kPrB * kZs;  // 暂存的 Jc_j
    for (uint t0 = 0; t0 < count; t0 += kPrB) {
        uint nb = min(kPrB, count - t0);
        GroupMemoryBarrierWithGroupSync();  // 前一批已读取完毕
        for (uint i = gi; i < nb * 6; i += kPairWg) {
            uint b = i / 6, m = i - 6 * b;
            uint2 e = u_pair_entries[off + t0 + b];
            Real v;
            if (m < 4) {  // Q[qr][qs] = sum_j Y_i[j][qr] Jp_j[j][qs]
                uint qr = m >> 1, qs = m & 1;
                v = u_Y[6 * e.x + qr] * u_Jp[6 * e.y + qs]
                  + u_Y[6 * e.x + 2 + qr] * u_Jp[6 * e.y + 2 + qs]
                  + u_Y[6 * e.x + 4 + qr] * u_Jp[6 * e.y + 4 + qs];
            } else if (diag) {  // (Y_i Bp)[row]，仅梯度使用
                uint row = m - 4;
                uint p = u_obs_point[e.x];
                v = u_Y[6 * e.x + row] * u_Bp[3 * p] + u_Y[6 * e.x + 2 + row] * u_Bp[3 * p + 1]
                  + u_Y[6 * e.x + 4 + row] * u_Bp[3 * p + 2];
            } else {
                continue;
            }
            g_pair[kPrQ + 6 * b + m] = v;
        }
        for (uint i = gi; i < nb * 2 * dof_j; i += kPairWg) {
            uint b = i / (2 * dof_j), m = i - b * 2 * dof_j;
            g_pair[kPrJ + kZs * b + m] = u_Jc[u_jc_off[u_pair_entries[off + t0 + b].y] + m];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint i = gi; i < nb * 2 * dof_i; i += kPairWg) {
            uint b = i / (2 * dof_i), m = i - b * 2 * dof_i;
            uint a = m >> 1, s = m & 1;
            uint bi = u_jc_off[u_pair_entries[off + t0 + b].x];
            g_pair[kPrZ + kZs * b + m] = u_Jc[bi + 2 * a] * g_pair[kPrQ + 6 * b + s]
                                       + u_Jc[bi + 2 * a + 1] * g_pair[kPrQ + 6 * b + 2 + s];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint b = 0; b < nb; b++) {
            uint zb = kPrZ + kZs * b, jb = kPrJ + kZs * b;
            // 延后 J^T J 的 Jc_i 保留从全局读取；双共享输入实测增加 45% 成本，而全局读通常命中 L1。
            uint bi = u_jc_off[u_pair_entries[off + t0 + b].x];
            [ForceUnroll]
            for (uint k = 0; k < kEl; k++) {
                uint el = gi + kPairWg * k;
                if (el >= nel) continue;
                uint r = elr[k], c = elc[k];
                if (diag && r < c) continue;  // 对称块仅计算下三角
                acc[k] = acc[k] + g_pair[zb + 2 * r] * g_pair[jb + 2 * c]
                                + g_pair[zb + 2 * r + 1] * g_pair[jb + 2 * c + 1];
                if (diag)
                    acc2[k] = acc2[k] + u_Jc[bi + 2 * r] * u_Jc[bi + 2 * c]
                                      + u_Jc[bi + 2 * r + 1] * u_Jc[bi + 2 * c + 1];
            }
            if (diag && gi < dof_i) {
                uint ox = u_pair_entries[off + t0 + b].x;
                gacc = gacc + u_Jc[bi + 2 * gi] * (g_pair[kPrQ + 6 * b + 4] - u_res[2 * ox])
                            + u_Jc[bi + 2 * gi + 1] * (g_pair[kPrQ + 6 * b + 5] - u_res[2 * ox + 1]);
            }
        }
    }

    [ForceUnroll]
    for (uint k = 0; k < kEl; k++) {
        uint el = gi + kPairWg * k;
        if (el >= nel) continue;
        uint r = elr[k], c = elc[k];
        if (diag && r < c) continue;
        Real v = -acc[k];
        if (diag) {
            Real jj = acc2[k];
            if (r == c)
                jj = jj * (Real(1.0) + Real(f0));  // 已合并 damp_S 的阻尼操作
            v = v + jj;
        }
        if (!isfinite(v)) continue;
        uint R = cols_i[r], C = cols_j[c];
        uint idx = R >= C ? pidx(R, C) : pidx(C, R);
        if (useAtomic)
            atomicAccum(u_S_a, idx, v);
        else
            u_S[idx] = u_S[idx] + v;
    }
    if (diag && gi < dof_i && isfinite(gacc)) {
        uint col = cols_i[gi];
        if (useAtomic)
            atomicAccum(u_g_a, col, -gacc);
        else
            u_g[col] = u_g[col] - gacc;
    }
}

// ================ 参数更新，Cholesky 解 dU 位于 u_g ================

// Bp-=Acp^T dU=Jp^T(Jc dU)，每观测一线程，u0=观测数。
void dpAccumImpl<let MD : int>(uint o) {
    if (o >= u0) return;
    uint dof, base;
    uint cols[MD];
    obsCamCols<MD>(o, dof, base, cols);
    uint pt = u_obs_point[o];
    Real d0 = Real(0.0), d1 = Real(0.0);
    for (uint a = 0; a < dof; a++) {
        Real x = u_g[cols[a]];
        d0 = d0 + u_Jc[base + 2 * a] * x;
        d1 = d1 + u_Jc[base + 2 * a + 1] * x;
    }
    for (int j = 0; j < 3; j++) {
        Real v = u_Jp[6 * o + 2 * j] * d0 + u_Jp[6 * o + 2 * j + 1] * d1;
        if (isfinite(v))
            atomicAccum(u_Bp_a, 3 * pt + j, -v);
    }
}

// dP=(App+l)^-1 Bp，points-=dP；W 使用相同阻尼预计算，Bp 已由 dp_accum 原地更新，u0=点数。
[shader("compute")] [numthreads(256, 1, 1)]
void point_update(uint3 t : SV_DispatchThreadID) {
    uint p = t.x;
    if (p >= u0) return;
    for (int i = 0; i < 3; i++) {
        Real d = u_W[9 * p + 3 * i] * u_Bp[3 * p] + u_W[9 * p + 3 * i + 1] * u_Bp[3 * p + 1]
               + u_W[9 * p + 3 * i + 2] * u_Bp[3 * p + 2];
        if (isfinite(d))
            u_points[3 * p + i] = u_points[3 * p + i] - d;
    }
}

// 帧位姿减去 dU，u0=pose_dim。
[shader("compute")] [numthreads(256, 1, 1)]
void cam_update(uint3 t : SV_DispatchThreadID) {
    uint i = t.x;
    if (i >= u0) return;
    Real d = u_g[i];
    if (!isfinite(d)) return;
    u_poses[i] = u_poses[i] - d;
}

// 成员外参减去 dU，每成员一线程；固定成员无自由列，跳过。u0=成员数。
[shader("compute")] [numthreads(64, 1, 1)]
void ext_update(uint3 t : SV_DispatchThreadID) {
    uint m = t.x;
    if (m >= u0) return;
    uint4 mi = u_member_info[m];
    if (mi.z == 0) return;
    uint j = 0;
    for (uint i = 0; i < 6; i++) {
        if (((mi.w >> i) & 1u) == 0) continue;
        Real d = u_g[mi.y + j];
        j++;
        if (isfinite(d)) u_exts[mi.x + i] = u_exts[mi.x + i] - d;
    }
}

// 每相机组一线程，按组表将自由内参列映回参数槽；存储全部内参但列仅含自由前缀，两种布局不能混同（D50）。u0=组数。
[shader("compute")] [numthreads(64, 1, 1)]
void intr_update(uint3 t : SV_DispatchThreadID) {
    uint g = t.x;
    if (g >= u0) return;
    uint4 gi = u_group_info[g];
    for (uint j = 0; j < gi.z; j++) {
        Real d = u_g[gi.y + j];
        if (isfinite(d)) u_intr[gi.x + j] = u_intr[gi.x + j] - d;
    }
}

// ================ Schur 分档入口，由主机按最大自由度选择 ================

[shader("compute")] [numthreads(128, 1, 1)]
void schur_obs_c(uint3 t : SV_DispatchThreadID) { schurObsImpl<kDofCompact>(t.x + u4); }
[shader("compute")] [numthreads(128, 1, 1)]
void schur_obs_m(uint3 t : SV_DispatchThreadID) { schurObsImpl<kDofMid>(t.x + u4); }
[shader("compute")] [numthreads(128, 1, 1)]
void schur_obs_w(uint3 t : SV_DispatchThreadID) { schurObsImpl<kDofWide>(t.x + u4); }
[shader("compute")] [numthreads(128, 1, 1)]
void schur_obs_x(uint3 t : SV_DispatchThreadID) { schurObsImpl<kDofRig>(t.x + u4); }

[shader("compute")] [numthreads(kPairWg, 1, 1)]
void schur_pair_c(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { schurPairImpl<kDofCompact>(wg.x + u4, gi); }
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void schur_pair_m(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { schurPairImpl<kDofMid>(wg.x + u4, gi); }
[shader("compute")] [numthreads(kPairWg, 1, 1)]
void schur_pair_w(uint3 wg : SV_GroupID, uint gi : SV_GroupIndex) { schurPairImpl<kDofWide>(wg.x + u4, gi); }

[shader("compute")] [numthreads(256, 1, 1)]
void dp_accum_c(uint3 t : SV_DispatchThreadID) { dpAccumImpl<kDofCompact>(t.x + u4); }
[shader("compute")] [numthreads(256, 1, 1)]
void dp_accum_m(uint3 t : SV_DispatchThreadID) { dpAccumImpl<kDofMid>(t.x + u4); }
[shader("compute")] [numthreads(256, 1, 1)]
void dp_accum_w(uint3 t : SV_DispatchThreadID) { dpAccumImpl<kDofWide>(t.x + u4); }
[shader("compute")] [numthreads(256, 1, 1)]
void dp_accum_x(uint3 t : SV_DispatchThreadID) { dpAccumImpl<kDofRig>(t.x + u4); }

// ================ 逐相机模型入口：普通/rig 的代价与雅可比 ================

[shader("compute")] [numthreads(256, 1, 1)]
void cost_snavely(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<SnavelyModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_snavely(uint3 t : SV_DispatchThreadID) { jac_impl<SnavelyModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_snavely_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<SnavelyModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_snavely_rig(uint3 t : SV_DispatchThreadID) { jac_impl<SnavelyModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_snavely_f(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<SnavelyFModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_snavely_f(uint3 t : SV_DispatchThreadID) { jac_impl<SnavelyFModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_snavely_f_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<SnavelyFModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_snavely_f_rig(uint3 t : SV_DispatchThreadID) { jac_impl<SnavelyFModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_pinhole_radial(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<PinholeRadialModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_pinhole_radial(uint3 t : SV_DispatchThreadID) { jac_impl<PinholeRadialModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_pinhole_radial_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<PinholeRadialModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_pinhole_radial_rig(uint3 t : SV_DispatchThreadID) { jac_impl<PinholeRadialModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_opencv(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<OpenCVModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_opencv(uint3 t : SV_DispatchThreadID) { jac_impl<OpenCVModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_opencv_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<OpenCVModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_opencv_rig(uint3 t : SV_DispatchThreadID) { jac_impl<OpenCVModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_simple_pinhole(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<SimplePinholeModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_simple_pinhole(uint3 t : SV_DispatchThreadID) { jac_impl<SimplePinholeModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_simple_pinhole_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<SimplePinholeModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_simple_pinhole_rig(uint3 t : SV_DispatchThreadID) { jac_impl<SimplePinholeModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_pinhole(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<PinholeModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_pinhole(uint3 t : SV_DispatchThreadID) { jac_impl<PinholeModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_pinhole_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<PinholeModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_pinhole_rig(uint3 t : SV_DispatchThreadID) { jac_impl<PinholeModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_opencv_fisheye(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<FisheyeModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_opencv_fisheye(uint3 t : SV_DispatchThreadID) { jac_impl<FisheyeModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_opencv_fisheye_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<FisheyeModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_opencv_fisheye_rig(uint3 t : SV_DispatchThreadID) { jac_impl<FisheyeModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_full_opencv(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<FullOpenCVModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_full_opencv(uint3 t : SV_DispatchThreadID) { jac_impl<FullOpenCVModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_full_opencv_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<FullOpenCVModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_full_opencv_rig(uint3 t : SV_DispatchThreadID) { jac_impl<FullOpenCVModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_equirect(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<EquirectModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_equirect(uint3 t : SV_DispatchThreadID) { jac_impl<EquirectModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_equirect_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<EquirectModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_equirect_rig(uint3 t : SV_DispatchThreadID) { jac_impl<EquirectModel, 1>(t.x); }

[shader("compute")] [numthreads(256, 1, 1)]
void cost_thin_prism_fisheye(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<ThinPrismFisheyeModel, 0>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_thin_prism_fisheye(uint3 t : SV_DispatchThreadID) { jac_impl<ThinPrismFisheyeModel, 0>(t.x); }
[shader("compute")] [numthreads(256, 1, 1)]
void cost_thin_prism_fisheye_rig(uint3 t : SV_DispatchThreadID, uint gi : SV_GroupIndex) { cost_impl<ThinPrismFisheyeModel, 1>(t.x, gi); }
[shader("compute")] [numthreads(128, 1, 1)]
void jac_thin_prism_fisheye_rig(uint3 t : SV_DispatchThreadID) { jac_impl<ThinPrismFisheyeModel, 1>(t.x); }

#include "cholesky.slang"
#include "cg.slang"
#include "prior.slang"
