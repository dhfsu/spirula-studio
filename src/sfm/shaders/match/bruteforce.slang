// 打包 uint8 描述子通过 ||q-t||²=||q||²+||t||²-2q·t 计算距离，预存范数并使用 DP4A；128 维上界 8323200<2²³，整数结果精确。
// 一次距离矩阵同时求行最近/次近邻与列最近邻，列打包 (d2<<6)|localQuery 并固定索引平局；若增加 B 侧比值测试，须同步保存次近邻。
// 训练块通过共享内存复用，避免每查询重复流读；多图描述子常驻缓冲，按索引分派而非逐对上传。

// 仅硬件打包整数点积加速时使用 DP4A，否则 NO_DOT4 展开相同整数算术。
// 不能仅检查扩展存在，Apple M2 模拟入口单对耗时 114 ms，而展开为 43 ms；统一经 udot4add 调用。
#ifdef NO_DOT4
uint udot4add(uint a, uint b, uint acc) {
    [unroll] for (uint s = 0; s < 32u; s += 8u)
        acc += ((a >> s) & 0xffu) * ((b >> s) & 0xffu);
    return acc;
}
#else
uint udot4add(uint a, uint b, uint acc) { return dot4add_u8packed(a, b, acc); }
#endif

// 32 个 uint 表示 128 字节，DeDoDe-G 用 DESC_WORDS=64；训练块减半保持共享内存 8 KB，距离上界变为 2²⁴，仍满足列打包范围。
#ifndef DESC_WORDS
#define DESC_WORDS 32
#endif

static const int DW = DESC_WORDS;
static const uint TQ = 64;   // match_pair 每组查询数，列打包仅留六位局部索引
static const uint TQR = 256; // match_rows 无列归约，可用更宽组，须与主机 kRowThreads 一致
static const uint TT = (DESC_WORDS >= 64) ? 32 : 64;  // 每共享块的训练描述子数

[[vk::push_constant]] cbuffer PC {
    uint dQ;       // 图像 A 首描述子索引
    uint nQuery;   // A 的特征数
    uint dT;       // 图像 B 首描述子索引
    uint nTrain;   // B 的特征数
    uint _f0, _f1, _f2, _f3;  // 推送常量中的浮点槽，此处不用
    uint outA;     // A 侧结果起始 uint4 索引
    uint outB;     // B 侧结果起始 uint4 索引
    uint numWG;    // 匹配使用的工作组数，供列归约
};

[[vk::binding(0)]] StructuredBuffer<uint>    desc;   // 全部常驻描述子
[[vk::binding(1)]] RWStructuredBuffer<uint>  norm;   // 逐描述子 ||d||²
[[vk::binding(2)]] RWStructuredBuffer<uint4> result; // (bestIdx, bestD2, secondD2, 0)
[[vk::binding(3)]] RWStructuredBuffer<uint>  colPartial;  // numWG*nTrain 个打包列候选

// 每次上传后为新描述子计算一次 ||d||²，u0=首索引，u1=数量。
[shader("compute")] [numthreads(64, 1, 1)]
void descriptor_norms(uint3 tid : SV_DispatchThreadID) {
    uint d = tid.x;
    if (d >= nQuery) return;
    uint idx = dQ + d;
    uint base = idx * uint(DW);
    uint s = 0;
    [unroll] for (int k = 0; k < DW; k++) {
        uint w = desc[base + uint(k)];
        s = udot4add(w, w, s);
    }
    norm[idx] = s;
}

groupshared uint sTrain[TT * uint(DW)];  // 两种宽度均占 8 KB
groupshared uint sCol[TT];               // 打包值 (d2<<6)|localQuery

// 训练流量为 ceil(nQuery/TQ)*nTrain；match_pair 受六位列索引限制，match_rows 可用 256 线程，将评分训练流读从八次减到两次。
// 每查询仍按训练索引顺序归约，结果逐位一致。

// 两入口共享行归约，WITH_COLS 可额外生成交叉检查列候选；纯行模式跨对只读共享数据、写互斥结果，整批无需图像对间屏障。
void matchImpl<let WITH_COLS : bool, let NQ : uint>(uint3 gid, uint3 lid) {
    const uint lq = lid.x;
    const uint q = gid.x * NQ + lq;
    const bool active = q < nQuery;

    uint qd[DW];
    [unroll] for (int k = 0; k < DW; k++) qd[k] = 0;  // 超出 nQuery 的线程不会读取这些值
    int nq = 0;
    if (active) {
        uint qbase = (dQ + q) * uint(DW);
        [unroll] for (int k = 0; k < DW; k++) qd[k] = desc[qbase + uint(k)];
        nq = int(norm[dQ + q]);
    }

    uint best = 0xffffffffu, second = 0xffffffffu, bestIdx = 0xffffffffu;

    for (uint t0 = 0; t0 < nTrain; t0 += TT) {
        const uint cnt = min(TT, nTrain - t0);

        GroupMemoryBarrierWithGroupSync();  // 前一块已完全消费
        for (uint e = lid.x; e < cnt * uint(DW); e += NQ)
            sTrain[e] = desc[(dT + t0 + e / uint(DW)) * uint(DW) + (e % uint(DW))];
        if (WITH_COLS)
            for (uint c = lid.x; c < cnt; c += NQ) sCol[c] = 0xffffffffu;
        GroupMemoryBarrierWithGroupSync();  // 当前块对全部线程可见

        for (uint c = 0; c < cnt; c++) {
            uint packed = 0xffffffffu;  // 无效 lane 不贡献结果
            if (active) {
                uint dot = 0;
                [unroll] for (int k = 0; k < DW; k++)
                    dot = udot4add(qd[k], sTrain[c * uint(DW) + uint(k)], dot);
                uint d2 = uint(nq + int(norm[dT + t0 + c]) - 2 * int(dot));
                uint j = t0 + c;
                if (d2 < best) { second = best; best = d2; bestIdx = j; }
                else if (d2 < second) { second = d2; }
                packed = (d2 << 6) | lq;
            }
            // 直接使用共享原子；先做 WaveActiveMin/WaveIsFirstLane 归约实测更慢，硬件处理争用比额外子组操作便宜。
            if (WITH_COLS && packed != 0xffffffffu) InterlockedMin(sCol[c], packed);
        }

        if (WITH_COLS) {
            GroupMemoryBarrierWithGroupSync();  // sCol 完成后才能写回
            for (uint c = lid.x; c < cnt; c += NQ)
                colPartial[gid.x * nTrain + t0 + c] = sCol[c];
        }
    }

    if (active) result[outA + q] = uint4(bestIdx, best, second, 0);
}

[shader("compute")] [numthreads(64, 1, 1)]
void match_pair(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    matchImpl<true, TQ>(gid, lid);
}

// 仅行匹配供无需交叉检查的评分路径使用，不写跨图像对共享状态。
[shader("compute")] [numthreads(256, 1, 1)]
void match_rows(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    matchImpl<false, TQR>(gid, lid);
}

// 归约各组列候选，仅 bestIdx 被下游使用；secondD2 写为无次近邻，避免误用时静默通过比值测试。
[shader("compute")] [numthreads(64, 1, 1)]
void reduce_cols(uint3 tid : SV_DispatchThreadID) {
    uint j = tid.x;
    if (j >= nTrain) return;

    uint bestD2 = 0xffffffffu, bestIdx = 0xffffffffu;
    for (uint w = 0; w < numWG; w++) {
        uint packed = colPartial[w * nTrain + j];
        if (packed == 0xffffffffu) continue;  // 该工作组没有有效查询
        uint d2 = packed >> 6;
        uint gi = w * TQ + (packed & 63u);
        if (d2 < bestD2 || (d2 == bestD2 && gi < bestIdx)) { bestD2 = d2; bestIdx = gi; }
    }
    result[outB + j] = uint4(bestIdx, bestD2, 0xffffffffu, 0);
}
