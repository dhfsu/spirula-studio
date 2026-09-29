// BA 问题每帧含 6 自由度位姿，每个 rig 成员含跨帧共享的 6 自由度 cam_from_rig，各相机组共享内参。
// 观测按点排序，并生成内核观测列表与约化相机系统列布局；没有 rig 时每张图独占一帧且没有成员外参。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/Env.h"
#include "sfm/core/Log.h"

// 相机模型注册表须与 ba.slang 入口一致；n_intr 是内核读取的参数数，可优化数量由问题的 Group::n_intr 决定（D50）。
struct ModelDesc {
    const char* name;
    uint32_t n_intr;
    const char* cost_entry;
    const char* jac_entry;
};
static const ModelDesc kModels[] = {
    {"snavely", 3, "cost_snavely", "jac_snavely"},                       // BAL，对数焦距
    {"snavely_f", 3, "cost_snavely_f", "jac_snavely_f"},                 // BAL，直接焦距
    {"pinhole_radial", 5, "cost_pinhole_radial", "jac_pinhole_radial"},  // COLMAP RADIAL
    {"opencv", 8, "cost_opencv", "jac_opencv"},                          // COLMAP OPENCV (D29)
    {"simple_pinhole", 3, "cost_simple_pinhole", "jac_simple_pinhole"},  // COLMAP SIMPLE_PINHOLE
    {"pinhole", 4, "cost_pinhole", "jac_pinhole"},                       // COLMAP PINHOLE
    {"opencv_fisheye", 8, "cost_opencv_fisheye", "jac_opencv_fisheye"},  // COLMAP OPENCV_FISHEYE (D29-C)
    {"full_opencv", 12, "cost_full_opencv", "jac_full_opencv"},          // COLMAP FULL_OPENCV (D34)
    {"thin_prism_fisheye", 12, "cost_thin_prism_fisheye", "jac_thin_prism_fisheye"},  // COLMAP THIN_PRISM_FISHEYE (D34)
    {"equirect", 2, "cost_equirect", "jac_equirect"},                    // COLMAP EQUIRECTANGULAR (D49)
};
static const int kNumModels = sizeof(kModels) / sizeof(kModels[0]);
// 6 帧参数 + 6 成员外参 + 至多 12 内参，必须与 ba.slang 一致；无 rig 时不超过 kMaxPlainDof。
static const uint32_t kMaxCamDof = 24;
static const uint32_t kMaxPlainDof = 18;
static const uint32_t kNoMember = 0xFFFFFFFFu;
static const uint32_t kExtAll = 0x3Fu;
inline uint32_t extFreeCount(uint32_t mask) {
    uint32_t n = 0;
    for (int i = 0; i < 6; i++) n += (mask >> i) & 1u;
    return n;
}

namespace sfm { struct PosePriors; }

struct BAProblem {
    uint32_t num_images = 0, num_points = 0, num_obs = 0;
    // 使用 BA 图像索引的位姿先验，见 Priors.h；空值保持普通求解路径。
    const sfm::PosePriors* priors = nullptr;
    // 无 rig 时 num_frames == num_images，image_frame 为恒等映射；有 rig 时图像须按帧排列，便于 CPU 任务独占帧行，finalizeTables 会检查。
    uint32_t num_frames = 0;
    std::vector<uint32_t> image_frame;    // 逐图像
    std::vector<uint32_t> image_member;   // 逐图像；kNoMember 表示无外参
    struct Member {
        uint32_t ext_offset;  // exts 中的偏移，共 6 项
        uint32_t ext_col;     // 列起点，n_free == 0 时不使用
        uint32_t n_free;      // 自由列数 popcount(mask)，0 表示固定
        // 按顺序选择六个轴角/平移参数，axial 仅优化旋转与 t.z。
        uint32_t mask = kExtAll;
    };
    std::vector<Member> members;

    // 观测按（点，图像）排序
    std::vector<uint32_t> obs_image, obs_point;
    std::vector<double> obs_xy;          // 每观测 2 项
    std::vector<uint32_t> obs_ranges;    // num_points+1

    // 相机分组
    struct Group {
        uint32_t intr_offset;  // 平铺内参数组中的偏移，共 kModels[model].n_intr 项
        uint32_t intr_col;     // 约化系统中的列起点
        uint32_t n_intr;       // 自由内参数量及其列数，不超过 kModels[model].n_intr；后续参数固定，但内核仍从 intr_offset 读取全部模型参数（D50）。
        uint32_t model;
    };
    std::vector<uint32_t> image_group;   // 逐图像
    std::vector<Group> groups;

    // 主机参数副本，使用 double
    std::vector<double> poses;   // 每帧 6 项
    std::vector<double> exts;    // 每成员 6 项，表示 cam_from_rig
    std::vector<double> intr;    // 平铺数组
    std::vector<double> points;  // 每三维点 3 项

    // 按（模型，rig）拼接的观测列表，用于专用内核分派
    struct ModelRange { uint32_t model, offset, count; bool rig; };
    std::vector<uint32_t> model_obs;
    std::vector<ModelRange> model_ranges;

    // Jc 块池的元素偏移，每块 2 × dof；Jp、Y 与残差按观测固定步长，无需偏移表。
    std::vector<uint32_t> jc_off;
    uint64_t jc_total = 0;

    // 按无序图像对聚合 Schur 项，每个共享点对应一对观测，a==b 为对角项；同对连续存放并分块，计数最高位置位表示需原子累加。
    // 由 buildPairTables 按需构建，纯 CG 路径跳过以节省主机与设备内存。
    std::vector<uint32_t> pair_entries;  // 每条目 2 项
    std::vector<uint32_t> pair_chunks;   // 每块 2 项：偏移、count|flag
    uint32_t num_pair_chunks = 0;
    bool use_pair_schur = false;

    // 按图像分组的 CSR 观测表，由 buildCamTables 按需构建；每相机分成固定大小块，每块一 warp，以原子累加改善占用率和负载均衡。
    std::vector<uint32_t> cam_obs_ranges;  // num_images+1
    std::vector<uint32_t> cam_obs;         // num_obs
    std::vector<uint32_t> cam_chunks;      // 每块 3 项：图像、起点、数量
    uint32_t num_cam_chunks = 0;

    // CG 的块 Jacobi 预条件器划分相机列；每块四个 uint 表示至多两段连续区间（col0,len0,col1,len1），对应位姿和内参。
    // 共享参数时的划分由 buildPrecBlocks 决定。
    std::vector<uint32_t> prec_blocks;
    uint32_t num_prec_blocks = 0;
    bool prec_exclusive = true;

    // 列布局：[帧位姿 | 自由成员外参 | 自由内参]。
    uint32_t pose_dim = 0;   // 6 * num_frames
    uint32_t ext_dim = 0;    // 6 × 可优化成员数
    uint32_t total_intr = 0;  // intr.size()，存储参数总数，自由参数在前
    uint32_t free_intr = 0;   // 其中占有优化列的参数数
    uint32_t n_dim = 0;      // 相机侧系统维度

    bool hasRigs() const { return num_frames != num_images || ext_dim != 0; }
    uint32_t memberFree(uint32_t img) const {
        const uint32_t m = image_member[img];
        return m == kNoMember ? 0 : members[m].n_free;
    }
    // 无 rig 时，每图像独占一帧。
    void identityFrames() {
        num_frames = num_images;
        image_frame.resize(num_images);
        for (uint32_t i = 0; i < num_images; i++) image_frame[i] = i;
        image_member.assign(num_images, kNoMember);
        members.clear();
        exts.clear();
        ext_dim = 0;
    }
};

// pair-Schur 要求各图像独占约化系统列；没有自由内参的组不占列，可优化成员或多图像帧则构成共享。
inline bool exclusiveGroups(const BAProblem& P) {
    if (P.hasRigs()) return false;
    std::vector<uint32_t> guse(P.groups.size(), 0);
    for (uint32_t g : P.image_group)
        if (++guse[g] > 1 && P.groups[g].n_intr) return false;
    return true;
}

// 独占参数时每图像一个 [位姿|内参] 块；共享时分别为每帧、每成员、每相机组建立块，顺序为帧、成员、组。
inline void buildPrecBlocks(BAProblem& P, bool exclusive) {
    P.prec_exclusive = exclusive;
    P.prec_blocks.clear();
    P.prec_blocks.reserve(4 * (P.num_frames + P.members.size() + P.groups.size()));
    auto push = [&](uint32_t c0, uint32_t l0, uint32_t c1, uint32_t l1) {
        P.prec_blocks.push_back(c0);
        P.prec_blocks.push_back(l0);
        P.prec_blocks.push_back(c1);
        P.prec_blocks.push_back(l1);
    };
    if (exclusive) {
        for (uint32_t i = 0; i < P.num_images; i++) {
            const BAProblem::Group& g = P.groups[P.image_group[i]];
            push(6 * i, 6, g.intr_col, g.n_intr);
        }
    } else {
        for (uint32_t f = 0; f < P.num_frames; f++) push(6 * f, 6, 0, 0);
        for (const BAProblem::Member& m : P.members) push(m.ext_col, m.n_free, 0, 0);
        for (const BAProblem::Group& g : P.groups) push(g.intr_col, g.n_intr, 0, 0);
    }
    P.num_prec_blocks = (uint32_t)(P.prec_blocks.size() / 4);
}

// pair-Schur 条目数为各点 t(t+1)/2 之和，用于建表前快速估算显存。
inline uint64_t pairEntryCount(const BAProblem& P) {
    uint64_t total = 0;
    for (uint32_t p = 0; p < P.num_points; p++) {
        uint64_t t = P.obs_ranges[p + 1] - P.obs_ranges[p];
        total += t * (t + 1) / 2;
    }
    return total;
}

static const uint64_t kMaxPairEntries = 400ull << 20;  // 约 3.2 GB 条目数据

// 粗层校正表：一段为轨迹在 k 帧簇内的连续观测；同一点的每对段 u >= v 对应一个首观测索引对。
template <class F>
inline void forCoarseRunPairs(const BAProblem& P, uint32_t k, F&& fn) {
    auto cl = [&](uint32_t o) { return P.image_frame[P.obs_image[o]] / k; };
    std::vector<uint32_t> runs;
    for (uint32_t p = 0; p < P.num_points; p++) {
        runs.clear();
        for (uint32_t o = P.obs_ranges[p]; o < P.obs_ranges[p + 1]; o++)
            if (o == P.obs_ranges[p] || cl(o) != cl(o - 1)) runs.push_back(o);
        for (size_t u = 0; u < runs.size(); u++)
            for (size_t v = 0; v <= u; v++) {
                const uint64_t cu = cl(runs[u]), cv = cl(runs[v]);
                fn(cu * (cu + 1) / 2 + cv, runs[u], runs[v]);
            }
    }
}

inline uint64_t coarseEntryCount(const BAProblem& P, uint32_t k) {
    uint64_t n = 0;
    forCoarseRunPairs(P, k, [&](uint64_t, uint32_t, uint32_t) { n++; });
    return n;
}

// 按簇对分组，key 保存各对范围，索引为 cu (cu + 1) / 2 + cv。
inline void buildCoarseEntries(const BAProblem& P, uint32_t k, std::vector<uint32_t>& ent,
                               std::vector<uint32_t>& key) {
    const uint64_t nc = (P.num_frames + k - 1) / k, nkeys = nc * (nc + 1) / 2;
    key.assign(nkeys + 1, 0);
    forCoarseRunPairs(P, k, [&](uint64_t kk, uint32_t, uint32_t) { key[kk + 1]++; });
    for (uint64_t i = 0; i < nkeys; i++) key[i + 1] += key[i];
    std::vector<uint32_t> fill(key.begin(), key.end() - 1);
    ent.resize(2 * (size_t)key[nkeys]);
    forCoarseRunPairs(P, k, [&](uint64_t kk, uint32_t a, uint32_t b) {
        const uint32_t e = fill[kk]++;
        ent[2 * (size_t)e] = a;
        ent[2 * (size_t)e + 1] = b;
    });
}

// 每簇 7 自由度，在 max_dim 允许下尽量缩小簇，同时保证条目数不超过观测数两倍；不足 4 簇或 SS_SFM_BA_COARSE=0 时禁用。
inline bool planCoarse(const BAProblem& P, uint32_t max_dim, uint32_t& k, uint32_t& dim,
                       uint64_t& entries) {
    k = dim = 0;
    entries = 0;
    const char* e = spirula::env("SFM_BA_COARSE");
    if (e && std::atoi(e) == 0) return false;
    const uint32_t nf = P.num_frames;
    for (uint32_t kk = std::max<uint32_t>(2, (7 * nf + max_dim - 1) / max_dim);
         (nf + kk - 1) / kk >= 4; kk *= 2) {
        const uint64_t n = coarseEntryCount(P, kk);
        if (n > 2 * (uint64_t)P.num_obs) continue;
        k = kk;
        dim = 7 * ((nf + kk - 1) / kk);
        entries = n;
        return true;
    }
    return false;
}

// 按图像生成 CSR 观测表，供 CG 的逐相机内核使用。
inline void buildCamTables(BAProblem& P) {
    P.cam_obs_ranges.assign(P.num_images + 1, 0);
    for (uint32_t o = 0; o < P.num_obs; o++) P.cam_obs_ranges[P.obs_image[o] + 1]++;
    for (uint32_t i = 0; i < P.num_images; i++) P.cam_obs_ranges[i + 1] += P.cam_obs_ranges[i];
    P.cam_obs.resize(P.num_obs);
    std::vector<uint32_t> fill(P.cam_obs_ranges.begin(), P.cam_obs_ranges.end() - 1);
    for (uint32_t o = 0; o < P.num_obs; o++) P.cam_obs[fill[P.obs_image[o]]++] = o;

    const uint32_t kChunk = 1024;
    P.cam_chunks.clear();
    for (uint32_t img = 0; img < P.num_images; img++)
        for (uint32_t o = P.cam_obs_ranges[img]; o < P.cam_obs_ranges[img + 1]; o += kChunk) {
            P.cam_chunks.push_back(img);
            P.cam_chunks.push_back(o);
            P.cam_chunks.push_back(std::min(kChunk, P.cam_obs_ranges[img + 1] - o));
        }
    P.num_cam_chunks = (uint32_t)(P.cam_chunks.size() / 3);
}

// 单图像观测涉及的列按雅可比顺序为 [帧 6 | 成员外参 n_free | 分组内参 n_intr]。
inline uint32_t imageColumns(const BAProblem& P, uint32_t img, uint32_t* cols) {
    uint32_t k = 0;
    const uint32_t f = P.image_frame[img];
    for (uint32_t i = 0; i < 6; i++) cols[k++] = 6 * f + i;
    const uint32_t m = P.image_member[img];
    if (m != kNoMember)
        for (uint32_t i = 0; i < P.members[m].n_free; i++) cols[k++] = P.members[m].ext_col + i;
    const BAProblem::Group& g = P.groups[P.image_group[img]];
    for (uint32_t i = 0; i < g.n_intr; i++) cols[k++] = g.intr_col + i;
    return k;
}

// 由观测、图像和分组表生成逐模型观测列表与 A_cp 偏移。
inline void finalizeTables(BAProblem& P) {
    if (P.image_frame.size() != P.num_images) P.identityFrames();
    if (P.image_member.size() != P.num_images) P.image_member.assign(P.num_images, kNoMember);
    for (uint32_t i = 1; i < P.num_images; i++)
        if (P.image_frame[i] < P.image_frame[i - 1])
            throw std::runtime_error("BA images must be ordered by frame");
    // 预计算逐图像的（模型，rig）桶与自由度，避免每观测追踪多层索引；rig 桶选择包含外参复合的内核。
    std::vector<uint8_t> img_bucket(P.num_images);
    std::vector<uint8_t> img_dof(P.num_images);
    const int nb = 2 * kNumModels;
    for (uint32_t i = 0; i < P.num_images; i++) {
        const BAProblem::Group& g = P.groups[P.image_group[i]];
        if (g.model >= (uint32_t)kNumModels)
            throw std::runtime_error("camera model index outside the registry");
        const bool rig = P.image_member[i] != kNoMember;
        img_bucket[i] = (uint8_t)(g.model + (rig ? kNumModels : 0));
        uint32_t dof = 6 + P.memberFree(i) + g.n_intr;
        if (dof > kMaxCamDof) throw std::runtime_error("camera dof exceeds kMaxCamDof");
        img_dof[i] = (uint8_t)dof;
    }

    // 一次计数分桶，桶内索引升序，桶按注册表顺序，忽略空桶。
    std::vector<uint32_t> cnt(nb, 0), off(nb, 0);
    for (uint32_t o = 0; o < P.num_obs; o++) cnt[img_bucket[P.obs_image[o]]]++;
    P.model_ranges.clear();
    uint32_t run = 0;
    for (int b = 0; b < nb; b++) {
        off[b] = run;
        if (cnt[b])
            P.model_ranges.push_back({(uint32_t)(b % kNumModels), run, cnt[b], b >= kNumModels});
        run += cnt[b];
    }
    P.model_obs.resize(P.num_obs);
    for (uint32_t o = 0; o < P.num_obs; o++) P.model_obs[off[img_bucket[P.obs_image[o]]]++] = o;

    P.jc_off.resize(P.num_obs);
    uint64_t acc = 0;
    for (uint32_t o = 0; o < P.num_obs; o++) {
        P.jc_off[o] = (uint32_t)acc;
        acc += 2 * (size_t)img_dof[P.obs_image[o]];
    }
    P.jc_total = acc;
    if (acc > 0xFFFFFFFFull) throw std::runtime_error("Jc pool exceeds 32-bit indexing");
    // 压缩三角形的 32 位索引限制约为 n_dim < 65k，仅在选择稠密求解时检查。
}

// pair-Schur 仅适用于图像独占内参列；共享参数或轨迹过长导致条目过大时，回退到逐观测原子内核。
inline void buildPairTables(BAProblem& P) {
    P.pair_entries.clear();
    P.pair_chunks.clear();
    P.num_pair_chunks = 0;
    P.use_pair_schur = false;
    if (P.num_obs == 0) return;

    // 独占条件：每相机组最多被一张图像引用
    if (!exclusiveGroups(P)) return;

    uint64_t total = pairEntryCount(P);
    if (total > kMaxPairEntries) {
        sfm::slog::diag(sfm::slog::Tag::Map, "[bal] pair-schur disabled (%llu entries)",
                   (unsigned long long)total);
        return;
    }

    // 按图像对键计数排序；点轨迹的图像索引严格递增，因此观测 i >= j 意味着 image_i >= image_j。
    auto key = [&](uint32_t oi, uint32_t oj) {
        uint64_t a = P.obs_image[oi], b = P.obs_image[oj];
        return a * (a + 1) / 2 + b;
    };
    uint64_t nkeys = (uint64_t)P.num_images * (P.num_images + 1) / 2;
    std::vector<uint32_t> cnt(nkeys + 1, 0);
    for (uint32_t p = 0; p < P.num_points; p++)
        for (uint32_t i = P.obs_ranges[p]; i < P.obs_ranges[p + 1]; i++)
            for (uint32_t j = P.obs_ranges[p]; j <= i; j++)
                cnt[key(i, j) + 1]++;
    for (uint64_t k = 0; k < nkeys; k++) cnt[k + 1] += cnt[k];
    std::vector<uint32_t> fill(cnt.begin(), cnt.end() - 1);
    P.pair_entries.resize(2 * total);
    for (uint32_t p = 0; p < P.num_points; p++)
        for (uint32_t i = P.obs_ranges[p]; i < P.obs_ranges[p + 1]; i++)
            for (uint32_t j = P.obs_ranges[p]; j <= i; j++) {
                uint32_t e = fill[key(i, j)]++;
                P.pair_entries[2 * e] = i;
                P.pair_entries[2 * e + 1] = j;
            }

    // 超过 kChunk 的图像对分块，并标记需使用原子累加。
    const uint32_t kChunk = 1024;
    for (uint64_t k = 0; k < nkeys; k++) {
        uint32_t off = cnt[k], n = cnt[k + 1] - cnt[k];
        if (!n) continue;
        uint32_t flag = n > kChunk ? 0x80000000u : 0;
        for (uint32_t o = 0; o < n; o += kChunk) {
            P.pair_chunks.push_back(off + o);
            P.pair_chunks.push_back(std::min(kChunk, n - o) | flag);
        }
    }
    P.num_pair_chunks = (uint32_t)(P.pair_chunks.size() / 2);
    P.use_pair_schur = true;
    sfm::slog::diag(sfm::slog::Tag::Map, "[bal] pair-schur: %llu entries, %u chunks",
               (unsigned long long)total, P.num_pair_chunks);
}


inline BAProblem loadBAL(const std::string& path, int model_id, bool shared_intrinsics) {
    if (model_id != 0 && model_id != 1)
        throw std::runtime_error("BAL loader supports snavely / snavely_f models");
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open " + path);
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::string text(len, 0);
    if (fread(text.data(), 1, len, fp) != (size_t)len) throw std::runtime_error("read failed");
    fclose(fp);

    char* s = text.data();
    auto nextInt = [&]() { return (uint32_t)strtoul(s, &s, 10); };
    auto nextDouble = [&]() { return strtod(s, &s); };

    BAProblem P;
    uint32_t nc = nextInt(), np = nextInt(), no = nextInt();
    P.num_images = nc;
    P.num_points = np;
    P.num_obs = no;
    sfm::slog::diag(sfm::slog::Tag::Map, "[bal] %u cameras, %u points, %u observations", nc, np,
                    no);

    std::vector<uint32_t> cam_idx(no), pnt_idx(no);
    std::vector<double> xy(2 * (size_t)no);
    for (uint32_t i = 0; i < no; i++) {
        cam_idx[i] = nextInt();
        pnt_idx[i] = nextInt();
        xy[2 * (size_t)i] = nextDouble();
        xy[2 * (size_t)i + 1] = nextDouble();
    }
    std::vector<double> cam9(9 * (size_t)nc);
    for (auto& v : cam9) v = nextDouble();
    P.points.resize(3 * (size_t)np);
    for (auto& v : P.points) v = nextDouble();

    // 按（点，图像）排序观测
    std::vector<uint32_t> order(no);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return pnt_idx[a] == pnt_idx[b] ? cam_idx[a] < cam_idx[b] : pnt_idx[a] < pnt_idx[b];
    });
    P.obs_image.resize(no);
    P.obs_point.resize(no);
    P.obs_xy.resize(2 * (size_t)no);
    for (uint32_t i = 0; i < no; i++) {
        uint32_t o = order[i];
        P.obs_image[i] = cam_idx[o];
        P.obs_point[i] = pnt_idx[o];
        P.obs_xy[2 * (size_t)i] = xy[2 * (size_t)o];
        P.obs_xy[2 * (size_t)i + 1] = xy[2 * (size_t)o + 1];
    }
    P.obs_ranges.assign(np + 1, 0);
    for (uint32_t i = 0; i < no; i++) P.obs_ranges[P.obs_point[i] + 1]++;
    for (uint32_t i = 0; i < np; i++) P.obs_ranges[i + 1] += P.obs_ranges[i];

    // 位姿与内参；BAL 相机布局为 [轴角(3), t(3), f, k1, k2]
    P.poses.resize(6 * (size_t)nc);
    for (uint32_t c = 0; c < nc; c++)
        for (int j = 0; j < 6; j++) P.poses[6 * (size_t)c + j] = cam9[9 * (size_t)c + j];

    const uint32_t ni = kModels[model_id].n_intr;  // 3
    auto camIntr = [&](uint32_t c, int j) {
        double v = cam9[9 * (size_t)c + 6 + j];
        if (model_id == 0 && j == 0) v = std::log(v);  // 对数焦距参数化
        return v;
    };
    if (shared_intrinsics) {
        P.groups.resize(1);
        P.image_group.assign(nc, 0);
        P.intr.assign(ni, 0.0);
        for (uint32_t c = 0; c < nc; c++)
            for (uint32_t j = 0; j < ni; j++) P.intr[j] += camIntr(c, j) / nc;
        P.groups[0] = {0, 0, ni, (uint32_t)model_id};
    } else {
        P.groups.resize(nc);
        P.image_group.resize(nc);
        P.intr.resize((size_t)ni * nc);
        for (uint32_t c = 0; c < nc; c++) {
            P.image_group[c] = c;
            for (uint32_t j = 0; j < ni; j++) P.intr[(size_t)ni * c + j] = camIntr(c, j);
            P.groups[c] = {ni * c, 0 /* 在下方修正 */, ni, (uint32_t)model_id};
        }
    }
    P.identityFrames();
    P.pose_dim = 6 * nc;
    P.total_intr = (uint32_t)P.intr.size();
    P.free_intr = P.total_intr;  // BAL 模型优化所读取的全部参数
    P.n_dim = P.pose_dim + P.free_intr;
    for (auto& g : P.groups) g.intr_col = P.pose_dim + g.intr_offset;

    finalizeTables(P);
    return P;
}
