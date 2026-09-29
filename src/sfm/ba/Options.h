// GPU 与 CPU 回退共用的 BA 求解配置。
#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// 求解器算术类型；CPU 表示主机双精度，用于无法执行设备内核的情况。
enum class RealCfg { F32, F64, DF64, CPU };

inline RealCfg realCfgFromName(const std::string& s) {
    return s == "float"  ? RealCfg::F32
         : s == "df"     ? RealCfg::DF64
         : s == "cpu"    ? RealCfg::CPU
                         : RealCfg::F64;
}

inline const char* realCfgName(RealCfg c) {
    switch (c) {
        case RealCfg::F32: return "float";
        case RealCfg::F64: return "double";
        case RealCfg::CPU: return "cpu";
        default: return "df";
    }
}
inline size_t realSize(RealCfg c) { return c == RealCfg::F32 ? 4 : 8; }

inline void packReals(std::vector<uint8_t>& out, const double* v, size_t n, RealCfg cfg) {
    out.resize(n * realSize(cfg));
    if (cfg == RealCfg::F32) {
        float* p = (float*)out.data();
        for (size_t i = 0; i < n; i++) p[i] = (float)v[i];
    } else if (cfg == RealCfg::DF64) {
        float* p = (float*)out.data();
        for (size_t i = 0; i < n; i++) {
            float hi = (float)v[i];
            p[2 * i] = hi;
            p[2 * i + 1] = (float)(v[i] - hi);
        }
    } else {
        memcpy(out.data(), v, n * 8);
    }
}

inline void unpackReals(std::vector<double>& out, const uint8_t* v, size_t n, RealCfg cfg) {
    out.resize(n);
    if (cfg == RealCfg::F32) {
        const float* p = (const float*)v;
        for (size_t i = 0; i < n; i++) out[i] = p[i];
    } else if (cfg == RealCfg::DF64) {
        const float* p = (const float*)v;
        for (size_t i = 0; i < n; i++) out[i] = (double)p[2 * i] + (double)p[2 * i + 1];
    } else {
        memcpy(out.data(), v, n * 8);
    }
}

enum class SolverSel { Auto, Dense, CG };
enum class CgFallback { Auto, On, Off };

// 可拆分问题的调用方可要求在分配前因超出预算抛出 BAOverBudget；CG 已接近问题数据加少量向量的最低开销，只能缩小问题或增加内存。
struct BAOverBudget : std::runtime_error {
    BAOverBudget(double need, double budget)
        : std::runtime_error("bundle adjustment needs more device memory than the budget allows"),
          need_mb(need), budget_mb(budget) {}
    double need_mb, budget_mb;
};

// 设备求解进度检查点：主机参数向量对应已完成的 LM 迭代，设备失败后从此恢复。
struct SolverCheckpoint {
    int iterations = 0;
    double damping = 0, cost = 0;
};

struct SolverOptions {
    RealCfg real = RealCfg::F64;
    float loss_param = 1.0f;      // Huber 的 delta 或 Cauchy 的 c，普通平方损失不使用
    int max_iters = 50;
    double init_damping = 1e-2;
    double rtol = 1e-6;
    int patience = 10;
    SolverSel solver = SolverSel::Auto;
    double vram_budget_mb = 0;    // 0 表示设备本地堆的 90%，CPU 为主机内存的一半
    // 可请求超预算时抛出 BAOverBudget，默认仍警告并尝试。
    bool over_budget_throws = false;
    int cg_max_iters = 100;       // 每个 LM 步的 CG 迭代上限
    double cg_tol = 0.1;          // 相对残差容差 eta
    // Nash-Sofer 条件可在二次模型收益低于累计收益指定比例时停止，0 禁用；此时残差约为该比例平方根，精确步应禁用提前停止。
    double cg_model_tol = 0.1;
    CgFallback cg_fallback = CgFallback::Auto;
    // 内核按 real/loss 编译，选择 ba_<real>_<loss>；spv_path 可用磁盘模块覆盖，便于修改着色器后无需重新链接。
    std::string loss = "trivial";
    std::string spv_path;
    // 规范 uuid:<hex>；空值沿用显式设备、环境选择、Auto 的共享优先级。
    std::string device_selector;
    int device = -1;
    // CPU 工作线程上限，0 使用全部核心；限制单次求解的任务数，不改变共享池宽度。
    int threads = 0;
    bool validate = false;
    bool verbose = true;
    bool profile = false;
    // 设备求解每隔数秒保存已接受参数与进度；空指针禁用检查点。
    SolverCheckpoint* checkpoint = nullptr;
};

struct SolverStats {
    double initial_cost = 0, final_cost = 0;
    int iterations = 0, accepted = 0;
    double solve_seconds = 0;
    double vram_mb = 0;           // CPU 路径使用主机内存
    const char* solver = "dense";
    double cg_iters_total = 0;    // 所有 LM 步累计的 CG 迭代数
    int cg_solves = 0;
    int cg_fallbacks = 0;         // 改用稠密求解器重算的 LM 迭代数
};
