// 轻量主机耗时统计，始终编译，仅 SS_SFM_MAP_PROF=1 时报告，不改变普通运行输出。
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "core/Env.h"
#include "sfm/core/Log.h"

namespace sfm {

// 多个原子工作线程共享累加器，C++17 用 CAS 循环实现浮点原子加，避免普通 double+= 的数据竞争；阶段级低频更新几乎无争用。
class ProfAcc {
public:
    ProfAcc& operator+=(double v) {
        double cur = v_.load(std::memory_order_relaxed);
        while (!v_.compare_exchange_weak(cur, cur + v, std::memory_order_relaxed)) {}
        return *this;
    }
    double get() const { return v_.load(std::memory_order_relaxed); }

private:
    std::atomic<double> v_{0};
};

struct MapProf {
    // 建图器主机阶段
    ProfAcc init_seed;   // initialize：种子搜索，含双视图 RANSAC
    ProfAcc seed_geom;   // 其中 seedGeometry 的双视图 RANSAC 耗时
    ProfAcc bootstrap;   // 焦距初始化，包含于 init_seed
    ProfAcc choose;      // 下一图像候选排序
    ProfAcc reg;         // PnP RANSAC、精化与内点重计数
    ProfAcc tri;         // 增长期的新点三角化
    ProfAcc retri;       // 轨迹补全与重三角化
    ProfAcc merge;       // 其中轨迹合并耗时
    ProfAcc filter;      // 相机约束、点过滤与图像过滤
    ProfAcc snapshot;    // 事务式精化的重建备份
    ProfAcc audit_check;  // 逐已配准图像的位姿审查
    ProfAcc audit_fix;    // 审查后的修复与精化耗时
    // 下列字段分别统计全局 BA 的各环节。
    ProfAcc ba_build;    // 由重建装配 BAProblem
    ProfAcc ba_init;     // 求解器初始化：设备、流水线与上传
    ProfAcc ba_solve;    // 求解器迭代
    ProfAcc ba_write;    // 下载与参数写回
    std::atomic<long> n_ba{0}, n_ba_iters{0}, n_choose{0}, n_reg_try{0}, n_reg_ok{0};
    std::atomic<long> n_seed_geom{0};  // 实际运行的双视图 RANSAC 数，即缓存未命中数
    std::atomic<long> n_merged{0};  // 轨迹合并吸收的观测数

    static bool enabled() {
        static bool e = spirula::env("SFM_MAP_PROF") != nullptr;
        return e;
    }

    // 计数累计不清零，建图与装配结束后可分别报告，what 标明阶段。
    void report(double total_s, const char* what = "mapper") const {
        if (!enabled()) return;
        const double init_seed = this->init_seed.get(), choose = this->choose.get();
        const double seed_geom = this->seed_geom.get(), bootstrap = this->bootstrap.get();
        const long n_seed_geom = this->n_seed_geom;
        const double reg = this->reg.get(), tri = this->tri.get(), retri = this->retri.get();
        const double merge = this->merge.get(), filter = this->filter.get();
        const double snapshot = this->snapshot.get(), ba_build = this->ba_build.get();
        const double audit_check = this->audit_check.get(), audit_fix = this->audit_fix.get();
        const double ba_init = this->ba_init.get(), ba_solve = this->ba_solve.get();
        const double ba_write = this->ba_write.get();
        const long n_ba = this->n_ba, n_ba_iters = this->n_ba_iters, n_choose = this->n_choose;
        const long n_reg_try = this->n_reg_try, n_reg_ok = this->n_reg_ok;
        const long n_merged = this->n_merged;
        double ba = ba_build + ba_init + ba_solve + ba_write;
        double accounted = init_seed + choose + reg + tri + retri + filter + snapshot + ba;
        slog::diag(slog::Tag::Map, "[prof] %s total %.2f s, accounted %.2f s (%.0f%%)\n"
                   "[prof]   seed search   %8.2f s  (two-view %.2f s over %ld pair(s), "
                   "focal bootstrap %.2f s)\n"
                   "[prof]   choose-next   %8.2f s  (%ld calls)\n"
                   "[prof]   register      %8.2f s  (%ld tries, %ld ok)\n"
                   "[prof]   triangulate   %8.2f s\n"
                   "[prof]   retriangulate %8.2f s  (of which merge %.2f s, %ld obs absorbed)\n"
                   "[prof]   filter        %8.2f s\n"
                   "[prof]   snapshot      %8.2f s\n"
                   "[prof]   audit         %8.2f s  (check %.2f + repair and refine %.2f)\n"
                   "[prof]   BA            %8.2f s  (%ld calls, %ld LM iters): "
                   "build %.2f + init %.2f + solve %.2f + write %.2f",
                   what, total_s, accounted, total_s > 0 ? 100.0 * accounted / total_s : 0.0,
                   init_seed, seed_geom, n_seed_geom, bootstrap, choose, n_choose, reg, n_reg_try,
                   n_reg_ok, tri, retri, merge,
                   n_merged, filter, snapshot, audit_check + audit_fix, audit_check, audit_fix,
                   ba, n_ba, n_ba_iters, ba_build, ba_init, ba_solve, ba_write);
    }
};

inline MapProf g_map_prof;

// RAII 计时器，离开作用域时将经过秒数加到 acc。
class ProfTimer {
public:
    explicit ProfTimer(ProfAcc& acc)
        : acc_(acc), t0_(std::chrono::steady_clock::now()) {}
    ~ProfTimer() {
        acc_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    }

private:
    ProfAcc& acc_;
    std::chrono::steady_clock::time_point t0_;
};

}  // 命名空间 sfm
