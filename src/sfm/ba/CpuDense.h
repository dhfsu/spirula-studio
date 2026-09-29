// 主机约化相机系统的稠密 SPD 求解器，原地使用与 GPU 相同的压缩下三角，第 r 行起点为 r(r+1)/2。
// 每步将当前块列复制到连续面板，使尾部更新的两个输入均连续；总复制量为 4n^2 字节，相比 n^3/3 次浮点运算较小。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sfm/ba/CpuParallel.h"

namespace bacpu {

class DenseSpd {
public:
    static constexpr uint32_t kBlock = 48;

    void init(uint32_t n) {
        n_ = n;
        a_.assign(elems(n), 0.0);
        panel_.resize((size_t)n * kBlock);
        panelT_.resize((size_t)n * kBlock);
        diag_.resize((size_t)kBlock * kBlock);
    }
    void release() {
        a_ = std::vector<double>();
        panel_ = std::vector<double>();
        panelT_ = std::vector<double>();
        diag_ = std::vector<double>();
    }

    static uint64_t elems(uint32_t n) { return (uint64_t)n * (n + 1) / 2; }
    static uint64_t rowOff(uint32_t r) { return (uint64_t)r * (r + 1) / 2; }
    bool allocated() const { return !a_.empty(); }
    size_t bytes() const {
        return (a_.capacity() + panel_.capacity() + panelT_.capacity() + diag_.capacity()) * 8;
    }
    double* row(uint32_t r) { return a_.data() + rowOff(r); }
    const double* row(uint32_t r) const { return a_.data() + rowOff(r); }
    const std::vector<double>& data() const { return a_; }

    void zero(Pool& pool, int nthreads) {
        const uint64_t n = a_.size();
        const int nt = taskCount((int64_t)n, 1 << 22, nthreads);
        pool.run(nt, nthreads, [&](int t, int) {
            int64_t lo, hi;
            taskRange((int64_t)n, nt, t, lo, hi);
            if (hi > lo) std::memset(a_.data() + lo, 0, (size_t)(hi - lo) * sizeof(double));
        });
    }

    // 原地分解并求解 L L^T x = g，结果 x 写回 g。
    void factorSolve(double* g, Pool& pool, int nthreads) {
        factor(pool, nthreads);
        solveInPlace(g, pool, nthreads);
    }
    void solve(double* g, Pool& pool, int nthreads) { solveInPlace(g, pool, nthreads); }

    // 指定 diag 时，小于对应对角值 rel 比例的主元由该对角值替代，与粗矩阵 cholesky.slang 的保护一致。
    void factor(Pool& pool, int nthreads, const double* diag = nullptr, double rel = 0) {
        diag_in_ = diag;
        rel_ = rel;
        const uint32_t nb = (n_ + kBlock - 1) / kBlock;
        for (uint32_t k = 0; k < nb; k++) {
            const uint32_t base = k * kBlock;
            const uint32_t m = std::min(kBlock, n_ - base);
            factorDiag(base, m);
            const uint32_t rest = n_ - base - m;
            if (!rest) break;

            for (uint32_t r = 0; r < m; r++)
                std::memcpy(&diag_[(size_t)r * m], row(base + r) + base, (r + 1) * sizeof(double));

            const uint32_t rbase = base + m;
            {
                const int nt = taskCount(rest, 64, nthreads);
                pool.run(nt, nthreads, [&](int t, int) {
                    int64_t lo, hi;
                    taskRange(rest, nt, t, lo, hi);
                    for (int64_t i = lo; i < hi; i++) trsmRow(row((uint32_t)(rbase + i)) + base, m);
                });
            }

            const uint32_t tb = (rest + kBlock - 1) / kBlock;
            {
                const int nt = taskCount(rest, 64, nthreads);
                pool.run(nt, nthreads, [&](int t, int) {
                    int64_t lo, hi;
                    taskRange(rest, nt, t, lo, hi);
                    for (int64_t i = lo; i < hi; i++)
                        std::memcpy(&panel_[(size_t)i * m], row((uint32_t)(rbase + i)) + base,
                                    m * sizeof(double));
                });
                pool.run((int)tb, nthreads, [&](int j, int) {
                    const uint32_t c0 = (uint32_t)j * kBlock;
                    const uint32_t nc = std::min(kBlock, rest - c0);
                    double* bt = &panelT_[(size_t)j * kBlock * kBlock];
                    for (uint32_t mm = 0; mm < m; mm++)
                        for (uint32_t c = 0; c < nc; c++)
                            bt[(size_t)mm * kBlock + c] = panel_[(size_t)(c0 + c) * m + mm];
                });
            }

            const int ntiles = (int)(tb * (tb + 1) / 2);
            pool.run(ntiles, nthreads, [&](int t, int) {
                uint32_t i = (uint32_t)((std::sqrt(8.0 * t + 1.0) - 1.0) * 0.5);
                while ((uint64_t)(i + 1) * (i + 2) / 2 <= (uint64_t)t) i++;
                while ((uint64_t)i * (i + 1) / 2 > (uint64_t)t) i--;
                const uint32_t j = (uint32_t)(t - (int)(i * (i + 1) / 2));
                updateTile(rbase, i, j, m, rest);
            });
        }
    }

private:
    // 对角块的右看式分解，主元保护与 chol_diag 一致。
    void factorDiag(uint32_t base, uint32_t m) {
        for (uint32_t j = 0; j < m; j++) {
            double* Rj = row(base + j) + base;
            double d;
            if (!diag_in_)
                d = std::sqrt(Rj[j] > 1e-30 ? Rj[j] : 1e-30);
            else
                d = std::sqrt(Rj[j] > rel_ * diag_in_[base + j] ? Rj[j]
                                                                 : std::max(diag_in_[base + j], 1e-30));
            Rj[j] = d;
            const double inv = 1.0 / d;
            for (uint32_t r = j + 1; r < m; r++) {
                double* Rr = row(base + r) + base;
                double f = Rr[j] * inv;
                Rr[j] = f;
                for (uint32_t c = j + 1; c <= r; c++) Rr[c] -= f * row(base + c)[base + j];
            }
        }
    }

    // 计算 L_ik = A_ik L_kk^-T 的一行。
    void trsmRow(double* a, uint32_t m) {
        for (uint32_t j = 0; j < m; j++) {
            const double* Lj = &diag_[(size_t)j * m];
            double v = a[j];
            for (uint32_t mm = 0; mm < j; mm++) v -= a[mm] * Lj[mm];
            a[j] = v / Lj[j];
        }
    }

    // 对尾部子矩阵的一个块执行 A_ij -= L_ik L_jk^T。
    void updateTile(uint32_t rbase, uint32_t i, uint32_t j, uint32_t m, uint32_t rest) {
        const uint32_t r0 = i * kBlock, c0 = j * kBlock;
        const uint32_t mr = std::min(kBlock, rest - r0), nc = std::min(kBlock, rest - c0);
        const double* A = &panel_[(size_t)r0 * m];
        const double* Bt = &panelT_[(size_t)j * kBlock * kBlock];
        const uint32_t gi = rbase + r0, gj = rbase + c0;

        if (i == j) {  // 对称块仅处理下三角
            for (uint32_t r = 0; r < mr; r++) {
                double* C = row(gi + r) + gj;
                const double* Ar = A + (size_t)r * m;
                for (uint32_t c = 0; c <= r; c++) {
                    const double* Ac = A + (size_t)c * m;
                    double acc = 0;
                    for (uint32_t mm = 0; mm < m; mm++) acc += Ar[mm] * Ac[mm];
                    C[c] -= acc;
                }
            }
            return;
        }

        // 2×8 累加器恰好占满基线构建的 16 个 SSE2 寄存器；n=6102、32 线程时测得 132 GFLOP/s，优于 4×4 的 108 与 6×4 的 90。
        constexpr uint32_t MR = 2, NR = 8;
        uint32_t r = 0;
        for (; r + MR <= mr; r += MR) {
            const double* A0 = A + (size_t)r * m;
            uint32_t c = 0;
            for (; c + NR <= nc; c += NR) {
                double acc[MR][NR] = {};
                for (uint32_t mm = 0; mm < m; mm++) {
                    const double* b = Bt + (size_t)mm * kBlock + c;
                    for (uint32_t p = 0; p < MR; p++) {
                        const double a = A0[p * m + mm];
                        for (uint32_t q = 0; q < NR; q++) acc[p][q] += a * b[q];
                    }
                }
                for (uint32_t p = 0; p < MR; p++) {
                    double* C = row(gi + r + p) + gj;
                    for (uint32_t q = 0; q < NR; q++) C[c + q] -= acc[p][q];
                }
            }
            for (; c < nc; c++) {
                const double* Bc = &panel_[(size_t)(c0 + c) * m];
                for (uint32_t p = 0; p < MR; p++) {
                    double a = 0;
                    for (uint32_t mm = 0; mm < m; mm++) a += A0[p * m + mm] * Bc[mm];
                    row(gi + r + p)[gj + c] -= a;
                }
            }
        }
        for (; r < mr; r++) {
            const double* Ar = A + (size_t)r * m;
            double* C = row(gi + r) + gj;
            for (uint32_t c = 0; c < nc; c++) {
                const double* Bc = &panel_[(size_t)(c0 + c) * m];
                double acc = 0;
                for (uint32_t mm = 0; mm < m; mm++) acc += Ar[mm] * Bc[mm];
                C[c] -= acc;
            }
        }
    }

    void solveInPlace(double* g, Pool& pool, int nthreads) {
        const uint32_t nb = (n_ + kBlock - 1) / kBlock;
        for (uint32_t k = 0; k < nb; k++) {
            const uint32_t base = k * kBlock, m = std::min(kBlock, n_ - base);
            for (uint32_t r = 0; r < m; r++) {
                const double* L = row(base + r) + base;
                double v = g[base + r];
                for (uint32_t j = 0; j < r; j++) v -= L[j] * g[base + j];
                g[base + r] = v / L[r];
            }
            const uint32_t rest = n_ - base - m;
            if (!rest) continue;
            const int nt = taskCount(rest, 256, nthreads);
            pool.run(nt, nthreads, [&](int t, int) {
                int64_t lo, hi;
                taskRange(rest, nt, t, lo, hi);
                for (int64_t i = lo; i < hi; i++) {
                    const double* L = row((uint32_t)(base + m + i)) + base;
                    double acc = 0;
                    for (uint32_t j = 0; j < m; j++) acc += L[j] * g[base + j];
                    g[base + m + i] -= acc;
                }
            });
        }
        for (int kk = (int)nb - 1; kk >= 0; kk--) {
            const uint32_t base = (uint32_t)kk * kBlock, m = std::min(kBlock, n_ - base);
            for (int r = (int)m - 1; r >= 0; r--) {
                double v = g[base + r];
                for (uint32_t j = (uint32_t)r + 1; j < m; j++)
                    v -= row(base + j)[base + r] * g[base + j];
                g[base + r] = v / row(base + r)[base + r];
            }
            if (!base) continue;
            const int nt = taskCount(base, 256, nthreads);
            pool.run(nt, nthreads, [&](int t, int) {
                int64_t lo, hi;
                taskRange(base, nt, t, lo, hi);
                for (uint32_t j = 0; j < m; j++) {
                    const double* L = row(base + j);
                    const double x = g[base + j];
                    for (int64_t i = lo; i < hi; i++) g[i] -= L[i] * x;
                }
            });
        }
    }

    std::vector<double> a_, panel_, panelT_, diag_;
    const double* diag_in_ = nullptr;
    double rel_ = 0;
    uint32_t n_ = 0;
};

}  // 命名空间 bacpu
