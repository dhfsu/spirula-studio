#pragma once
// 编译时恰好选择 REAL_FLOAT/REAL_DOUBLE/REAL_DF，下游统一使用 Real。
// AWord 为同缓冲的原子视图元素类型，atomicAccum 执行无锁累加，load/store 为普通访问；rexp/rlog/rsin/rcos 提供可微超越函数。

#include "df.slang"
#include "dmath.slang"

#if defined(REAL_FLOAT)

typealias Real = float;
typealias AWord = float;

void atomicAccum(RWStructuredBuffer<AWord> buf, uint i, Real v) {
    __atomic_add(buf[i], v, MemoryOrder.Relaxed);
}
Real loadAccum(RWStructuredBuffer<AWord> buf, uint i) { return buf[i]; }
void storeAccum(RWStructuredBuffer<AWord> buf, uint i, Real v) { buf[i] = v; }

[Differentiable] Real rexp(Real x) { return exp(x); }
[Differentiable] Real rlog(Real x) { return log(x); }
[Differentiable] Real rsin(Real x) { return sin(x); }
[Differentiable] Real rcos(Real x) { return cos(x); }
void rsincos_raw(Real x, out Real s, out Real c) { s = sin(x); c = cos(x); }
float rToFloat(Real x) { return x; }

#elif defined(REAL_DOUBLE)

typealias Real = double;
typealias AWord = double;

void atomicAccum(RWStructuredBuffer<AWord> buf, uint i, Real v) {
    __atomic_add(buf[i], v, MemoryOrder.Relaxed);
}
Real loadAccum(RWStructuredBuffer<AWord> buf, uint i) { return buf[i]; }
void storeAccum(RWStructuredBuffer<AWord> buf, uint i, Real v) { buf[i] = v; }

[Differentiable] Real rexp(Real x) { return dbl_exp(x); }
[Differentiable] Real rlog(Real x) { return dbl_log(x); }
[Differentiable] Real rsin(Real x) { return dbl_sin(x); }
[Differentiable] Real rcos(Real x) { return dbl_cos(x); }
void rsincos_raw(Real x, out Real s, out Real c) { dbl_sincos(x, s, c); }
float rToFloat(Real x) { return float(x); }

#elif defined(REAL_DF)

typealias Real = DF;
typealias AWord = uint64_t;

// DF 布局为 {hi,lo} 共八字节，小端 uint64 视图的低字存 hi，使两个视图字节完全一致。
uint64_t packReal(Real v) { return uint64_t(asuint(v.hi)) | (uint64_t(asuint(v.lo)) << 32); }
Real unpackReal(uint64_t w) { return DF(asfloat(uint(w)), asfloat(uint(w >> 32))); }

void atomicAccum(RWStructuredBuffer<AWord> buf, uint i, Real v) {
    uint64_t old_w = buf[i];
    for (;;) {
        uint64_t new_w = packReal(unpackReal(old_w) + v);
        uint64_t got = __atomic_compare_exchange(buf[i], old_w, new_w,
            MemoryOrder.Relaxed, MemoryOrder.Relaxed);
        if (got == old_w) break;
        old_w = got;
    }
}
Real loadAccum(RWStructuredBuffer<AWord> buf, uint i) { return unpackReal(buf[i]); }
void storeAccum(RWStructuredBuffer<AWord> buf, uint i, Real v) { buf[i] = packReal(v); }

[Differentiable] Real rexp(Real x) { return df_exp(x); }
[Differentiable] Real rsin(Real x) { return df_sin(x); }
[Differentiable] Real rcos(Real x) { return df_cos(x); }
void rsincos_raw(Real x, out Real s, out Real c) { df_sincos_impl(x, s, c); }
[BackwardDerivative(df_rlog_bwd)]
Real rlog(Real x) {
    // 以 float log 为初值，经 df_exp 做一次 Newton 精化
    DF y = DF(log(x.hi));
    DF u = df_mul(x, df_exp(df_neg(y))) - DF(1.0f);
    return y + u - DF(0.5f) * (u * u);
}
void df_rlog_bwd(inout DifferentialPair<DF> x, DF g) {
    x = diffPair(x.p, df_div(g, x.p));
}
float rToFloat(Real x) { return x.hi; }

#else
#error "define one of REAL_FLOAT / REAL_DOUBLE / REAL_DF"
#endif

// r>=0 的 atan2(r,z) 返回 [0,pi]；fp32 初值后对 g(t)=z sin(t)-r cos(t)=0 做 Newton，达到当前 Real 精度。
// 自定义反向直接使用 dtheta/dr=z/(r²+z²)、dtheta/dz=-r/(r²+z²)，不对迭代过程求导。
[BackwardDerivative(rAtan2_bwd)]
Real rAtan2(Real r, Real z) {
    Real theta = Real(atan2(rToFloat(r), rToFloat(z)));
    [ForceUnroll]
    for (int i = 0; i < 3; i++) {
        Real s, c;
        rsincos_raw(theta, s, c);
        theta = theta - (z * s - r * c) / (z * c + r * s);
    }
    return theta;
}
void rAtan2_bwd(inout DifferentialPair<Real> r, inout DifferentialPair<Real> z, Real g) {
    Real d = r.p * r.p + z.p * z.p;
    r = diffPair(r.p, g * z.p / d);
    z = diffPair(z.p, -(g * r.p) / d);
}
