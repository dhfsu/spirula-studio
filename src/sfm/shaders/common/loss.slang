#pragma once
// 稳健损失总成本为 sum(0.5*rho(s))，s 为残差范数平方，残差与雅可比乘 sqrt(rho'(s))，不做 Triggs 修正。
// 损失类型编译期选择，Huber delta 或 Cauchy c 通过推送常量逐问题设置。

#include "real.slang"

interface ILoss {
    static Real cost(Real s, float param);
    static Real weight(Real s, float param); // rho'(s)
};

struct TrivialLoss : ILoss {
    static Real cost(Real s, float param) { return s; }
    static Real weight(Real s, float param) { return Real(1.0); }
};

struct HuberLoss : ILoss {
    // s<=d² 时 rho(s)=s，否则 rho(s)=2d sqrt(s)-d²。
    static Real cost(Real s, float param) {
        Real d2 = Real(param) * Real(param);
        if (s <= d2) return s;
        return Real(2.0) * Real(param) * sqrt(s) - d2;
    }
    static Real weight(Real s, float param) {
        Real d2 = Real(param) * Real(param);
        if (s <= d2) return Real(1.0);
        return Real(param) / sqrt(s);
    }
};

struct CauchyLoss : ILoss {
    // rho(s) = c^2 log(1 + s/c^2);  rho'(s) = 1 / (1 + s/c^2)
    static Real weight(Real s, float param) {
        Real c2 = Real(param) * Real(param);
        return Real(1.0) / (Real(1.0) + s / c2);
    }
    static Real cost(Real s, float param) {
        Real c2 = Real(param) * Real(param);
        return c2 * rlog(Real(1.0) + s / c2);
    }
};

#if defined(LOSS_HUBER)
typealias Loss = HuberLoss;
#elif defined(LOSS_CAUCHY)
typealias Loss = CauchyLoss;
#else
typealias Loss = TrivialLoss;
#endif
