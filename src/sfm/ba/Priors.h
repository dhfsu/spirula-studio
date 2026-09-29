// BA 的相机侧传感器先验：相对旋转、重力方向和相机中心线性约束。
// 先验数量为 O(frames)，少于 O(observations) 重投影，因此在主机求值，再以稀疏 6×6 帧块交给任一求解器。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "sfm/ba/Problem.h"
#include "sfm/core/Pose.h"

namespace sfm {

// 两图像相机坐标间满足 R_j ~ R_ji R_i，R 为世界到相机旋转。
struct PriorRotation {
    uint32_t i = 0, j = 0;
    Mat3 R_ji = mat3Identity();
    double sigma = 0.01;   // 弧度
};

// R_i up_w ~ u，即世界向上轴在图像 i 相机坐标系中的方向。
struct PriorUp {
    uint32_t i = 0;
    Vec3 u{0, -1, 0};
    double sigma = 0.05;   // 弧度
};

// 相机中心约束 sum_k A_k c_{img_k} ~ b，各轴按 1/sigma 加权，0 表示忽略；一项为绝对位置，两项为位移，三项为无速度惯性约束。
struct PriorCentre {
    int n = 0;
    uint32_t img[3] = {0, 0, 0};
    Mat3 A[3] = {mat3Identity(), mat3Identity(), mat3Identity()};
    Vec3 b;
    Vec3 sigma{1, 1, 1};
};

// 建图器使用重建图像 ID，buildBundle 将其映射为 BA 索引。
struct PosePriors {
    Vec3 up_w{0, 0, 1};    // 全部 PriorUp 共用的世界向上轴
    double huber = 1.345;  // 每因子的阈值，以标准差为单位
    std::vector<PriorRotation> rotations;
    std::vector<PriorUp> ups;
    std::vector<PriorCentre> centres;
    bool empty() const { return rotations.empty() && ups.empty() && centres.empty(); }
    size_t size() const { return rotations.size() + ups.size() + centres.size(); }
};

// 先验正规方程以 6×6 帧块 CSR 保存，双向块均列出；[6a+b] 将 x[6 col+b] 映射到 y[6 row+a]，并附 pose_dim 梯度。
class PriorAssembler {
public:
    void init(const BAProblem& P) {
        const PosePriors* pr = P.priors;
        nfact_ = 0;
        fact_.clear();
        rows_.clear();
        cols_.clear();
        erow_.clear();
        blk_.clear();
        g_.clear();
        nframes_ = P.num_frames;
        if (!pr || pr->empty()) return;
        auto frameOf = [&](uint32_t img) { return P.image_frame[img]; };
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> id;
        auto note = [&](Fact& f) {
            for (int a = 0; a < f.nf; a++)
                for (int b = 0; b < f.nf; b++)
                    id.emplace(std::make_pair(f.frame[a], f.frame[b]), 0u);
        };
        for (const PriorRotation& r : pr->rotations) {
            if (r.i >= P.num_images || r.j >= P.num_images || !(r.sigma > 0)) continue;
            Fact f;
            f.kind = 0;
            f.index = (uint32_t)(&r - pr->rotations.data());
            f.addFrame(frameOf(r.i));
            f.addFrame(frameOf(r.j));
            note(f);
            fact_.push_back(f);
        }
        for (const PriorUp& u : pr->ups) {
            if (u.i >= P.num_images || !(u.sigma > 0)) continue;
            Fact f;
            f.kind = 1;
            f.index = (uint32_t)(&u - pr->ups.data());
            f.addFrame(frameOf(u.i));
            note(f);
            fact_.push_back(f);
        }
        for (const PriorCentre& c : pr->centres) {
            bool ok = c.n >= 1 && c.n <= 3;
            for (int k = 0; ok && k < c.n; k++) ok = c.img[k] < P.num_images;
            if (!ok) continue;
            Fact f;
            f.kind = 2;
            f.index = (uint32_t)(&c - pr->centres.data());
            for (int k = 0; k < c.n; k++) f.addFrame(frameOf(c.img[k]));
            note(f);
            fact_.push_back(f);
        }
        nfact_ = fact_.size();
        // std::map 按 (row, col) 顺序生成 CSR。
        rows_.assign(nframes_ + 1, 0);
        uint32_t e = 0;
        for (auto& kv : id) {
            kv.second = e++;
            rows_[kv.first.first + 1]++;
            cols_.push_back(kv.first.second);
            erow_.push_back(kv.first.first);
        }
        for (uint32_t f = 0; f < nframes_; f++) rows_[f + 1] += rows_[f];
        for (Fact& f : fact_)
            for (int a = 0; a < f.nf; a++)
                for (int b = 0; b < f.nf; b++)
                    f.entry[a][b] = id.at({f.frame[a], f.frame[b]});
        blk_.assign(36 * (size_t)e, 0.0);
        g_.assign(6 * (size_t)nframes_, 0.0);
    }

    bool empty() const { return nfact_ == 0; }
    uint32_t numEntries() const { return (uint32_t)cols_.size(); }
    const std::vector<uint32_t>& rows() const { return rows_; }
    const std::vector<uint32_t>& cols() const { return cols_; }
    const std::vector<uint32_t>& entryRow() const { return erow_; }
    const std::vector<double>& blocks() const { return blk_; }
    const std::vector<double>& gradient() const { return g_; }

    // 给定参数下的 0.5 sum rho(|r|^2)，每帧 6 个位姿参数、每成员 6 个外参。
    double cost(const BAProblem& P, const double* poses, const double* exts) const {
        double c = 0;
        Eval ev;
        for (const Fact& f : fact_) {
            evaluate(P, poses, exts, f, ev);
            c += 0.5 * robustCost(ev.r);
        }
        return c;
    }

    // 计算块与梯度，对角块按 (1 + damping) 阻尼，与观测内核一致，返回代价。
    double assemble(const BAProblem& P, const double* poses, const double* exts, double damping) {
        std::fill(blk_.begin(), blk_.end(), 0.0);
        std::fill(g_.begin(), g_.end(), 0.0);
        double c = 0;
        Eval ev;
        for (const Fact& f : fact_) {
            evaluate(P, poses, exts, f, ev);
            const double s = ev.r[0] * ev.r[0] + ev.r[1] * ev.r[1] + ev.r[2] * ev.r[2];
            c += 0.5 * robustCost(ev.r);
            const double w = robustWeight(s);
            for (int a = 0; a < f.nf; a++) {
                double* ga = &g_[6 * (size_t)f.frame[a]];
                for (int p = 0; p < 6; p++) {
                    double v = 0;
                    for (int m = 0; m < 3; m++) v += ev.J[a][m][p] * ev.r[m];
                    ga[p] += w * v;
                }
                for (int b = 0; b < f.nf; b++) {
                    double* B = &blk_[36 * (size_t)f.entry[a][b]];
                    for (int p = 0; p < 6; p++)
                        for (int q = 0; q < 6; q++) {
                            double v = 0;
                            for (int m = 0; m < 3; m++) v += ev.J[a][m][p] * ev.J[b][m][q];
                            B[6 * p + q] += w * v;
                        }
                }
            }
        }
        if (damping != 0)
            for (uint32_t e = 0; e < cols_.size(); e++)
                if (cols_[e] == erow_[e])
                    for (int p = 0; p < 6; p++) blk_[36 * (size_t)e + 7 * p] *= 1.0 + damping;
        return c;
    }

    // 供测试读取单因子的残差与雅可比；frames 为 J 各列对应的帧槽位。
    int debugFactor(const BAProblem& P, const double* poses, const double* exts, size_t k,
                    double r[3], double J[3][3][6], uint32_t frames[3]) const {
        Eval ev;
        evaluate(P, poses, exts, fact_[k], ev);
        for (int m = 0; m < 3; m++) r[m] = ev.r[m];
        for (int a = 0; a < 3; a++)
            for (int m = 0; m < 3; m++)
                for (int q = 0; q < 6; q++) J[a][m][q] = ev.J[a][m][q];
        for (int a = 0; a < 3; a++) frames[a] = fact_[k].frame[a];
        return fact_[k].nf;
    }
    size_t numFactors() const { return nfact_; }

private:
    struct Fact {
        int kind = 0;      // 0 为旋转，1 为向上方向，2 为中心
        uint32_t index = 0;
        int nf = 0;
        uint32_t frame[3] = {0, 0, 0};
        uint32_t entry[3][3] = {};
        void addFrame(uint32_t f) {
            for (int k = 0; k < nf; k++)
                if (frame[k] == f) return;
            frame[nf++] = f;
        }
        int slot(uint32_t f) const {
            for (int k = 0; k < nf; k++)
                if (frame[k] == f) return k;
            return -1;
        }
    };
    struct Eval {
        double r[3];
        double J[3][3][6];   // 每帧槽位为 3 × [轴角 3 | 平移 3]
    };

    // 图像的变换链由帧块与成员外参组成。
    struct Cam {
        uint32_t frame = 0;
        Mat3 Rf, Rm, Rc;   // 帧、成员与相机旋转：camera = Rm Rf
        Vec3 tf, tm;
        Mat3 Jl;           // 帧轴角的左雅可比
    };

    static Cam camOf(const BAProblem& P, const double* poses, const double* exts, uint32_t img) {
        Cam c;
        c.frame = P.image_frame[img];
        const double* q = poses + 6 * (size_t)c.frame;
        const Vec3 aa{q[0], q[1], q[2]};
        c.Rf = angleAxisToRotation(aa);
        c.tf = {q[3], q[4], q[5]};
        c.Jl = so3LeftJacobian(aa);
        const uint32_t m = P.image_member[img];
        if (m == kNoMember) {
            c.Rm = mat3Identity();
            c.tm = {0, 0, 0};
            c.Rc = c.Rf;
        } else {
            const double* e = exts + P.members[m].ext_offset;
            c.Rm = angleAxisToRotation({e[0], e[1], e[2]});
            c.tm = {e[3], e[4], e[5]};
            c.Rc = mul(c.Rm, c.Rf);
        }
        return c;
    }

    // 将 d r / d delta_cam（行数 × 3）映射到帧轴角列：delta_cam = Rm delta_f，delta_f = Jl(a) da。
    static void rotCols(const Mat3& D, const Cam& c, double scale, double J[3][6]) {
        const Mat3 M = mul(mul(D, c.Rm), c.Jl);
        for (int m = 0; m < 3; m++)
            for (int p = 0; p < 3; p++) J[m][p] += scale * M[3 * m + p];
    }

    double robustCost(const double r[3]) const {
        const double s = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        const double k2 = huber_ * huber_;
        return s <= k2 ? s : 2.0 * huber_ * std::sqrt(s) - k2;
    }
    double robustWeight(double s) const {
        const double k2 = huber_ * huber_;
        return s <= k2 ? 1.0 : huber_ / std::sqrt(s);
    }

    void evaluate(const BAProblem& P, const double* poses, const double* exts, const Fact& f,
                  Eval& ev) const {
        const PosePriors& pr = *P.priors;
        huber_ = pr.huber;
        for (int a = 0; a < 3; a++)
            for (int m = 0; m < 3; m++)
                for (int p = 0; p < 6; p++) ev.J[a][m][p] = 0;
        if (f.kind == 0) {
            const PriorRotation& q = pr.rotations[f.index];
            const Cam ci = camOf(P, poses, exts, q.i), cj = camOf(P, poses, exts, q.j);
            const Mat3 E = mul(mul(cj.Rc, transpose(ci.Rc)), transpose(q.R_ji));
            const Vec3 e = so3Log(E);
            const double inv = 1.0 / q.sigma;
            ev.r[0] = e.x * inv;
            ev.r[1] = e.y * inv;
            ev.r[2] = e.z * inv;
            // Exp(dj) E = Exp(e + Jl^-1(e) dj); E Exp(-R di) = Exp(e - Jr^-1(e) R di).
            rotCols(so3LeftJacobianInv(e), cj, inv, ev.J[f.slot(cj.frame)]);
            rotCols(mul(so3RightJacobianInv(e), q.R_ji), ci, -inv, ev.J[f.slot(ci.frame)]);
            return;
        }
        if (f.kind == 1) {
            const PriorUp& q = pr.ups[f.index];
            const Cam c = camOf(P, poses, exts, q.i);
            const Vec3 g = mul(c.Rc, pr.up_w);
            const double inv = 1.0 / q.sigma;
            ev.r[0] = (g.x - q.u.x) * inv;
            ev.r[1] = (g.y - q.u.y) * inv;
            ev.r[2] = (g.z - q.u.z) * inv;
            // Exp(d) g = g + d x g = g - [g]x d.
            rotCols(crossMatrix(g), c, -inv, ev.J[f.slot(c.frame)]);
            return;
        }
        const PriorCentre& q = pr.centres[f.index];
        Vec3 sum{0, 0, 0};
        for (int k = 0; k < q.n; k++) {
            const Cam c = camOf(P, poses, exts, q.img[k]);
            // c=-Rf^T a，其中 a=tf+Rm^T tm；dc/dt=-Rf^T，dc/dd=-Rf^T[a]x。
            const Vec3 a = c.tf + mul(transpose(c.Rm), c.tm);
            const Vec3 centre = mul(transpose(c.Rf), a) * -1.0;
            sum = sum + mul(q.A[k], centre);
            const Mat3 dt = mul(q.A[k], transpose(c.Rf));
            const Mat3 dd = mul(mul(dt, crossMatrix(a)), c.Jl);
            double (*J)[6] = ev.J[f.slot(c.frame)];
            for (int m = 0; m < 3; m++) {
                const double sg = (&q.sigma.x)[m];
                const double inv = sg > 0 ? 1.0 / sg : 0.0;
                for (int p = 0; p < 3; p++) {
                    J[m][p] += -inv * dd[3 * m + p];
                    J[m][3 + p] += -inv * dt[3 * m + p];
                }
            }
        }
        const Vec3 d = sum - q.b;
        for (int m = 0; m < 3; m++) {
            const double sg = (&q.sigma.x)[m];
            ev.r[m] = sg > 0 ? (&d.x)[m] / sg : 0.0;
        }
    }

    std::vector<Fact> fact_;
    size_t nfact_ = 0;
    uint32_t nframes_ = 0;
    mutable double huber_ = 1.345;
    std::vector<uint32_t> rows_, cols_, erow_;
    std::vector<double> blk_, g_;
};

}  // 命名空间 sfm
