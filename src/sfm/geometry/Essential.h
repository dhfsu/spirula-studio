// 由 F 与内参通过 E=K2^T F K1 构造本质矩阵，分解相对位姿并按正深度选择解；输入为针孔像素与 3×3 内参。
#pragma once

#include <cmath>
#include <vector>

#include "sfm/geometry/LinAlg.h"
#include "sfm/geometry/Triangulation.h"

namespace sfm {

struct Pose {
    Mat3 R = mat3Identity();  // 世界到相机的旋转
    Vec3 t;                   // 世界到相机的平移，相对位姿采用单位尺度
};

inline Mat3 essentialFromFundamental(const Mat3& F, const Mat3& K1, const Mat3& K2) {
    return mul(mul(transpose(K2), F), K1);
}

// 将近似正交矩阵投影为 det=+1 的正旋转。
inline Mat3 nearestRotation(const Mat3& R) {
    Svd3 s = svd3(R);
    Mat3 out = mul(s.U, transpose(s.V));
    if (det3(out) < 0)
        for (int c = 0; c < 3; c++) { out[c] = -out[c]; out[3 + c] = -out[3 + c]; out[6 + c] = -out[6 + c]; }
    return out;
}

// 由本质矩阵得到四组 (R,t) 候选。
inline std::vector<Pose> decomposeEssential(const Mat3& E) {
    Svd3 s = svd3(E);
    Mat3 U = s.U, V = s.V;
    if (det3(U) < 0) for (int c = 0; c < 3; c++) { U[c] = -U[c]; U[3 + c] = -U[3 + c]; U[6 + c] = -U[6 + c]; }
    if (det3(V) < 0) for (int c = 0; c < 3; c++) { V[c] = -V[c]; V[3 + c] = -V[3 + c]; V[6 + c] = -V[6 + c]; }
    Mat3 W = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    Mat3 R1 = mul(mul(U, W), transpose(V));
    Mat3 R2 = mul(mul(U, transpose(W)), transpose(V));
    Vec3 t = {U[2], U[5], U[8]};  // U 的第三列
    return {{R1, t}, {R1, {-t.x, -t.y, -t.z}}, {R2, t}, {R2, {-t.x, -t.y, -t.z}}};
}

inline Mat34 poseToP(const Pose& p) {
    return {p.R[0], p.R[1], p.R[2], p.t.x, p.R[3], p.R[4], p.R[5], p.t.y,
            p.R[6], p.R[7], p.R[8], p.t.z};
}

// 选择双相机前方对应最多的位姿；n1/n2 已乘 K^-1，返回有效点数与掩码。
inline Pose recoverRelativePose(const std::vector<Vec2>& n1, const std::vector<Vec2>& n2,
                                const std::vector<int>& idx, const Mat3& E, int& bestCount,
                                std::vector<char>* mask = nullptr) {
    Mat34 P1 = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    Pose best;
    bestCount = -1;
    for (const Pose& cand : decomposeEssential(E)) {
        Mat34 P2 = poseToP(cand);
        int count = 0;
        std::vector<char> m(idx.size(), 0);
        for (size_t k = 0; k < idx.size(); k++) {
            int i = idx[k];
            // 将 z=1 归一化坐标提升为单位视线，交给统一三角化器。
            Vec3 b1 = Vec3{n1[i].x, n1[i].y, 1.0}.normalized();
            Vec3 b2 = Vec3{n2[i].x, n2[i].y, 1.0}.normalized();
            Vec3 X = triangulateDLT(P1, P2, b1, b2);
            double z1 = X.z;
            Vec3 Xc2 = mul(cand.R, X) + cand.t;
            if (z1 > 0 && Xc2.z > 0) { count++; m[k] = 1; }
        }
        if (count > bestCount) { bestCount = count; best = cand; if (mask) *mask = m; }
    }
    return best;
}

// 单位视线版本按四种候选投票，正深度由 dot(X_cam,b)>0 判断；鱼眼侧向或后向观测不能简单采用 z>0（D45）。
inline Pose recoverRelativePoseBearing(const std::vector<Vec3>& b1, const std::vector<Vec3>& b2,
                                       const std::vector<int>& idx, const Mat3& E, int& bestCount,
                                       std::vector<char>* mask = nullptr) {
    Mat34 P1 = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    Pose best;
    bestCount = -1;
    for (const Pose& cand : decomposeEssential(E)) {
        Mat34 P2 = poseToP(cand);
        int count = 0;
        std::vector<char> m(idx.size(), 0);
        for (size_t k = 0; k < idx.size(); k++) {
            int i = idx[k];
            Vec3 X = triangulateDLT(P1, P2, b1[i], b2[i]);
            Vec3 Xc2 = mul(cand.R, X) + cand.t;
            if (X.dot(b1[i]) > 0 && Xc2.dot(b2[i]) > 0) { count++; m[k] = 1; }
        }
        if (count > bestCount) { bestCount = count; best = cand; if (mask) *mask = m; }
    }
    return best;
}

}  // 命名空间 sfm
