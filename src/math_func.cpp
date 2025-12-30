/**
 * @file math_func.cpp
 * @brief Mathematical utility functions (C++ Refactored)
 */

#include "../include/math_func.h"

#include <cmath>       // 替代 <math.h>，提供 std::abs, std::sqrt 等
#include <immintrin.h> // 保留 AVX512 头文件

// ==========================================
// 稠密矩阵计算 (Dense Matrix Operations)
// ==========================================

// 向量与矩阵相乘: y = A * x
void vector_x_matrix(FLOAT *y, FLOAT* a, FLOAT* x) {
    int n = NumberOfParticles;

    // j 在内层循环声明，OpenMP 会自动将其设为 private
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        FLOAT vxm = 0;
        for (int j = 0; j < n; j++) {
            vxm += a[i * n + j] * x[j];
        }
        y[i] = vxm;
    }
}

// ==========================================
// 向量基础运算 (Vector Operations)
// ==========================================

// 计算内积 (Dot Product)
// OpenMP 对应实现
FLOAT dot_product(FLOAT* x, FLOAT* y) {
    FLOAT dot_p = 0;
    
    #pragma omp parallel for reduction(+:dot_p)
    for (int i = 0; i < NumberOfParticles; i++) {
        dot_p += x[i] * y[i];
    }
    return dot_p;
}

// 计算向量的 L1 范数 (Vector Norm)
// sum(|x[i]|)
FLOAT vector_norm(FLOAT* x) {
    FLOAT norm = 0;

    #pragma omp parallel for reduction(+:norm)
    for (int i = 0; i < NumberOfParticles; i++) {
        // [关键修正] 使用 std::abs 替代宏 ABS
        // std::abs 会根据 x[i] 是 float 还是 double 自动调用正确的版本
        norm += std::abs(x[i]);
    }
    return norm;
}

// ip_omp: 与 dot_product 功能相同，保留以兼容接口
FLOAT ip_omp(FLOAT* x, FLOAT* y) {
    FLOAT dot_p = 0;

    #pragma omp parallel for reduction(+:dot_p)
    for (int i = 0; i < NumberOfParticles; i++) {
        dot_p += x[i] * y[i];
    }
    return dot_p;
}

// ==========================================
// ELL 稀疏矩阵计算 (Sparse Matrix ELL Format)
// ==========================================

// 稀疏矩阵向量乘法 (SpMV) - ELL 格式
// 适用于普通 ELL 格式 (MatA_ELLVal, MatA_ELLCol)
void mvm_ell_normal(FLOAT *mat_ellVal, int *mat_ellVCol, FLOAT *x, FLOAT *Ax) {
    int N = NumberOfParticles;

    // 1. 重置 Ax 向量
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        Ax[i] = 0;
    }

    // 2. 计算 A * x
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        // 遍历 ELL 矩阵的一行 (最大长度 Ell_Length)
        for (int j = 0; j < Ell_Length; j++) {
            // 获取列索引
            int xInd = mat_ellVCol[i * Ell_Length + j];

            // 如果遇到填充符 (ELL_NULL)，说明该行非零元素结束
            if (xInd == ELL_NULL) {
                break;
            }

            // 累加计算: Ax[i] += A[i][j] * x[col]
            Ax[i] += mat_ellVal[i * Ell_Length + j] * x[xInd];
        }
    }
}