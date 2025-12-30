/**
 * @file cg_cg.cpp
 * @brief Conjugate Gradient Solver with Internal Profiling
 */

#include "../include/cg_cg.h"
#include "../include/math_func.h"

#include <cstdio>
#include <cmath>
#include <chrono> // [新增]

// 辅助计时宏 (为了代码整洁)
#define TIC(start_var) auto start_var = std::chrono::high_resolution_clock::now()
#define TOC(start_var, accumulator) do { \
    auto end_var = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double, std::milli> dur = end_var - start_var; \
    accumulator += dur.count(); \
} while(0)


/*
 * ==========================================
 * 1. 线性系统对象 (System Objects)
 * ==========================================
 * MatA_ELLVal : ELL 格式矩阵数值数组 [N * Ell_Length]
 * MatA_ELLCol : ELL 格式列索引数组 [N * Ell_Length]
 * x           : 未知量向量 (压力)，在求解过程中原地更新 [N]
 * b           : 右端项向量 (源项) [N]
 *
 * ==========================================
 * 2. CG 工作向量 (Work Vectors)
 * ==========================================
 * Ax          : A*x 矩阵向量积缓存 (仅用于计算初始残差) [N]
 * r           : 残差向量, r = b - A*x [N]
 * p           : 搜索方向向量 [N]
 * Ap          : A*p 矩阵向量积缓存 (每次迭代更新) [N]
 *
 * ==========================================
 * 3. 标量系数 (Scalars)
 * ==========================================
 * alpha       : 步长系数, alpha = (r.r) / (p.Ap)
 * beta        : 方向更新系数, beta = (r_new.r_new) / (r_old.r_old)
 * rk          : 当前残差的模平方 (dot(r, r)) - 对应旧的 r
 * rk_new      : 更新后残差的模平方 (dot(r_new, r_new))
 * rk1         : 当前残差范数 ||r|| (即 sqrt(rk_new)), 用于收敛判断
 * r0          : 初始残差范数 ||r0||, 用于计算相对误差容限
 */
void cg_method_ell_normal( void ) {
    FLOAT alpha, beta;
    FLOAT r0, rk, rk1;
    FLOAT rk_new;

    // 1. 初始化部分 (算一次 SpMV 和一些向量操作)
    // 为了不破坏主循环的统计纯度，初始化部分的时间通常算作 setup 或忽略，
    // 但这里我们也简单计入对应类别
    
    TIC(t_init_spmv);
    mvm_ell_normal(MatA_ELLVal, MatA_ELLCol, x, Ax);
    TOC(t_init_spmv, time_spmv);

    TIC(t_init_vec);
    #pragma omp parallel for
    for (int i = 0; i < NumberOfParticles; i++) {
        r[i] = b[i] - Ax[i];
        p[i] = r[i];
    }
    TOC(t_init_vec, time_axpy);

    TIC(t_init_dot);
    rk = ip_omp(r, r);
    r0 = std::sqrt(rk);
    TOC(t_init_dot, time_dot);

    if (r0 < EPS) return; 

    // Main loop
    for (int k = 0; k < TMAX; k++) {

        // --- SpMV: Ap = A * p ---
        TIC(t_spmv);
        mvm_ell_normal(MatA_ELLVal, MatA_ELLCol, p, Ap);
        TOC(t_spmv, time_spmv);

        // --- Dot Product: alpha calc ---
        TIC(t_dot1);
        FLOAT pAp = ip_omp(p, Ap);
        TOC(t_dot1, time_dot);
        
        alpha = rk / pAp;

        // --- AXPY: x update, r update ---
        TIC(t_axpy1);
        #pragma omp parallel for
        for (int i = 0; i < NumberOfParticles; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        TOC(t_axpy1, time_axpy);

        // --- Dot Product: convergence check ---
        TIC(t_dot2);
        rk_new = ip_omp(r, r);
        rk1 = std::sqrt(rk_new);
        TOC(t_dot2, time_dot);

        if ((rk1 / r0) <= EPS) {
            cg_sum_loop_count_profile += (k + 1);
            // 这里去掉了原本的 printf Loop count，避免 benchmark 刷屏
            break;
        }

        // --- Beta Calculation (Scalar ops) ---
        beta = rk_new / rk;
        rk = rk_new; 

        // --- AXPY: p update ---
        TIC(t_axpy2);
        #pragma omp parallel for
        for (int i = 0; i < NumberOfParticles; i++) {
            p[i] = r[i] + beta * p[i];
        }
        TOC(t_axpy2, time_axpy);
    }
}