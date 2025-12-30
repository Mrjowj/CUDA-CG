/**
 * @file solver_main.cpp
 * @brief CG Solver Main Program (Reads data, solves, and profiles)
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "io_utils.h"                // 公共IO库
#include "../include/cg_cg.h"        // CG 求解器
#include "../include/math_func.h"
#include "../include/extern_gloval_variables.h"

// ==========================================
// 1. 全局变量定义 (核心求解器变量)
// ==========================================
FLOAT *MatA_ELLVal;
int *MatA_ELLCol;
FLOAT *x;   // 解向量
FLOAT *b;   // 右端项
FLOAT *p, *r, *Ax, *Ap; // CG工作向量

int NumberOfParticles; // 粒子数/矩阵维度 N

// ==========================================
// 2. Profiling 计时变量 (对应 cg_cg.cpp 的 extern)
// ==========================================
// 这些变量会在 CG 内部被累加，单位：毫秒
double time_spmv = 0.0;   // 稀疏矩阵乘法耗时 (Ap = A*p)
double time_dot  = 0.0;   // 内积/范数耗时 (dot product)
double time_axpy = 0.0;   // 向量更新耗时 (x+=alpha*p, ...)

// 统计迭代次数
int cg_sum_loop_count_profile = 0;
int iTimeStep_gloval = 0; // 控制 CG 内部 printf 的开关

// ==========================================
// 4. 内存管理
// ==========================================
void allocate_solver_memory(int N) {
    NumberOfParticles = N;
    
    MatA_ELLVal = new FLOAT[N * Ell_Length];
    MatA_ELLCol = new int[N * Ell_Length];
    x  = new FLOAT[N];
    b  = new FLOAT[N];
    p  = new FLOAT[N];
    r  = new FLOAT[N];
    Ax = new FLOAT[N];
    Ap = new FLOAT[N];

    // 初始化解向量 x 为 0 (初始猜测)
    std::fill(x, x + N, 0.0);
}

void free_solver_memory() {
    delete[] MatA_ELLVal; delete[] MatA_ELLCol;
    delete[] x; delete[] b;
    delete[] p; delete[] r; delete[] Ax; delete[] Ap;
}

// ==========================================
// 5. 主函数
// ==========================================
int main() {
    // 数据文件路径
    std::string prefix = "data/system_data"; 

    // --- A. 读取元数据 ---
    FILE* f_meta = fopen((prefix + "_meta.txt").c_str(), "r");
    if (!f_meta) {
        fprintf(stderr, "Error: Meta file not found at '%s_meta.txt'.\n", prefix.c_str());
        fprintf(stderr, "Hint: Did you run the generator? Check the path.\n");
        return 1;
    }
    int N = 0;
    fscanf(f_meta, "%d", &N);
    fclose(f_meta);

    printf("Initialize Solver for N = %d...\n", N);

    // --- B. 分配内存并读取数据 ---
    allocate_solver_memory(N);

    read_binary_file((prefix + "_A_val.bin").c_str(), MatA_ELLVal, N * Ell_Length);
    read_binary_file((prefix + "_A_col.bin").c_str(), MatA_ELLCol, N * Ell_Length);
    read_binary_file((prefix + "_b.bin").c_str(), b, N);

    // (可选) 读取真值用于验证
    std::vector<FLOAT> x_true(N);
    read_binary_file((prefix + "_x_true.bin").c_str(), x_true.data(), N);

    // --- C. 性能测试 (Benchmark) ---
    printf("\n=== Starting CG Solver (Benchmarking 50 runs) ===\n");
    
    const int NUM_RUNS = 50;
    double total_wall_time_ms = 0.0;

    // [Step 1] 重置所有统计变量
    cg_sum_loop_count_profile = 0;
    time_spmv = 0.0;
    time_dot  = 0.0;
    time_axpy = 0.0;
    
    // [Step 2] 关闭 CG 内部的 Debug 输出 (避免刷屏)
    iTimeStep_gloval = 0; 

    for (int run = 0; run < NUM_RUNS; run++) {
        // [关键] 每次运行前必须重置 x 为初始猜测 (0.0)
        std::fill(x, x + NumberOfParticles, 0.0);

        // --- 单次 Wall Clock 计时开始 ---
        auto start_time = std::chrono::high_resolution_clock::now();

        // 调用求解器 (计时逻辑在 cg_cg.cpp 内部也会累积 time_spmv 等)
        cg_method_ell_normal(); 

        // --- 单次 Wall Clock 计时结束 ---
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        total_wall_time_ms += elapsed.count();
        
        // 进度条
        if ((run + 1) % 10 == 0) {
            printf("[Benchmark] Progress: %d / %d runs completed...\n", run + 1, NUM_RUNS);
        }
    }

    // --- D. 计算统计数据并输出报告 ---
    
    // 1. 计算平均值
    double avg_wall_time = total_wall_time_ms / NUM_RUNS;
    double avg_iters     = (double)cg_sum_loop_count_profile / NUM_RUNS;
    
    double avg_spmv = time_spmv / NUM_RUNS;
    double avg_dot  = time_dot  / NUM_RUNS;
    double avg_axpy = time_axpy / NUM_RUNS;
    
    // 2. 计算内部统计的总和 (用于计算 Overhead)
    double sum_internal = avg_spmv + avg_dot + avg_axpy;
    double overhead     = avg_wall_time - sum_internal;
    // 修正微小的计时误差导致的负 overhead
    if (overhead < 0) overhead = 0.0; 

    printf("\n======================================================\n");
    printf("   CG Solver Performance Profile (Avg over %d runs)   \n", NUM_RUNS);
    printf("======================================================\n");
    printf(" Total Wall Time    : %10.4f ms  (100.0%%)\n", avg_wall_time);
    printf("------------------------------------------------------\n");
    printf(" Breakdown:\n");
    printf("  > SpMV (Matrix*v) : %10.4f ms  (%5.1f%%)\n", avg_spmv, (avg_spmv / avg_wall_time) * 100.0);
    printf("  > Dot Product     : %10.4f ms  (%5.1f%%)\n", avg_dot,  (avg_dot  / avg_wall_time) * 100.0);
    printf("  > Vector Update   : %10.4f ms  (%5.1f%%)\n", avg_axpy, (avg_axpy / avg_wall_time) * 100.0);
    printf("  > Other/Overhead  : %10.4f ms  (%5.1f%%)\n", overhead, (overhead / avg_wall_time) * 100.0);
    printf("------------------------------------------------------\n");
    printf(" Avg Iterations     : %10.2f\n", avg_iters);
    printf(" Time per Iteration : %10.4f ms\n", avg_wall_time / avg_iters);
    printf("======================================================\n");

    // --- E. 验证结果 ---
    double error_sq = 0.0;
    double true_sq = 0.0;
    for(int i=0; i<N; i++) {
        double diff = x[i] - x_true[i];
        error_sq += diff * diff;
        true_sq += x_true[i] * x_true[i];
    }
    double rel_err = std::sqrt(error_sq) / std::sqrt(true_sq);

    printf("\nVerification Result:\n");
    printf("Relative Error: %e\n", rel_err);
    if (rel_err < 1e-4) printf("Status: CONVERGED [SUCCESS]\n");
    else printf("Status: FAILED / DIVERGED\n");

    free_solver_memory();
    return 0;
}