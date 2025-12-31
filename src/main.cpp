/**
 * @file main.cpp
 * @brief Unified CG Solver Benchmark (CPU & Full GPU)
 * @details 支持一键切换 CPU/GPU 模式，自动加载对应格式的数据。
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
#include "../include/cg_cg.h"        // CPU 求解器接口
#include "../include/kernel.h"      // GPU 求解器接口
#include "../include/extern_gloval_variables.h"

// ==========================================
// 0. 配置区域 (一键切换)
// ==========================================
// true = 使用 Full GPU 求解器 (Column-Major)
// false = 使用 CPU OpenMP 求解器 (Row-Major)
const bool USE_GPU = true; 

// ==========================================
// 1. 全局变量 (CPU 求解器必须)
// ==========================================
// 即便跑 GPU，为了编译通过 cg_cg.cpp，这些变量也必须定义
FLOAT *MatA_ELLVal;
int *MatA_ELLCol;
FLOAT *x;   
FLOAT *b;   
FLOAT *p, *r, *Ax, *Ap; 
int NumberOfParticles;

// Profiling 变量
double time_spmv = 0.0, time_dot = 0.0, time_axpy = 0.0;
int cg_sum_loop_count_profile = 0;
int iTimeStep_gloval = 0; 
// 其他占位符
FLOAT Re_forNumberDensity, Re2_forNumberDensity, Re_forGradient, Re2_forGradient, Re_forLaplacian, Re2_forLaplacian, N0_forNumberDensity, N0_forGradient, N0_forLaplacian, Lambda, collisionDistance, collisionDistance2, FluidDensity;
FLOAT *Acceleration, *MinimumPressure, *Position, *Velocity, *NumberDensity, *VelocityAfterCollision;
int *normalELLCol_diagonalIndex, *Bucket_ParticleNum, profile_sparse_mat_sum_non_zeros, profile_sparse_rate_time_step, ell_max_row_width_debug, cg_iter_total;
double profile_sparse_rate_sum, Time;

// ==========================================
// 2. 内存管理 (CPU 模式专用)
// ==========================================
void allocate_cpu_memory(int N) {
    MatA_ELLVal = new FLOAT[N * Ell_Length];
    MatA_ELLCol = new int[N * Ell_Length];
    x  = new FLOAT[N];
    b  = new FLOAT[N];
    p  = new FLOAT[N];
    r  = new FLOAT[N];
    Ax = new FLOAT[N];
    Ap = new FLOAT[N];
}

void free_cpu_memory() {
    delete[] MatA_ELLVal; delete[] MatA_ELLCol;
    delete[] x; delete[] b;
    delete[] p; delete[] r; delete[] Ax; delete[] Ap;
}

// ==========================================
// 3. 主程序
// ==========================================
int main() {
    std::string prefix = "data/system_data";
    
    // --- Step A: 读取元数据 ---
    FILE* f_meta = fopen((prefix + "_meta.txt").c_str(), "r");
    if (!f_meta) {
        printf("Error: Meta file not found. Run generator first!\n");
        return 1;
    }
    int N = 0;
    fscanf(f_meta, "%d", &N);
    fclose(f_meta);
    NumberOfParticles = N; // 设置全局变量，供 CPU 求解器使用

    printf("\n==================================================\n");
    printf("  CG Solver Benchmark | Mode: %s | N = %d\n", USE_GPU ? "FULL GPU" : "CPU OpenMP", N);
    printf("==================================================\n");

    // 用于存放最终结果以便验证
    std::vector<FLOAT> h_x_result(N);
    // 用于存放右端项 b (GPU 初始化需要)
    std::vector<FLOAT> h_b(N);

    // GPU 句柄
    CgCudaHandle gpu_handle = nullptr;

    // --- Step B: 数据加载与初始化 ---
    if (USE_GPU) {
        // === GPU 模式初始化 ===
        printf("[Init] Loading Column-Major Data for GPU...\n");
        std::vector<FLOAT> val_cm(N * Ell_Length);
        std::vector<int>   col_cm(N * Ell_Length);
        
        read_binary_file((prefix + "_A_val_cm.bin").c_str(), val_cm.data(), N * Ell_Length);
        read_binary_file((prefix + "_A_col_cm.bin").c_str(), col_cm.data(), N * Ell_Length);
        read_binary_file((prefix + "_b.bin").c_str(), h_b.data(), N); // 读取 b 到局部向量

        printf("[Init] Uploading data to GPU...\n");
        gpu_handle = cg_cuda_init(N, Ell_Length, val_cm.data(), col_cm.data(), h_b.data());

    } else {
        // === CPU 模式初始化 ===
        printf("[Init] Allocating CPU Memory & Loading Row-Major Data...\n");
        allocate_cpu_memory(N);
        
        read_binary_file((prefix + "_A_val.bin").c_str(), MatA_ELLVal, N * Ell_Length);
        read_binary_file((prefix + "_A_col.bin").c_str(), MatA_ELLCol, N * Ell_Length);
        read_binary_file((prefix + "_b.bin").c_str(), b, N); // 读取 b 到全局变量
    }

    // 读取真值 (用于验证)
    std::vector<FLOAT> x_true(N);
    read_binary_file((prefix + "_x_true.bin").c_str(), x_true.data(), N);

    // --- Step C: Benchmark 循环 ---
    const int NUM_RUNS = 50;
    double total_ms = 0.0;
    int total_iters = 0;

    printf("\n[Benchmark] Starting %d runs...\n", NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
        
        // 1. 重置解向量 (Reset)
        if (USE_GPU) {
            // GPU 模式下，每次 solve 前，GPU 显存内的 x 需要清零。
            // 注意：这依赖于你已经在 src/cg_cuda.cu 的 cg_cuda_solve 开头添加了 cudaMemset(d_x, 0)
            // 如果没加，请务必加上！否则只有第1次是正确的。
        } else {
            // CPU 模式：手动清零全局 x 数组
            std::fill(x, x + N, 0.0);
            // 重置 Profiling 计数器
            cg_sum_loop_count_profile = 0; 
        }

        // 2. 计时开始
        auto t1 = std::chrono::high_resolution_clock::now();

        // 3. 执行求解
        int iters = 0;
        if (USE_GPU) {
            // Full GPU Solve
            iters = cg_cuda_solve(gpu_handle, h_x_result.data(), 300, EPS);
        } else {
            // CPU Solve
            cg_method_ell_normal(); // 结果存在全局变量 x 中
            iters = (cg_sum_loop_count_profile > 0) ? cg_sum_loop_count_profile : 300; 
        }

        // 4. 计时结束
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t2 - t1;
        total_ms += ms.count();
        total_iters += iters;

        if ((run + 1) % 5 == 0) printf("   > Completed %d / 50 runs...\n", run + 1);
    }

    // --- Step D: 结果输出与验证 ---
    printf("\n[Result] Average Time: %.4f ms\n", total_ms / NUM_RUNS);
    printf("[Result] Average Iter: %.2f\n", (double)total_iters / NUM_RUNS);

    // 提取结果用于验证
    if (!USE_GPU) {
        // 如果是 CPU 模式，把全局 x 拷贝到 h_x_result 以便统一验证
        for(int i=0; i<N; i++) h_x_result[i] = x[i];
    }

    // 计算误差
    double err_sq = 0, true_sq = 0;
    for(int i=0; i<N; ++i) {
        double diff = h_x_result[i] - x_true[i];
        err_sq += diff*diff;
        true_sq += x_true[i]*x_true[i];
    }
    double rel_err = std::sqrt(err_sq/true_sq);

    printf("[Verify] Relative Error: %e ", rel_err);
    if (rel_err < 1e-4) printf("[PASS]\n");
    else printf("[FAIL]\n");

    // --- Step E: 资源清理 ---
    if (USE_GPU) {
        cg_cuda_free(gpu_handle);
    } else {
        free_cpu_memory();
    }

    return 0;
}