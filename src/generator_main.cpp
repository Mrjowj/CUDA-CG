/**
 * @file generator_main.cpp
 * @brief Standalone program to generate MPS-like sparse matrix data (Physically-based).
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <random>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <string>
#include <filesystem> // C++17 标准库，用于自动创建文件夹

#include "io_utils.h"                // 公共IO库
#include "../include/math_func.h"    // 需要用到 mvm 计算 b
#include "../include/extern_gloval_variables.h" // 宏定义等

namespace fs = std::filesystem;

// ==========================================
// 1. 全局变量定义 (为 math_func 提供存储空间)
// ==========================================
FLOAT *MatA_ELLVal;
int *MatA_ELLCol;
FLOAT *x; // 这里临时用来存 x_true
FLOAT *b; // 临时用来存计算出的 b

// 以下变量为占位符，为了满足链接器对 math_func 或 cg_cg 的依赖
FLOAT *p, *r, *Ax, *Ap; 
int NumberOfParticles;
// 其他未使用的 extern 变量可暂时忽略，如果链接报错请补充定义

// ==========================================
// 2. 内存管理
// ==========================================
void allocate_gen_memory(int N) {
    NumberOfParticles = N;
    
    // 分配 ELL 矩阵内存
    MatA_ELLVal = new FLOAT[N * Ell_Length];
    MatA_ELLCol = new int[N * Ell_Length];
    
    // 分配向量内存
    x = new FLOAT[N];
    b = new FLOAT[N];
    
    // 初始化清零
    std::fill(MatA_ELLVal, MatA_ELLVal + N * Ell_Length, 0.0);
    std::fill(MatA_ELLCol, MatA_ELLCol + N * Ell_Length, ELL_NULL);
}

void free_gen_memory() {
    delete[] MatA_ELLVal; delete[] MatA_ELLCol;
    delete[] x; delete[] b;
}

// ==========================================
// 3. 物理模拟矩阵生成器
// ==========================================

/**
 * @brief 生成模拟 MPS 压力方程的矩阵 (Hard Mode)
 * @param N 粒子数
 * @param avg_neighbors 平均每个粒子的邻居数 (控制稀疏度的核心参数)
 * MPS中通常为 20~40。值越小，矩阵越稀疏。
 */
void generate_mps_like_matrix(int N, int avg_neighbors) {
    // 检查：平均邻居数不能超过物理显存限制
    if (avg_neighbors > Ell_Length) {
        printf("Error: avg_neighbors (%d) > Ell_Length (%d)\n", avg_neighbors, Ell_Length);
        exit(1);
    }

    std::vector<std::map<int, FLOAT>> graph(N);
    std::mt19937 gen(12345);
    
    // === 核心修改 ===
    // 使用 avg_neighbors 作为正态分布的均值
    // 标准差设为均值的 1/3，模拟邻居数量的自然波动
    double std_dev = avg_neighbors / 3.0;
    std::normal_distribution<double> dist_neighbor_count((double)avg_neighbors, std_dev);
    
    // 模拟权重 (1/r^2)，生成正数，稍后取负
    std::uniform_real_distribution<FLOAT> dist_weight(0.1, 10.0);

    // 自由表面比例 (约 10%)
    std::vector<bool> is_surface(N, false);
    std::uniform_real_distribution<double> dist_surface(0.0, 1.0);
    for(int i=0; i<N; i++) {
        if(dist_surface(gen) < 0.1) is_surface[i] = true;
    }

    printf("[Generator] MPS Topology: Avg Neighbors = %d (StdDev=%.1f)\n", avg_neighbors, std_dev);

    // 1. 构建连接 (Topology)
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    
    for (int i = 0; i < N; i++) {
        // 确定该行的期望邻居数
        int target = (int)dist_neighbor_count(gen);
        
        // 物理限制：至少 2 个邻居，至多不超过 Ell_Length - 1
        if (target < 2) target = 2;
        if (target >= Ell_Length) target = Ell_Length - 1;

        int current = graph[i].size();
        int needed = target - current;

        for (int k = 0; k < needed; k++) {
            int j = dist_idx(gen);
            
            // 避免自环、重复、超限
            if (i == j) continue;
            if (graph[i].count(j)) continue;
            if (graph[j].size() >= (size_t)(Ell_Length - 1)) continue;

            FLOAT weight = -dist_weight(gen); // 负权重 (Laplacian 特性)
            graph[i][j] = weight;
            graph[j][i] = weight; // 保持对称性
        }
    }

    // 2. 对角线处理 (Diagonal & BC)
    for (int i = 0; i < N; i++) {
        if (is_surface[i]) {
            FLOAT sum_abs = 0.0;
            for (auto const& [col, val] : graph[i]) sum_abs += std::abs(val);
            graph[i][i] = sum_abs + 1.0; 
        } else {
            // 内部流体：弱对角占优 (Near-Singular)
            FLOAT sum_abs = 0.0;
            for (auto const& [col, val] : graph[i]) sum_abs += std::abs(val);
            graph[i][i] = sum_abs * (1.0 + 1.0e-7); 
        }
    }

    // 3. 转换为 ELL 格式
    long long total_non_zeros = 0; // 统计一下实际非零元

    for (int i = 0; i < N; i++) {
        int col_idx = 0;
        for (auto const& [j, val] : graph[i]) {
            if (col_idx >= Ell_Length) break;
            MatA_ELLVal[i * Ell_Length + col_idx] = val;
            MatA_ELLCol[i * Ell_Length + col_idx] = j;
            col_idx++;
        }
        total_non_zeros += col_idx;
    }

    // 计算并打印实际的 "Blank Ratio"
    double actual_fill_rate = (double)total_non_zeros / (double)(N * Ell_Length);
    printf("[Generator] Actual Fill Rate: %.2f%% (Equivalent Blank Ratio: %.2f)\n", 
           actual_fill_rate * 100.0, 1.0 - actual_fill_rate);
}

// ==========================================
// 4. 主程序
// ==========================================

int main() {
    // ---------------------------------------------------------
    // A. 配置参数
    // ---------------------------------------------------------
    int N = 200000;         // 粒子数
    int avg_neighbors = 64; // 平均邻居数 (替代了 blank_ratio)
    
    std::string output_dir = "../data";
    std::string case_name  = "system_data";
    std::string prefix     = output_dir + "/" + case_name;

    // ---------------------------------------------------------
    // B. 环境准备
    // ---------------------------------------------------------
    // 自动创建输出目录
    try {
        if (!fs::exists(output_dir)) {
            if (fs::create_directories(output_dir)) {
                printf("[Info] Created directory: %s\n", output_dir.c_str());
            }
        }
    } catch (const fs::filesystem_error& e) {
        fprintf(stderr, "Error creating directory: %s\n", e.what());
        return 1;
    }

    // 分配内存
    allocate_gen_memory(N);

    // ---------------------------------------------------------
    // C. 生成数据
    // ---------------------------------------------------------
    // 1. 生成模拟 MPS 的稀疏矩阵 A
    generate_mps_like_matrix(N, avg_neighbors);

    // 2. 生成随机真解 x_true
    printf("[Generator] Generating True Solution...\n");
    std::mt19937 gen(time(NULL));
    std::uniform_real_distribution<FLOAT> dist(-10.0, 10.0);
    for(int i=0; i<N; i++) x[i] = dist(gen);

    // 3. 计算右端项 b = A * x_true
    // 创建一个临时数组存放结果，因为 mvm_ell_normal 的输出参数是独立的指针
    printf("[Generator] Calculating RHS (b = A * x)...\n");
    FLOAT* temp_Ax = new FLOAT[N];
    
    // 调用 math_func 中的矩阵乘法
    mvm_ell_normal(MatA_ELLVal, MatA_ELLCol, x, temp_Ax);
    
    // 将结果复制到全局变量 b
    for(int i=0; i<N; i++) b[i] = temp_Ax[i];
    delete[] temp_Ax;

    // ---------------------------------------------------------
    // D. 保存文件
    // ---------------------------------------------------------
    printf("[Generator] Saving files to %s_*.bin ...\n", prefix.c_str());
    
    write_binary_file((prefix + "_A_val.bin").c_str(), MatA_ELLVal, N * Ell_Length);
    write_binary_file((prefix + "_A_col.bin").c_str(), MatA_ELLCol, N * Ell_Length);
    write_binary_file((prefix + "_b.bin").c_str(), b, N);
    write_binary_file((prefix + "_x_true.bin").c_str(), x, N);

    // 保存元数据 (粒子数 N) 供读取器使用
    FILE* f_meta = fopen((prefix + "_meta.txt").c_str(), "w");
    if (f_meta) {
        fprintf(f_meta, "%d", N);
        fclose(f_meta);
    } else {
        perror("Failed to write meta file");
    }

    printf("Data generation complete. N=%d\n", N);

    // ---------------------------------------------------------
    // E. 清理
    // ---------------------------------------------------------
    free_gen_memory();
    return 0;
}