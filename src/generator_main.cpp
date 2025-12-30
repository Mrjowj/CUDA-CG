/**
 * @file generator_main.cpp
 * @brief Standalone program to generate MPS-like sparse matrix data (Row-Major & Column-Major).
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
#include <filesystem> 
#include <omp.h> // 需要 OpenMP 头文件

#include "io_utils.h"                // 公共IO库
#include "../include/math_func.h"    // 需要用到 mvm 计算 b
#include "../include/extern_gloval_variables.h" // 宏定义等

namespace fs = std::filesystem;

// ==========================================
// 1. 全局变量定义 (为 math_func 提供存储空间)
// ==========================================
// 这些主要存储默认的 Row-Major 数据
FLOAT *MatA_ELLVal;
int *MatA_ELLCol;
FLOAT *x; 
FLOAT *b; 

// 占位符
FLOAT *p, *r, *Ax, *Ap; 
int NumberOfParticles;

// ==========================================
// 3. 内存管理
// ==========================================
void allocate_gen_memory(int N) {
    NumberOfParticles = N;
    
    // 分配 Row-Major 内存 (用于生成和 CPU 验证)
    MatA_ELLVal = new FLOAT[N * Ell_Length];
    MatA_ELLCol = new int[N * Ell_Length];
    
    x = new FLOAT[N];
    b = new FLOAT[N];
    
    std::fill(MatA_ELLVal, MatA_ELLVal + N * Ell_Length, 0.0);
    std::fill(MatA_ELLCol, MatA_ELLCol + N * Ell_Length, ELL_NULL);
}

void free_gen_memory() {
    delete[] MatA_ELLVal; delete[] MatA_ELLCol;
    delete[] x; delete[] b;
}

// ==========================================
// 4. 物理模拟矩阵生成器 (Row-Major)
// ==========================================
void generate_mps_like_matrix(int N, int avg_neighbors) {
    // ... (此处保持原本的生成逻辑不变，生成到 MatA_ELLVal/Col 中) ...
    // 为了节省篇幅，请保留你之前版本中完整的 generate_mps_like_matrix 代码
    // ... 核心逻辑省略 ...
    
    // 如果需要我把那部分代码再贴一遍请告诉我，
    // 这里假设逻辑与上一版完全一致，填充的是 MatA_ELLVal (Row-Major)
    
    // (简略重现关键部分以便编译通过)
    if (avg_neighbors > Ell_Length) exit(1);
    std::vector<std::map<int, FLOAT>> graph(N);
    std::mt19937 gen(12345);
    double std_dev = avg_neighbors / 3.0;
    std::normal_distribution<double> dist_neighbor_count((double)avg_neighbors, std_dev);
    std::uniform_real_distribution<FLOAT> dist_weight(0.1, 10.0);
    std::vector<bool> is_surface(N, false);
    std::uniform_real_distribution<double> dist_surface(0.0, 1.0);
    for(int i=0; i<N; i++) if(dist_surface(gen) < 0.1) is_surface[i] = true;
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    for (int i = 0; i < N; i++) {
        int target = (int)dist_neighbor_count(gen);
        if (target < 2) target = 2; if (target >= Ell_Length) target = Ell_Length - 1;
        int needed = target - (int)graph[i].size();
        for (int k = 0; k < needed; k++) {
            int j = dist_idx(gen);
            if (i == j || graph[i].count(j) || graph[j].size() >= (size_t)(Ell_Length - 1)) continue;
            FLOAT weight = -dist_weight(gen);
            graph[i][j] = weight; graph[j][i] = weight;
        }
    }
    for (int i = 0; i < N; i++) {
        if (is_surface[i]) graph[i][i] = 1.0e6;
        else {
            FLOAT sum_abs = 0.0;
            for (auto const& [col, val] : graph[i]) sum_abs += std::abs(val);
            graph[i][i] = sum_abs * (1.0 + 1.0e-7);
        }
    }
    std::fill(MatA_ELLVal, MatA_ELLVal + N * Ell_Length, 0.0);
    std::fill(MatA_ELLCol, MatA_ELLCol + N * Ell_Length, ELL_NULL);
    for (int i = 0; i < N; i++) {
        int col_idx = 0;
        for (auto const& [j, val] : graph[i]) {
            if (col_idx >= Ell_Length) break;
            MatA_ELLVal[i * Ell_Length + col_idx] = val;
            MatA_ELLCol[i * Ell_Length + col_idx] = j;
            col_idx++;
        }
    }
    printf("[Generator] Row-Major Matrix Generated.\n");
}

// ==========================================
// 5. 主程序
// ==========================================

int main() {
    // A. 配置
    int N = 200000;         
    int avg_neighbors = 64; 
    
    std::string output_dir = "data";
    std::string case_name  = "system_data";
    std::string prefix     = output_dir + "/" + case_name;

    // B. 环境准备
    try {
        if (!fs::exists(output_dir)) fs::create_directories(output_dir);
    } catch (const fs::filesystem_error& e) {
        fprintf(stderr, "Error creating directory: %s\n", e.what());
        return 1;
    }

    allocate_gen_memory(N);

    // ---------------------------------------------------------
    // C. 生成与计算
    // ---------------------------------------------------------
    
    // 1. 生成 Row-Major 矩阵 (默认存放在全局变量 MatA_ELLVal 中)
    generate_mps_like_matrix(N, avg_neighbors);

    // 2. 生成 Column-Major 矩阵 (转换)
    printf("[Generator] Converting to Column-Major format (for GPU)...\n");
    std::vector<FLOAT> val_cm; // cm = Column Major
    std::vector<int>   col_cm;
    
    convert_ell_to_column_major_cpp(
        MatA_ELLVal, MatA_ELLCol, 
        val_cm, col_cm, 
        N, Ell_Length
    );

    // 3. 生成 x_true 和计算 b
    printf("[Generator] Calculating RHS...\n");
    std::mt19937 gen(time(NULL));
    std::uniform_real_distribution<FLOAT> dist(-10.0, 10.0);
    for(int i=0; i<N; i++) x[i] = dist(gen);

    // 计算 b = A * x (使用 Row-Major 版的 CPU 函数计算)
    FLOAT* temp_Ax = new FLOAT[N];
    mvm_ell_normal(MatA_ELLVal, MatA_ELLCol, x, temp_Ax);
    for(int i=0; i<N; i++) b[i] = temp_Ax[i];
    delete[] temp_Ax;

    // ---------------------------------------------------------
    // D. 保存文件 (两套格式)
    // ---------------------------------------------------------
    printf("[Generator] Saving files to %s ...\n", output_dir.c_str());
    
    // === 保存 Row-Major (CPU / Legacy 兼容) ===
    write_binary_file((prefix + "_A_val.bin").c_str(), MatA_ELLVal, N * Ell_Length);
    write_binary_file((prefix + "_A_col.bin").c_str(), MatA_ELLCol, N * Ell_Length);

    // === 保存 Column-Major (GPU 优化) ===
    // 命名惯例：加上 _cm 后缀
    write_binary_file((prefix + "_A_val_cm.bin").c_str(), val_cm.data(), N * Ell_Length);
    write_binary_file((prefix + "_A_col_cm.bin").c_str(), col_cm.data(), N * Ell_Length);

    // === 保存向量与元数据 ===
    write_binary_file((prefix + "_b.bin").c_str(), b, N);
    write_binary_file((prefix + "_x_true.bin").c_str(), x, N);

    FILE* f_meta = fopen((prefix + "_meta.txt").c_str(), "w");
    if (f_meta) {
        fprintf(f_meta, "%d", N);
        fclose(f_meta);
    }

    printf("Data generation complete. N=%d\n", N);
    free_gen_memory();
    return 0;
}