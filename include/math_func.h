#ifndef MATH_FUNC
#define MATH_FUNC

#include "extern_gloval_variables.h"
#include <vector>
#include <omp.h>    // OpenMP 支持
#include <cassert>  // 用于调试断言

// static inline double weight(double distance, double re);
// 汎用的な計算に用いる関数群を定義する
static inline FLOAT weight( FLOAT distance, FLOAT re )
{
    if( distance >= re )
    {
        return 0.0;
    }
    else
    {
        return (re/distance) - 1.0;
    }
}

void vector_x_matrix(FLOAT *y, FLOAT* a, FLOAT* x);
FLOAT dot_product(FLOAT* x, FLOAT* y);
FLOAT vector_norm(FLOAT* x);
FLOAT ip_omp(FLOAT* x, FLOAT* y);

void mvm_ell_normal(FLOAT *mat_ellVal, int *mat_ellVCol, FLOAT  *x, FLOAT  *Ax);

/**
 * @brief C++ 模板函数：将 ELL 矩阵从行主序转置为列主序 (Row-Major -> Column-Major)
 * * 注意：由于这是一个模板函数，为了避免链接错误，实现必须放在头文件中。
 * * @tparam T 数据类型 (float 或 double)
 * @param src_val       [输入] 行主序的数值指针 (MatA_ELLVal)
 * @param src_col       [输入] 行主序的列索引指针 (MatA_ELLCol)
 * @param dst_val       [输出] 列主序的数值容器 (std::vector, 会自动 resize)
 * @param dst_col       [输出] 列主序的列索引容器 (std::vector, 会自动 resize)
 * @param num_particles [输入] 粒子总数 (N)
 * @param ell_length    [输入] 最大邻居数 (Ell_Length)
 */
template <typename T>
void convert_ell_to_column_major_cpp(
    const T* __restrict__ src_val, 
    const int* __restrict__ src_col,
    std::vector<T>& dst_val, 
    std::vector<int>& dst_col,
    const int num_particles, 
    const int ell_length
) {
    // 1. 预分配内存 (避免多次 realloc)
    size_t total_size = static_cast<size_t>(num_particles) * ell_length;
    
    // 如果大小不够，进行 resize (包含初始化开销，但多线程写入下最安全)
    if (dst_val.size() != total_size) dst_val.resize(total_size);
    if (dst_col.size() != total_size) dst_col.resize(total_size);

    // 获取 vector 内部数据的裸指针，以便并行快速访问
    T* d_val_ptr = dst_val.data();
    int* d_col_ptr = dst_col.data();

    // 2. 并行转置
    // collapse(2) 将两层循环合并为一个任务空间，增加并行度
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_particles; ++i) {
        for (int j = 0; j < ell_length; ++j) {
            // 源索引 (Row-Major): [i * K + j]
            // CPU 原始格式: 粒子 i 的所有邻居在一起
            int src_idx = i * ell_length + j;

            // 目标索引 (Column-Major): [j * N + i]
            // GPU 优化格式: 所有粒子的 "第 j 个邻居" 在一起 (Coalesced Access)
            int dst_idx = j * num_particles + i;

            d_val_ptr[dst_idx] = src_val[src_idx];
            d_col_ptr[dst_idx] = src_col[src_idx];
        }
    }
}

#endif /* MATH_FUNC */
