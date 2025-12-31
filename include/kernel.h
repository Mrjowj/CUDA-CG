/**
 * @file cg_cuda.h
 * @brief Header for Full-GPU Conjugate Gradient Solver
 */
#ifndef CG_CUDA_H
#define CG_CUDA_H

#include "extern_gloval_variables.h"

// 不透明句柄，隐藏 GPU 内部复杂的显存指针
typedef void* CgCudaHandle;

/**
 * @brief 初始化全 GPU CG 求解器环境
 * * 1. 分配所有需要的显存 (A, b, x, r, p, Ax, Ap, buffer)
 * 2. 将矩阵 A (Col-Major) 和向量 b 传输到 GPU
 * 3. 将解向量 x 初始化为 0
 * * @param N           粒子数 (矩阵行数)
 * @param ell_width   ELL 矩阵宽度
 * @param h_val_cm    Host端 矩阵数值 (Column-Major)
 * @param h_col_cm    Host端 矩阵索引 (Column-Major)
 * @param h_b         Host端 右端项向量 b
 * @return CgCudaHandle 上下文句柄
 */
CgCudaHandle cg_cuda_init(int N, int ell_width, 
                          const FLOAT* h_val_cm, 
                          const int* h_col_cm, 
                          const FLOAT* h_b);

/**
 * @brief 执行 CG 求解 (Full GPU Resident Loop)
 * * 数据流：
 * - 初始：x, b, A 已经在显存中
 * - 循环：所有迭代计算 (SpMV, Dot, AXPY) 均在 GPU 完成，
 * CPU 仅负责 Kernel Launch 和读取标量 (alpha, beta, residual)
 * - 结束：将最终解 x 传回 Host
 * * @param handle 句柄
 * @param h_x_out [输出] 存放最终解的 Host 数组
 * @param max_iter 最大迭代次数
 * @param tol      收敛容差
 * @return int     实际迭代次数
 */
int cg_cuda_solve(CgCudaHandle handle, FLOAT* h_x_out, int max_iter, FLOAT tol);

/**
 * @brief 释放 GPU 资源
 */
void cg_cuda_free(CgCudaHandle handle);

#endif // CG_CUDA_H